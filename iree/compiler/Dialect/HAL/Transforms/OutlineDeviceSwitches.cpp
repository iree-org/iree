// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Given a #hal.match.* expression tree returns a boolean value indicating
// whether the expression evaluates to true.
static Value buildConditionExpression(Location loc, Value device,
                                      Attribute conditionAttr,
                                      OpBuilder funcBuilder) {
  if (auto matchAttr = conditionAttr.dyn_cast<IREE::HAL::MatchAlwaysAttr>()) {
    // #hal.match.always -> true
    return funcBuilder.createOrFold<ConstantIntOp>(loc, 1, 1);
  } else if (auto matchAttr =
                 conditionAttr.dyn_cast<IREE::HAL::MatchAnyAttr>()) {
    // #hal.match.any<[a, b, c]> -> or(or(a, b), c)
    auto conditionAttrs = matchAttr.conditions().cast<ArrayAttr>();
    auto conditionValues =
        llvm::to_vector<4>(llvm::map_range(conditionAttrs, [&](Attribute attr) {
          return buildConditionExpression(loc, device, attr, funcBuilder);
        }));
    Value resultValue = conditionValues[0];
    for (int i = 1; i < conditionValues.size(); ++i) {
      resultValue =
          funcBuilder.createOrFold<OrOp>(loc, resultValue, conditionValues[i]);
    }
    return resultValue;
  } else if (auto matchAttr =
                 conditionAttr.dyn_cast<IREE::HAL::MatchAllAttr>()) {
    // #hal.match.all<[a, b, c]> -> and(and(a, b), c)
    auto conditionAttrs = matchAttr.conditions().cast<ArrayAttr>();
    auto conditionValues =
        llvm::to_vector<4>(llvm::map_range(conditionAttrs, [&](Attribute attr) {
          return buildConditionExpression(loc, device, attr, funcBuilder);
        }));
    Value resultValue = conditionValues[0];
    for (int i = 1; i < conditionValues.size(); ++i) {
      resultValue =
          funcBuilder.createOrFold<AndOp>(loc, resultValue, conditionValues[i]);
    }
    return resultValue;
  } else if (auto matchAttr =
                 conditionAttr.dyn_cast<IREE::HAL::DeviceMatchIDAttr>()) {
    // #hal.device.match.id<"pattern"> -> hal.device.match.id
    return funcBuilder.createOrFold<IREE::HAL::DeviceMatchIDOp>(
        loc, funcBuilder.getI1Type(), device, matchAttr.patternAttr());
  }
  llvm_unreachable("unhandled condition expression attribute");
  return {};
}

// Outlines a condition region from a switch op into a standalone function.
static FuncOp outlineConditionRegion(StringRef funcName,
                                     Region &conditionRegion,
                                     ArrayRef<Type> resultTypes,
                                     OpBuilder moduleBuilder) {
  auto &entryBlock = conditionRegion.front();
  auto funcType = moduleBuilder.getFunctionType(
      llvm::to_vector<4>(
          llvm::map_range(entryBlock.getArguments(),
                          [](BlockArgument arg) { return arg.getType(); })),
      resultTypes);
  auto funcOp = moduleBuilder.create<FuncOp>(
      conditionRegion.getLoc(), funcName, funcType, ArrayRef<NamedAttribute>{});
  funcOp.getBody().takeBody(conditionRegion);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  // Replace hal.return statements with normal std.return. This ensures that
  // normal matchers/inlining/etc works as we continue transformation.
  for (auto &block : funcOp.getBlocks()) {
    if (auto returnOp = dyn_cast<IREE::HAL::ReturnOp>(block.back())) {
      OpBuilder builder(returnOp);
      builder.create<mlir::ReturnOp>(
          returnOp.getLoc(), llvm::to_vector<4>(returnOp.getOperands()));
      returnOp.erase();
    }
  }

  return funcOp;
}

// Outlines each switch condition region into its own function and replaces the
// switch op with conditioned calls to those functions.
//
// Since switch conditions are evaluated in the order they are defined we can
// trivially turn the switch into a chain of if-else blocks.
//   if condition_0_match:
//     call outlined_condition_0
//   else
//     if condition_1_match:
//       call outlined_condition_1
//     else ...
static void buildConditionDispatchTable(IREE::HAL::DeviceSwitchOp switchOp,
                                        StringRef baseFuncName,
                                        OpBuilder moduleBuilder,
                                        OpBuilder funcBuilder) {
  // Split the block containing the switch op such that all ops before the
  // switch are before and the switch and the following ops are after.
  // We'll have all of our outlined regions bounce over to the afterBlock with
  // the results of the call and use that to replace the switch op.
  auto *beforeBlock = funcBuilder.getBlock();
  auto *afterBlock = beforeBlock->splitBlock(switchOp);
  auto finalValues =
      llvm::to_vector<4>(afterBlock->addArguments(switchOp.getResultTypes()));

  // Create the blocks we'll use for all our conditions so that we can reference
  // them when inserting the branch ops.
  SmallVector<Block *, 4> conditionMatchBlocks(
      switchOp.condition_regions().size());
  SmallVector<Block *, 4> conditionFallthroughBlocks(
      switchOp.condition_regions().size());
  for (int i = 0; i < conditionMatchBlocks.size(); ++i) {
    conditionMatchBlocks[i] = funcBuilder.createBlock(afterBlock);
    conditionFallthroughBlocks[i] = funcBuilder.createBlock(afterBlock);
  }

  funcBuilder.setInsertionPoint(beforeBlock, beforeBlock->end());
  int argOffset = 0;
  for (auto condition : llvm::enumerate(llvm::zip(
           switchOp.conditions().getValue(), switchOp.condition_regions()))) {
    auto conditionAttr = std::get<0>(condition.value());
    auto &conditionRegion = std::get<1>(condition.value());

    // Get the arguments from the switch that we want to carry along in the
    // block arguments.
    auto regionOperands = conditionRegion.front().getArguments();
    auto regionArgs = switchOp.args().slice(argOffset, regionOperands.size());
    argOffset += regionOperands.size();

    // Outline the region into a function.
    std::string regionFuncName =
        (baseFuncName + "_").str() + std::to_string(condition.index());
    auto regionFuncOp = outlineConditionRegion(
        regionFuncName, conditionRegion,
        switchOp.getOperation()->getResultTypes(), moduleBuilder);

    // Insert the branch based on the match. We either match and jump to a block
    // that will call the function or don't match and need to fall through.
    auto isMatch = buildConditionExpression(
        switchOp.getLoc(), switchOp.device(), conditionAttr, funcBuilder);
    auto *matchBlock = conditionMatchBlocks[condition.index()];
    auto *fallthroughBlock = conditionFallthroughBlocks[condition.index()];
    funcBuilder.create<CondBranchOp>(switchOp.getLoc(), isMatch, matchBlock,
                                     fallthroughBlock);

    // Block that calls the outlined function and then jumps out of the chain.
    funcBuilder.setInsertionPointToStart(matchBlock);
    auto matchResults =
        funcBuilder.create<CallOp>(switchOp.getLoc(), regionFuncOp, regionArgs);
    funcBuilder.create<BranchOp>(switchOp.getLoc(), afterBlock,
                                 matchResults.getResults());

    // Block that we enter to check the next condition.
    funcBuilder.setInsertionPointToStart(fallthroughBlock);
    if (condition.index() + 1 < conditionFallthroughBlocks.size()) {
      // Just continue on - the next loop iteration for the following condition
      // will add its IR to the block.
    } else {
      // Fallthrough of all expressions; die if we expected return values.
      if (switchOp.getNumResults() > 0) {
        funcBuilder.create<IREE::UnreachableOp>(switchOp.getLoc());
      } else {
        funcBuilder.create<BranchOp>(switchOp.getLoc(), afterBlock);
      }
    }
  }

  // Remove the switch op and replace its results with the final joined results.
  switchOp.replaceAllUsesWith(finalValues);
  switchOp.erase();
}

class OutlineDeviceSwitchesPass
    : public PassWrapper<OutlineDeviceSwitchesPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto funcOps = llvm::to_vector<16>(moduleOp.getOps<FuncOp>());
    for (auto &funcOp : funcOps) {
      OpBuilder moduleBuilder(funcOp);
      moduleBuilder.setInsertionPointAfter(funcOp);
      for (auto &block : funcOp) {
        auto switchOps =
            llvm::to_vector<4>(block.getOps<IREE::HAL::DeviceSwitchOp>());
        for (auto switchOp : llvm::enumerate(switchOps)) {
          std::string baseFuncName = (funcOp.getName() + "_switch_").str() +
                                     std::to_string(switchOp.index());
          OpBuilder funcBuilder(switchOp.value());
          buildConditionDispatchTable(switchOp.value(), baseFuncName,
                                      moduleBuilder, funcBuilder);
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createOutlineDeviceSwitchesPass() {
  return std::make_unique<OutlineDeviceSwitchesPass>();
}

static PassRegistration<OutlineDeviceSwitchesPass> pass(
    "iree-hal-outline-device-switches",
    "Outlines hal.device.switch condition regions");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
