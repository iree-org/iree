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
#include "mlir/IR/BlockAndValueMapping.h"
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

// Inlines a condition region from a switch op into the function at the given
// point. This assumes that the insertion point will only be reached if the
// condition the region is predicated on is true.
static void inlineConditionRegion(OperandRange regionArgs,
                                  Region &conditionRegion, Block *exitBlock,
                                  OpBuilder funcBuilder) {
  assert(!conditionRegion.empty() && "source regions must not be empty");

  // Remap arguments from the function values captured by the switch into the
  // entry block arguments for the region.
  auto &entryBlock = conditionRegion.front();
  assert(regionArgs.size() == entryBlock.getNumArguments() &&
         "switch capture args must match region args");
  for (auto argPair : llvm::zip(regionArgs, entryBlock.getArguments())) {
    auto outerValue = std::get<0>(argPair);
    auto innerValue = std::get<1>(argPair);
    assert(outerValue.getType() == innerValue.getType() &&
           "capture arg types must match");
    innerValue.replaceAllUsesWith(outerValue);
  }

  // Splice in the region blocks.
  auto *insertBlock = funcBuilder.getBlock();
  auto postInsertBlockIt = std::next(insertBlock->getIterator())->getIterator();
  auto *insertRegion = insertBlock->getParent();
  insertRegion->getBlocks().splice(postInsertBlockIt,
                                   conditionRegion.getBlocks());
  auto newBlocks = llvm::make_range(std::next(insertBlock->getIterator()),
                                    postInsertBlockIt);
  auto *firstNewBlock = &*newBlocks.begin();

  // Handle the hal.return ops which will transfer control to the exitBlock.
  for (auto &newBlock : newBlocks) {
    if (auto returnOp =
            dyn_cast<IREE::HAL::ReturnOp>(newBlock.getTerminator())) {
      OpBuilder branchBuilder(returnOp);
      branchBuilder.create<BranchOp>(returnOp.getLoc(), exitBlock,
                                     returnOp.getOperands());
      returnOp.erase();
    }
  }

  // Splice the instructions of the inlined entry block into the insert block.
  insertBlock->getOperations().splice(insertBlock->end(),
                                      firstNewBlock->getOperations());
  firstNewBlock->erase();
}

// Inlines each switch condition region into the parent function predicated on
// the switch condition expression.
//
// Since switch conditions are evaluated in the order they are defined we can
// trivially turn the switch into a chain of if-else blocks.
//   if condition_0_match:
//     <inlined condition_0>
//   else
//     if condition_1_match:
//       <inlined condition_1>
//     else ...
static void buildConditionDispatchTable(IREE::HAL::DeviceSwitchOp switchOp,
                                        OpBuilder funcBuilder) {
  // Split the block containing the switch op such that all ops before the
  // switch are before and the switch and the following ops are after.
  // We'll have all of our inlined regions bounce over to the afterBlock with
  // the results of the call and use that to replace the switch op.
  auto *beforeBlock = funcBuilder.getBlock();
  auto *afterBlock = beforeBlock->splitBlock(switchOp);
  auto finalValues =
      llvm::to_vector<4>(afterBlock->addArguments(switchOp.getResultTypes()));

  // Create the blocks we'll use for all our conditions so that we can
  // reference them when inserting the branch ops.
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
    auto regionOperands = conditionRegion.getArguments();
    auto regionArgs = switchOp.args().slice(argOffset, regionOperands.size());
    argOffset += regionOperands.size();

    // Insert the branch based on the match. We either match and jump to a
    // block that will contain the inlined region or don't match and need to
    // fall through.
    auto isMatch = buildConditionExpression(
        switchOp.getLoc(), switchOp.device(), conditionAttr, funcBuilder);
    auto *matchBlock = conditionMatchBlocks[condition.index()];
    auto *fallthroughBlock = conditionFallthroughBlocks[condition.index()];
    funcBuilder.create<CondBranchOp>(switchOp.getLoc(), isMatch, matchBlock,
                                     fallthroughBlock);

    // Block that contains the inlined region and then jumps out of the chain.
    funcBuilder.setInsertionPointToStart(matchBlock);
    inlineConditionRegion(regionArgs, conditionRegion, afterBlock, funcBuilder);

    // Block that we enter to check the next condition.
    funcBuilder.setInsertionPointToStart(fallthroughBlock);
    if (condition.index() + 1 < conditionFallthroughBlocks.size()) {
      // Just continue on - the next loop iteration for the following
      // condition will add its IR to the block.
    } else {
      // Fallthrough of all expressions; die if we expected return values.
      funcBuilder.create<IREE::UnreachableOp>(switchOp.getLoc());
    }
  }

  // Remove the switch op and replace its results with the final joined
  // results.
  switchOp.replaceAllUsesWith(finalValues);
}

class InlineDeviceSwitchesPass
    : public PassWrapper<InlineDeviceSwitchesPass, OperationPass<FuncOp>> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<IREE::HAL::DeviceSwitchOp, 4> switchOps;
    funcOp.walk([&](IREE::HAL::DeviceSwitchOp switchOp) {
      switchOps.push_back(switchOp);
    });
    for (auto switchOp : switchOps) {
      OpBuilder funcBuilder(switchOp);
      buildConditionDispatchTable(switchOp, funcBuilder);
      switchOp.erase();
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createInlineDeviceSwitchesPass() {
  return std::make_unique<InlineDeviceSwitchesPass>();
}

static PassRegistration<InlineDeviceSwitchesPass> pass(
    "iree-hal-inline-device-switches",
    "Inlines hal.device.switch condition regions");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
