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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Utils/TypeConversionUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// TODO(b/150312935): remove this when the SPIR-V and LLVM targets use
// hal.interface.
static void makeLegacyExecutableDispatchABI(
    IREE::Flow::DispatchEntryOp dispatchEntryOp, FuncOp thunkOp) {
  auto *context = thunkOp.getContext();

  auto implOp = thunkOp.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      (thunkOp.getName() + "_impl").str());
  implOp.setAttr("iree.executable.export", UnitAttr::get(context));

  // Destroy the IO op and replace with the original entry.
  SymbolTable::setSymbolVisibility(implOp, SymbolTable::Visibility::Public);
  auto originalName = thunkOp.getName();
  thunkOp.erase();
  implOp.setName(originalName);

  // Reset function type to memrefs with output args.
  SmallVector<Type, 4> inputTypes;
  for (const auto &oldType : implOp.getType().getInputs()) {
    inputTypes.push_back(
        convertLegacyTypeToMemRef(legalizeLegacyType(oldType)));
  }
  SmallVector<Type, 4> outputTypes;
  for (const auto &oldType : implOp.getType().getResults()) {
    outputTypes.push_back(
        convertLegacyTypeToMemRef(legalizeLegacyType(oldType)));
  }
  inputTypes.append(outputTypes.begin(), outputTypes.end());
  auto funcType = FunctionType::get(inputTypes, {}, context);
  implOp.setType(funcType);

  // Rewrite the entry block to match the new args for inputs.
  auto &entryBlock = implOp.getBlocks().front();
  auto oldArgs = entryBlock.getArguments().vec();
  OpBuilder entryBuilder(&entryBlock);
  entryBuilder.setInsertionPointToStart(&entryBlock);
  for (auto arg : entryBlock.getArguments()) {
    Type oldType = arg.getType();
    arg.setType(convertLegacyTypeToMemRef(legalizeLegacyType(oldType)));
    auto loadInputOp = entryBuilder.create<IREE::LoadInputOp>(
        dispatchEntryOp.getLoc(), oldType, arg);
    arg.replaceAllUsesWith(loadInputOp.getResult());
    loadInputOp.setOperand(arg);
  }

  // Add output args and replace returns with stores.
  auto outputArgs = llvm::to_vector<4>(entryBlock.addArguments(outputTypes));
  SmallVector<Operation *, 4> deadOps;
  implOp.walk([&](mlir::ReturnOp returnOp) {
    OpBuilder returnBuilder(returnOp);
    for (auto operand : llvm::enumerate(returnOp.getOperands())) {
      returnBuilder.create<IREE::StoreOutputOp>(
          returnOp.getLoc(), operand.value(), outputArgs[operand.index()]);
    }
    returnBuilder.create<mlir::ReturnOp>(returnOp.getLoc());
    deadOps.push_back(returnOp);
  });
  for (auto *deadOp : deadOps) deadOp->erase();
}

// TODO(b/150312935): remove this when the SPIR-V and LLVM targets use
// hal.interface.
static void makeLegacyExecutableReductionABI(
    IREE::Flow::ReductionEntryOp reductionEntryOp, FuncOp thunkOp) {
  auto *context = thunkOp.getContext();

  auto implOp = thunkOp.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      (thunkOp.getName() + "_impl").str());
  implOp.setAttr("iree.executable.export", UnitAttr::get(context));
  implOp.setAttr("iree.executable.reduction", UnitAttr::get(context));
  implOp.setAttr("iree.executable.reduction.apply",
                 FlatSymbolRefAttr::get(reductionEntryOp.apply_ref(), context));
  implOp.setAttr("iree.executable.reduction.dimension",
                 IntegerAttr::get(IntegerType::get(32, context),
                                  reductionEntryOp.dimension()));

  // Remove any blocks that may exist within the implementation function as the
  // backend will be replacing the body with its own implementation.
  implOp.getBlocks().clear();

  // Destroy the IO op and replace with the original entry.
  SymbolTable::setSymbolVisibility(implOp, SymbolTable::Visibility::Public);
  auto originalName = thunkOp.getName();
  thunkOp.erase();
  implOp.setName(originalName);

  // Reset function type to memrefs with output args.
  SmallVector<Type, 4> inputTypes;
  for (const auto &oldType : implOp.getType().getInputs()) {
    inputTypes.push_back(
        convertLegacyTypeToMemRef(legalizeLegacyType(oldType)));
  }
  for (const auto &oldType : implOp.getType().getResults()) {
    inputTypes.push_back(
        convertLegacyTypeToMemRef(legalizeLegacyType(oldType)));
  }
  auto funcType = FunctionType::get(inputTypes, {}, context);
  implOp.setType(funcType);
}

class RewriteLegacyIOPass
    : public OperationPass<RewriteLegacyIOPass, IREE::Flow::ExecutableOp> {
 public:
  void runOnOperation() override {
    auto executableOp = getOperation();
    auto moduleOp = executableOp.getInnerModule();

    // Rewrite entry functions.
    for (auto &op : executableOp.getBlock()) {
      if (auto entryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(&op)) {
        auto thunkOp = moduleOp.lookupSymbol<FuncOp>(entryOp.function_ref());
        makeLegacyExecutableDispatchABI(entryOp, thunkOp);
      } else if (auto entryOp = dyn_cast<IREE::Flow::ReductionEntryOp>(&op)) {
        auto thunkOp = moduleOp.lookupSymbol<FuncOp>(entryOp.function_ref());
        makeLegacyExecutableReductionABI(entryOp, thunkOp);
      }
    }

    // Drop unneeded interface ops.
    auto interfaceOps =
        llvm::to_vector<4>(moduleOp.getOps<IREE::HAL::InterfaceOp>());
    for (auto interfaceOp : interfaceOps) {
      interfaceOp.erase();
    }
  }
};

std::unique_ptr<OpPassBase<IREE::Flow::ExecutableOp>>
createRewriteLegacyIOPass() {
  return std::make_unique<RewriteLegacyIOPass>();
}

static PassRegistration<RewriteLegacyIOPass> pass(
    "iree-hal-rewrite-legacy-io",
    "Rewrites hal.interface usage to legacy iree.load_input/iree.store_output");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
