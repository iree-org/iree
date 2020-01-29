// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/HAL/Target/LegacyUtil.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Utils/TypeConversionUtils.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static void makeLegacyExecutableDispatchABI(
    IREE::Flow::DispatchEntryOp dispatchEntryOp, ModuleOp moduleOp,
    IREE::HAL::ExecutableEntryPointOp targetEntryOp) {
  // Add export attr so that the function is not DCE'd.
  auto funcOp = moduleOp.lookupSymbol<FuncOp>(dispatchEntryOp.function_ref());
  funcOp.setAttr("iree.executable.export",
                 UnitAttr::get(moduleOp.getContext()));
  funcOp.setAttr("iree.executable.workload",
                 dispatchEntryOp.workload().getValue());
  funcOp.setAttr("iree.executable.workgroup_size",
                 targetEntryOp.workgroup_size());

  // Reset function type to memrefs with output args.
  SmallVector<Type, 4> inputTypes;
  for (const auto &oldType : funcOp.getType().getInputs()) {
    inputTypes.push_back(
        convertLegacyTypeToMemRef(legalizeLegacyType(oldType)));
  }
  SmallVector<Type, 4> outputTypes;
  for (const auto &oldType : funcOp.getType().getResults()) {
    outputTypes.push_back(
        convertLegacyTypeToMemRef(legalizeLegacyType(oldType)));
  }
  inputTypes.append(outputTypes.begin(), outputTypes.end());
  auto funcType = FunctionType::get(inputTypes, {}, moduleOp.getContext());
  funcOp.setType(funcType);

  // Rewrite the entry block to match the new args for inputs.
  auto &entryBlock = funcOp.getBlocks().front();
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
  funcOp.walk([&](mlir::ReturnOp returnOp) {
    OpBuilder returnBuilder(returnOp);
    for (auto operand : llvm::enumerate(returnOp.getOperands())) {
      returnBuilder.create<IREE::StoreOutputOp>(
          returnOp.getLoc(), operand.value(), outputArgs[operand.index()]);
    }
    returnBuilder.create<IREE::ReturnOp>(returnOp.getLoc());
    deadOps.push_back(returnOp);
  });
  for (auto *deadOp : deadOps) deadOp->erase();
}

static void makeLegacyExecutableReductionABI(
    IREE::Flow::ReductionEntryOp reductionEntryOp, ModuleOp moduleOp,
    IREE::HAL::ExecutableEntryPointOp targetEntryOp) {
  // Add export attr so that the function is not DCE'd.
  auto funcOp = moduleOp.lookupSymbol<FuncOp>(reductionEntryOp.function_ref());
  funcOp.setAttr("iree.executable.export",
                 UnitAttr::get(moduleOp.getContext()));
  funcOp.setAttr("iree.executable.reduction",
                 UnitAttr::get(moduleOp.getContext()));
  funcOp.setAttr("iree.executable.reduction.apply",
                 FlatSymbolRefAttr::get(reductionEntryOp.apply_ref(),
                                        moduleOp.getContext()));
  funcOp.setAttr("iree.executable.reduction.dimension",
                 IntegerAttr::get(IntegerType::get(32, moduleOp.getContext()),
                                  reductionEntryOp.dimension()));

  // Reset function type to memrefs with output args.
  SmallVector<Type, 4> inputTypes;
  for (const auto &oldType : funcOp.getType().getInputs()) {
    inputTypes.push_back(
        convertLegacyTypeToMemRef(legalizeLegacyType(oldType)));
  }
  for (const auto &oldType : funcOp.getType().getResults()) {
    inputTypes.push_back(
        convertLegacyTypeToMemRef(legalizeLegacyType(oldType)));
  }
  auto funcType = FunctionType::get(inputTypes, {}, moduleOp.getContext());
  funcOp.setType(funcType);
}

void makeLegacyExecutableABI(IREE::Flow::ExecutableOp executableOp,
                             ModuleOp moduleOp,
                             IREE::HAL::ExecutableOp targetExecutableOp) {
  for (auto &op : executableOp.getBlock()) {
    if (auto entryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(&op)) {
      auto targetEntryOp =
          targetExecutableOp.lookupSymbol<IREE::HAL::ExecutableEntryPointOp>(
              entryOp.sym_name());
      makeLegacyExecutableDispatchABI(entryOp, moduleOp, targetEntryOp);
    } else if (auto entryOp = dyn_cast<IREE::Flow::ReductionEntryOp>(&op)) {
      auto targetEntryOp =
          targetExecutableOp.lookupSymbol<IREE::HAL::ExecutableEntryPointOp>(
              entryOp.sym_name());
      makeLegacyExecutableReductionABI(entryOp, moduleOp, targetEntryOp);
    }
  }
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
