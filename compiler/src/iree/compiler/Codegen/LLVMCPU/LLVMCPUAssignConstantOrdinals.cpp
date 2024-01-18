// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {

struct LLVMCPUAssignConstantOrdinalsPass
    : public LLVMCPUAssignConstantOrdinalsBase<
          LLVMCPUAssignConstantOrdinalsPass> {
  LLVMCPUAssignConstantOrdinalsPass() = default;
  void runOnOperation() override {
    auto variantOp = getOperation();

    // Ignore non-LLVMCPU variants.
    // TODO(benvanik): a way to nest this in the pipeline via dynamic passes.
    if (variantOp.getTarget().getBackend().getValue() != "llvm-cpu")
      return;

    // Get a constant key -> ordinal mapping.
    auto keyOrdinals = variantOp.gatherConstantOrdinals();
    if (keyOrdinals.empty())
      return;

    // Update placeholders to hold the concrete ordinal values.
    // Eventually MLIR or LLVM will inline them.
    auto moduleOp = variantOp.getInnerModule();
    for (auto globalOp :
         llvm::make_early_inc_range(moduleOp.getOps<LLVM::GlobalOp>())) {
      auto keyAttr = globalOp->getAttr(
          IREE::HAL::ExecutableConstantBlockOp::getKeyAttrName());
      if (!keyAttr)
        continue;
      auto it = keyOrdinals.find(keyAttr);
      if (it == keyOrdinals.end()) {
        globalOp.emitOpError()
            << "no constant block providing key '" << keyAttr << "'";
        return signalPassFailure();
      }
      globalOp->removeAttr(
          IREE::HAL::ExecutableConstantBlockOp::getKeyAttrName());
      globalOp.setConstantAttr(UnitAttr::get(globalOp.getContext()));
      globalOp.setValueAttr(IntegerAttr::get(
          IntegerType::get(globalOp.getContext(), 32), it->second));
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPUAssignConstantOrdinalsPass() {
  return std::make_unique<LLVMCPUAssignConstantOrdinalsPass>();
}

} // namespace mlir::iree_compiler
