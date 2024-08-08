// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUASSIGNCONSTANTORDINALSPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

struct LLVMCPUAssignConstantOrdinalsPass
    : public impl::LLVMCPUAssignConstantOrdinalsPassBase<
          LLVMCPUAssignConstantOrdinalsPass> {
  void runOnOperation() override {
    auto variantOp = getOperation();

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
} // namespace mlir::iree_compiler
