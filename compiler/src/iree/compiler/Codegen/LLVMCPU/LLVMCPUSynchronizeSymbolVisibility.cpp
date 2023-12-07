// Copyright 2021 The IREE Authors
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

static void setVisibilityFromLinkage(SymbolOpInterface op,
                                     LLVM::Linkage linkage) {
  SymbolTable::Visibility visibility = op.getVisibility();
  switch (linkage) {
  case LLVM::Linkage::Private:
  case LLVM::Linkage::Internal:
    visibility = SymbolTable::Visibility::Private;
    break;
  default:
    visibility = SymbolTable::Visibility::Public;
    break;
  }
  op.setVisibility(visibility);
}

struct LLVMCPUSynchronizeSymbolVisibilityPass
    : public LLVMCPUSynchronizeSymbolVisibilityBase<
          LLVMCPUSynchronizeSymbolVisibilityPass> {
  LLVMCPUSynchronizeSymbolVisibilityPass() = default;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    for (auto &op : moduleOp.getOps()) {
      if (auto globalOp = dyn_cast<LLVM::GlobalOp>(op)) {
        setVisibilityFromLinkage(globalOp, globalOp.getLinkage());
      } else if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        setVisibilityFromLinkage(funcOp, funcOp.getLinkage());
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUSynchronizeSymbolVisibilityPass() {
  return std::make_unique<LLVMCPUSynchronizeSymbolVisibilityPass>();
}

} // namespace mlir::iree_compiler
