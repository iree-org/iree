// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-test-llvmgpu-query-mma"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTLLVMGPUQUERYMMAPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct TestLLVMGPUQueryMMAPass final
    : impl::TestLLVMGPUQueryMMAPassBase<TestLLVMGPUQueryMMAPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    llvm::SmallDenseMap<Operation *, SmallVector<IREE::GPU::MMAIntrinsic>>
        mmaMap = queryMMAIntrinsics(moduleOp);
    for (const auto &[op, mmaAttrs] : mmaMap) {
      if (auto variantOp = llvm::dyn_cast<IREE::HAL::ExecutableVariantOp>(op)) {
        llvm::outs() << "Executable Variant Name: " << variantOp.getName()
                     << "\n";
      } else {
        llvm::outs() << "Executable Variant Name: " << "Unnamed Operation"
                     << "\n";
      }
      llvm::outs() << "MMA Intrinsics: ";
      llvm::interleave(mmaAttrs, llvm::outs(), " ");
      llvm::outs() << "\n";
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
