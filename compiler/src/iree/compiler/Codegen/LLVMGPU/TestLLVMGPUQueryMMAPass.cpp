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
    llvm::SmallDenseMap<IREE::HAL::ExecutableVariantOp,
                        SmallVector<IREE::GPU::MMAIntrinsic>>
        mmaMap = queryMMAIntrinsics(moduleOp);
    for (const auto &[op, mmaAttrs] : mmaMap) {
      llvm::outs() << "Executable Variant Name: "
                   << cast<IREE::HAL::ExecutableVariantOp>(*op).getName()
                   << "\n";
      llvm::outs() << "MMA Intrinsics: ";
      llvm::interleave(mmaAttrs, llvm::outs(), " ");
      llvm::outs() << "\n";
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
