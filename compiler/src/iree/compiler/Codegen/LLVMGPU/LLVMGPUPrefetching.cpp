// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUPREFETCHSHAREDMEMORYPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct LLVMGPUPrefetchSharedMemoryPass final
    : impl::LLVMGPUPrefetchSharedMemoryPassBase<
          LLVMGPUPrefetchSharedMemoryPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    SmallVector<scf::ForOp> loops;
    funcOp.walk([&loops](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops) {
      FailureOr<scf::ForOp> newLoop = prefetchSharedMemoryCopy(rewriter, forOp);
      // The only possible failure is the analysis failure, which does not cause
      // the pass to fail. Therefore we discard any failures at this point.
      (void)newLoop;
      break; // TODO: Fix nested loop handling.
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
