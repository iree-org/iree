// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUPREFETCHSHAREDMEMORYPASS
#define GEN_PASS_DEF_LLVMGPUOPTIMIZEANDPREFETCHSHAREDMEMORYPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

static void prefetchSharedMemory(FunctionOpInterface funcOp) {
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

struct LLVMGPUPrefetchSharedMemoryPass final
    : impl::LLVMGPUPrefetchSharedMemoryPassBase<
          LLVMGPUPrefetchSharedMemoryPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    prefetchSharedMemory(funcOp);
  }
};

/// This pass is a convenient wrapper around the prefetching passes. It will
/// query the pipeline options, making sure prefetch is enabled. Then invoke all
/// relevant optimization passes before invoking the
/// LLVMGPUPrefetchSharedMemoryPass.
struct LLVMGPUOptimizeAndPrefetchSharedMemoryPass final
    : impl::LLVMGPUOptimizeAndPrefetchSharedMemoryPassBase<
          LLVMGPUOptimizeAndPrefetchSharedMemoryPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    std::optional<OpPassManager> maybePipeline =
        getFunctionOpInterfacePassManager(funcOp);
    if (!maybePipeline) {
      funcOp.emitOpError(
          "unhandled function-like container during executable lowering");
      return signalPassFailure();
    }
    OpPassManager &pipeline = maybePipeline.value();

    pipeline.addPass(createFissionTransferOpsInControlFlowPass());
    pipeline.addPass(createHoistStaticallyBoundAllocationsPass());
    pipeline.addPass(createRemoveSingleIterationLoopPass());
    pipeline.addPass(createLLVMGPUPrefetchSharedMemoryPass());
    if (failed(runPipeline(pipeline, funcOp))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
