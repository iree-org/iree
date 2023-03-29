// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SANDBOX_PASSES_H_
#define IREE_CODEGEN_SANDBOX_PASSES_H_

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates a pass to perform SplitReduction transformations of `LinalgOp`s.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgSplitReductionPass(
    bool enableReassociateFpReductions = false, int64_t size = 0);

/// Struct to control pass options for `LinalgSingleTilingExpert` pass.
struct LinalgSingleTilingExpertPassOptions {
  std::string anchorFuncOpName = "";
  std::string anchorOpName = "";
  bool generalize = false;
  SmallVector<int64_t> iteratorInterchange = {};
  bool vectorize = false;
  bool enableVectorMasking = false;
  bool vectorizePadding = false;
  int64_t tilingLevel = -1;
};

/// Creates a pass to drive one level tiling + vectorization of `LinalgOp`s.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgSingleTilingExpertPass();
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgSingleTilingExpertPass(
    const LinalgSingleTilingExpertPassOptions &passOptions);

/// Struct to control pass options for `LinalgVectorLoweringPass` pass.
struct LinalgVectorLoweringPassOptions {
  int vectorLoweringStage = 0;
  std::string splitVectorTransfersTo = "";
  std::string lowerVectorTransposeTo = "eltwise";
  bool lowerVectorTransposeToAVX2 = false;
  std::string lowerVectorMultiReductionTo = "innerparallel";
  std::string lowerVectorContractionTo = "outerproduct";
  bool unrollVectorTransfers = true;
  int maxTransferRank = 1;
};

/// Creates a pass to drive the lowering of vector operations in a staged
/// manner.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgVectorLoweringPass(
    int64_t vectorLoweringStage = 0);
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgVectorLoweringPass(
    const LinalgVectorLoweringPassOptions &options);

//===----------------------------------------------------------------------===//
// Transforms that tie together individual drivers.
//===----------------------------------------------------------------------===//

/// Add staged lowering of vector ops. `passManager` is expected to be a
/// `builtin.func` op pass manager.
void addLowerToVectorTransforms(OpPassManager &passManager,
                                LinalgVectorLoweringPassOptions options);

//===----------------------------------------------------------------------===//
// IREE specific pass creation methods to allow invocation from within IREEs
// backend pipelines
//===----------------------------------------------------------------------===//

namespace iree_compiler {
/// Creates a pass to drive one level tiling + vectorization of `LinalgOp`s with
/// additional parameters that allow controlling the value of the pass options
/// when building the pipeline.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgSingleTilingExpertPass(
    int64_t tilingLevel, bool vectorize);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerSandboxPasses();
}  // namespace iree_compiler

}  // namespace mlir
#endif  // IREE_CODEGEN_SANDBOX_PASSES_H_
