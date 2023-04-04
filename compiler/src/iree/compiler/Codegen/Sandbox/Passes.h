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
//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerSandboxPasses();
}  // namespace iree_compiler

}  // namespace mlir
#endif  // IREE_CODEGEN_SANDBOX_PASSES_H_
