// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/CPU/ReductionStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensions.h"
#include "iree/compiler/Codegen/TransformStrategies/CPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/AbstractReductionStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::cpu::ReductionConfig;
using iree_compiler::cpu::ReductionStrategy;
using transform_ext::RegisterMatchCallbacksOp;

mlir::iree_compiler::cpu::ReductionStrategy::ReductionStrategy(
    const transform_ext::MatchedReductionCaptures &captures,
    const ReductionConfig &reductionConfig)
    : AbstractReductionStrategy(captures, {}) {
  configure(reductionConfig);
  LLVM_DEBUG(DBGS() << "use CPU reduction strategy\n");
}

void mlir::iree_compiler::cpu::ReductionStrategy::configure(
    const ReductionConfig &config) {
  // Block-level
  // ===========
  // Tile all the parallel dimensions to 8 for now.
  int64_t numParallelLoops = captures.reductionRank - 1;
  workgroupTileSizes.append(numParallelLoops, 8);
  vectorSize = config.vectorSize;
}

/// Builds the transform IR tiling reductions for CUDA targets. Supports
/// reductions in the last dimension, with optional leading and trailing
/// elementwise operations.
void mlir::iree_compiler::cpu::buildReductionStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const ReductionStrategy &strategy) {
  // Step 1. Tiling to the block/workgroup level. Keep everything fused.
  auto [maybeLeadingHBlock, gridFillH, gridReductionH, maybeTiledTrailingHBlock,
        forall] =
      buildReductionStrategyBlockDistribution(b, variantH,
                                              strategy.workgroupTileSizes);

  // Step 2. Naive first strategy to tile the most minor dimension by
  // strategy.getVectorSize().
  for (auto [val, rank] : SmallVector<std::pair<Value, int64_t>>{
           {maybeLeadingHBlock, strategy.captures.maybeLeadingRank},
           {gridReductionH, strategy.captures.reductionRank},
           {maybeTiledTrailingHBlock, strategy.captures.maybeTrailingRank}}) {
    if (rank == 0)
      continue;
    SmallVector<int64_t> tileSizes(rank - 1, 0);
    tileSizes.push_back(strategy.getVectorSize());
    buildTileFuseToScfFor(b, variantH, val, {},
                          getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)));
  }

  // Step 3-5. Common trailing steps.
  vector::LowerVectorsOptions lowerVectorsOptions;
  lowerVectorsOptions
      .setVectorTransformsOptions(vector::VectorContractLowering::OuterProduct)
      .setVectorMultiReductionLowering(
          vector::VectorMultiReductionLowering::InnerParallel)
      .setVectorTransferSplit(vector::VectorTransferSplit::LinalgCopy)
      .setVectorTransposeLowering(vector::VectorTransposeLowering::EltWise)
      .setTransposeAVX2Lowering(false)
      .setUnrollVectorTransfers(true);
  buildCommonTrailingStrategy(b, variantH, lowerVectorsOptions);
}
