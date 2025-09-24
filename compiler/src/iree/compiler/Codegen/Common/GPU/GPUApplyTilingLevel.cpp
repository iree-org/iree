// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-tiling-level"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYTILINGLEVELPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUApplyTilingLevelPass final
    : impl::GPUApplyTilingLevelPassBase<GPUApplyTilingLevelPass> {
  using GPUApplyTilingLevelPassBase::GPUApplyTilingLevelPassBase;
  void runOnOperation() override;
};
} // namespace

static llvm::SmallDenseSet<TilingInterface>
getTiledOps(Operation *funcOp, IREE::GPU::TilingLevel tilingLevel) {
  llvm::SmallDenseSet<TilingInterface> targets;
  unsigned opaqueLevel = llvm::to_underlying(tilingLevel);
  funcOp->walk([&](TilingInterface target) {
    // TODO: This would probably be easier with a lowering config interface
    // method that checks whether a particular level is tiled.
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(target)) {
      if (loweringConfig.hasTilingLevel(opaqueLevel)) {
        targets.insert(target);
      }
    }
  });
  return targets;
}

void GPUApplyTilingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  if (tilingLevel != IREE::GPU::TilingLevel::Reduction &&
      tilingLevel != IREE::GPU::TilingLevel::Thread &&
      tilingLevel != IREE::GPU::TilingLevel::Subgroup &&
      tilingLevel != IREE::GPU::TilingLevel::PartialReduction) {
    funcOp.emitError() << "unsupported tiling level: "
                       << IREE::GPU::stringifyEnum(tilingLevel) << "\n";
    return signalPassFailure();
  }

  llvm::SmallDenseSet<TilingInterface> targetOps =
      getTiledOps(funcOp, tilingLevel);

  IRRewriter rewriter(funcOp);
  if (failed(applyTileAndFuseToEachRoot(rewriter, targetOps, tilingLevel,
                                        allowZeroSlices))) {
    funcOp.emitError() << "tiling of level "
                       << IREE::GPU::stringifyEnum(tilingLevel) << " failed\n";
    return signalPassFailure();
  }

  MLIRContext *context = &getContext();

  // Swap `collapse_shape` with `extract_slice` to enable more loop fusion
  // opportunity. Currently this is only needed for convolution IGEMM path.
  // TODO(vivian): Move the pattern to `GPUFuseAndHoistParallelLoopsPass`.
  if (normalizeLoops) {
    funcOp->walk(
        [&](scf::ForOp forOp) { (void)normalizeLoopBounds(rewriter, forOp); });
    funcOp->walk([&](scf::ForallOp forallOp) {
      (void)normalizeLoopBounds(rewriter, forallOp);
    });

    RewritePatternSet patterns(context);
    populateSwapExtractWithCollapsePattern(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // Apply cleanup patterns.
  {
    RewritePatternSet patterns(context);
    // Merge consecutive insert/extract slice ops to simplify later loop
    // hoisting patterns.
    tensor::populateFoldTensorEmptyPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "tiling cleanup failed\n";
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
