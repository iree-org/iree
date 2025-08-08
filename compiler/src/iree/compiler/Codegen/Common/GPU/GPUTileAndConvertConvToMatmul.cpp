// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUTILEANDCONVERTCONVTOMATMULPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUTileAndConvertConvToMatmulPass final
    : impl::GPUTileAndConvertConvToMatmulPassBase<
          GPUTileAndConvertConvToMatmulPass> {
  void runOnOperation() override;
};
} // namespace

// TODO : Upstream utility that does this pruning is broken for LinalgOp. Drop
// this if that gets fixed.
static SmallVector<NamedAttribute> getPrunedAttributeList(linalg::LinalgOp op) {
  const StringLiteral memoAttr =
      linalg::LinalgDialect::kMemoizedIndexingMapsAttrName;
  SmallVector<NamedAttribute> prunedAttributeList;
  for (auto attr : op->getDiscardableAttrs()) {
    if (attr.getName() != memoAttr) {
      prunedAttributeList.push_back(attr);
    }
  }
  return prunedAttributeList;
}

// Helper to remove unit filter loop dimensions from input map of convolution
// operations so that they can become contractions.
void static removeUnitExtentDimsfromMaps(linalg::LinalgOp linalgOp,
                                         RewriterBase &rewriter) {
  auto convDimsOrFailure = linalg::inferConvolutionDims(linalgOp);
  if (failed(convDimsOrFailure)) {
    return;
  }
  const mlir::linalg::ConvolutionDimensions &convDims = *convDimsOrFailure;
  // We cant make strided convolutions into contractions directly so bail out.
  if (llvm::any_of(convDims.strides,
                   [](int64_t stride) { return stride != 1; })) {
    return;
  }
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  if (indexingMaps.empty())
    return;
  AffineMap inputMap = indexingMaps[0];
  AffineMap filterMap = indexingMaps[1];

  // Check that all filter loop dimensions are unit and then make them zero.
  DenseMap<AffineExpr, AffineExpr> dimMap;
  Value filter = linalgOp.getDpsInputs()[1];
  auto filterType = llvm::cast<ShapedType>(filter.getType());
  ArrayRef<int64_t> filterShape = filterType.getShape();
  for (auto filterLoop : convDims.filterLoop) {
    std::optional<int64_t> maybeDim = filterMap.getResultPosition(
        getAffineDimExpr(filterLoop, filterMap.getContext()));
    if (!maybeDim || filterShape[maybeDim.value()] != 1) {
      return;
    }
    dimMap[rewriter.getAffineDimExpr(filterLoop)] =
        getAffineConstantExpr(0, filterMap.getContext());
  }
  SmallVector<AffineMap> newIndexingMaps;
  newIndexingMaps.push_back(inputMap.replace(dimMap));
  // No changes to the filter and output map.
  newIndexingMaps.push_back(filterMap);
  newIndexingMaps.push_back(indexingMaps[2]);
  // Create the new contraction op and replace the old convolution op.
  auto newOp = rewriter.create<linalg::GenericOp>(
      linalgOp.getLoc(), linalgOp.getDpsInits().getType(),
      linalgOp.getDpsInputs(), linalgOp.getDpsInits(), newIndexingMaps,
      linalgOp.getIteratorTypesArray(), /*bodyBuild=*/nullptr,
      getPrunedAttributeList(linalgOp));
  rewriter.inlineRegionBefore(linalgOp->getRegion(0), newOp.getRegion(),
                              newOp.getRegion().begin());
  rewriter.replaceOp(linalgOp, newOp.getResults());
}

void GPUTileAndConvertConvToMatmulPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  // Collect candiates that need to be tiled to convert to matmul.
  IRRewriter rewriter(funcOp);
  SmallVector<linalg::LinalgOp> convCandidates;
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
    if (!loweringConfig) {
      return;
    }
    if (!getMmaKind(loweringConfig)) {
      return;
    }
    auto convDimsOrFailure = linalg::inferConvolutionDims(linalgOp);
    if (failed(convDimsOrFailure)) {
      return;
    }
    convCandidates.push_back(linalgOp);
  });
  // Handle convolution operations by tiling the filter dimensions to 1 so that
  // they can become contractions.
  llvm::SmallDenseSet<TilingInterface> targets;
  llvm::SmallDenseMap<TilingInterface, SmallVector<OpFoldResult>> targetTileMap;
  auto zero = rewriter.getIndexAttr(0);
  auto one = rewriter.getIndexAttr(1);
  for (auto candidate : convCandidates) {
    SmallVector<OpFoldResult> directTileSizes(candidate.getNumLoops(), zero);
    auto convDimsOrFailure = linalg::inferConvolutionDims(candidate);
    for (auto loopDim : convDimsOrFailure->filterLoop) {
      directTileSizes[loopDim] = one;
    }
    auto tilingOp = dyn_cast<TilingInterface>(*candidate);
    targets.insert(tilingOp);
    targetTileMap[tilingOp] = directTileSizes;
  }
  IREE::GPU::TilingLevel reductionLevel = IREE::GPU::TilingLevel::Reduction;
  if (failed(applyTileAndFuseToEachRoot(rewriter, targets, reductionLevel,
                                        /*allowZeroSlices=*/true,
                                        targetTileMap))) {
    funcOp.emitError() << "tiling of level  convolution failed\n";
  }
  // Collect candiates again since the old candidates are not valid
  // after convolution tiling.
  convCandidates = {};
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
    if (!loweringConfig) {
      return;
    }
    // Currently we only convert convolutions that have a MMA attr
    // in there configurations as this is meant to be used for
    // lowering the convolutions to matmul intrinsic. If we
    // want to do this for all convolutions we can drop this check
    // and move this pass to the common directory.
    if (!getMmaKind(loweringConfig)) {
      return;
    }
    convCandidates.push_back(linalgOp);
  });

  // Remove unit extent filter reductions dims from input maps of convolution
  // operations which would make them contractions.
  for (auto candidate : convCandidates) {
    rewriter.setInsertionPoint(candidate);
    removeUnitExtentDimsfromMaps(candidate, rewriter);
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
