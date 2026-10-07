// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/DispatchCreation/CollapseDimensions.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_CONVERTCONVTOIM2COLPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc" // IWYU pragma: export

namespace {

/// Returns the loops of `convOp` that its input indexing map can use: depth
/// loops, then batch and output image loops, in the order they appear in the
/// output map, followed by input channel and filter window loops in the order
/// they appear in the filter map.
static SmallVector<unsigned>
getInputLoopOrder(linalg::LinalgOp convOp,
                  const linalg::ConvolutionDimensions &convDims) {
  AffineMap filterMap =
      convOp.getMatchingIndexingMap(convOp.getDpsInputOperand(/*i=*/1));
  AffineMap outputMap =
      convOp.getMatchingIndexingMap(convOp.getDpsInitOperand(/*i=*/0));
  SmallVector<unsigned> loopOrder;
  auto appendInMapOrder = [&](AffineMap map, ArrayRef<unsigned> loops) {
    for (AffineExpr expr : map.getResults()) {
      unsigned loop = cast<AffineDimExpr>(expr).getPosition();
      if (llvm::is_contained(loops, loop)) {
        loopOrder.push_back(loop);
      }
    }
  };
  appendInMapOrder(outputMap, convDims.depth);
  appendInMapOrder(outputMap, llvm::to_vector(llvm::concat<const unsigned>(
                                  convDims.batch, convDims.outputImage)));
  appendInMapOrder(filterMap, llvm::to_vector(llvm::concat<const unsigned>(
                                  convDims.inputChannel, convDims.filterLoop)));
  return loopOrder;
}

/// Returns the input indexing map of `convOp` with each loop in `filterLoops`
/// that has a static extent of one replaced by zero. A loop of extent one only
/// takes the value zero, so the returned map addresses the same input element
/// as the original one in every iteration of `convOp`.
static AffineMap getInputMapWithoutUnitWindows(linalg::LinalgOp convOp,
                                               ArrayRef<unsigned> filterLoops) {
  MLIRContext *context = convOp.getContext();
  unsigned numLoops = convOp.getNumLoops();
  SmallVector<int64_t> loopSizes = convOp.getStaticLoopRanges();
  SmallVector<AffineExpr> replacements =
      llvm::map_to_vector(llvm::seq<unsigned>(numLoops), [&](unsigned loop) {
        return getAffineDimExpr(loop, context);
      });
  for (unsigned loop : filterLoops) {
    if (loopSizes[loop] == 1) {
      replacements[loop] = getAffineConstantExpr(0, context);
    }
  }
  AffineMap inputMap =
      convOp.getMatchingIndexingMap(convOp.getDpsInputOperand(0));
  return simplifyAffineMap(inputMap.replaceDimsAndSymbols(
      replacements, /*symReplacements=*/{}, numLoops, /*numResultSyms=*/0));
}

/// A convolution input materialized by `gatherInput`.
struct GatheredInput {
  // The gathered tensor. It has one dimension per loop the input access uses,
  // in contraction loop order.
  Value value;
  // Indexing map from the convolution's loops to `value`: a projected
  // permutation selecting the gathered loops.
  AffineMap contractionMap;
};

/// Materializes the access `inputMap` makes into the input of `convOp`.
///
/// The gather is an all-parallel copy whose iteration space consists of the
/// loops `inputMap` uses, ordered as in `loopOrder`. For an NCHW convolution
/// with `inputMap = (n, oc, oh, ow, c, kh, kw) -> (n, c, oh + kh, ow + kw)` and
/// loop order (n, oh, ow, c, kh, kw), the gather loops are
/// (n, oh, ow, c, kh, kw), and
///   - the gather reads the input with `inputMap` rewritten onto its own
///     loops: `(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1 + d4, d2 + d5)`;
///   - the convolution then reads the gathered tensor with
///     `(n, oc, oh, ow, c, kh, kw) -> (n, oh, ow, c, kh, kw)`.
static GatheredInput gatherInput(RewriterBase &rewriter,
                                 linalg::LinalgOp convOp, AffineMap inputMap,
                                 ArrayRef<unsigned> loopOrder) {
  MLIRContext *context = rewriter.getContext();
  Location loc = convOp.getLoc();
  Value input = convOp.getDpsInputs().front();
  SmallVector<OpFoldResult> loopSizes =
      llvm::map_to_vector(convOp.createLoopRanges(rewriter, loc),
                          [](Range range) { return range.size; });

  SmallVector<unsigned> gatherLoops = llvm::filter_to_vector(
      loopOrder, [&](unsigned loop) { return inputMap.isFunctionOfDim(loop); });
  unsigned gatherRank = gatherLoops.size();

  // `loopToGatherDim` renames each gathered loop to its gather dimension; the
  // remaining loops do not occur in `inputMap`, so their placeholder entry is
  // never used. `gatherDimToLoop` is the reverse selection.
  SmallVector<AffineExpr> loopToGatherDim(convOp.getNumLoops(),
                                          getAffineConstantExpr(0, context));
  SmallVector<AffineExpr> gatherDimToLoop;
  SmallVector<OpFoldResult> gatherSizes;
  for (auto [gatherDim, loop] : llvm::enumerate(gatherLoops)) {
    loopToGatherDim[loop] = getAffineDimExpr(gatherDim, context);
    gatherDimToLoop.push_back(getAffineDimExpr(loop, context));
    gatherSizes.push_back(loopSizes[loop]);
  }

  AffineMap readMap = inputMap.replaceDimsAndSymbols(
      loopToGatherDim, /*symReplacements=*/{}, gatherRank,
      /*numResultSyms=*/0);
  Value init = tensor::EmptyOp::create(rewriter, loc, gatherSizes,
                                       getElementTypeOrSelf(input.getType()));
  auto gatherOp = linalg::GenericOp::create(
      rewriter, loc, init.getType(), input, init,
      ArrayRef<AffineMap>{readMap, rewriter.getMultiDimIdentityMap(gatherRank)},
      SmallVector<utils::IteratorType>(gatherRank,
                                       utils::IteratorType::parallel),
      [](OpBuilder &b, Location nestedLoc, ValueRange args) {
        linalg::YieldOp::create(b, nestedLoc, args[0]);
      });
  return {gatherOp.getResult(0),
          AffineMap::get(convOp.getNumLoops(), /*symbolCount=*/0,
                         gatherDimToLoop, context)};
}

/// Rewrites a convolution as a contraction over an explicitly gathered input.
///
/// The convolved input access (e.g. `(n, c, oh * s + kh * d, ow * s + kw * d)`)
/// is the only part of a convolution that a contraction cannot express. This
/// pattern materializes it with an all-parallel copy indexed by the loops the
/// access uses, then rewires the convolution to read that copy through a
/// projected permutation. The payload is kept as is, so floating-point and
/// integer bodies, including operand extensions, are supported unchanged, and
/// any layout, stride, dilation, group or depth multiplier works without
/// per-layout handling. Loops that every operand keeps contiguous are then
/// collapsed, which turns the standard layouts into plain (batch) matmuls.
///
/// Filter window loops of extent one are substituted with zero before deciding
/// what to gather. When that leaves the input access a projected permutation,
/// as for an unstrided 1x1 convolution, the input is read in place.
struct ConvertConvToIm2Col final : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaConvolutionOpInterface(linalgOp)) {
      return rewriter.notifyMatchFailure(linalgOp, "not a convolution");
    }
    // The gather is built on tensors, and the rewrite moves the op's body
    // region into the contraction.
    if (!linalgOp.hasPureTensorSemantics() || linalgOp->getNumRegions() != 1) {
      return rewriter.notifyMatchFailure(
          linalgOp, "expected tensor semantics and a body region");
    }
    // Requiring exactly an input and a filter excludes quantized convolutions
    // with zero-point operands; they are lowered before im2col runs.
    FailureOr<linalg::ConvolutionDimensions> convDims =
        linalg::inferConvolutionDims(linalgOp);
    if (failed(convDims)) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "failed to infer convolution dims");
    }
    // Pooling ops use the filter only for its shape; gathering their windows
    // would just copy the input.
    if (linalgOp.getMatchingBlockArgument(linalgOp.getDpsInputOperand(1))
            .use_empty()) {
      return rewriter.notifyMatchFailure(linalgOp, "filter values are unused");
    }

    AffineMap inputMap =
        getInputMapWithoutUnitWindows(linalgOp, convDims->filterLoop);
    SmallVector<unsigned> loopOrder = getInputLoopOrder(linalgOp, *convDims);

    // Gather the input unless its access is already a projected permutation.
    Value newInput = linalgOp.getDpsInputs().front();
    AffineMap newInputMap = inputMap;
    if (!inputMap.isProjectedPermutation()) {
      GatheredInput gathered =
          gatherInput(rewriter, linalgOp, inputMap, loopOrder);
      newInput = gathered.value;
      newInputMap = gathered.contractionMap;
    }

    // Rebuild the op as a generic around the new input. Named and generic
    // convolutions alike keep their body region, so the payload is moved
    // unchanged.
    SmallVector<Value> inputs = linalgOp.getDpsInputs();
    inputs.front() = newInput;
    SmallVector<AffineMap> contractionMaps = linalgOp.getIndexingMapsArray();
    contractionMaps.front() = newInputMap;
    auto contractionOp = linalg::GenericOp::create(
        rewriter, linalgOp.getLoc(), linalgOp->getResultTypes(), inputs,
        linalgOp.getDpsInits(), contractionMaps,
        linalgOp.getIteratorTypesArray());
    rewriter.inlineRegionBefore(linalgOp->getRegion(0),
                                contractionOp.getRegion(),
                                contractionOp.getRegion().end());
    rewriter.replaceOp(linalgOp, contractionOp->getResults());

    // Collapse the contraction's contiguous loops, so standard layouts become
    // (batch) matmuls with one loop per dimension kind. The reshapes this
    // introduces also keep the gather from being fused back into the
    // contraction, which would recreate the convolution.
    SmallVector<ReassociationIndices> collapsibleLoops =
        DispatchCreation::getCollapsibleLoops(contractionOp);
    if (collapsibleLoops.empty()) {
      return success();
    }
    // The loops are contiguous in every operand by construction, which is
    // the only precondition of collapsing.
    FailureOr<linalg::CollapseResult> collapsed =
        linalg::collapseOpIterationDims(contractionOp, collapsibleLoops,
                                        rewriter);
    assert(succeeded(collapsed) && "collapsible groups are not contiguous");
    rewriter.replaceOp(contractionOp, collapsed->results);
    return success();
  }
};

struct ConvertConvToIm2ColPass final
    : impl::ConvertConvToIm2ColPassBase<ConvertConvToIm2ColPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertConvToIm2Col>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::GlobalOptimization
