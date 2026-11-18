// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/DispatchCreation/CollapseDimensions.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <functional>
#include <optional>

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_CONVERTCONVTOIM2COLPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc" // IWYU pragma: export

namespace {

/// The loops of an op the rewrite applies to, grouped by the operands whose
/// indexing maps use them. The op has an input, a filter and an output; the
/// names follow the roles the loops have in a convolution, but the grouping
/// only looks at the indexing maps. It therefore also describes what unit
/// dimension folding leaves of a convolution whose filter window has extent
/// one along a strided axis, e.g. `(c, oh * 2, ow * 2)`, which is no longer a
/// convolution.
struct OperandLoops {
  // Used by the input, the filter and the output: groups and depthwise
  // channels.
  SmallVector<unsigned> depth;
  // Used by the input and the output only: batch and output image.
  SmallVector<unsigned> image;
  // Used by the input and the filter only: input channels and filter window.
  SmallVector<unsigned> reduced;
  // The loops in `reduced` that the input map only uses inside compound
  // expressions such as `oh * 2 + kh`: the filter window.
  SmallVector<unsigned> window;
  // Used by the filter and the output only: output channels.
  SmallVector<unsigned> outputChannel;
};

/// Returns true if `expr` is a sum of loops scaled by positive constants, the
/// form of a strided and dilated window access.
static bool isStridedSum(AffineExpr expr) {
  if (isa<AffineDimExpr>(expr)) {
    return true;
  }
  auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binaryExpr) {
    return false;
  }
  if (binaryExpr.getKind() == AffineExprKind::Add) {
    return isStridedSum(binaryExpr.getLHS()) &&
           isStridedSum(binaryExpr.getRHS());
  }
  if (binaryExpr.getKind() == AffineExprKind::Mul) {
    auto scale = dyn_cast<AffineConstantExpr>(binaryExpr.getRHS());
    return scale && scale.getValue() > 0 && isStridedSum(binaryExpr.getLHS());
  }
  return false;
}

/// Returns the loops of `convOp` that its input indexing map can use: depth
/// loops, then batch and output image loops, in the order they appear in the
/// output map, followed by input channel and filter window loops in the order
/// they appear in the filter map.
static SmallVector<unsigned> getInputLoopOrder(linalg::LinalgOp convOp,
                                               const OperandLoops &loops) {
  AffineMap filterMap =
      convOp.getMatchingIndexingMap(convOp.getDpsInputOperand(/*i=*/1));
  AffineMap outputMap =
      convOp.getMatchingIndexingMap(convOp.getDpsInitOperand(/*i=*/0));
  SmallVector<unsigned> loopOrder;
  auto appendInMapOrder = [&](AffineMap map, ArrayRef<unsigned> selected) {
    for (AffineExpr expr : map.getResults()) {
      unsigned loop = cast<AffineDimExpr>(expr).getPosition();
      if (llvm::is_contained(selected, loop)) {
        loopOrder.push_back(loop);
      }
    }
  };
  appendInMapOrder(outputMap, loops.depth);
  appendInMapOrder(outputMap, loops.image);
  appendInMapOrder(filterMap, loops.reduced);
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

/// The access a convolution makes into its input, after filter window loops
/// of extent one are substituted with zero.
struct InputAccess {
  // Indexing map from the convolution's loops to its input.
  AffineMap inputMap;
  // The loops `inputMap` uses, in the order the gather iterates them.
  SmallVector<unsigned> gatherLoops;
  // `inputMap` rewritten onto the gather loops.
  AffineMap readMap;
};

static InputAccess getInputAccess(linalg::LinalgOp convOp,
                                  const OperandLoops &loops) {
  MLIRContext *context = convOp.getContext();
  AffineMap inputMap = getInputMapWithoutUnitWindows(convOp, loops.window);
  SmallVector<unsigned> gatherLoops = llvm::filter_to_vector(
      getInputLoopOrder(convOp, loops),
      [&](unsigned loop) { return inputMap.isFunctionOfDim(loop); });
  // `loopToGatherDim` renames each gathered loop to its gather dimension; the
  // remaining loops do not occur in `inputMap`, so their placeholder entry is
  // never used.
  SmallVector<AffineExpr> loopToGatherDim(convOp.getNumLoops(),
                                          getAffineConstantExpr(0, context));
  for (auto [gatherDim, loop] : llvm::enumerate(gatherLoops)) {
    loopToGatherDim[loop] = getAffineDimExpr(gatherDim, context);
  }
  AffineMap readMap = inputMap.replaceDimsAndSymbols(
      loopToGatherDim, /*symReplacements=*/{}, gatherLoops.size(),
      /*numResultSyms=*/0);
  return {inputMap, std::move(gatherLoops), readMap};
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

/// Materializes `access` into the input of `convOp`.
///
/// The gather is an all-parallel copy whose iteration space consists of the
/// gather loops of `access`. For an NCHW convolution with
/// `inputMap = (n, oc, oh, ow, c, kh, kw) -> (n, c, oh + kh, ow + kw)`, the
/// gather loops are (n, oh, ow, c, kh, kw), and
///   - the gather reads the input with `inputMap` rewritten onto its own
///     loops: `(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1 + d4, d2 + d5)`;
///   - the convolution then reads the gathered tensor with
///     `(n, oc, oh, ow, c, kh, kw) -> (n, oh, ow, c, kh, kw)`.
static GatheredInput gatherInput(RewriterBase &rewriter,
                                 linalg::LinalgOp convOp,
                                 const InputAccess &access) {
  MLIRContext *context = rewriter.getContext();
  Location loc = convOp.getLoc();
  Value input = convOp.getDpsInputs().front();
  SmallVector<OpFoldResult> loopSizes =
      llvm::map_to_vector(convOp.createLoopRanges(rewriter, loc),
                          [](Range range) { return range.size; });

  unsigned gatherRank = access.gatherLoops.size();
  SmallVector<AffineExpr> gatherDimToLoop;
  SmallVector<OpFoldResult> gatherSizes;
  for (unsigned loop : access.gatherLoops) {
    gatherDimToLoop.push_back(getAffineDimExpr(loop, context));
    gatherSizes.push_back(loopSizes[loop]);
  }

  Value init = tensor::EmptyOp::create(rewriter, loc, gatherSizes,
                                       getElementTypeOrSelf(input.getType()));
  auto gatherOp = linalg::GenericOp::create(
      rewriter, loc, init.getType(), input, init,
      ArrayRef<AffineMap>{access.readMap,
                          rewriter.getMultiDimIdentityMap(gatherRank)},
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
/// The pattern matches on that input access and not on the convolution
/// interface. Once unit dimensions are folded, a convolution whose filter
/// window has extent one along a strided axis reads its input with
/// `(c, oh * s, ow * s)` or `(c, oh * s, ow * s + kw * d)`. Neither is a
/// convolution to the interface, yet both keep a contraction from having one
/// loop per dimension kind in the same way, so they are gathered as well.
///
/// Filter window loops of extent one are substituted with zero before deciding
/// what to gather. When that leaves the input access a projected permutation,
/// as for an unstrided 1x1 convolution, the input is read in place.
///
/// The pattern only checks that the rewrite is valid; whether it is worth
/// applying to a given convolution is decided by `controlFn`.
///
/// TODO: Upstream this pattern and `getCollapsibleLoops` to MLIR, where they
/// would generalize `linalg::populateConvertConv2DToImg2ColPatterns` beyond
/// its four named 2-D convolutions.
struct ConvertConvToIm2Col final : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using ControlFn = std::function<bool(linalg::LinalgOp, const OperandLoops &)>;

  ConvertConvToIm2Col(MLIRContext *context, ControlFn controlFn)
      : OpInterfaceRewritePattern(context), controlFn(std::move(controlFn)) {}

  /// Returns the loops of `linalgOp` grouped by the operands that use them if
  /// the rewrite is valid for it, and otherwise sets `reason` and fails.
  static FailureOr<OperandLoops> matchInputAccess(linalg::LinalgOp linalgOp,
                                                  StringRef &reason) {
    // The gather is built on tensors, and the rewrite moves the op's body
    // region into the contraction.
    if (!linalgOp.hasPureTensorSemantics() || linalgOp->getNumRegions() != 1) {
      reason = "expected tensor semantics and a body region";
      return failure();
    }
    // A preset compilation info or lowering config refers to the loops of the
    // original op, which the rewrite replaces.
    if (getCompilationInfo(linalgOp) || getLoweringConfig(linalgOp)) {
      reason = "has a preset compilation info or lowering config";
      return failure();
    }
    // Requiring exactly an input and a filter excludes quantized convolutions
    // with zero-point operands; they are lowered before im2col runs.
    if (linalgOp.getNumDpsInputs() != 2 || linalgOp.getNumDpsInits() != 1) {
      reason = "expected an input, a filter and an output";
      return failure();
    }
    AffineMap inputMap =
        linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
    AffineMap filterMap =
        linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(1));
    AffineMap outputMap =
        linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
    if (!filterMap.isProjectedPermutation() ||
        !outputMap.isProjectedPermutation()) {
      reason = "filter and output maps are not projected permutations";
      return failure();
    }
    // An input read through a projected permutation needs no gather: the op is
    // a contraction already, and is left to the passes that handle those.
    if (inputMap.isProjectedPermutation()) {
      reason = "input access is a projected permutation";
      return failure();
    }
    if (inputMap.getNumSymbols() != 0 ||
        !llvm::all_of(inputMap.getResults(), isStridedSum)) {
      reason = "input access is not a sum of scaled loops";
      return failure();
    }

    OperandLoops loops;
    for (unsigned loop : llvm::seq<unsigned>(linalgOp.getNumLoops())) {
      bool inFilter = filterMap.isFunctionOfDim(loop);
      bool inOutput = outputMap.isFunctionOfDim(loop);
      if (!inputMap.isFunctionOfDim(loop)) {
        if (inFilter && inOutput) {
          loops.outputChannel.push_back(loop);
        }
        continue;
      }
      if (inFilter && inOutput) {
        loops.depth.push_back(loop);
      } else if (inOutput) {
        loops.image.push_back(loop);
      } else if (inFilter) {
        loops.reduced.push_back(loop);
      } else {
        // The gather is laid out by the output and filter maps, which do not
        // place this loop.
        reason = "input loop is used by neither the filter nor the output";
        return failure();
      }
    }
    // Without a loop the input and the filter reduce over together, the op is
    // not a contraction of the two, and gathering the input gains nothing.
    if (loops.reduced.empty()) {
      reason = "input and filter share no reduced loop";
      return failure();
    }
    loops.window = llvm::filter_to_vector(loops.reduced, [&](unsigned loop) {
      return !llvm::is_contained(inputMap.getResults(),
                                 getAffineDimExpr(loop, linalgOp.getContext()));
    });
    return loops;
  }

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    StringRef reason;
    FailureOr<OperandLoops> loops = matchInputAccess(linalgOp, reason);
    if (failed(loops)) {
      return rewriter.notifyMatchFailure(linalgOp, reason);
    }
    if (!controlFn(linalgOp, *loops)) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "rejected by the control function");
    }

    InputAccess access = getInputAccess(linalgOp, *loops);

    // Gather the input unless its access is already a projected permutation.
    Value newInput = linalgOp.getDpsInputs().front();
    AffineMap newInputMap = access.inputMap;
    if (!access.inputMap.isProjectedPermutation()) {
      GatheredInput gathered = gatherInput(rewriter, linalgOp, access);
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

private:
  // Decides which valid convolutions are rewritten.
  ControlFn controlFn;
};

/// Returns true if im2col can pay off for `convOp`, independent of the target.
/// The gather copies every input element once per filter window position it
/// falls into, and the contraction only amortizes that copy by reusing each
/// gathered element across output channels. Without an output channel loop of
/// extent other than one, i.e. for depthwise convolutions with a unit
/// multiplier and for pooling, nothing reuses the gather and the rewrite only
/// adds memory traffic. A dynamic output channel extent may exceed one, so it
/// counts as reuse.
static bool hasOutputChannelReuse(linalg::LinalgOp convOp,
                                  const OperandLoops &loops) {
  SmallVector<int64_t> loopSizes = convOp.getStaticLoopRanges();
  return llvm::any_of(loops.outputChannel,
                      [&](unsigned loop) { return loopSizes[loop] != 1; });
}

/// Identifies the gather the rewrite materializes for a convolution. Equal
/// keys yield identical gathers, which CSE merges into one.
struct GatherKey {
  Value input;
  AffineMap readMap;
  SmallVector<int64_t> sizes;

  bool operator==(const GatherKey &other) const {
    return input == other.input && readMap == other.readMap &&
           sizes == other.sizes;
  }
};

/// Returns the key of the gather the rewrite materializes for `convOp`, or
/// std::nullopt if it reads its input in place or the gather shape is dynamic.
static std::optional<GatherKey> getGatherKey(linalg::LinalgOp convOp,
                                             const OperandLoops &loops) {
  InputAccess access = getInputAccess(convOp, loops);
  if (access.inputMap.isProjectedPermutation()) {
    return std::nullopt;
  }
  SmallVector<int64_t> loopSizes = convOp.getStaticLoopRanges();
  SmallVector<int64_t> sizes = llvm::map_to_vector(
      access.gatherLoops, [&](unsigned loop) { return loopSizes[loop]; });
  if (ShapedType::isDynamicShape(sizes)) {
    return std::nullopt;
  }
  return GatherKey{convOp.getDpsInputs().front(), access.readMap,
                   std::move(sizes)};
}

struct ConvertConvToIm2ColPass final
    : impl::ConvertConvToIm2ColPassBase<ConvertConvToIm2ColPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // A convolution without output channel reuse still pays off when its
    // gather is the one another rewritten convolution materializes anyway,
    // e.g. the input window sum of a quantized convolution with a weight zero
    // point. Collect those gathers up front, so the decision does not depend
    // on the order in which the convolutions are rewritten.
    SmallVector<GatherKey> reusedGathers;
    getOperation()->walk([&](linalg::LinalgOp linalgOp) {
      StringRef reason;
      FailureOr<OperandLoops> loops =
          ConvertConvToIm2Col::matchInputAccess(linalgOp, reason);
      if (failed(loops) || !hasOutputChannelReuse(linalgOp, *loops)) {
        return;
      }
      if (std::optional<GatherKey> key = getGatherKey(linalgOp, *loops)) {
        reusedGathers.push_back(std::move(*key));
      }
    });
    auto controlFn = [&](linalg::LinalgOp linalgOp, const OperandLoops &loops) {
      if (hasOutputChannelReuse(linalgOp, loops)) {
        return true;
      }
      std::optional<GatherKey> key = getGatherKey(linalgOp, loops);
      return key && llvm::is_contained(reusedGathers, *key);
    };

    RewritePatternSet patterns(context);
    patterns.add<ConvertConvToIm2Col>(context, controlFn);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::GlobalOptimization
