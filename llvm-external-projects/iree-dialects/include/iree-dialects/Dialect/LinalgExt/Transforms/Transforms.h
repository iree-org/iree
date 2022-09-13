// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace scf {
class ForOp;
class ForeachThreadOp;
} // namespace scf
namespace linalg {
class LinalgOp;
}

namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

/// Pattern to swap a `TilingInterface` op -> `tensor::ExtractSliceOp`.
struct SwapTilingInterfaceOp : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  FailureOr<Operation *>
  returningMatchAndRewrite(tensor::ExtractSliceOp sliceOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(sliceOp, rewriter);
  }
};

/// Pattern to rewrite a scf::ForEachThreadOp to the async dialect.
struct ForeachThreadOpToAsyncRewriter
    : public OpRewritePattern<scf::ForeachThreadOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<Operation *>
  returningMatchAndRewrite(scf::ForeachThreadOp foreachThreadOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(scf::ForeachThreadOp foreachThreadOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(foreachThreadOp, rewriter);
  }
};

/// Pattern to rewrite a ForeachThreadOp to an scf::ForOp.
struct ForeachThreadOpToScfForRewriter
    : public OpRewritePattern<scf::ForeachThreadOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<scf::ForOp>
  returningMatchAndRewrite(scf::ForeachThreadOp foreachThreadOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(scf::ForeachThreadOp foreachThreadOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(foreachThreadOp, rewriter);
  }
};

struct FusionResult {
  TilingInterface consumerOp;
  SmallVector<TilingInterface> fusedOps;
};

/// Pattern to fuse the producers of a tileable op.
struct LinalgExtFusionPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  LinalgExtFusionPattern(MLIRContext *context, ArrayRef<int64_t> operandsToFuse)
      : OpInterfaceRewritePattern<TilingInterface>(context),
        operandsToFuse(operandsToFuse.begin(), operandsToFuse.end()) {}

  FailureOr<FusionResult>
  returningMatchAndRewrite(TilingInterface consumerOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface consumerOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(consumerOp, rewriter);
  }

private:
  SmallVector<int64_t> operandsToFuse;
};

//===----------------------------------------------------------------------===//
// Transformations exposed as patterns, moved from upstream MLIR as IREE still
// heavily relies on patterns that compose through filters.
// TODO: Deprecate all the patterns below.
//===----------------------------------------------------------------------===//
///
/// Linalg tiling pattern.
///
/// Apply the `tiling` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `tiling` for more details.
// TODO: TiledOpInterface
struct LinalgTilingPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  /// Construct a generic pattern applied to all LinalgOp that verify `filter`.
  LinalgTilingPattern(MLIRContext *context, linalg::LinalgTilingOptions options,
                      linalg::LinalgTransformationFilter f =
                          linalg::LinalgTransformationFilter(),
                      PatternBenefit benefit = 1);

  /// Construct a pattern specifically applied to `opName`.
  LinalgTilingPattern(StringRef opName, MLIRContext *context,
                      linalg::LinalgTilingOptions options,
                      linalg::LinalgTransformationFilter f =
                          linalg::LinalgTransformationFilter(),
                      PatternBenefit benefit = 1);

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<linalg::TiledLinalgOp>
  returningMatchAndRewrite(linalg::LinalgOp op,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  linalg::LinalgTransformationFilter filter;
  /// Options to control tiling;
  linalg::LinalgTilingOptions options;
};

template <typename... OpTypes>
class TilingPatterns;

template <>
class TilingPatterns<> {
public:
  static void insert(RewritePatternSet &patterns,
                     const linalg::LinalgTilingOptions &options,
                     const linalg::LinalgTransformationFilter &f) {}
};

template <typename OpTy, typename... OpTypes>
class TilingPatterns<OpTy, OpTypes...> {
public:
  static void insert(RewritePatternSet &patterns,
                     const linalg::LinalgTilingOptions &options,
                     const linalg::LinalgTransformationFilter &f) {
    patterns.add<LinalgTilingPattern>(OpTy::getOperationName(),
                                      patterns.getContext(), options, f);
    TilingPatterns<OpTypes...>::insert(patterns, options, f);
  }
};

// ///
// /// Linalg padding pattern.
// ///
// /// Apply the `padding` transformation as a pattern.
// /// `filter` controls LinalgTransformMarker matching and update when
// specified.
// /// See `padding` for more details.
// struct LinalgPaddingPattern
//     : public OpInterfaceRewritePattern<linalg::LinalgOp> {
//   /// Construct a generic pattern applied to all LinalgOp that verify
//   `filter`. LinalgPaddingPattern(
//       MLIRContext *context,
//       linalg::LinalgPaddingOptions options = linalg::LinalgPaddingOptions(),
//       linalg::LinalgTransformationFilter f =
//           linalg::LinalgTransformationFilter(),
//       PatternBenefit benefit = 1);

//   /// Construct a pattern specifically applied to `opName`.
//   LinalgPaddingPattern(
//       StringRef opName, MLIRContext *context,
//       linalg::LinalgPaddingOptions options = linalg::LinalgPaddingOptions(),
//       linalg::LinalgTransformationFilter f =
//           linalg::LinalgTransformationFilter(),
//       PatternBenefit benefit = 1);

//   /// `matchAndRewrite` implementation that returns the significant
//   transformed
//   /// pieces of IR.
//   FailureOr<linalg::LinalgOp> returningMatchAndRewrite(linalg::LinalgOp op,
//                                                PatternRewriter &rewriter)
//                                                const;

//   LogicalResult matchAndRewrite(linalg::LinalgOp op,
//                                 PatternRewriter &rewriter) const override {
//     return returningMatchAndRewrite(op, rewriter);
//   }

// private:
//   /// LinalgTransformMarker handles special attribute manipulations.
//   linalg::LinalgTransformationFilter filter;
//   /// Options to control padding and hoisting.
//   linalg::LinalgPaddingOptions options;
// };

// /// Rewrites 2-D convolution ops with size-1 window dimensions into 1-D
// /// convolution ops.
// struct DownscaleSizeOneWindowed2DConvolution final
//     : public OpRewritePattern<Conv2DNhwcHwcfOp> {
//   DownscaleSizeOneWindowed2DConvolution(
//       MLIRContext *context,
//       linalg::LinalgTransformationFilter f =
//       linalg::LinalgTransformationFilter(), PatternBenefit benefit = 1) :
//       OpRewritePattern<Conv2DNhwcHwcfOp>(context, benefit),
//         filter(std::move(f)) {}

//   FailureOr<Conv1DNwcWcfOp>
//   returningMatchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
//                            PatternRewriter &rewriter) const;

//   LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
//                                 PatternRewriter &rewriter) const override {
//     return returningMatchAndRewrite(convOp, rewriter);
//   }

// private:
//   /// LinalgTransformMarker handles special attribute manipulations.
//   linalg::LinalgTransformationFilter filter;
// };

// /// Rewrites 2-D depthwise convolution ops with size-1 (w, kw) or (h, kh)
// /// dimensions into 1-D depthwise convolution ops.
// struct DownscaleDepthwiseConv2DNhwcHwcOp final
//     : public OpRewritePattern<DepthwiseConv2DNhwcHwcOp> {
//   DownscaleDepthwiseConv2DNhwcHwcOp(
//       MLIRContext *context,
//       linalg::LinalgTransformationFilter f =
//       linalg::LinalgTransformationFilter(), PatternBenefit benefit = 1) :
//       OpRewritePattern<DepthwiseConv2DNhwcHwcOp>(context, benefit),
//         filter(std::move(f)) {}

//   FailureOr<DepthwiseConv1DNwcWcOp>
//   returningMatchAndRewrite(DepthwiseConv2DNhwcHwcOp convOp,
//                            PatternRewriter &rewriter) const;

//   LogicalResult matchAndRewrite(DepthwiseConv2DNhwcHwcOp convOp,
//                                 PatternRewriter &rewriter) const override {
//     return returningMatchAndRewrite(convOp, rewriter);
//   }

// private:
//   /// LinalgTransformMarker handles special attribute manipulations.
//   linalg::LinalgTransformationFilter filter;
// };

// ///
// /// Linalg tile and fuse tensor ops pattern.
// ///
// /// Apply tiling and fusion as a pattern.
// /// `filter` controls LinalgTransformMarker matching and update when
// specified.
// /// See `tileConsumerAndFuseProducers` for more details.
// struct LinalgTileAndFuseTensorOpsPattern : public RewritePattern {
//   // Entry point to match any LinalgOp.
//   LinalgTileAndFuseTensorOpsPattern(
//       MLIRContext *context, linalg::LinalgTilingAndFusionOptions options,
//       linalg::LinalgTransformationFilter f =
//       linalg::LinalgTransformationFilter(), PatternBenefit benefit = 1);
//   // Entry point to match a specific linalg::LinalgOp.
//   LinalgTileAndFuseTensorOpsPattern(
//       StringRef opName, MLIRContext *context,
//       linalg::LinalgTilingAndFusionOptions options,
//       linalg::LinalgTransformationFilter f =
//       linalg::LinalgTransformationFilter(), PatternBenefit benefit = 1);

//   /// `matchAndRewrite` implementation that returns the significant
//   transformed
//   /// pieces of IR.
//   FailureOr<TileLoopNest>
//   returningMatchAndRewrite(Operation *op, PatternRewriter &rewriter) const;

//   LogicalResult matchAndRewrite(Operation *op,
//                                 PatternRewriter &rewriter) const override {
//     return returningMatchAndRewrite(op, rewriter);
//   }

// private:
//   /// LinalgTransformMarker handles special attribute manipulations.
//   linalg::LinalgTransformationFilter filter;
//   /// Tile sizes and interchange used to tile the root operation.
//   linalg::LinalgTilingAndFusionOptions options;
// };

// ///
// /// Linalg generalization pattern.
// ///
// /// Apply the `generalization` transformation as a pattern.
// /// `filter` controls LinalgTransformMarker matching and update when
// specified.
// /// See `generalization` for more details.
// struct LinalgGeneralizationPattern
//     : public OpInterfaceRewritePattern<linalg::LinalgOp> {
//   /// Construct a generic pattern applied to all LinalgOp that verify
//   `filter`. LinalgGeneralizationPattern(
//       MLIRContext *context,
//       linalg::LinalgTransformationFilter f =
//       linalg::LinalgTransformationFilter(), PatternBenefit benefit = 1);

//   /// Construct a pattern specifically applied to `opName`.
//   LinalgGeneralizationPattern(
//       StringRef opName, MLIRContext *context,
//       linalg::LinalgTransformationFilter f =
//       linalg::LinalgTransformationFilter(), PatternBenefit benefit = 1);

//   /// `matchAndRewrite` implementation that returns the significant
//   transformed
//   /// pieces of IR.
//   FailureOr<GenericOp>
//   returningMatchAndRewrite(linalg::LinalgOp op, PatternRewriter &rewriter)
//   const;

//   LogicalResult matchAndRewrite(linalg::LinalgOp op,
//                                 PatternRewriter &rewriter) const override {
//     return returningMatchAndRewrite(op, rewriter);
//   }

// private:
//   /// LinalgTransformMarker handles special attribute manipulations.
//   linalg::LinalgTransformationFilter filter;
// };

// ///
// /// Linalg peeling patterns.
// ///

// /// Compute the loops to peel and return them in a SmallVector. Loops will be
// /// peeled in order of appearance in the SmallVector. This order will impact
// the
// /// output IR. If an inner-to-outer order is provided, the peeled iterations
// of
// /// the outer loops will also contain the peeled inner loops. If an
// /// outer-to-inner order is provided, the peeled iterations of the outer
// loops
// /// will not contain any peeled inner loops.
// using LoopsToPeelComputationFunction = std::function<void(
//     OpBuilder &, Operation *, SmallVectorImpl<scf::ForOp> &)>;

// struct LinalgPeelOptions {
//   LoopsToPeelComputationFunction loopsToPeelComputationFunction = nullptr;
// };

// /// `filter` controls LinalgTransformMarker matching and update when
// specified. struct LinalgPeelingPattern : public
// OpInterfaceRewritePattern<linalg::LinalgOp> {
//   /// Construct a generic pattern applied to all LinalgOp that verify
//   `filter`. LinalgPeelingPattern(
//       MLIRContext *context,
//       linalg::LinalgTransformationFilter f =
//       linalg::LinalgTransformationFilter(), linalg::LinalgPeelOptions options
//       = linalg::LinalgPeelOptions(), PatternBenefit benefit = 1);

//   /// Construct a pattern specifically applied to `opName`.
//   LinalgPeelingPattern(
//       StringRef opName, MLIRContext *context,
//       linalg::LinalgPeelOptions options = linalg::LinalgPeelOptions(),
//       linalg::LinalgTransformationFilter f =
//       linalg::LinalgTransformationFilter(), PatternBenefit benefit = 1);

//   LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
//                                 PatternRewriter &rewriter) const override;

// private:
//   /// LinalgTransformMarker handles special attribute manipulations.
//   const linalg::LinalgTransformationFilter filter;
//   /// Peeling options.
//   const linalg::LinalgPeelOptions options;
// };

// ///
// /// Linalg vectorization patterns.
// ///
// /// Empty for now, used for SFINAE purposes only.
// struct LinalgVectorizationOptions {};

// /// `filter` controls LinalgTransformMarker matching and update when
// specified.
// /// See `vectorizeLinalgOp` for more details.
// struct LinalgVectorizationPattern : public
// OpInterfaceRewritePattern<linalg::LinalgOp> {
//   /// Construct a generic pattern applied to all LinalgOp that verify
//   `filter`. LinalgVectorizationPattern(
//       MLIRContext *context,
//       linalg::LinalgTransformationFilter f =
//       linalg::LinalgTransformationFilter(),
//       linalg::LinalgVectorizationOptions options =
//       linalg::LinalgVectorizationOptions(), PatternBenefit benefit = 1);

//   /// Construct a pattern specifically applied to `opName`.
//   LinalgVectorizationPattern(
//       StringRef opName, MLIRContext *context,
//       linalg::LinalgVectorizationOptions options =
//       linalg::LinalgVectorizationOptions(),
//       linalg::LinalgTransformationFilter f =
//       linalg::LinalgTransformationFilter(), PatternBenefit benefit = 1);

//   LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
//                                 PatternRewriter &rewriter) const override;

// private:
//   /// LinalgTransformMarker handles special attribute manipulations.
//   linalg::LinalgTransformationFilter filter;
// };

///
/// Linalg promotion patterns.
///
/// Apply the `promoteSubViews` transformation as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `promoteSubViews` for more details.
struct LinalgBasePromotionPattern : public RewritePattern {
  /// Entry point to match any LinalgOp
  /// OpInterface. MatchAnyOpTag-based constructor
  /// with a mandatory `filter`.
  LinalgBasePromotionPattern(
      MLIRContext *context, linalg::LinalgTransformationFilter f,
      linalg::LinalgPromotionOptions options = linalg::LinalgPromotionOptions(),
      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        filter(std::move(f)), options(std::move(options)) {}
  /// Entry point to match a specific Linalg op.
  LinalgBasePromotionPattern(StringRef opName, MLIRContext *context,
                             linalg::LinalgPromotionOptions options,
                             linalg::LinalgTransformationFilter f =
                                 linalg::LinalgTransformationFilter(),
                             PatternBenefit benefit = 1)
      : RewritePattern(opName, benefit, context, {}), filter(std::move(f)),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();
    if (failed(promoteSubviewsPrecondition(op, options)))
      return failure();

    // TODO: We cannot use root update here. This
    // pattern is creating other ops, so if the
    // promotion fails, those need to be cleaned
    // up, which doesnt seem to be happening here.
    // So to fail properly, we should be cloning
    // the op and deleting the previous op. This
    // needs more investigation.
    rewriter.startRootUpdate(op);
    Optional<linalg::LinalgOp> promotedOp =
        promoteSubViews(rewriter, op, options);
    if (!promotedOp) {
      rewriter.cancelRootUpdate(op);
      return op->emitError("subview promotion failed");
    }
    rewriter.finalizeRootUpdate(op);
    filter.replaceLinalgTransformationFilter(rewriter, op);
    return success();
  }

private:
  /// LinalgTransformMarker handles special
  /// attribute manipulations.
  linalg::LinalgTransformationFilter filter;
  /// Promotion options.
  linalg::LinalgPromotionOptions options;
};

template <typename OpTy>
struct LinalgPromotionPattern : public LinalgBasePromotionPattern {
  /// SFINAE: This constructor can only trigger for
  /// concrete ops that have a static
  /// `getOperationName` method.
  template <typename ConcreateOpTy = OpTy>
  LinalgPromotionPattern(MLIRContext *context,
                         linalg::LinalgPromotionOptions options,
                         linalg::LinalgTransformationFilter f =
                             linalg::LinalgTransformationFilter(),
                         PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(OpTy::getOperationName(), context, options,
                                   f, benefit) {}
  /// This constructor is available to anyone.
  LinalgPromotionPattern(StringRef opName, MLIRContext *context,
                         linalg::LinalgPromotionOptions options,
                         linalg::LinalgTransformationFilter f =
                             linalg::LinalgTransformationFilter(),
                         PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(opName, context, options, f, benefit) {}
};

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
