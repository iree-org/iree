// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_CONVERT1X1FILTERCONV2DTOMATMULPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

/// Create a linalg.transpose operation that permutes the dimensions of
/// `source` according to `perm`. Return the transposed tensor value.
static Value createTranspose(OpBuilder &builder, Value source,
                             SmallVector<int64_t> perm) {
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, source.getLoc(), source);
  applyPermutationToVector(mixedSizes, perm);
  Type elemType = cast<RankedTensorType>(source.getType()).getElementType();
  Value empty =
      tensor::EmptyOp::create(builder, source.getLoc(), mixedSizes, elemType)
          .getResult();
  return linalg::TransposeOp::create(builder, source.getLoc(), source, empty,
                                     perm)
      ->getResult(0);
}

// Converts 1x1 filter convolution ops to matmul-like operations
template <typename Conv2DOpType>
class Convert1x1FilterConvToMatmul : public OpRewritePattern<Conv2DOpType> {
public:
  using OpRewritePattern<Conv2DOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOpType convOp,
                                PatternRewriter &rewriter) const override {
    auto filterShapeType = dyn_cast<RankedTensorType>(
        convOp.getDpsInputOperand(1)->get().getType());
    if (!filterShapeType) {
      return failure();
    }

    constexpr bool isNCHWFchw =
        std::is_same_v<linalg::Conv2DNchwFchwOp, Conv2DOpType>;
    constexpr bool isNHWCHwcf =
        std::is_same_v<linalg::Conv2DNhwcHwcfOp, Conv2DOpType>;
    constexpr bool isNHWCFhwc =
        std::is_same_v<linalg::Conv2DNhwcFhwcOp, Conv2DOpType>;
    static_assert(isNCHWFchw || isNHWCHwcf || isNHWCFhwc);

    auto filterShape = filterShapeType.getShape();

    constexpr int64_t numLoops = 7;

    // Adjusting dimension indices based on Conv2DOpType.
    // Filter layouts:
    // - HWCF: [Kh, Kw, Cin, Cout]
    // - FHWC: [Cout, Kh, Kw, Cin]
    // - FCHW: [Cout, Cin, Kh, Kw]
    constexpr int khIndex = isNHWCHwcf ? 0 : (isNHWCFhwc ? 1 : 2);
    constexpr int kwIndex = isNHWCHwcf ? 1 : (isNHWCFhwc ? 2 : 3);
    // Loop indices for Kh and Kw in the iteration space
    constexpr int khLoopIndex = (isNHWCHwcf || isNHWCFhwc) ? 4 : 5;
    constexpr int kwLoopIndex = (isNHWCHwcf || isNHWCFhwc) ? 5 : 6;

    if (filterShape[khIndex] != 1 || filterShape[kwIndex] != 1) {
      return failure();
    }

    // Set insertion point before conv op to ensure transpose is created first
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(convOp);

    // Permutation for FHWC layout transpose:
    // FHWC: [Cout, Kh, Kw, Cin] -> [Cin, Kh, Kw, Cout] (perm = [3, 1, 2, 0])
    SmallVector<int64_t> permutation{3, 1, 2, 0};

    // For FHWC layout, insert a transpose on the filter to swap Cout and Cin
    // dimensions. This ensures that after unit extent dims are folded, the
    // filter has the correct [Cin, Cout] shape for matmul.
    if (isNHWCFhwc) {
      Value filterOperand = convOp.getDpsInputOperand(1)->get();

      // Create the transpose operation before generalization
      Value transposedFilter =
          createTranspose(rewriter, filterOperand, permutation);

      // Update the conv op to use the transposed filter
      rewriter.modifyOpInPlace(
          convOp, [&]() { convOp.setOperand(1, transposedFilter); });
    }

    SmallVector<AffineExpr> dimReplacements;
    for (int i = 0; i < numLoops; i++) {
      if (llvm::is_contained({khLoopIndex, kwLoopIndex}, i)) {
        dimReplacements.push_back(
            getAffineConstantExpr(0, rewriter.getContext()));
      } else {
        dimReplacements.push_back(getAffineDimExpr(i, rewriter.getContext()));
      }
    }

    SmallVector<AffineMap> newMaps = convOp.getIndexingMapsArray();
    AffineMap inputMap = newMaps[0];
    SmallVector<AffineExpr> newExprs =
        llvm::map_to_vector(inputMap.getResults(), [&](AffineExpr resultExpr) {
          return resultExpr.replaceDims(dimReplacements);
        });
    newMaps[0] = AffineMap::get(inputMap.getNumDims(), inputMap.getNumSymbols(),
                                newExprs, rewriter.getContext());

    // Update the indexing map for the filter (operand 1) if transposed
    if (isNHWCFhwc) {
      // After transpose, the filter indexing needs to be updated
      // FHWC: (d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6) becomes
      //       (d0, d1, d2, d3, d4, d5, d6) -> (d6, d4, d5, d3)
      AffineMap filterMap = newMaps[1];
      SmallVector<AffineExpr> filterExprs = llvm::map_to_vector(
          filterMap.getResults(), [&](AffineExpr expr) { return expr; });

      // Apply the permutation to the filter map expressions
      SmallVector<AffineExpr> permutedExprs;
      for (int64_t idx : permutation) {
        permutedExprs.push_back(filterExprs[idx]);
      }

      newMaps[1] = AffineMap::get(filterMap.getNumDims(),
                                  filterMap.getNumSymbols(), permutedExprs,
                                  rewriter.getContext());
    }

    FailureOr<linalg::GenericOp> genericOp =
        linalg::generalizeNamedOp(rewriter, convOp);
    if (failed(genericOp))
      return failure();
    genericOp->setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newMaps));

    return success();
  }
};

struct Convert1X1FilterConv2DToMatmulPass
    : public impl::Convert1X1FilterConv2DToMatmulPassBase<
          Convert1X1FilterConv2DToMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<Convert1x1FilterConvToMatmul<linalg::Conv2DNhwcHwcfOp>,
                    Convert1x1FilterConvToMatmul<linalg::Conv2DNhwcFhwcOp>,
                    Convert1x1FilterConvToMatmul<linalg::Conv2DNchwFchwOp>>(
        context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
