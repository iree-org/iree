// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_CONVERT1X1FILTERCONV2DTOMATMULPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
template <typename Conv2DOpType>
class Convert1x1FilterConvToMatmul : public OpRewritePattern<Conv2DOpType> {
public:
  using OpRewritePattern<Conv2DOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOpType convOp,
                                PatternRewriter &rewriter) const override {
    auto filterShapeType = llvm::dyn_cast<RankedTensorType>(
        convOp.getDpsInputOperand(1)->get().getType());
    if (!filterShapeType)
      return failure();

    constexpr bool isNCHW =
        std::is_same_v<linalg::Conv2DNchwFchwOp, Conv2DOpType>;
    constexpr bool isNHWC =
        std::is_same_v<linalg::Conv2DNhwcHwcfOp, Conv2DOpType>;
    static_assert(isNCHW || isNHWC);

    auto filterShape = filterShapeType.getShape();

    constexpr int64_t numLoops = 7;

    // Adjusting dimension indices based on Conv2DOpType.
    constexpr int khIndex = isNHWC ? 0 : 2;
    constexpr int kwIndex = isNHWC ? 1 : 3;
    constexpr int khLoopIndex = isNHWC ? 4 : 5;
    constexpr int kwLoopIndex = isNHWC ? 5 : 6;

    if (filterShape[khIndex] != 1 || filterShape[kwIndex] != 1)
      return failure();

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

    auto genericOp = linalg::generalizeNamedOp(rewriter, convOp).value();
    genericOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newMaps));
    return success();
  }
};

struct Convert1X1FilterConv2DToMatmulPass
    : public impl::Convert1X1FilterConv2DToMatmulPassBase<
          Convert1X1FilterConv2DToMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<Convert1x1FilterConvToMatmul<linalg::Conv2DNhwcHwcfOp>,
                    Convert1x1FilterConvToMatmul<linalg::Conv2DNchwFchwOp>>(
        context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
