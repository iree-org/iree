// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
template <typename Conv2DOpType>
class Convert1x1FilterConvToMatmul : public OpRewritePattern<Conv2DOpType> {
public:
  using OpRewritePattern<Conv2DOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOpType convOp,
                                PatternRewriter &rewriter) const override {
    auto inputShapeType = llvm::dyn_cast<RankedTensorType>(
        convOp.getDpsInputOperand(0)->get().getType());
    auto filterShapeType = llvm::dyn_cast<RankedTensorType>(
        convOp.getDpsInputOperand(1)->get().getType());
    auto outputShapeType = llvm::dyn_cast<RankedTensorType>(
        convOp.getDpsInitOperand(0)->get().getType());

    const bool isNCHW = isa<linalg::Conv2DNchwFchwOp>(convOp);
    const bool isNHWC = isa<linalg::Conv2DNhwcHwcfOp>(convOp);
    if (!isNCHW & !isNHWC)
      return failure();

    if (!inputShapeType || !filterShapeType || !outputShapeType)
      return failure();

    auto inputShape = inputShapeType.getShape();
    auto filterShape = filterShapeType.getShape();
    auto outputShape = outputShapeType.getShape();

    // Adjusting dimension indices based on Conv2DOpType.
    const int nIndex = 0;
    const int kcIndex = isNHWC ? 2 : 1;
    const int kfIndex = isNHWC ? 3 : 0;
    const int khIndex = isNHWC ? 0 : 2;
    const int kwIndex = isNHWC ? 1 : 3;
    const int ohIndex = isNHWC ? 1 : 2;
    const int owIndex = isNHWC ? 2 : 3;
    const int ocIndex = isNHWC ? 3 : 1;

    bool isInputHWDynamic = ShapedType::isDynamic(inputShape[ohIndex]) &&
                            ShapedType::isDynamic(inputShape[owIndex]);

    // We cannot merge the width and height if they are both dynamic as we
    // cannot expand them back to their dynamic values.
    if (isInputHWDynamic)
      return failure();

    if (filterShape[khIndex] != 1 || filterShape[kwIndex] != 1)
      return failure();

    // TODO(ataei): Support conversion to linalg.batch_matmul.
    if (inputShape[0] != 1)
      return failure();

    if (!llvm::all_of(convOp.getStrides(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();
    if (!llvm::all_of(convOp.getDilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    auto combineDims = [](int64_t a, int64_t b) {
      if (ShapedType::isDynamic(a) || ShapedType::isDynamic(b))
        return ShapedType::kDynamic;
      return a * b;
    };

    SmallVector<ReassociationIndices> reassociationInputOutputIndices;
    SmallVector<ReassociationIndices> reassociationFilterIndices;
    SmallVector<int64_t> reshapedInputShape(2, 0);
    SmallVector<int64_t> reshapedFilterShape(2, 0);
    SmallVector<int64_t> reshapedOutputShape(2, 0);
    if (isNHWC) {
      // Generate reassociation indices.
      reassociationInputOutputIndices = {{nIndex, ohIndex, owIndex}, {ocIndex}};
      reassociationFilterIndices = {{khIndex, kwIndex, kcIndex}, {kfIndex}};

      // Generate matmul shapes from 1x1 conv.
      reshapedInputShape = {
          combineDims(inputShape[ohIndex], inputShape[owIndex]),
          inputShape[ocIndex]};
      reshapedFilterShape = {filterShape[kcIndex], filterShape[kfIndex]};
      reshapedOutputShape = {
          combineDims(outputShape[ohIndex], outputShape[owIndex]),
          outputShape[ocIndex]};
    } else if (isNCHW) {
      // Generate reassociation indices.
      reassociationInputOutputIndices = {{nIndex, ocIndex}, {ohIndex, owIndex}};
      reassociationFilterIndices = {{kfIndex}, {kcIndex, khIndex, kwIndex}};

      // Generate matmul shapes from 1x1 conv.
      reshapedInputShape = {
          inputShape[ocIndex],
          combineDims(inputShape[ohIndex], inputShape[owIndex])};
      reshapedFilterShape = {filterShape[kfIndex], filterShape[kcIndex]};
      reshapedOutputShape = {
          outputShape[ocIndex],
          combineDims(outputShape[ohIndex], outputShape[owIndex])};
    }

    auto reshapedInputType = RankedTensorType::get(
        reshapedInputShape, inputShapeType.getElementType());

    auto reshapedFilterType = RankedTensorType::get(
        reshapedFilterShape, filterShapeType.getElementType());

    auto reshapedOutputType = RankedTensorType::get(
        reshapedOutputShape, outputShapeType.getElementType());

    Value input = convOp.getDpsInputOperand(0)->get();
    Value filter = convOp.getDpsInputOperand(1)->get();
    Value output = convOp.getDpsInitOperand(0)->get();
    auto loc = convOp.getLoc();

    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, reassociationInputOutputIndices);
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, reassociationFilterIndices);
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, reassociationInputOutputIndices);

    SmallVector<Value, 2> matmulInput;
    if (isNHWC) {
      matmulInput = {reshapedInput, reshapedFilter};
    } else if (isNCHW) {
      matmulInput = {reshapedFilter, reshapedInput};
    }
    auto matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType, matmulInput, ArrayRef<Value>{reshapedOutput});

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputShapeType, matmulResult.getResults()[0],
        reassociationInputOutputIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

struct Convert1X1FilterConv2DToMatmulPass
    : public Convert1X1FilterConv2DToMatmulBase<
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

std::unique_ptr<Pass> createConvert1X1FilterConv2DToMatmulPass() {
  return std::make_unique<Convert1X1FilterConv2DToMatmulPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
