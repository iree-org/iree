// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_CONVERTCONV2DTOIM2COLOPPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = llvm::isa<IntegerType>(x.getType());
  if (isInt)
    return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = llvm::isa<IntegerType>(x.getType());
  if (isInt)
    return builder.create<arith::MulIOp>(loc, x, y);
  return builder.create<arith::MulFOp>(loc, x, y);
}

namespace {

using ControlFnTy = std::function<bool(Operation *)>;

// Convert linalg.conv_2d_nhwc_hwcf into linalg.generic (for img2col packing)
// and linalg.matmul.
//
// A convolution operaton can be written as a matrix-matrix multiplication by
// unfolding the cross correlation between input and filter and explicitly copy
// overlapped sliding window inputs.
//
// Consider 2D input X with single channel input and output and 2x2 filter W:
// [x(0, 0)  , x(0, 1)  , ...,   x(0, n)  ]
// [x(1, 0)  , x(1, 1)  , ...,   x(1, n)  ]
// [.        ,  .       ,.   ,      .     ]            [w(0, 0), w(0, 1)]
// [.        ,  .       , .  ,      .     ]    (conv)  [w(1, 0), w(1, 1)]
// [.        ,  .       ,   .,      .     ]
// [x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]
//
// The packed input data (img2col) is a matrix with |rows| = output spatial
// size, |columns| = filter spatial size. To compute the output Y(i, j) we need
// to calculate the dot product between filter window at input X(x, y)) and the
// filter which will look like the following where r.h.s is the img2col matrix
// and l.h.s is the flattned filter:
//
// clang-format off
// [x(0, 0), x(0, 1), x(1, 0), x(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)] (matmul) [w(0, 0), w(0, 1), w(1, 0), w(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)]
// [   .   ,    .   ,    .   ,    .   ]
// clang-format on
//
// In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
// and output (N, Ho, Wo, D) the convolutin is the following matrix-matrix
// multiplication (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in
// the N input. For the case where N > 1 its a batched matrxi-matrix
// multplication.
class ConvertConv2DNhwcHwcf final
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  ConvertConv2DNhwcHwcf(MLIRContext *context,
                        std::optional<ControlFnTy> controlFn)
      : OpRewritePattern<linalg::Conv2DNhwcHwcfOp>(context),
        controlFn(controlFn) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    if (controlFn.has_value() && !controlFn.value()(convOp)) {
      return rewriter.notifyMatchFailure(convOp, "controlFn failed.");
    }

    auto inputType = llvm::cast<ShapedType>(convOp.getInputs()[0].getType());
    auto filterType = llvm::cast<ShapedType>(convOp.getInputs()[1].getType());
    auto outputType = llvm::cast<ShapedType>(convOp.getOutputs()[0].getType());

    if (!filterType.hasStaticShape() || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(convOp, [](Diagnostic &diag) {
        diag << "[unimplemented] "
             << "expected 'filterType' and 'inputType' to have static shapes.";
      });
    }

    // TODO: Support dilation.
    if (!hasAllOneValues(convOp.getDilations())) {
      return rewriter.notifyMatchFailure(convOp, [](Diagnostic &diag) {
        diag << "[unimplemented] "
             << "expected no dilations (expected dilations to all be one).";
      });
    }

    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];

    auto filterShape = filterType.getShape();
    auto outputShape = outputType.getShape();

    const int n = outputShape[0];
    const int oh = outputShape[1];
    const int ow = outputShape[2];
    const int oc = outputShape[3];
    const int fh = filterShape[0];
    const int fw = filterShape[1];
    const int ic = filterShape[2];

    auto loc = convOp.getLoc();

    SmallVector<int64_t> colTensorShape = {n, oh, ow, fh * fw * ic};

    SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1, 2}, {3}};

    Value colTensor = rewriter.create<tensor::EmptyOp>(
        loc, colTensorShape, inputType.getElementType());
    SmallVector<int64_t> strides(convOp.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(convOp.getDilations().getValues<int64_t>());
    SmallVector<OpFoldResult> kernelSize = {rewriter.getIndexAttr(fh),
                                            rewriter.getIndexAttr(fw)};
    OpFoldResult zero = rewriter.getIndexAttr(0);
    OpFoldResult one = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> mOffset = {zero, zero};
    SmallVector<OpFoldResult> mBasis = {rewriter.getIndexAttr(ow), one};
    SmallVector<OpFoldResult> kOffset = {zero};
    SmallVector<OpFoldResult> kBasis = {one};
    SmallVector<int64_t> batchPos = {0};
    SmallVector<int64_t> mPos = {1, 2};
    SmallVector<int64_t> kPos = {3};
    Value img2ColTensor = rewriter
                              .create<IREE::LinalgExt::Im2colOp>(
                                  loc, input, /*output=*/colTensor, strides,
                                  dilations, kernelSize, mOffset, mBasis,
                                  kOffset, kBasis, batchPos, mPos, kPos)
                              .getResult(0);

    SmallVector<ReassociationIndices> filterReassocIndices = {{0, 1, 2}, {3}};
    auto reshapedFilterType =
        RankedTensorType::get({fh * fw * ic, oc}, inputType.getElementType());

    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, filterReassocIndices);

    SmallVector<AffineMap> indexingMaps =
        getIGEMMContractionIndexingMaps(convOp).value();
    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    SmallVector<utils::IteratorType> genericIterators = {
        parallel, parallel, parallel, parallel, reduction};
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, outputType,
        /*inputs=*/ValueRange{img2ColTensor, reshapedFilter},
        /*outputs=*/ValueRange{output}, indexingMaps, genericIterators,
        [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value lhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[0],
                                           args[2].getType(),
                                           /*isUnsignedCast=*/false);
          Value rhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[1],
                                           args[2].getType(),
                                           /*isUnsignedCast=*/false);
          Value mul = createMul(nestedLoc, lhs, rhs, nestedBuilder);
          Value add = createAdd(nestedLoc, mul, args[2], nestedBuilder);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
        },
        linalg::getPrunedAttributeList(convOp));
    Value result = genericOp.getResults().front();

    rewriter.replaceOp(convOp, result);

    return success();
  }

private:
  std::optional<ControlFnTy> controlFn;
};

// For nchw, because the channels are to the left of the image shape dimensions,
// the position of the contraction dimension in the resulting matmul is
// reversed. This swaps the LHS and RHS of the matmul when compared with nhwc
// (i.e. (D, C x Kh x Kw) * (C x Kh x Kw, Ho x Wo))
class ConvertConv2DNchwFchw final
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  ConvertConv2DNchwFchw(MLIRContext *context,
                        std::optional<ControlFnTy> controlFn)
      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context),
        controlFn(controlFn) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    if (controlFn.has_value() && !controlFn.value()(convOp)) {
      return rewriter.notifyMatchFailure(convOp, "controlFn failed.");
    }

    auto inputType = llvm::cast<ShapedType>(convOp.getInputs()[0].getType());
    auto filterType = llvm::cast<ShapedType>(convOp.getInputs()[1].getType());
    auto outputType = llvm::cast<ShapedType>(convOp.getOutputs()[0].getType());

    if (!filterType.hasStaticShape() || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(convOp, [](Diagnostic &diag) {
        diag << "[unimplemented] "
             << "expected 'filterType' and 'inputType' to have static shapes.";
      });
    }

    // TODO: Support dilation.
    if (!hasAllOneValues(convOp.getDilations()))
      return rewriter.notifyMatchFailure(convOp, [](Diagnostic &diag) {
        diag << "[unimplemented] "
             << "expected no dilations (expected dilations to all be one).";
      });

    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];

    auto filterShape = filterType.getShape();
    auto outputShape = outputType.getShape();

    const int n = outputShape[0];
    const int oc = outputShape[1];
    const int oh = outputShape[2];
    const int ow = outputShape[3];
    const int ic = filterShape[1];
    const int fh = filterShape[2];
    const int fw = filterShape[3];

    auto loc = convOp.getLoc();

    SmallVector<int64_t> colTensorShape = {n, oh, ow, fh * fw * ic};

    Value colTensor = rewriter.create<tensor::EmptyOp>(
        loc, colTensorShape, inputType.getElementType());
    SmallVector<int64_t> strides(convOp.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(convOp.getDilations().getValues<int64_t>());
    SmallVector<OpFoldResult> kernelSize = {rewriter.getIndexAttr(fh),
                                            rewriter.getIndexAttr(fw)};
    OpFoldResult zero = rewriter.getIndexAttr(0);
    OpFoldResult one = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> mOffset = {zero, zero};
    SmallVector<OpFoldResult> mBasis = {rewriter.getIndexAttr(ow), one};
    SmallVector<OpFoldResult> kOffset = {zero};
    SmallVector<OpFoldResult> kBasis = {one};
    SmallVector<int64_t> batchPos = {0};
    SmallVector<int64_t> mPos = {2, 3};
    SmallVector<int64_t> kPos = {1};
    Value img2ColTensor = rewriter
                              .create<IREE::LinalgExt::Im2colOp>(
                                  loc, input, /*output=*/colTensor, strides,
                                  dilations, kernelSize, mOffset, mBasis,
                                  kOffset, kBasis, batchPos, mPos, kPos)
                              .getResult(0);

    SmallVector<ReassociationIndices> filterReassocIndices = {{0}, {1, 2, 3}};
    auto reshapedFilterType =
        RankedTensorType::get({oc, fh * fw * ic}, inputType.getElementType());
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, filterReassocIndices);

    SmallVector<AffineMap> indexingMaps =
        getIGEMMContractionIndexingMaps(convOp).value();
    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    SmallVector<utils::IteratorType> genericIterators = {
        parallel, parallel, parallel, parallel, reduction};
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, outputType,
        /*inputs=*/ValueRange{reshapedFilter, img2ColTensor},
        /*outputs=*/ValueRange{output}, indexingMaps, genericIterators,
        [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value lhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[0],
                                           args[2].getType(),
                                           /*isUnsignedCast=*/false);
          Value rhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[1],
                                           args[2].getType(),
                                           /*isUnsignedCast=*/false);
          Value mul = createMul(nestedLoc, lhs, rhs, nestedBuilder);
          Value add = createAdd(nestedLoc, mul, args[2], nestedBuilder);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
        },
        linalg::getPrunedAttributeList(convOp));
    Value result = genericOp.getResults().front();

    rewriter.replaceOp(convOp, result);

    return success();
  }

private:
  std::optional<ControlFnTy> controlFn;
};

struct ConvertConv2DToIm2ColOpPass final
    : impl::ConvertConv2DToIm2ColOpPassBase<ConvertConv2DToIm2ColOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateConv2DToIm2colOpPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateConv2DToIm2colOpPatterns(RewritePatternSet &patterns,
                                      std::optional<ControlFnTy> controlFn) {
  patterns.insert<ConvertConv2DNhwcHwcf, ConvertConv2DNchwFchw>(
      patterns.getContext(), std::move(controlFn));
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
