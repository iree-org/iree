// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

static Value createAdd(Location loc, Value x, Value y, bool isInt,
                       OpBuilder &builder) {
  if (isInt)
    return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, bool isInt,
                       OpBuilder &builder) {
  if (isInt)
    return builder.create<arith::MulIOp>(loc, x, y);
  return builder.create<arith::MulFOp>(loc, x, y);
}

namespace {

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

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
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

    SmallVector<int64_t> colTensorShape = {n, oh, ow, fh, fw, ic};

    Value colTensor = rewriter.create<tensor::EmptyOp>(
        loc, colTensorShape, inputType.getElementType());

    AffineExpr nDim, ohDim, owDim, khDim, kwDim, icDim;
    bindDims(getContext(), nDim, ohDim, owDim, khDim, kwDim, icDim);

    auto shSym = rewriter.getAffineConstantExpr(
        convOp.getStrides().getValues<int64_t>()[0]);
    auto swSym = rewriter.getAffineConstantExpr(
        convOp.getStrides().getValues<int64_t>()[1]);

    SmallVector<AffineExpr> inputExprs = {nDim, ohDim * shSym + khDim,
                                          owDim * swSym + kwDim, icDim};

    auto nloops = colTensorShape.size();

    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    SmallVector<utils::IteratorType, 3> img2colIterators(nloops, parallel);

    SmallVector<AffineMap> img2colIndexingMaps = {
        AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

    auto img2ColTensor = rewriter.create<linalg::GenericOp>(
        loc, colTensor.getType(),
        /*inputs=*/input, /*outputs=*/colTensor, img2colIndexingMaps,
        img2colIterators,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });

    SmallVector<ReassociationIndices> img2ColTensorReassocIndices;
    SmallVector<ReassociationIndices> outputReassocIndices;
    RankedTensorType reshapedImg2ColTensorType, reshapedOutputType;
    if (n == 1) {
      img2ColTensorReassocIndices = {{0, 1, 2}, {3, 4, 5}};
      outputReassocIndices = {{0, 1, 2}, {3}};

      reshapedImg2ColTensorType = RankedTensorType::get(
          {oh * ow, fh * fw * ic}, inputType.getElementType());
      reshapedOutputType =
          RankedTensorType::get({oh * ow, oc}, outputType.getElementType());
    } else {
      img2ColTensorReassocIndices = {{0}, {1, 2}, {3, 4, 5}};
      outputReassocIndices = {{0}, {1, 2}, {3}};

      reshapedImg2ColTensorType = RankedTensorType::get(
          {n, oh * ow, fh * fw * ic}, inputType.getElementType());
      reshapedOutputType =
          RankedTensorType::get({n, oh * ow, oc}, outputType.getElementType());
    }

    SmallVector<ReassociationIndices> filterReassocIndices = {{0, 1, 2}, {3}};
    auto reshapedFilterType =
        RankedTensorType::get({fh * fw * ic, oc}, inputType.getElementType());

    Value reshapedImg2ColTensor = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
        img2ColTensorReassocIndices);

    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, filterReassocIndices);

    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, outputReassocIndices);

    Value result;
    if (n == 1) {
      auto matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, reshapedOutputType,
          ArrayRef<Value>{reshapedImg2ColTensor, reshapedFilter},
          ArrayRef<Value>{reshapedOutput});
      result = matmulOp.getResults().front();
    } else {
      // For cases where batch is not 1, we need to keep the batch dimension
      // separate. Because the filter does not share the same batch dimension,
      // the batch dimension is only used in indexing the input and output. Thus
      // we cannot use existing linalg named ops like linalg.batch_matmul.
      // i.e. (B x) M x K * K x N = (B x) M x N
      AffineExpr bDim, mDim, nDim, kDim;
      bindDims(getContext(), bDim, mDim, nDim, kDim);
      auto lhsMap = AffineMap::get(4, 0, {bDim, mDim, kDim}, getContext());
      auto rhsMap = AffineMap::get(4, 0, {kDim, nDim}, getContext());
      auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, getContext());
      SmallVector<utils::IteratorType> genericIterators = {parallel, parallel,
                                                           parallel, reduction};
      bool isInt = llvm::isa<IntegerType>(outputType.getElementType());
      auto genericOp = rewriter.create<linalg::GenericOp>(
          loc, reshapedOutputType,
          /*inputs=*/ValueRange{reshapedImg2ColTensor, reshapedFilter},
          /*outputs=*/ValueRange{reshapedOutput},
          ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value mul = createMul(loc, args[0], args[1], isInt, nestedBuilder);
            Value add = createAdd(loc, mul, args[2], isInt, nestedBuilder);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
          });
      result = genericOp.getResults().front();
    }

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputType, result, outputReassocIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

// Similar to the conv pattern above except there is no reduction among the
// input channels so each convolution can be a matrix-vector product and
// by transposing both input filter so channels are outer most the computation
// is a batched matrix-vector product.
class ConvertDepthwiseConv2DNhwcHwc final
    : public OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcOp> {
public:
  using OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNhwcHwcOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputType =
        llvm::cast<RankedTensorType>(convOp.getInputs()[0].getType());
    auto filterType =
        llvm::cast<RankedTensorType>(convOp.getInputs()[1].getType());
    auto outputType =
        llvm::cast<RankedTensorType>(convOp.getOutputs()[0].getType());

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

    auto loc = convOp.getLoc();

    auto transposeOperand = [&](Value operand, ArrayRef<int64_t> indices) {
      auto operandTensorType = llvm::cast<RankedTensorType>(operand.getType());
      auto nloops = indices.size();
      auto inputShape = operandTensorType.getShape();

      SmallVector<AffineExpr> exprs =
          llvm::map_to_vector(indices, [&](int64_t index) -> AffineExpr {
            return rewriter.getAffineDimExpr(index);
          });

      SmallVector<int64_t> targetShape = llvm::map_to_vector(
          indices, [&](int64_t index) -> int64_t { return inputShape[index]; });

      Value outputTensor = rewriter.create<tensor::EmptyOp>(
          loc, targetShape, operandTensorType.getElementType());

      SmallVector<utils::IteratorType> loopAttributeTypes(
          nloops, utils::IteratorType::parallel);

      SmallVector<AffineMap> indexingMaps = {
          inversePermutation(
              AffineMap::get(nloops, 0, exprs, rewriter.getContext())),
          AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

      auto transposedOp = rewriter.create<linalg::GenericOp>(
          loc, outputTensor.getType(),
          /*inputs=*/operand, /*outputs=*/outputTensor, indexingMaps,
          loopAttributeTypes,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
          });

      return transposedOp.getResult(0);
    };

    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];

    // Transpose input, filter so channels are outermost
    auto inputT = transposeOperand(input, {0, 3, 1, 2});
    auto filterT = transposeOperand(filter, {2, 0, 1});
    auto filterTShape =
        llvm::cast<RankedTensorType>(filterT.getType()).getShape();
    auto outputShape = outputType.getShape();

    const int n = outputShape[0];
    const int oh = outputShape[1];
    const int ow = outputShape[2];
    const int c = outputShape[3];
    const int fh = filterTShape[1];
    const int fw = filterTShape[2];

    SmallVector<int64_t> colTensorShape = {n, c, oh, ow, fh, fw};
    Value transposedOutputTensor = transposeOperand(output, {0, 3, 1, 2});

    AffineExpr nDim, cDim, ohDim, owDim, khDim, kwDim;
    bindDims(getContext(), nDim, cDim, ohDim, owDim, khDim, kwDim);

    auto shSym = rewriter.getAffineConstantExpr(
        convOp.getStrides().getValues<int64_t>()[0]);
    auto swSym = rewriter.getAffineConstantExpr(
        convOp.getStrides().getValues<int64_t>()[1]);

    SmallVector<AffineExpr> inputExprs = {nDim, cDim, ohDim * shSym + khDim,
                                          owDim * swSym + kwDim};

    auto nloops = colTensorShape.size();

    SmallVector<utils::IteratorType> loopAttributeTypes(
        nloops, utils::IteratorType::parallel);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

    Value colTensor = rewriter.create<tensor::EmptyOp>(
        loc, colTensorShape, inputType.getElementType());

    auto img2ColTensor = rewriter.create<linalg::GenericOp>(
        loc, colTensor.getType(),
        /*inputs=*/inputT, /*outputs=*/colTensor, indexingMaps,
        loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });

    SmallVector<ReassociationIndices> img2ColTensorReassocIndices = {
        {0, 1}, {2, 3}, {4, 5}};
    SmallVector<ReassociationIndices> filterReassociationIndice = {{0}, {1, 2}};
    SmallVector<ReassociationIndices> outputReassociationIndice = {{0, 1},
                                                                   {2, 3}};

    auto reshapedImg2ColTensorType = RankedTensorType::get(
        {n * c, oh * ow, fh * fw}, inputType.getElementType());
    auto reshapedFilterTensorType =
        RankedTensorType::get({c, fh * fw}, filterType.getElementType());
    auto reshapedOutputTensorType =
        RankedTensorType::get({n * c, oh * ow}, outputType.getElementType());

    Value reshapedImg2ColTensor = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
        img2ColTensorReassocIndices);
    Value reshapedFilterTensor = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterTensorType, filterT, filterReassociationIndice);
    Value reshapedoutputTensor = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputTensorType, transposedOutputTensor,
        outputReassociationIndice);

    auto batchMatVecResult = rewriter.create<linalg::BatchMatvecOp>(
        loc, TypeRange{reshapedoutputTensor.getType()},
        ValueRange{reshapedImg2ColTensor, reshapedFilterTensor},
        ValueRange{reshapedoutputTensor});

    SmallVector<ReassociationIndices> batchMatVecReassociationIndice = {{0, 1},
                                                                        {2, 3}};

    Value batchMatVecResultReshaped = rewriter.create<tensor::ExpandShapeOp>(
        loc, transposedOutputTensor.getType(), batchMatVecResult.getResult(0),
        batchMatVecReassociationIndice);

    auto transposedResult =
        transposeOperand(batchMatVecResultReshaped, {0, 2, 3, 1});

    rewriter.replaceOp(convOp, ArrayRef<Value>{transposedResult});
    return success();
  }
};

// For nchw, because the channels are to the left of the image shape dimensions,
// the position of the contraction dimension in the resulting matmul is
// reversed. This swaps the LHS and RHS of the matmul when compared with nhwc
// (i.e. (D, C x Kh x Kw) * (C x Kh x Kw, Ho x Wo))
class ConvertConv2DNchwFchw final
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
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

    SmallVector<int64_t> colTensorShape = {n, ic, fh, fw, oh, ow};

    Value colTensor = rewriter.create<tensor::EmptyOp>(
        loc, colTensorShape, inputType.getElementType());

    AffineExpr nDim, icDim, khDim, kwDim, ohDim, owDim;
    bindDims(getContext(), nDim, icDim, khDim, kwDim, ohDim, owDim);

    auto shSym = rewriter.getAffineConstantExpr(
        convOp.getStrides().getValues<int64_t>()[0]);
    auto swSym = rewriter.getAffineConstantExpr(
        convOp.getStrides().getValues<int64_t>()[1]);

    SmallVector<AffineExpr> inputExprs = {nDim, icDim, ohDim * shSym + khDim,
                                          owDim * swSym + kwDim};

    auto nloops = colTensorShape.size();

    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    SmallVector<utils::IteratorType, 3> img2colIterators(nloops, parallel);

    SmallVector<AffineMap> img2colIndexingMaps = {
        AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

    auto img2ColTensor = rewriter.create<linalg::GenericOp>(
        loc, colTensor.getType(),
        /*inputs=*/input, /*outputs=*/colTensor, img2colIndexingMaps,
        img2colIterators,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });

    SmallVector<ReassociationIndices> filterReassocIndices = {{0}, {1, 2, 3}};
    auto reshapedFilterType =
        RankedTensorType::get({oc, fh * fw * ic}, inputType.getElementType());
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, filterReassocIndices);

    SmallVector<ReassociationIndices> img2ColTensorReassocIndices;
    SmallVector<ReassociationIndices> outputReassocIndices;
    RankedTensorType reshapedImg2ColTensorType, reshapedOutputType;
    if (n == 1) {
      img2ColTensorReassocIndices = {{0, 1, 2, 3}, {4, 5}};
      outputReassocIndices = {{0, 1}, {2, 3}};

      reshapedImg2ColTensorType = RankedTensorType::get(
          {fh * fw * ic, oh * ow}, inputType.getElementType());
      reshapedOutputType =
          RankedTensorType::get({oc, oh * ow}, outputType.getElementType());
    } else {
      img2ColTensorReassocIndices = {{0}, {1, 2, 3}, {4, 5}};
      outputReassocIndices = {{0}, {1}, {2, 3}};

      reshapedImg2ColTensorType = RankedTensorType::get(
          {n, fh * fw * ic, oh * ow}, inputType.getElementType());
      reshapedOutputType =
          RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
    }

    Value reshapedImg2ColTensor = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
        img2ColTensorReassocIndices);

    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, outputReassocIndices);

    Value result;
    if (n == 1) {
      auto matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, reshapedOutputType,
          ArrayRef<Value>{reshapedFilter, reshapedImg2ColTensor},
          ArrayRef<Value>{reshapedOutput});
      result = matmulOp.getResults().front();
    } else {
      // For cases where batch is not 1, we need to keep the batch dimension
      // separate. Because the filter does not share the same batch dimension,
      // the batch dimension is only used in indexing the input and output. Thus
      // we cannot use existing linalg named ops like linalg.batch_matmul.
      // i.e. M x K * (B x) K x N = (B x) M x N
      AffineExpr bDim, mDim, nDim, kDim;
      bindDims(getContext(), bDim, mDim, nDim, kDim);
      auto lhsMap = AffineMap::get(4, 0, {mDim, kDim}, getContext());
      auto rhsMap = AffineMap::get(4, 0, {bDim, kDim, nDim}, getContext());
      auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, getContext());
      SmallVector<utils::IteratorType> genericIterators = {parallel, parallel,
                                                           parallel, reduction};
      bool isInt = llvm::isa<IntegerType>(outputType.getElementType());
      auto genericOp = rewriter.create<linalg::GenericOp>(
          loc, reshapedOutputType,
          /*inputs=*/ValueRange{reshapedFilter, reshapedImg2ColTensor},
          /*outputs=*/ValueRange{reshapedOutput},
          ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value mul = createMul(loc, args[0], args[1], isInt, nestedBuilder);
            Value add = createAdd(loc, mul, args[2], isInt, nestedBuilder);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
          });
      result = genericOp.getResults().front();
    }

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputType, result, outputReassocIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

struct ConvertConv2DToImg2ColPass
    : ConvertConv2DToImg2ColBase<ConvertConv2DToImg2ColPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertConv2DNhwcHwcf, ConvertDepthwiseConv2DNhwcHwc,
                    ConvertConv2DNchwFchw>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertConv2DToImg2ColPass() {
  return std::make_unique<ConvertConv2DToImg2ColPass>();
}

} // namespace mlir::iree_compiler::Preprocessing
