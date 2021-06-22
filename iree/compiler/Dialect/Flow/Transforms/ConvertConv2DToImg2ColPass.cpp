// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// clang-format off
//
// Convert linalg.conv_2d_input_nhwc_filter_hwcf op into img2col packing
// operation (linalg.generic) + linalg.matmul. See details below:
// A convolution operaton can be written as a matrix-matrix multiplication by
// unfolding the cross corrolation between input and filter and explicitly copy
// overlapped sliding window inputs.
// Consider 2D input X with single channel input and output and 2x2 filter W:
// [x(0, 0)  , x(0, 1)  , ...,   x(0, n)  ]
// [x(1, 0)  , x(1, 1)  , ...,   x(1, n)  ]
// [.        ,  .       ,.   ,      .     ]            [w(0, 0), w(0, 1)]
// [.        ,  .       , .  ,      .     ]    (conv)  [w(1, 0), w(1, 1)]
// [.        ,  .       ,   .,      .     ]
// [x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]
// The packed input data (img2col) is a matrix with |rows| = output spatial
// size, |columns| = filter spatial size. To compute the output Y(i, j) we need
// to calculate the dot product between filter window at input X(x, y)) and the
// filter which will look like the following where r.h.s is the img2col matrix and
// l.h.s is the flattned filter:
// [x(0, 0), x(0, 1), x(1, 0), x(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)] (matmul) [w(0, 0), w(0, 1), w(1, 0), w(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)]
// [   .   ,    .   ,    .   ,    .   ]
// In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
// and output (N, Ho, Wo, D) the convolutin is the following matrix-matrix multiplication
// (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in the N input.
// For the case where N > 1 its a batched matrxi-matrix multplication.
//
// clang-format on
class Conv2DImg2ColMatmulConversion
    : public OpRewritePattern<linalg::ConvInputNHWCFilterHWCFOp> {
 public:
  using OpRewritePattern<linalg::ConvInputNHWCFilterHWCFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ConvInputNHWCFilterHWCFOp convOp,
                                PatternRewriter &rewriter) const override {
    ShapedType inputShapeType =
        convOp.getInputOperand(0)->get().getType().cast<ShapedType>();
    ShapedType filterShapeType =
        convOp.getInputOperand(1)->get().getType().cast<ShapedType>();
    ShapedType outputShapeType =
        convOp.getOutputOperand(0)->get().getType().cast<ShapedType>();

    if (!filterShapeType || !inputShapeType) return failure();
    if (!filterShapeType.hasStaticShape() || !inputShapeType.hasStaticShape())
      return failure();

    Value input = convOp.getInputOperand(0)->get();
    Value filter = convOp.getInputOperand(1)->get();
    Value output = convOp.getOutputOperand(0)->get();
    auto filterShape = filterShapeType.getShape();
    auto outputShape = outputShapeType.getShape();

    // TODO(ataei): Support for batched version.
    if (inputShapeType.getShape()[0] > 1) return failure();

    // TODO(ataei) : Support padding & dilation.
    if (!llvm::all_of(convOp.dilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    auto loc = convOp.getLoc();

    // col tensor shape (n, d1, d1, k1, k2, ci)
    SmallVector<int64_t, 4> colTensorShape = {outputShape[0], outputShape[1],
                                              outputShape[2], filterShape[0],
                                              filterShape[1], filterShape[2]};

    Value colTensor = rewriter.create<linalg::InitTensorOp>(
        loc, colTensorShape, inputShapeType.getElementType());

    auto n = rewriter.getAffineDimExpr(0);
    auto d = [&](int i) { return rewriter.getAffineDimExpr(i); };
    auto k = [&](int i) { return rewriter.getAffineDimExpr(i + 2); };
    auto ci = rewriter.getAffineDimExpr(5);

    auto s = [&](unsigned i) {
      return rewriter.getAffineConstantExpr(
          convOp.strides().getValue<int64_t>({i}));
    };

    SmallVector<AffineExpr, 4> inputExprs = {n, d(1) * s(0) + k(1),
                                             d(2) * s(1) + k(2), ci};

    auto nloops = colTensorShape.size();

    SmallVector<StringRef, 3> loopAttributeTypes(nloops, "parallel");

    SmallVector<AffineMap, 4> indexingMaps = {
        AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

    auto img2ColTensor = rewriter.create<linalg::GenericOp>(
        loc, colTensor.getType(),
        /*inputs=*/input, /*outputs=*/colTensor, indexingMaps,
        loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });

    SmallVector<linalg::ReassociationIndices>
        img2ColTensorReassociationIndices = {{0, 1, 2}, {3, 4, 5}};

    SmallVector<linalg::ReassociationIndices>
        filterAndOutputReassociationIndices = {{0, 1, 2}, {3}};

    auto reshapedImg2ColTensorType = RankedTensorType::get(
        {outputShape[1] * outputShape[2],
         filterShape[0] * filterShape[1] * filterShape[2]},
        inputShapeType.getElementType());

    auto reshapedFilterType = RankedTensorType::get(
        {filterShape[0] * filterShape[1] * filterShape[2], filterShape[3]},
        inputShapeType.getElementType());

    auto reshapedOutputType =
        RankedTensorType::get({outputShape[1] * outputShape[2], outputShape[3]},
                              outputShapeType.getElementType());

    Value reshapedImg2ColTensor =
        rewriter.create<linalg::TensorCollapseShapeOp>(
            loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
            img2ColTensorReassociationIndices);

    Value reshapedFilter = rewriter.create<linalg::TensorCollapseShapeOp>(
        loc, reshapedFilterType, filter, filterAndOutputReassociationIndices);

    Value reshapedOutput = rewriter.create<linalg::TensorCollapseShapeOp>(
        loc, reshapedOutputType, output, filterAndOutputReassociationIndices);

    auto matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType,
        ArrayRef<Value>{reshapedImg2ColTensor, reshapedFilter},
        ArrayRef<Value>{reshapedOutput});

    auto reshapedResult = rewriter.create<linalg::TensorExpandShapeOp>(
        loc, outputShapeType, matmulResult.getResults()[0],
        filterAndOutputReassociationIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

// Similar to the conv pattern above except there is no reduction among the
// input channles so each convolution can be a matrix-vector product and
// by transposing both input filter so channles are outer most the computation
// is a batched matrix-vector product.
class DepthwiseConv2DNHWCHWCImg2ColMatmulConversion
    : public OpRewritePattern<linalg::DepthwiseConvInputNHWCFilterHWCOp> {
 public:
  using OpRewritePattern<
      linalg::DepthwiseConvInputNHWCFilterHWCOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::DepthwiseConvInputNHWCFilterHWCOp convOp,
      PatternRewriter &rewriter) const override {
    RankedTensorType inputTensorType =
        convOp.getInputOperand(0)->get().getType().dyn_cast<RankedTensorType>();
    RankedTensorType filterTensorType =
        convOp.getInputOperand(1)->get().getType().dyn_cast<RankedTensorType>();
    RankedTensorType outputTensorType = convOp.getOutputOperand(0)
                                            ->get()
                                            .getType()
                                            .dyn_cast<RankedTensorType>();

    if (!filterTensorType || !inputTensorType) return failure();
    if (!filterTensorType.hasStaticShape() || !inputTensorType.hasStaticShape())
      return failure();

    // TODO(ataei) : Support padding & dilation.
    if (!llvm::all_of(convOp.dilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    auto loc = convOp.getLoc();

    auto transposeOperand = [&](Value operand,
                                ArrayRef<int64_t> indices) -> Value {
      RankedTensorType operandTensorType =
          operand.getType().cast<RankedTensorType>();
      auto nloops = indices.size();
      auto inputShape = operandTensorType.getShape();

      SmallVector<AffineExpr, 4> exprs = llvm::to_vector<4>(
          llvm::map_range(indices, [&](int64_t index) -> AffineExpr {
            return rewriter.getAffineDimExpr(index);
          }));

      SmallVector<int64_t> targetShape = llvm::to_vector<4>(llvm::map_range(
          indices,
          [&](int64_t index) -> int64_t { return inputShape[index]; }));

      Value outputTensor = rewriter.create<linalg::InitTensorOp>(
          loc, targetShape, operandTensorType.getElementType());

      SmallVector<StringRef> loopAttributeTypes(nloops, "parallel");

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

    Value input = convOp.getInputOperand(0)->get();
    Value filter = convOp.getInputOperand(1)->get();
    Value output = convOp.getOutputOperand(0)->get();

    // Transpose input, filter so channels are outermost
    auto transposedInput = transposeOperand(input, {0, 3, 1, 2});
    auto transposedFilter = transposeOperand(filter, {2, 0, 1});
    auto transposedFilterShape =
        transposedFilter.getType().cast<RankedTensorType>().getShape();
    auto outputShape = output.getType().cast<RankedTensorType>().getShape();

    // Transposed output tensor shape (n, ci, d1, d2)
    SmallVector<int64_t, 4> transposedOutputTensorShape = {
        outputShape[0], transposedFilterShape[0], outputShape[1],
        outputShape[2]};

    // col tensor shape (n, ci, d1, d2, k1, k2)
    SmallVector<int64_t, 4> colTensorShape = {
        outputShape[0], transposedFilterShape[0], outputShape[1],
        outputShape[2], transposedFilterShape[1], transposedFilterShape[2]};
    Value transposedOutputTensor = transposeOperand(output, {0, 3, 1, 2});

    auto n = rewriter.getAffineDimExpr(0);
    auto ci = rewriter.getAffineDimExpr(1);
    auto d = [&](int i) { return rewriter.getAffineDimExpr(i + 1); };
    auto k = [&](int i) { return rewriter.getAffineDimExpr(i + 3); };

    auto s = [&](unsigned i) {
      return rewriter.getAffineConstantExpr(
          convOp.strides().getValue<int64_t>({i}));
    };

    SmallVector<AffineExpr> inputExprs = {n, ci, d(1) * s(0) + k(1),
                                          d(2) * s(1) + k(2)};

    auto nloops = colTensorShape.size();

    SmallVector<StringRef> loopAttributeTypes(nloops, "parallel");

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

    Value colTensor = rewriter.create<linalg::InitTensorOp>(
        loc, colTensorShape, inputTensorType.getElementType());

    auto img2ColTensor = rewriter.create<linalg::GenericOp>(
        loc, colTensor.getType(),
        /*inputs=*/transposedInput, /*outputs=*/colTensor, indexingMaps,
        loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });

    SmallVector<linalg::ReassociationIndices>
        img2ColTensorReassociationIndices = {{0, 1}, {2, 3}, {4, 5}};
    SmallVector<linalg::ReassociationIndices> filterReassociationIndice = {
        {0}, {1, 2}};
    SmallVector<linalg::ReassociationIndices> outputReassociationIndice = {
        {0, 1}, {2, 3}};

    auto reshapedImg2ColTensorType = RankedTensorType::get(
        {outputShape[0] * transposedFilterShape[0],
         outputShape[1] * outputShape[2],
         transposedFilterShape[1] * transposedFilterShape[2]},
        inputTensorType.getElementType());
    auto reshapedFilterTensorType = RankedTensorType::get(
        {transposedFilterShape[0],
         transposedFilterShape[1] * transposedFilterShape[2]},
        filterTensorType.getElementType());
    auto reshapedOutputTensorType = RankedTensorType::get(
        {transposedOutputTensorShape[0] * transposedOutputTensorShape[1],
         transposedOutputTensorShape[2] * transposedOutputTensorShape[3]},
        outputTensorType.getElementType());

    Value reshapedImg2ColTensor =
        rewriter.create<linalg::TensorCollapseShapeOp>(
            loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
            img2ColTensorReassociationIndices);
    Value reshapedFilterTensor = rewriter.create<linalg::TensorCollapseShapeOp>(
        loc, reshapedFilterTensorType, transposedFilter,
        filterReassociationIndice);
    Value reshapedoutputTensor = rewriter.create<linalg::TensorCollapseShapeOp>(
        loc, reshapedOutputTensorType, transposedOutputTensor,
        outputReassociationIndice);

    auto batchMatVecResult = rewriter.create<linalg::BatchMatvecOp>(
        loc, TypeRange{reshapedoutputTensor.getType()},
        ValueRange{reshapedImg2ColTensor, reshapedFilterTensor},
        ValueRange{reshapedoutputTensor});

    SmallVector<linalg::ReassociationIndices> batchMatVecReassociationIndice = {
        {0, 1}, {2, 3}};

    Value batchMatVecResultReshaped =
        rewriter.create<linalg::TensorExpandShapeOp>(
            loc, transposedOutputTensor.getType(),
            batchMatVecResult.getResult(0), batchMatVecReassociationIndice);

    auto transposedResult =
        transposeOperand(batchMatVecResultReshaped, {0, 2, 3, 1});

    rewriter.replaceOp(convOp, ArrayRef<Value>{transposedResult});
    return success();
  }
};

struct ConvertConv2DToImg2ColPass
    : ConvertConv2DToImg2ColBase<ConvertConv2DToImg2ColPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<Conv2DImg2ColMatmulConversion,
                    DepthwiseConv2DNHWCHWCImg2ColMatmulConversion>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertConv2DToImg2ColPass() {
  return std::make_unique<ConvertConv2DToImg2ColPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
