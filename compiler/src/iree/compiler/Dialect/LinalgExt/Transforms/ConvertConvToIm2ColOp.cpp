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

#define GEN_PASS_DEF_CONVERTCONVTOIM2COLOPPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

static bool hasAllOneValues(ArrayRef<int64_t> attr) {
  return llvm::all_of(attr, [](int64_t element) { return element == 1; });
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

// Helper to convert a shape into basis for im2col op.
static SmallVector<int64_t> getBasisFromShape(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> basis(shape.size());
  int64_t cummulativeProduct = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    basis[i] = cummulativeProduct;
    cummulativeProduct *= shape[i];
  }
  return basis;
}

namespace {

using ControlFnTy = std::function<bool(Operation *)>;
// Converts non-depthwise convs into into linalg.generic (for img2col packing)
// and linalg.matmul.
// The following explains this for a linalg.conv_2d_nhwc_hwcf op.
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

class ConvertConvGeneric final
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  ConvertConvGeneric(MLIRContext *context, std::optional<ControlFnTy> controlFn)
      : OpInterfaceRewritePattern(context), controlFn(controlFn) {}
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (controlFn.has_value() && !controlFn.value()(linalgOp)) {
      return rewriter.notifyMatchFailure(linalgOp, "controlFn failed.");
    }

    auto igemmConvDetailsOrFailure =
        LinalgExt::getIGEMMGenericConvDetails(linalgOp);
    if (failed(igemmConvDetailsOrFailure))
      return rewriter.notifyMatchFailure(linalgOp,
                                         "Failed to extract IGEMM details");

    LinalgExt::IGEMMGenericConvDetails igemmConvDetails =
        *igemmConvDetailsOrFailure;

    SmallVector<AffineMap> igemmContractionMaps =
        igemmConvDetails.igemmContractionMaps;
    mlir::linalg::ConvolutionDimensions convDims = igemmConvDetails.convDims;
    SmallVector<ReassociationIndices> filterReassocIndices =
        igemmConvDetails.filterReassocIndices;
    bool isOutputChannelFirst = igemmConvDetails.isOutputChannelFirst;
    SmallVector<int64_t> igemmLoopBounds = igemmConvDetails.igemmLoopBounds;
    SmallVector<utils::IteratorType> igemmLoopIterators =
        igemmConvDetails.igemmLoopIterators;

    Value input = linalgOp.getDpsInputs()[0];
    Value filter = linalgOp.getDpsInputs()[1];
    Value output = linalgOp.getDpsInits()[0];
    auto inputType = llvm::cast<ShapedType>(input.getType());
    auto filterType = llvm::cast<ShapedType>(filter.getType());
    auto outputType = llvm::cast<ShapedType>(output.getType());

    ArrayRef<int64_t> filterShape = filterType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    AffineMap inputMap = indexingMaps[0];
    AffineMap filterMap = indexingMaps[1];
    AffineMap outputMap = indexingMaps[2];

    SmallVector<OpFoldResult> kernelSizes;
    for (auto filterLoop : convDims.filterLoop) {
      std::optional<int64_t> maybeDim = filterMap.getResultPosition(
          getAffineDimExpr(filterLoop, filterMap.getContext()));
      if (!maybeDim) {
        return rewriter.notifyMatchFailure(linalgOp,
                                           "Failed to infer filter shape.");
      }
      kernelSizes.push_back(
          rewriter.getIndexAttr(filterShape[maybeDim.value()]));
    }

    // Shape of the resulting tensor from im2col.
    SmallVector<int64_t> colTensorShape;
    SmallVector<int64_t> batchPos;
    for (auto batch : convDims.batch) {
      std::optional<int64_t> maybeBatch = inputMap.getResultPosition(
          getAffineDimExpr(batch, inputMap.getContext()));
      if (!maybeBatch) {
        return rewriter.notifyMatchFailure(linalgOp,
                                           "Failed to infer batch shape.");
      }
      batchPos.push_back(maybeBatch.value());
      colTensorShape.push_back(inputShape[maybeBatch.value()]);
    }

    SmallVector<int64_t> mPos;
    SmallVector<int64_t> mShape;
    for (auto outputImage : convDims.outputImage) {
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(outputImage)) {
          mPos.push_back(idx);
        }
      }
      for (auto [idx, e] : llvm::enumerate(outputMap.getResults())) {
        if (e.isFunctionOfDim(outputImage)) {
          mShape.push_back(outputShape[idx]);
          colTensorShape.push_back(outputShape[idx]);
        }
      }
    }

    SmallVector<int64_t> kPos;
    for (auto reductionDim : convDims.inputChannel) {
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(reductionDim)) {
          kPos.push_back(idx);
        }
      }
    }
    // The index at which the reduction dimension bounds starts in
    // igemmLoopBounds.
    int64_t reductionBoundIndex = convDims.batch.size() +
                                  convDims.outputImage.size() +
                                  convDims.outputChannel.size();
    SmallVector<int64_t> kShape(igemmLoopBounds.begin() + reductionBoundIndex,
                                igemmLoopBounds.end());
    colTensorShape.insert(colTensorShape.end(), kShape.begin(), kShape.end());

    SmallVector<OpFoldResult> mBasis =
        getAsIndexOpFoldResult(getContext(), getBasisFromShape(mShape));
    SmallVector<OpFoldResult> kBasis =
        getAsIndexOpFoldResult(getContext(), getBasisFromShape(kShape));

    SmallVector<OpFoldResult> kOffset(kBasis.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> mOffset(mBasis.size(), rewriter.getIndexAttr(0));
    auto loc = linalgOp.getLoc();
    Value colTensor = rewriter.create<tensor::EmptyOp>(
        loc, colTensorShape, inputType.getElementType());
    Value img2ColTensor =
        rewriter
            .create<IREE::LinalgExt::Im2colOp>(
                loc, input, /*output=*/colTensor, convDims.strides,
                convDims.dilations, kernelSizes, mOffset, mBasis, kOffset,
                kBasis, batchPos, mPos, kPos)
            .getResult(0);

    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, filter, filterReassocIndices);

    auto genericGEMMOp = rewriter.create<linalg::GenericOp>(
        loc, outputType,
        /*inputs=*/
        isOutputChannelFirst ? ValueRange{reshapedFilter, img2ColTensor}
                             : ValueRange{img2ColTensor, reshapedFilter},
        /*outputs=*/ValueRange{output}, igemmContractionMaps,
        igemmLoopIterators,
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
        });
    genericGEMMOp->setDiscardableAttrs(getPrunedAttributeList(linalgOp));
    Value result = genericGEMMOp.getResults().front();

    rewriter.replaceOp(linalgOp, result);
    return success();
  }

private:
  std::optional<ControlFnTy> controlFn;
};

struct ConvertConvToIm2ColOpPass final
    : impl::ConvertConvToIm2ColOpPassBase<ConvertConvToIm2ColOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateConvToIm2colOpPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateConvToIm2colOpPatterns(RewritePatternSet &patterns,
                                    std::optional<ControlFnTy> controlFn) {
  patterns.insert<ConvertConvGeneric>(patterns.getContext(),
                                      std::move(controlFn));
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
