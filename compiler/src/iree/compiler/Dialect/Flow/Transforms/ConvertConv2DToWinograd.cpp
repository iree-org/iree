// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/WinogradConstants.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/SetVector.h"
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

#define INDEX2(y, x, dimy, dimx) (x + dimx * y)
#define INDEX4(z, y, x, w, dimz, dimy, dimx, dimw) \
  (w + dimw * (x + dimx * (y + dimy * z)))

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

static DenseElementsAttr foldFilterTransform(
    ArrayRef<int64_t> shape, int64_t inputTileSize, int64_t kernelSize,
    Type outputType, const float *G, bool isSplat, float splatValue,
    DenseElementsAttr::iterator_range<APFloat> &input, Type elementType) {
  const int64_t numElements =
      inputTileSize * inputTileSize * shape[2] * shape[3];
  SmallVector<APFloat> output(numElements, APFloat(0.0f));
  for (int d0 = 0; d0 < inputTileSize; d0++) {
    for (int d1 = 0; d1 < inputTileSize; d1++) {
      for (int d2 = 0; d2 < shape[2]; d2++) {
        for (int d3 = 0; d3 < shape[3]; d3++) {
          APFloat accum(0.0f);
          for (int d4 = 0; d4 < kernelSize; d4++) {
            for (int d5 = 0; d5 < kernelSize; d5++) {
              APFloat ival(splatValue);
              if (!isSplat) {
                ival = input[INDEX4(d4, d5, d2, d3, shape[0], shape[1],
                                    shape[2], shape[3])];
              }
              int idx0 = INDEX2(d0, d4, inputTileSize, kernelSize);
              int idx1 = INDEX2(d1, d5, inputTileSize, kernelSize);
              accum = accum + APFloat(G[idx0]) * ival * APFloat(G[idx1]);
            }
          }
          int odx = INDEX4(d0, d1, d2, d3, inputTileSize, inputTileSize,
                           shape[2], shape[3]);
          output[odx] = accum;
        }
      }
    }
  }
  return DenseElementsAttr::get(outputType, output);
}

static Value createCollapseOrExpand(
    Value tensor, Location loc, PatternRewriter &rewriter,
    SmallVectorImpl<int64_t> &outputShape,
    SmallVectorImpl<ReassociationIndices> &reassociations, bool collapse) {
  auto tensorType = tensor.getType().cast<ShapedType>();
  auto elementTy = tensorType.getElementType();
  auto resultType = RankedTensorType::get(outputShape, elementTy);
  if (collapse)
    return rewriter.create<tensor::CollapseShapeOp>(loc, resultType, tensor,
                                                    reassociations);
  return rewriter.create<tensor::ExpandShapeOp>(loc, resultType, tensor,
                                                reassociations);
}

namespace {

class ConvertConv2DNhwcHwcf final
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    // Check that strides = 1
    if (!hasAllOneValues(convOp.getStrides())) return failure();

    // Check that dilations = 1
    if (!hasAllOneValues(convOp.getDilations())) return failure();

    // Check that kernel size = 3x3
    Value kernel = convOp.getInputs()[1];
    auto kernelType = kernel.getType().cast<ShapedType>();
    if (!kernelType) return failure();
    ArrayRef<int64_t> kernelShape = kernelType.getShape();
    Type elementType = kernelType.getElementType();
    const int64_t kh = kernelShape[0];
    const int64_t kw = kernelShape[1];
    if ((kh != 3) || (kw != 3)) return failure();
    const int64_t kernelSize = kh;

    // TODO: Make this a user-settable parameter once we have support
    // for more tile sizes
    const int64_t outputTileSize = 6;
    const int64_t inputTileSize = outputTileSize + kernelSize - 1;

    // Determine if filter transform can be constant folded (only true for
    // inference)
    Location loc = convOp.getLoc();
    bool constantFoldFilter{false};
    Value foldedKernel;
    auto constOp = kernel.getDefiningOp<arith::ConstantOp>();
    if (constOp) {
      auto kernel = constOp.getValue().cast<DenseIntOrFPElementsAttr>();
      if (!kernel) return failure();
      ShapedType type = constOp.getType().cast<ShapedType>();
      Type elementType = type.getElementType();
      assert(elementType.isa<FloatType>());
      ArrayRef<int64_t> shape = type.getShape();
      DenseElementsAttr::iterator_range<APFloat> nonSplatValues =
          kernel.getValues<APFloat>();
      bool isSplat = kernel.isSplat();
      float splatValue{0.0};
      if (isSplat) {
        splatValue = kernel.getSplatValue<APFloat>().convertToFloat();
      }
      SmallVector<int64_t> resultShape{inputTileSize * inputTileSize, shape[2],
                                       shape[3]};
      auto resultType = RankedTensorType::get(resultShape, elementType);
      auto foldedKernelAttr =
          foldFilterTransform(shape, inputTileSize, kernelSize, resultType,
                              IREE::LinalgExt::Winograd::G_6x6_3x3, isSplat,
                              splatValue, nonSplatValues, elementType);
      foldedKernel = rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          constOp, foldedKernelAttr);
      constantFoldFilter = true;
    }
    if (!constantFoldFilter) return failure();

    // Create winograd input transform op
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value input = convOp.getInputs()[0];
    auto inputType = input.getType().cast<ShapedType>();
    if (!inputType) return failure();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    for (int i = 0; i < inputShape.size(); i++) {
      if (ShapedType::isDynamic(inputShape[i])) return failure();
    }
    assert(inputShape.size() == 4);

    SmallVector<int64_t, 2> imageDimensions = {1, 2};
    const size_t numImageDims = imageDimensions.size();
    SmallVector<int64_t> resultShape(6, inputTileSize);
    llvm::SmallSetVector<int64_t, 2> imageDimensionsSet(imageDimensions.begin(),
                                                        imageDimensions.end());
    int outputIndex;
    for (int i = 0; i < inputShape.size(); i++) {
      outputIndex = i + numImageDims;
      if (!imageDimensionsSet.contains(i)) {
        resultShape[outputIndex] = inputShape[i];
      } else {
        resultShape[outputIndex] =
            std::ceil((float)(inputShape[i] - kernelSize + 1) / outputTileSize);
      }
    }
    Value emptyTensor =
        rewriter.create<tensor::EmptyOp>(loc, resultShape, elementType);
    auto winogradInputOp =
        rewriter.create<IREE::LinalgExt::WinogradInputTransformOp>(
            loc, emptyTensor.getType(), ValueRange{input},
            ValueRange{emptyTensor}, outputTileSize, kernelSize,
            imageDimensions);
    Value winogradInput = winogradInputOp.getResult()[0];

    // Add collapse shape
    SmallVector<int64_t> collapsedShape = {
        resultShape[0] * resultShape[1],
        resultShape[2] * resultShape[3] * resultShape[4], resultShape[5]};
    SmallVector<ReassociationIndices> reassociations = {{0, 1}, {2, 3, 4}, {5}};
    Value collapsedWinogradInput = createCollapseOrExpand(
        winogradInput, loc, rewriter, collapsedShape, reassociations, true);

    // Add BatchMatmulOp
    SmallVector<int64_t> bmmShape(collapsedShape.begin(), collapsedShape.end());
    Value output = convOp.getOutputs()[0];
    auto outputType = output.getType().cast<RankedTensorType>();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    bmmShape[2] = outputShape[3];
    auto bmmOutputType = RankedTensorType::get(bmmShape, elementType);
    emptyTensor = rewriter.create<tensor::EmptyOp>(loc, bmmShape, elementType);
    auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zero},
                                                  ValueRange{emptyTensor});
    auto bmmOp = rewriter.create<linalg::BatchMatmulOp>(
        loc, bmmOutputType, ValueRange({collapsedWinogradInput, foldedKernel}),
        ValueRange({fillOp.result()}));
    Value bmmResult = bmmOp.getResult(0);

    // Add expand shape
    SmallVector<int64_t> expandedShape = {resultShape[0], resultShape[1],
                                          resultShape[2], resultShape[3],
                                          resultShape[4], outputShape[3]};
    reassociations = {{0, 1}, {2, 3, 4}, {5}};
    Value expandedBmmResult = createCollapseOrExpand(
        bmmResult, loc, rewriter, expandedShape, reassociations, false);

    // Convert back into original domain
    SmallVector<int64_t> paddedResultShape(outputShape.size(), 0);
    for (int i = 0; i < outputShape.size(); i++) {
      if (!imageDimensionsSet.contains(i)) {
        paddedResultShape[i] = outputShape[i];
      } else {
        paddedResultShape[i] = resultShape[i + numImageDims] * outputTileSize;
      }
    }
    emptyTensor =
        rewriter.create<tensor::EmptyOp>(loc, paddedResultShape, elementType);
    auto winogradOutputOp =
        rewriter.create<IREE::LinalgExt::WinogradOutputTransformOp>(
            loc, emptyTensor.getType(), ValueRange{expandedBmmResult},
            ValueRange{emptyTensor}, outputTileSize, kernelSize,
            imageDimensions);
    Value paddedOutput = winogradOutputOp.getResult()[0];

    // Extract slice
    SmallVector<OpFoldResult> offsets(outputShape.size(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(outputShape.size(),
                                      rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes;
    for (int i = 0; i < outputShape.size(); i++)
      sizes.push_back(rewriter.getIndexAttr(outputShape[i]));
    auto winogradOutput = rewriter.create<tensor::ExtractSliceOp>(
        loc, outputType, paddedOutput, offsets, sizes, strides);

    Value result = convOp.getResult(0);
    result.replaceAllUsesWith(winogradOutput);
    return success();
  }
};

struct ConvertConv2DToWinogradPass
    : ConvertConv2DToWinogradBase<ConvertConv2DToWinogradPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertConv2DNhwcHwcf>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertConv2DToWinogradPass() {
  return std::make_unique<ConvertConv2DToWinogradPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
