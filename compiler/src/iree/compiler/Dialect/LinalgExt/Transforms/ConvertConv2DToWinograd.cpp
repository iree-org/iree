// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/WinogradConstants.h"
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
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

static inline int index(int y, int x, int dimy, int dimx) {
  return (x + dimx * y);
}

static inline int index(int z, int y, int x, int w, int dimz, int dimy,
                        int dimx, int dimw) {
  return (w + dimw * (x + dimx * (y + dimy * z)));
}

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(attr, [](APInt element) { return element.isOne(); });
}

// TODO: Make this a user-settable parameter once we have support
// for more tile sizes
static constexpr int64_t outputTileSize = 6;

/// This function computes the Winograd filter transform when
/// the filter is known to be a constant. Specifically, this
/// function computes matmul(G, matmul(F, transpose(G))) where
/// F is a tile of the convolution filter of size m x m
/// (single input channel, single output channel) and G has
/// shape m x (m + r - 1) where r is the output tile size and
/// (m + r - 1) is the input tile size.
/// The time complexity of this function is O(ic * oc)
/// where ic is the number of input channels and oc is the
/// number of output channels since input tile size and kernel size
/// are constants. So for large ic and oc, this function is
/// time intensive.
/// TODO: Codegen this as a kernel and run once at initialization
static DenseElementsAttr
foldFilterTransform(ArrayRef<int64_t> shape, int64_t inputTileSize,
                    int64_t kernelSize, ShapedType outputType, const float *G,
                    bool isSplat, float splatValue,
                    DenseElementsAttr::iterator_range<APFloat> &input,
                    FloatType floatType, bool isNchw) {
  const int &kh = isNchw ? shape[2] : shape[0];
  const int &kw = isNchw ? shape[3] : shape[1];
  const int &ic = isNchw ? shape[1] : shape[2];
  const int &oc = isNchw ? shape[0] : shape[3];
  const int64_t numElements = inputTileSize * inputTileSize * ic * oc;
  SmallVector<APFloat> output(numElements, APFloat(0.0f));
  for (int d0 = 0; d0 < inputTileSize; d0++) {
    for (int d1 = 0; d1 < inputTileSize; d1++) {
      for (int d2 = 0; d2 < ic; d2++) {
        for (int d3 = 0; d3 < oc; d3++) {
          APFloat accum(0.0f);
          for (int d4 = 0; d4 < kernelSize; d4++) {
            for (int d5 = 0; d5 < kernelSize; d5++) {
              APFloat ival(splatValue);
              if (!isSplat) {
                if (!isNchw) {
                  ival = input[index(d4, d5, d2, d3, kh, kw, ic, oc)];
                } else {
                  ival = input[index(d3, d2, d4, d5, oc, ic, kh, kw)];
                }
              }
              int idx0 = index(d0, d4, inputTileSize, kernelSize);
              int idx1 = index(d1, d5, inputTileSize, kernelSize);
              accum = accum + APFloat(G[idx0]) * ival * APFloat(G[idx1]);
            }
          }
          int odx = index(d0, d1, d2, d3, inputTileSize, inputTileSize, ic, oc);
          output[odx] = accum;
          if (floatType.isF16()) {
            bool losesInfo;
            output[odx].convert(APFloat::IEEEhalf(),
                                APFloat::rmNearestTiesToEven, &losesInfo);
          }
        }
      }
    }
  }
  return DenseElementsAttr::get(outputType, output);
}

template <typename T>
static bool hasValidStridesAndDilations(Operation *op) {
  auto convOp = dyn_cast<T>(op);
  // Check that strides = 1
  if (!hasAllOneValues(convOp.getStrides())) {
    return false;
  }

  // Check that dilations = 1
  if (!hasAllOneValues(convOp.getDilations())) {
    return false;
  }
  return true;
}

static bool isValidConv2d(Operation *op, bool &isNchw) {
  isNchw = isa<linalg::Conv2DNchwFchwOp>(op);
  const bool isNhwc = isa<linalg::Conv2DNhwcHwcfOp>(op);
  if (!(isNchw || isNhwc)) {
    return false;
  }
  return (isNchw ? hasValidStridesAndDilations<linalg::Conv2DNchwFchwOp>(op)
                 : hasValidStridesAndDilations<linalg::Conv2DNhwcHwcfOp>(op));
}

namespace {

template <typename ConvOp>
class FoldWinogradFilterTransform final : public OpRewritePattern<ConvOp> {
public:
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp convOp,
                                PatternRewriter &rewriter) const override {

    bool isNchw;
    if (!isValidConv2d(convOp, isNchw)) {
      return failure();
    }

    // Check that kernel size = 3x3
    Value kernel = convOp.getInputs()[1];
    auto kernelType = kernel.getType().cast<ShapedType>();
    if (!kernelType) {
      return failure();
    }
    ArrayRef<int64_t> kernelShape = kernelType.getShape();
    if (kernelShape.size() != 4) {
      return failure();
    }
    const int64_t kh = isNchw ? kernelShape[2] : kernelShape[0];
    const int64_t kw = isNchw ? kernelShape[3] : kernelShape[1];
    if ((kh != 3) || (kw != 3)) {
      return failure();
    }
    const int64_t kernelSize = kh;
    const int64_t inputTileSize = outputTileSize + kernelSize - 1;

    DenseIntOrFPElementsAttr kernelAttr;
    if (!matchPattern(kernel, m_Constant(&kernelAttr))) {
      return failure();
    }

    Operation *constOp = kernel.getDefiningOp();
    ShapedType type = constOp->getResult(0).getType().cast<ShapedType>();
    auto elemType = type.getElementType().cast<FloatType>();
    ArrayRef<int64_t> shape = type.getShape();
    DenseElementsAttr::iterator_range<APFloat> nonSplatValues =
        kernelAttr.getValues<APFloat>();
    bool isSplat = kernelAttr.isSplat();
    float splatValue{0.0};
    if (isSplat) {
      splatValue = kernelAttr.getSplatValue<APFloat>().convertToFloat();
    }
    SmallVector<int64_t> resultShape{inputTileSize * inputTileSize, shape[2],
                                     shape[3]};
    if (isNchw) {
      resultShape[1] = shape[1];
      resultShape[2] = shape[0];
    }
    auto resultType = RankedTensorType::get(resultShape, elemType);
    auto foldedKernelAttr =
        foldFilterTransform(shape, inputTileSize, kernelSize, resultType,
                            IREE::LinalgExt::Winograd::G_6x6_3x3, isSplat,
                            splatValue, nonSplatValues, elemType, isNchw);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, foldedKernelAttr);
    return success();
  }
};

} // namespace

static Value
createCollapse(Value tensor, Location loc, PatternRewriter &rewriter,
               SmallVectorImpl<int64_t> &outputShape,
               SmallVectorImpl<ReassociationIndices> &reassociations) {
  auto tensorType = tensor.getType().cast<ShapedType>();
  auto elementTy = tensorType.getElementType();
  auto resultType = RankedTensorType::get(outputShape, elementTy);
  return rewriter.create<tensor::CollapseShapeOp>(loc, resultType, tensor,
                                                  reassociations);
}

static Value
createExpand(Value tensor, Location loc, PatternRewriter &rewriter,
             SmallVectorImpl<int64_t> &outputShape,
             SmallVectorImpl<ReassociationIndices> &reassociations) {
  auto tensorType = tensor.getType().cast<ShapedType>();
  auto elementTy = tensorType.getElementType();
  auto resultType = RankedTensorType::get(outputShape, elementTy);
  return rewriter.create<tensor::ExpandShapeOp>(loc, resultType, tensor,
                                                reassociations);
}

namespace {

/// Convert conv2d to a sequence of ops that implement the
/// Winograd transformation. The Winograd transformation
/// is parameterized by the output tile size(r). The larger
/// the tile size, the greater the computational savings,
/// but this comes at the cost of accuracy.
///
/// For now, we restrict this transform to convolutions
/// where the filter size = 3x3, though extensions to larger
/// filter sizes are possible. We refer to the
/// filter size as (m). The input tile size (i) is defined as
/// m + r - 1. For a given output tile size, the Winograd
/// transformation defines 3 constant matrices:
///
/// B: i x i [used in input transform]
/// G: m x i [used in the filter transform]
/// A: i x r [used in output transform]
///
/// The choice of these matrices is not unique and affects
/// the accuracy of the approach.
///
/// Given a convolution of the form
///
/// y = conv2d(x, f)
///
/// where x: (N, H, W, C) | (N, C, H, W)
///       f: (H, W, C, F) | (F, C, H, W)
///
/// this pattern converts the convolution to the following
/// sequence:
///
/// f_winograd = winograd.filter_transform(f) [folded]
/// x_winograd = winograd.input_transform(x)
/// x_winograd_c = collapse(x_winograd)
/// y_winograd = batch_matmul(x_winograd_c, f_winograd)
/// y_winograd_e = expand(y_winograd)
/// y_padded = winograd.output_transform(y_winograd_e)
/// y = extract_slice(y_padded)
///
/// where the dimensions of the tensors above are:
///
/// f_winograd:   (i * i, C, F)
/// x_winograd:   (i, i, N, H', W', C)
/// x_winograd_c: (i * i, N * H' * W', C)
/// y_winograd:   (i * i, N * H' * W', F)
/// y_winograd_e: (i, i, N, H', W', F)
/// y_padded:     (N, r * H', r * W', F) | (N, F, r * H', r * W')
///
/// H': ceil((H - m + 1) / r)
/// W': ceil((W - m + 1) / r)
///
/// The winograd input transform extracts a tile of the input
/// of size i x i and computes matmul(transpose(B), matmul(tile(x), B)).
/// The winograd filter transform extracts a tile of the filter
/// of size m x m and computes matmul(G, matmul(tile(f), transpose(G)).
/// These two are then combined using elementwise multiplication
/// (which becomes a batch matmul when combining over multiple channels).
/// The winograd output filter extracts a tile of the result of size
/// i x i and computes matmul(transpose(A), matmul(tile(y_winograd_e), A)).
///
/// For more information and additional references,
/// see here:
///
/// https://github.com/nod-ai/MLIRWinogradTalk/blob/main/MLIRSummit2022.Nodai.Menon.pdf
///
template <typename ConvOp>
class ConvertConvToWinograd final : public OpRewritePattern<ConvOp> {
public:
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp convOp,
                                PatternRewriter &rewriter) const override {

    bool isNchw;
    if (!isValidConv2d(convOp, isNchw)) {
      return failure();
    }

    // Check that kernel has been constant folded (by validating rank = 3)
    Value kernel = convOp.getInputs()[1];
    auto kernelType = kernel.getType().cast<ShapedType>();
    if (!kernelType) {
      return failure();
    }
    Type elementType = kernelType.getElementType();
    ArrayRef<int64_t> kernelShape = kernelType.getShape();
    if (kernelShape.size() != 3) {
      return failure();
    }

    const int64_t kernelSize = 3;
    const int64_t inputTileSize = outputTileSize + kernelSize - 1;

    // Create winograd input transform op
    Location loc = convOp.getLoc();
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value input = convOp.getInputs()[0];
    auto inputType = input.getType().cast<ShapedType>();
    if (!inputType) {
      return failure();
    }
    SmallVector<int64_t> inputShape(inputType.getShape());
    if (llvm::any_of(inputShape, ShapedType::isDynamic)) {
      return failure();
    }
    assert(inputShape.size() == 4);
    if (isNchw) {
      permute<IREE::LinalgExt::Permutation::NCHW_TO_NHWC>(inputShape);
    }

    const std::array<int64_t, 2> nhwcImageDimensions{1, 2};
    const std::array<int64_t, 2> nchwImageDimensions{2, 3};
    const size_t numImageDims = nhwcImageDimensions.size();
    SmallVector<int64_t> resultShape(6, inputTileSize);
    llvm::SmallSetVector<int64_t, 2> imageDimensionsSet(
        nhwcImageDimensions.begin(), nhwcImageDimensions.end());
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
    auto &imageDimensions = isNchw ? nchwImageDimensions : nhwcImageDimensions;
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
    Value collapsedWinogradInput = createCollapse(
        winogradInput, loc, rewriter, collapsedShape, reassociations);

    // Add BatchMatmulOp
    SmallVector<int64_t> bmmShape(collapsedShape.begin(), collapsedShape.end());
    Value output = convOp.getOutputs()[0];
    auto outputType = output.getType().cast<RankedTensorType>();
    SmallVector<int64_t> outputShape(outputType.getShape());
    if (isNchw) {
      permute<IREE::LinalgExt::Permutation::NCHW_TO_NHWC>(outputShape);
    }
    bmmShape[2] = outputShape[3];
    auto bmmOutputType = RankedTensorType::get(bmmShape, elementType);
    emptyTensor = rewriter.create<tensor::EmptyOp>(loc, bmmShape, elementType);
    auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zero},
                                                  ValueRange{emptyTensor});
    auto bmmOp = rewriter.create<linalg::BatchMatmulOp>(
        loc, bmmOutputType, ValueRange({collapsedWinogradInput, kernel}),
        ValueRange({fillOp.result()}));
    Value bmmResult = bmmOp.getResult(0);

    // Add expand shape
    SmallVector<int64_t> expandedShape = {resultShape[0], resultShape[1],
                                          resultShape[2], resultShape[3],
                                          resultShape[4], outputShape[3]};
    reassociations = {{0, 1}, {2, 3, 4}, {5}};
    Value expandedBmmResult =
        createExpand(bmmResult, loc, rewriter, expandedShape, reassociations);

    // Convert back into original domain
    SmallVector<int64_t> paddedResultShape(outputShape.size(), 0);
    for (int i = 0; i < outputShape.size(); i++) {
      if (!imageDimensionsSet.contains(i)) {
        paddedResultShape[i] = outputShape[i];
      } else {
        paddedResultShape[i] = resultShape[i + numImageDims] * outputTileSize;
      }
    }
    if (isNchw) {
      permute<IREE::LinalgExt::Permutation::NHWC_TO_NCHW>(paddedResultShape);
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
    for (const int64_t shape : outputType.getShape())
      sizes.push_back(rewriter.getIndexAttr(shape));
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
    patterns.insert<FoldWinogradFilterTransform<linalg::Conv2DNchwFchwOp>,
                    FoldWinogradFilterTransform<linalg::Conv2DNhwcHwcfOp>,
                    ConvertConvToWinograd<linalg::Conv2DNhwcHwcfOp>,
                    ConvertConvToWinograd<linalg::Conv2DNchwFchwOp>>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertConv2DToWinogradPass() {
  return std::make_unique<ConvertConv2DToWinogradPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
