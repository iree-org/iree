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

static Value
createCollapse(Value tensor, Location loc, PatternRewriter &rewriter,
               SmallVectorImpl<int64_t> &outputShape,
               SmallVectorImpl<ReassociationIndices> &reassociations) {
  auto tensorType = cast<ShapedType>(tensor.getType());
  auto elementTy = tensorType.getElementType();
  auto resultType = RankedTensorType::get(outputShape, elementTy);
  return rewriter.create<tensor::CollapseShapeOp>(loc, resultType, tensor,
                                                  reassociations);
}

static Value
createExpand(Value tensor, Location loc, PatternRewriter &rewriter,
             SmallVectorImpl<int64_t> &outputShape,
             SmallVectorImpl<ReassociationIndices> &reassociations) {
  auto tensorType = cast<ShapedType>(tensor.getType());
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

    bool isNchwFchw;
    if (!isValidConv2d(convOp, isNchwFchw)) {
      return failure();
    }

    // Create winograd filter transform op.
    Value kernel = convOp.getInputs()[1];
    auto kernelType = cast<ShapedType>(kernel.getType());
    if (!kernelType) {
      return failure();
    }
    SmallVector<int64_t> kernelShape(kernelType.getShape());
    if (llvm::any_of(kernelShape, ShapedType::isDynamic)) {
      return failure();
    }
    const int64_t kh = isNchwFchw ? kernelShape[2] : kernelShape[0];
    const int64_t kw = isNchwFchw ? kernelShape[3] : kernelShape[1];
    if ((kh != 3) || (kw != 3)) {
      return failure();
    }
    assert(kernelShape.size() == 4);
    Type elementType = kernelType.getElementType();

    const int64_t kernelSize = 3;
    const int64_t inputTileSize = outputTileSize + kernelSize - 1;

    Location loc = convOp.getLoc();
    const std::array<int64_t, 2> hwcfKernelDims{0, 1};
    const std::array<int64_t, 2> fchwKernelDims{2, 3};
    SmallVector<int64_t> filterResultShape(4, inputTileSize);
    filterResultShape[2] = isNchwFchw ? kernelShape[1] : kernelShape[2];
    filterResultShape[3] = isNchwFchw ? kernelShape[0] : kernelShape[3];
    Value kernelInit =
        rewriter.create<tensor::EmptyOp>(loc, filterResultShape, elementType);
    auto &kernelDims = isNchwFchw ? fchwKernelDims : hwcfKernelDims;
    auto winogradFilterOp =
        rewriter.create<IREE::LinalgExt::WinogradFilterTransformOp>(
            loc, kernelInit.getType(), ValueRange{kernel},
            ValueRange{kernelInit}, outputTileSize, kernelSize, kernelDims);
    Value winogradFilter = winogradFilterOp.getResult()[0];

    // Add collapse shape
    SmallVector<int64_t> collapsedFilterShape;
    collapsedFilterShape.push_back(filterResultShape[0] * filterResultShape[1]);
    collapsedFilterShape.push_back(filterResultShape[2]);
    collapsedFilterShape.push_back(filterResultShape[3]);
    SmallVector<ReassociationIndices> filterReassociations = {{0, 1}, {2}, {3}};
    Value collapsedWinogradFilter =
        createCollapse(winogradFilter, loc, rewriter, collapsedFilterShape,
                       filterReassociations);

    // Create winograd input transform op.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value input = convOp.getInputs()[0];
    auto inputType = cast<ShapedType>(input.getType());
    if (!inputType) {
      return failure();
    }
    SmallVector<int64_t> inputShape(inputType.getShape());
    if (llvm::any_of(inputShape, ShapedType::isDynamic)) {
      return failure();
    }
    assert(inputShape.size() == 4);
    if (isNchwFchw) {
      permute<IREE::LinalgExt::Permutation::NCHW_TO_NHWC>(inputShape);
    }

    const std::array<int64_t, 2> nhwcImageDims{1, 2};
    const std::array<int64_t, 2> nchwImageDims{2, 3};
    const size_t numImageDims = nhwcImageDims.size();
    SmallVector<int64_t> resultShape(6, inputTileSize);
    llvm::SmallSetVector<int64_t, 2> imageDimsSet(nhwcImageDims.begin(),
                                                  nhwcImageDims.end());
    int outputIndex;
    for (int i = 0; i < inputShape.size(); i++) {
      outputIndex = i + numImageDims;
      if (!imageDimsSet.contains(i)) {
        resultShape[outputIndex] = inputShape[i];
      } else {
        resultShape[outputIndex] =
            std::ceil((float)(inputShape[i] - kernelSize + 1) / outputTileSize);
      }
    }
    Value emptyTensor =
        rewriter.create<tensor::EmptyOp>(loc, resultShape, elementType);
    auto &imageDims = isNchwFchw ? nchwImageDims : nhwcImageDims;
    auto winogradInputOp =
        rewriter.create<IREE::LinalgExt::WinogradInputTransformOp>(
            loc, emptyTensor.getType(), ValueRange{input},
            ValueRange{emptyTensor}, outputTileSize, kernelSize, imageDims);
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
    auto outputType = cast<RankedTensorType>(output.getType());
    SmallVector<int64_t> outputShape(outputType.getShape());
    if (isNchwFchw) {
      permute<IREE::LinalgExt::Permutation::NCHW_TO_NHWC>(outputShape);
    }
    bmmShape[2] = outputShape[3];
    auto bmmOutputType = RankedTensorType::get(bmmShape, elementType);
    emptyTensor = rewriter.create<tensor::EmptyOp>(loc, bmmShape, elementType);
    auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zero},
                                                  ValueRange{emptyTensor});
    auto bmmOp = rewriter.create<linalg::BatchMatmulOp>(
        loc, bmmOutputType,
        ValueRange({collapsedWinogradInput, collapsedWinogradFilter}),
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
      if (!imageDimsSet.contains(i)) {
        paddedResultShape[i] = outputShape[i];
      } else {
        paddedResultShape[i] = resultShape[i + numImageDims] * outputTileSize;
      }
    }
    if (isNchwFchw) {
      permute<IREE::LinalgExt::Permutation::NHWC_TO_NCHW>(paddedResultShape);
    }
    emptyTensor =
        rewriter.create<tensor::EmptyOp>(loc, paddedResultShape, elementType);
    auto winogradOutputOp =
        rewriter.create<IREE::LinalgExt::WinogradOutputTransformOp>(
            loc, emptyTensor.getType(), ValueRange{expandedBmmResult},
            ValueRange{emptyTensor}, outputTileSize, kernelSize, imageDims);
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
    patterns.insert<ConvertConvToWinograd<linalg::Conv2DNhwcHwcfOp>,
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
