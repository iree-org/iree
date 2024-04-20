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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {
namespace {

/// Pattern to remove unit dims from winograd ops after tililng. Tiling is
/// expected to tile most dimensions to 1, so the winograd op is only a small
/// tile of rank 2 for decomposition.
template <typename TransformOp>
struct FoldWinogradOpUnitDims : public OpRewritePattern<TransformOp> {
  using OpRewritePattern<TransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransformOp transformOp,
                                PatternRewriter &rewriter) const override {
    auto originalType =
        dyn_cast<RankedTensorType>(transformOp.getOriginalOperandType());
    auto transformedType =
        dyn_cast<RankedTensorType>(transformOp.getTransformedOperandType());
    if (!originalType || !transformedType) {
      return failure();
    }
    auto originalShape = originalType.getShape();
    auto transformedShape = transformedType.getShape();
    auto hwDims = transformOp.hwDimensions();
    SetVector<int64_t> hwDimsSet(hwDims.begin(), hwDims.end());
    if (!llvm::all_of(llvm::enumerate(originalShape),
                      [&](auto it) {
                        return it.value() == 1 ||
                               hwDimsSet.contains(it.index());
                      }) ||
        originalShape.size() == 2) {
      return failure();
    }
    Location loc = transformOp->getLoc();
    SmallVector<int64_t> newOriginalShape = llvm::map_to_vector(
        hwDims, [&](int64_t dim) { return originalShape[dim]; });
    auto newOriginalType = originalType.clone(newOriginalShape);
    SmallVector<int64_t> newTransformedShape(transformedShape.begin(),
                                             transformedShape.begin() + 2);
    auto newTransformedType = transformedType.clone(newTransformedShape);
    RankedTensorType newInputType = newOriginalType;
    RankedTensorType newOutputType = newTransformedType;
    if (std::is_same_v<TransformOp, WinogradOutputTransformOp>) {
      newInputType = newTransformedType;
      newOutputType = newOriginalType;
    }

    auto newInput = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, transformOp.getInputs()[0], newInputType);
    auto newInit = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, transformOp.getOutputs()[0], newOutputType);
    auto rankReducedTransform = clone(rewriter, transformOp, newOutputType,
                                      ValueRange{newInput, newInit});
    auto insertSliceOp = tensor::createCanonicalRankReducingInsertSliceOp(
        rewriter, loc, rankReducedTransform->getResult(0),
        transformOp.getOutputs()[0]);
    rewriter.replaceOp(transformOp, insertSliceOp);

    return success();
  }
};

/// Pattern to decompose the tiled WinogradInputTransformOp.
/// The input should be just a single tile of the input image of size `i x i`,
/// where `i` is the input tile size and `i = m + r - 1`. `m` is the filter
/// size, and `r` is the output tile size. This tile might not be a full tile.
///
/// The tile of the input transform decomposes into two matrix multiplications:
/// `matmul(transpose(B), matmul(tile(x), B))`
/// The matrix `B` is a precomputed constant.
///
/// The result of the decomposition will look like the following:
/// ```
/// %tf = iree_linalg_ext.winograd.input_transform
///     output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
///     ins(%in_tile : tensor<?x?xf32>)
///     outs(%out_tile : tensor<8x8xf32>) -> tensor<8x8xf32>
/// ```
/// Decomposes to
/// ```
/// %B = arith.constant dense<[...]>
/// %BT = arith.constant dense<[...]>
/// %scratch = linalg.fill ins(%zero) outs(%empty) -> tensor<8x8xf32>
/// %padded = tensor.insert_slice %in_tile into %scratch
///     : tensor<?x?xf16> into tensor<8x8xf16>
/// %init_0 = linalg.fill ins(%zero) outs(%out_tile) -> tensor<8x8xf32>
/// %mm_0 = linalg.matmul ins(%padded, %B : tensor<8x8xf16>, tensor<8x8xf32>)
///     outs(%init_0 : tensor<8x8xf32>) -> tensor<8x8xf32>
/// %init_1 = linalg.fill ins(%zero) outs(%out_tile) -> tensor<8x8xf32>
/// %mm_1 = linalg.matmul ins(%BT, %mm_0 : tensor<8x8xf16>, tensor<8x8xf32>)
///     outs(%init_1 : tensor<8x8xf32>) -> tensor<8x8xf32>
/// ````
struct DecomposeWinogradInputTransform
    : public OpRewritePattern<WinogradInputTransformOp> {
  using OpRewritePattern<WinogradInputTransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WinogradInputTransformOp transformOp,
                                PatternRewriter &rewriter) const override {
    Location loc = transformOp.getLoc();
    Value dynamicSlice = transformOp.input();
    Value outputSlice = transformOp.output();
    if (transformOp.getInputOperandRank() != 2 ||
        transformOp.getOutputOperandRank() != 2) {
      return rewriter.notifyMatchFailure(transformOp, "Winograd op not tiled");
    }
    auto one = rewriter.getIndexAttr(1);
    auto zero = rewriter.getIndexAttr(0);
    const int64_t inputTileSize = transformOp.getInputTileSize();
    ArrayRef<int64_t> imageDims = transformOp.getImageDimensions();
    llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                  imageDims.end());
    SmallVector<int64_t> inputTileSquare(imageDims.size(), inputTileSize);
    Type elementType = transformOp.getOutputOperandType().getElementType();
    Value zeroF32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value scratch =
        rewriter.create<tensor::EmptyOp>(loc, inputTileSquare, elementType);
    const float *BT{nullptr};
    const float *B{nullptr};
    B = IREE::LinalgExt::Winograd::B_6x6_3x3;
    BT = IREE::LinalgExt::Winograd::BT_6x6_3x3;
    Value BTV = IREE::LinalgExt::createValueFrom2DConstant(
        BT, inputTileSize, inputTileSize, loc, rewriter);
    Value BV = IREE::LinalgExt::createValueFrom2DConstant(
        B, inputTileSize, inputTileSize, loc, rewriter);

    auto inputExtractSliceOp =
        dynamicSlice.getDefiningOp<tensor::ExtractSliceOp>();
    SmallVector<OpFoldResult> mixedSizes = inputExtractSliceOp.getMixedSizes();
    // Harcoding input rank as 4 here - since we'd be getting a tiled version
    // with rank 2. We are always expected to either have a rank 4 version of
    // this op, or rank 2 (tiled). And at this point in the flow, it is
    // guaranteed to be a rank 2 version of the op as ensured by the assertion
    // above. Copy input slice into zeroed padded scratch space
    SmallVector<OpFoldResult> offsets(2, zero);
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(2, one);
    for (int i = 0; i < 4; i++) {
      if (imageDimsSet.contains(i)) {
        sizes.push_back(mixedSizes[i]);
      }
    }
    linalg::FillOp fillOp = rewriter.create<linalg::FillOp>(
        loc, ValueRange{zeroF32}, ValueRange{scratch});
    Value inputSlice = rewriter.create<tensor::InsertSliceOp>(
        loc, dynamicSlice, fillOp.result(), offsets, sizes, strides);

    // Create computation
    Value result, AMatrix, BMatrix;
    linalg::MatmulOp matmulOp;
    Type tensorType = outputSlice.getType();
    for (int i = 0; i < 2; i++) {
      fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32},
                                               ValueRange{outputSlice});
      if (i == 0) {
        AMatrix = inputSlice;
        BMatrix = BV;
      } else {
        AMatrix = BTV;
        BMatrix = result;
      }
      matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, tensorType, ValueRange{AMatrix, BMatrix}, fillOp.result());
      result = matmulOp.getResult(0);
    }
    transformOp.getResult()[0].replaceAllUsesWith(result);
    return success();
  }
};

/// Pattern to decompose the tiled WinogradOutputTransformOp.
/// The input should be just a single tile of the winograd output image of size
/// `i x i`, where `i` is the input tile size and `i = m + r - 1`. `m` is the
/// filter size, and `r` is the output tile size.
///
/// The tile of the input transform decomposes into two matrix multiplications:
/// `matmul(transpose(A), matmul(x, A))`
/// The matrix `A` is a precomputed constant.
///
/// The result of the decomposition will look like the following:
/// ```
/// %tf = iree_linalg_ext.winograd.output_transform
///     output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
///     ins(%in_tile : tensor<8x8xf32>)
///     outs(%out_tile : tensor<6x6xf32>) -> tensor<6x6xf32>
/// ```
/// Decomposes to
/// ```
/// %A = arith.constant dense<[...]>
/// %AT = arith.constant dense<[...]>
/// %init_0 = linalg.fill ins(%zero) outs(%empty) -> tensor<8x6xf32>
/// %mm_0 = linalg.matmul ins(%in_tile, %A : tensor<8x8xf16>, tensor<8x6xf32>)
///     outs(%init_0 : tensor<8x8xf32>) -> tensor<8x8xf32>
/// %init_1 = linalg.fill ins(%zero) outs(%out_tile) -> tensor<6x6xf32>
/// %mm_1 = linalg.matmul ins(%AT, %mm_0 : tensor<6x8xf16>, tensor<8x6xf32>)
///     outs(%init_1 : tensor<6x6xf32>) -> tensor<6x6xf32>
/// ````
struct DecomposeWinogradOutputTransform
    : public OpRewritePattern<WinogradOutputTransformOp> {
  using OpRewritePattern<WinogradOutputTransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WinogradOutputTransformOp transformOp,
                                PatternRewriter &rewriter) const override {
    Location loc = transformOp.getLoc();
    Value inputSlice = transformOp.input();
    Value outputSlice = transformOp.output();
    if (transformOp.getInputOperandRank() != 2 ||
        transformOp.getOutputOperandRank() != 2) {
      return rewriter.notifyMatchFailure(transformOp, "Winograd op not tiled");
    }
    ShapedType outputType = transformOp.getOutputOperandType();
    Type elementType = outputType.getElementType();
    const float *AT{nullptr};
    const float *A{nullptr};
    A = IREE::LinalgExt::Winograd::A_6x6_3x3;
    AT = IREE::LinalgExt::Winograd::AT_6x6_3x3;
    const int64_t inputTileSize = transformOp.getInputTileSize();
    const int64_t outputTileSize = transformOp.getOutputTileSize();
    /// The two values below are the transpose(A) [ATV]
    /// and A [AV] constant matrices that convert the output
    /// tile from the Winograd domain to the original domain.
    Value ATV = IREE::LinalgExt::createValueFrom2DConstant(
        AT, outputTileSize, inputTileSize, loc, rewriter);
    Value AV = IREE::LinalgExt::createValueFrom2DConstant(
        A, inputTileSize, outputTileSize, loc, rewriter);
    Value zeroF32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    SmallVector<int64_t> scratchShape = {inputTileSize, outputTileSize};
    Value scratch =
        rewriter.create<tensor::EmptyOp>(loc, scratchShape, elementType);
    // Create computation
    Value result, AMatrix, BMatrix;
    linalg::MatmulOp matmulOp;
    linalg::FillOp fillOp;
    Value tmp;
    for (int i = 0; i < 2; i++) {
      tmp = i == 0 ? scratch : outputSlice;
      fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32},
                                               ValueRange{tmp});
      if (i == 0) {
        AMatrix = inputSlice;
        BMatrix = AV;
      } else {
        AMatrix = ATV;
        BMatrix = result;
      }
      matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, tmp.getType(), ValueRange{AMatrix, BMatrix}, fillOp.result());
      result = matmulOp.getResult(0);
    }
    transformOp.getResult()[0].replaceAllUsesWith(result);
    return success();
  }
};

} // namespace

namespace {
struct DecomposeWinogradTransformPass
    : public DecomposeWinogradTransformBase<DecomposeWinogradTransformPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void DecomposeWinogradTransformPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<FoldWinogradOpUnitDims<WinogradInputTransformOp>>(context);
  patterns.add<FoldWinogradOpUnitDims<WinogradOutputTransformOp>>(context);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
  patterns.add<DecomposeWinogradInputTransform>(context);
  patterns.add<DecomposeWinogradOutputTransform>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDecomposeWinogradTransformPass() {
  return std::make_unique<DecomposeWinogradTransformPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
