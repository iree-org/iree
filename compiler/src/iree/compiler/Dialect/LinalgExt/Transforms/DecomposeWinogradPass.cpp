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
#include "mlir/IR/BuiltinTypes.h"
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

/// Pattern to decompose the tiled WinogradFilterTransformOp.
/// The input should be just a single tile of the input filter of size `m x m`,
/// where `m` is the kernel size. The output size should be `i x i`, where
/// `i = m + r - 1` and `r` is the output tile size.
///
/// The tile of the filter transform decomposes into two matrix multiplications:
/// `matmul(G, matmul(tile(f), transpose(G)))`
/// The matrix `G` is a precomputed constant.
///
/// The result of the decomposition will look like the following:
/// ```
/// %tf = iree_linalg_ext.winograd.filter_transform
///     output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
///     ins(%in_tile : tensor<3x3xf32>)
///     outs(%out_tile : tensor<8x8xf32>) -> tensor<8x8xf32>
/// ```
/// Decomposes to
/// ```
/// %G = arith.constant dense<[...]>
/// %GT = arith.constant dense<[...]>
/// %init_0 = linalg.fill ins(%zero) outs(%out_tile) -> tensor<3x8xf32>
/// %mm_0 = linalg.matmul ins(%in_tile, %GT : tensor<3x3xf32>, tensor<3x8xf32>)
///     outs(%init_0 : tensor<3x8xf32>) -> tensor<3x8xf32>
/// %init_1 = linalg.fill ins(%zero) outs(%out_tile) -> tensor<8x8xf32>
/// %mm_1 = linalg.matmul ins(%G, %mm_0 : tensor<8x3xf32>, tensor<3x8xf32>)
///     outs(%init_1 : tensor<8x8xf32>) -> tensor<8x8xf32>
/// ````
struct DecomposeWinogradFilterTransform
    : public OpRewritePattern<WinogradFilterTransformOp> {
  using OpRewritePattern<WinogradFilterTransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WinogradFilterTransformOp transformOp,
                                PatternRewriter &rewriter) const override {
    Location loc = transformOp.getLoc();

    if (transformOp.getInputRank() != 2 || transformOp.getOutputRank() != 2) {
      return rewriter.notifyMatchFailure(transformOp, "Winograd op not tiled");
    }
    const int64_t inputTileSize = transformOp.getInputTileSize();
    const int64_t kernelSize = transformOp.getKernelSize();
    ArrayRef<int64_t> kernelDims = transformOp.getKernelDimensions();
    llvm::SmallSetVector<int64_t, 2> kernelDimsSet(kernelDims.begin(),
                                                   kernelDims.end());
    Type elementType = transformOp.getOutputType().getElementType();
    Value zeroF32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    /// The two values below are the transpose(G) [GT]
    /// and G [G] constant matrices that convert the filter
    /// tile from the original domain to the Winograd domain.
    Value GT = IREE::LinalgExt::createValueFrom2DConstant(
        IREE::LinalgExt::Winograd::GT_6x6_3x3, kernelSize, inputTileSize, loc,
        rewriter);
    Value G = IREE::LinalgExt::createValueFrom2DConstant(
        IREE::LinalgExt::Winograd::G_6x6_3x3, inputTileSize, kernelSize, loc,
        rewriter);

    // Create matmul(input, GT)
    SmallVector<int64_t> initShape(kernelDims.size(), inputTileSize);
    initShape[0] = kernelSize;
    Value inputSlice = transformOp.input();
    Value outputSlice = transformOp.output();
    auto tensorType = transformOp.getOutputType().clone(initShape);
    Value init = rewriter.create<tensor::EmptyOp>(loc, initShape, elementType);
    linalg::FillOp fillOp = rewriter.create<linalg::FillOp>(
        loc, ValueRange{zeroF32}, ValueRange{init});
    linalg::MatmulOp matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, tensorType, ValueRange{inputSlice, GT}, fillOp.result());

    // Create matmul(G, matmul(input, GT))
    fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32},
                                             ValueRange{outputSlice});
    matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, outputSlice.getType(), ValueRange{G, matmulOp.getResult(0)},
        fillOp.result());
    rewriter.replaceOp(transformOp, matmulOp.getResult(0));
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
    if (transformOp.getInputRank() != 2 || transformOp.getOutputRank() != 2) {
      return rewriter.notifyMatchFailure(transformOp, "Winograd op not tiled");
    }

    /// The two values below are the transpose(B) [BT]
    /// and B [B] constant matrices that convert the input
    /// tile from the original domain to the Winograd domain.
    Location loc = transformOp.getLoc();
    const int64_t inputTileSize = transformOp.getInputTileSize();
    Value BT = IREE::LinalgExt::createValueFrom2DConstant(
        IREE::LinalgExt::Winograd::BT_6x6_3x3, inputTileSize, inputTileSize,
        loc, rewriter);
    Value B = IREE::LinalgExt::createValueFrom2DConstant(
        IREE::LinalgExt::Winograd::B_6x6_3x3, inputTileSize, inputTileSize, loc,
        rewriter);

    // Pad the input slice.
    Value dynamicSlice = transformOp.input();
    Type elementType = transformOp.getOutputType().getElementType();
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    SmallVector<int64_t> inputTileSquare(
        transformOp.getImageDimensions().size(), inputTileSize);
    auto inputSliceType = RankedTensorType::get(inputTileSquare, elementType);
    Value inputSlice = tensor::createPadHighOp(
        inputSliceType, dynamicSlice, zero, /*nofold=*/false, loc, rewriter);

    // Create computation
    Value result, AMatrix, BMatrix;
    linalg::MatmulOp matmulOp;
    for (int i = 0; i < 2; i++) {
      auto fillOp = rewriter.create<linalg::FillOp>(
          loc, ValueRange{zero}, ValueRange{transformOp.output()});
      if (i == 0) {
        AMatrix = inputSlice;
        BMatrix = B;
      } else {
        AMatrix = BT;
        BMatrix = result;
      }
      matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, transformOp.getOutputType(), ValueRange{AMatrix, BMatrix},
          fillOp.result());
      result = matmulOp.getResult(0);
    }
    rewriter.replaceOp(transformOp, result);
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
    if (transformOp.getInputRank() != 2 || transformOp.getOutputRank() != 2) {
      return rewriter.notifyMatchFailure(transformOp, "Winograd op not tiled");
    }
    ShapedType outputType = transformOp.getOutputType();
    Type elementType = outputType.getElementType();
    const int64_t inputTileSize = transformOp.getInputTileSize();
    const int64_t outputTileSize = transformOp.getOutputTileSize();
    /// The two values below are the transpose(A) [AT]
    /// and A [A] constant matrices that convert the output
    /// tile from the Winograd domain to the original domain.
    Value AT = IREE::LinalgExt::createValueFrom2DConstant(
        IREE::LinalgExt::Winograd::AT_6x6_3x3, outputTileSize, inputTileSize,
        loc, rewriter);
    Value A = IREE::LinalgExt::createValueFrom2DConstant(
        IREE::LinalgExt::Winograd::A_6x6_3x3, inputTileSize, outputTileSize,
        loc, rewriter);
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
        BMatrix = A;
      } else {
        AMatrix = AT;
        BMatrix = result;
      }
      matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, tmp.getType(), ValueRange{AMatrix, BMatrix}, fillOp.result());
      result = matmulOp.getResult(0);
    }
    rewriter.replaceOp(transformOp, result);
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
  patterns.add<FoldWinogradOpUnitDims<WinogradFilterTransformOp>>(context);
  patterns.add<FoldWinogradOpUnitDims<WinogradInputTransformOp>>(context);
  patterns.add<FoldWinogradOpUnitDims<WinogradOutputTransformOp>>(context);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
  patterns.add<DecomposeWinogradFilterTransform>(context);
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
