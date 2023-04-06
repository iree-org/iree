// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/WinogradConstants.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

namespace {

static void computeLoopParams(SmallVectorImpl<Value> &lbs,
                              SmallVectorImpl<Value> &ubs,
                              SmallVectorImpl<Value> &steps, Value tensor,
                              int numImageDims, Location loc,
                              OpBuilder &builder) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<OpFoldResult> dimValues =
      tensor::createDimValues(builder, loc, tensor);
  for (int i = numImageDims; i < dimValues.size(); i++) {
    lbs.push_back(zero);
    ubs.push_back(getValueOrCreateConstantIndexOp(builder, loc, dimValues[i]));
    steps.push_back(one);
  }
}

class ReifyWinogradInputTransform final
    : public OpRewritePattern<WinogradInputTransformOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  /// The input to this op is either (N, H, W, C) or (N, C, H, W)
  /// but the output to this op is always (T, T, N, H', W', C).
  /// Since the first two dimensions are used for the inner matrix
  /// multiplication, we create the loop nest over (N, H', W', C).
  LogicalResult matchAndRewrite(WinogradInputTransformOp inputOp,
                                PatternRewriter &rewriter) const override {
    Location loc = inputOp.getLoc();
    auto funcOp = inputOp->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          inputOp, "Could not find parent of type funcOp");
    }

    const float *BT{nullptr};
    const float *B{nullptr};
    const int64_t inputTileSize = inputOp.getInputTileSize();
    const int64_t outputTileSize = inputOp.getOutputTileSize();
    switch (outputTileSize) {
    case 6:
      B = IREE::LinalgExt::Winograd::B_6x6_3x3;
      BT = IREE::LinalgExt::Winograd::BT_6x6_3x3;
      break;
    default:
      return failure();
    }
    /// The two values below are the transpose(B) [BTV]
    /// and B [BV] constant matrices that convert the input
    /// tile to the Winograd domain.
    Value BTV = IREE::LinalgExt::createValueFrom2DConstant(
        BT, inputTileSize, inputTileSize, loc, rewriter);
    Value BV = IREE::LinalgExt::createValueFrom2DConstant(
        B, inputTileSize, inputTileSize, loc, rewriter);

    Value input = inputOp.input();
    Value output = inputOp.output();
    auto outputType = output.getType().cast<ShapedType>();
    auto inputType = input.getType().cast<ShapedType>();
    SmallVector<int64_t> inputShape(inputType.getShape());
    const bool isNchw = inputOp.isNchw();
    if (isNchw) {
      permute<Permutation::NCHW_TO_NHWC>(inputShape);
    }
    Type elementType = outputType.getElementType();
    const std::array<int64_t, 2> imageDims = inputOp.nhwcImageDimensions();
    const size_t numImageDims = imageDims.size();
    llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                  imageDims.end());
    SmallVector<int64_t> inputTileSquare(imageDims.size(), inputTileSize);

    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    Value zeroF32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value scratch =
        rewriter.create<tensor::EmptyOp>(loc, inputTileSquare, elementType);

    rewriter.setInsertionPoint(inputOp);
    SmallVector<Value> lbs, ubs, steps;
    computeLoopParams(lbs, ubs, steps, output, numImageDims, loc, rewriter);
    // Construct loops
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, ValueRange({output}),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
            ValueRange iterArgs) -> scf::ValueVector { return {iterArgs[0]}; });

    // Extract input slice
    auto one = rewriter.getIndexAttr(1);
    auto zero = rewriter.getIndexAttr(0);
    auto inputTileSizeAttr = rewriter.getIndexAttr(inputTileSize);
    SmallVector<OpFoldResult> strides(inputOp.getInputOperandRank(), one);
    SmallVector<OpFoldResult> sizes(inputOp.getInputOperandRank(), one);
    SmallVector<OpFoldResult> offsets(inputOp.getInputOperandRank(), zero);
    SmallVector<Value> ivs;
    for (scf::ForOp loop : loopNest.loops) {
      ivs.push_back(loop.getInductionVar());
    }
    for (int i = 0; i < inputShape.size(); i++) {
      if (!imageDimsSet.contains(i)) {
        offsets[i] = ivs[i];
      } else {
        rewriter.setInsertionPointToStart(loopNest.loops[i].getBody());
        AffineExpr dim0;
        auto it = rewriter.getAffineConstantExpr(inputTileSize);
        auto ot = rewriter.getAffineConstantExpr(outputTileSize);
        auto delta = rewriter.getAffineConstantExpr(inputShape[i]);
        bindDims(rewriter.getContext(), dim0);
        AffineMap scaleMap =
            AffineMap::get(1, 0, {dim0 * ot}, rewriter.getContext());
        offsets[i] = rewriter.createOrFold<AffineApplyOp>(loc, scaleMap,
                                                          ValueRange{ivs[i]});
        AffineMap minMap =
            AffineMap::get(1, 0, {-dim0 + delta, it}, rewriter.getContext());
        sizes[i] = rewriter.createOrFold<AffineMinOp>(
            loc, minMap,
            ValueRange{
                getValueOrCreateConstantIndexOp(rewriter, loc, offsets[i])});
      }
    }
    rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());
    auto tensorType = RankedTensorType::get(
        SmallVector<int64_t>(numImageDims, ShapedType::kDynamic), elementType);
    if (isNchw) {
      permute<Permutation::NHWC_TO_NCHW>(offsets);
      permute<Permutation::NHWC_TO_NCHW>(sizes);
    }
    Value dynamicSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, tensorType, input, offsets, sizes, strides);

    // Copy input slice into zeroed padded scratch space
    strides = SmallVector<OpFoldResult>(numImageDims, one);
    offsets = SmallVector<OpFoldResult>(numImageDims, zero);
    SmallVector<OpFoldResult> sliceSizes;
    for (const int64_t dim : inputOp.imageDimensions())
      sliceSizes.push_back(sizes[dim]);
    linalg::FillOp fillOp = rewriter.create<linalg::FillOp>(
        loc, ValueRange{zeroF32}, ValueRange{scratch});
    Value inputSlice = rewriter.create<tensor::InsertSliceOp>(
        loc, dynamicSlice, fillOp.result(), offsets, sliceSizes, strides);

    // Extract output slice
    strides = SmallVector<OpFoldResult>(inputOp.getOutputOperandRank(), one);
    offsets = SmallVector<OpFoldResult>(numImageDims, zero);
    offsets.append(ivs.begin(), ivs.end());
    sizes = SmallVector<OpFoldResult>(inputOp.getOutputOperandRank(), one);
    sizes[0] = sizes[1] = inputTileSizeAttr;
    tensorType = RankedTensorType::get(inputTileSquare, elementType);
    Value iterArg = loopNest.loops.back().getRegionIterArg(0);
    Value outputSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, tensorType, iterArg, offsets, sizes, strides);

    // Create computation
    Value result, AMatrix, BMatrix;
    linalg::MatmulOp matmulOp;
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

    // Insert results into output slice
    Value updatedOutput = rewriter.create<tensor::InsertSliceOp>(
        loc, result, iterArg, offsets, sizes, strides);

    // Replace returned value
    if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
            loopNest.loops.back().getBody()->getTerminator())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, updatedOutput);
    }
    inputOp.getResults()[0].replaceAllUsesWith(loopNest.results[0]);
    return success();
  }
};

} // namespace

namespace {

class ReifyWinogradOutputTransform final
    : public OpRewritePattern<WinogradOutputTransformOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  /// The input to this op is always (T, T, N, H', W', C)
  /// but the output is either (N, H, W, C) or (N, C, H, W).
  LogicalResult matchAndRewrite(WinogradOutputTransformOp outputOp,
                                PatternRewriter &rewriter) const override {
    Location loc = outputOp.getLoc();
    auto funcOp = outputOp->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          outputOp, "Could not find parent of type funcOp");
    }

    const float *AT{nullptr};
    const float *A{nullptr};
    const int64_t inputTileSize = outputOp.getInputTileSize();
    const int64_t outputTileSize = outputOp.getOutputTileSize();
    switch (outputTileSize) {
    case 6:
      A = IREE::LinalgExt::Winograd::A_6x6_3x3;
      AT = IREE::LinalgExt::Winograd::AT_6x6_3x3;
      break;
    default:
      return failure();
    }
    /// The two values below are the transpose(A) [ATV]
    /// and A [AV] constant matrices that convert the output
    /// tile from the Winograd domain to the original domain.
    Value ATV = IREE::LinalgExt::createValueFrom2DConstant(
        AT, outputTileSize, inputTileSize, loc, rewriter);
    Value AV = IREE::LinalgExt::createValueFrom2DConstant(
        A, inputTileSize, outputTileSize, loc, rewriter);

    Value input = outputOp.input();
    Value output = outputOp.output();
    auto outputType = output.getType().cast<ShapedType>();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    Type elementType = outputType.getElementType();
    const std::array<int64_t, 2> imageDims = outputOp.nhwcImageDimensions();
    const size_t numImageDims = imageDims.size();
    llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                  imageDims.end());
    SmallVector<int64_t> inputTileSquare(imageDims.size(), inputTileSize);

    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    Value zeroF32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    SmallVector<int64_t> scratchShape = {inputTileSize, outputTileSize};
    Value scratch =
        rewriter.create<tensor::EmptyOp>(loc, scratchShape, elementType);

    rewriter.setInsertionPoint(outputOp);
    SmallVector<Value> lbs, ubs, steps;
    computeLoopParams(lbs, ubs, steps, input, numImageDims, loc, rewriter);
    // Construct loops
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, ValueRange({output}),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
            ValueRange iterArgs) -> scf::ValueVector { return {iterArgs[0]}; });

    // Extract input slice
    rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());
    auto one = rewriter.getIndexAttr(1);
    auto zero = rewriter.getIndexAttr(0);
    auto inputTileSizeAttr = rewriter.getIndexAttr(inputTileSize);
    auto outputTileSizeAttr = rewriter.getIndexAttr(outputTileSize);
    SmallVector<OpFoldResult> strides(outputOp.getInputOperandRank(), one);
    SmallVector<OpFoldResult> sizes(outputOp.getInputOperandRank(), one);
    SmallVector<OpFoldResult> offsets(numImageDims, zero);
    sizes[0] = sizes[1] = inputTileSizeAttr;
    SmallVector<Value> ivs;
    for (scf::ForOp loop : loopNest.loops) {
      ivs.push_back(loop.getInductionVar());
    }
    offsets.append(ivs.begin(), ivs.end());
    auto tensorType = RankedTensorType::get(inputTileSquare, elementType);
    tensor::ExtractSliceOp extractSliceOp =
        rewriter.create<tensor::ExtractSliceOp>(loc, tensorType, input, offsets,
                                                sizes, strides);
    Value inputSlice = extractSliceOp.getResult();

    // Extract output slice
    strides = SmallVector<OpFoldResult>(outputOp.getOutputOperandRank(), one);
    offsets = SmallVector<OpFoldResult>(outputOp.getOutputOperandRank(), zero);
    sizes = SmallVector<OpFoldResult>(outputOp.getOutputOperandRank(), one);
    for (int i = 0; i < outputShape.size(); i++) {
      if (!imageDimsSet.contains(i)) {
        offsets[i] = ivs[i];
      } else {
        rewriter.setInsertionPointToStart(loopNest.loops[i].getBody());
        AffineExpr dim0;
        auto ot = rewriter.getAffineConstantExpr(outputTileSize);
        bindDims(rewriter.getContext(), dim0);
        AffineMap scaleMap =
            AffineMap::get(1, 0, {dim0 * ot}, rewriter.getContext());
        offsets[i] = rewriter.createOrFold<AffineApplyOp>(loc, scaleMap,
                                                          ValueRange{ivs[i]});
        sizes[i] = outputTileSizeAttr;
      }
    }
    rewriter.setInsertionPointAfter(extractSliceOp);
    tensorType = RankedTensorType::get(
        SmallVector<int64_t>(numImageDims, outputTileSize), elementType);
    Value iterArg = loopNest.loops.back().getRegionIterArg(0);
    if (outputOp.isNchw()) {
      permute<Permutation::NHWC_TO_NCHW>(offsets);
      permute<Permutation::NHWC_TO_NCHW>(sizes);
    }
    Value outputSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, tensorType, iterArg, offsets, sizes, strides);

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

    // Insert results into output slice
    Value updatedOutput = rewriter.create<tensor::InsertSliceOp>(
        loc, result, iterArg, offsets, sizes, strides);

    // Replace returned value
    if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
            loopNest.loops.back().getBody()->getTerminator())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, updatedOutput);
    }
    outputOp.getResults()[0].replaceAllUsesWith(loopNest.results[0]);
    return success();
  }
};

} // namespace

namespace {
struct TileAndDecomposeWinogradTransformPass
    : public TileAndDecomposeWinogradTransformBase<
          TileAndDecomposeWinogradTransformPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void TileAndDecomposeWinogradTransformPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(&getContext());
  patterns.insert<ReifyWinogradInputTransform, ReifyWinogradOutputTransform>(
      context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createTileAndDecomposeWinogradTransformPass() {
  return std::make_unique<TileAndDecomposeWinogradTransformPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
