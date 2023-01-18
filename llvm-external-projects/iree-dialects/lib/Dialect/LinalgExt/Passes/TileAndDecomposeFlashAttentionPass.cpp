// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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

// Computes a reduction along the rows of a 2d tensor of shape MxN
// to produce a tensor of shape M
template <typename T>
static Value computeRowwiseReduction(Value a, Value output, Location loc,
                                     OpBuilder &builder) {
  SmallVector<utils::IteratorType> iteratorTypes{utils::IteratorType::reduction,
                                                 utils::IteratorType::parallel};
  AffineMap id = AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{id, rowMap};
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), a, output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<T>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static Value computeNewMax(Value oldMax, Value currentMax, Value output,
                           Location loc, OpBuilder &builder) {
  SmallVector<utils::IteratorType> iteratorTypes{utils::IteratorType::parallel};
  auto identityMap = AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps(3, identityMap);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, currentMax.getType(), ValueRange{oldMax, currentMax}, output,
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::MaxFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

// Computes alpha * oldSum + beta * currentSum
static Value computeNewSum(Value oldSum, Value currentSum, Value alpha,
                           Value beta, Value output, Location loc,
                           OpBuilder &builder) {
  SmallVector<utils::IteratorType> iteratorTypes{utils::IteratorType::parallel};
  auto identityMap = AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps(5, identityMap);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, currentSum.getType(), ValueRange{oldSum, currentSum, alpha, beta},
      output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value first = b.create<arith::MulFOp>(loc, args[2], args[0]);
        Value second = b.create<arith::MulFOp>(loc, args[3], args[1]);
        Value result = b.create<arith::AddFOp>(loc, first, second);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

// Computes c = exp(a - b) where a, b, c are 1D
static Value subtractAndExponentiate1D(Value a, Value b, Value output,
                                       Location loc, OpBuilder &builder) {
  SmallVector<utils::IteratorType> iteratorTypes{utils::IteratorType::parallel};
  auto identityMap = AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps(3, identityMap);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, a.getType(), ValueRange{a, b}, output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[0], args[1]);
        Value result = b.create<math::ExpOp>(loc, diff);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static Value computeSoftmax(Value qkTranspose, Value currentMax,
                            Value currentWeight, Value newSum, Value output,
                            Location loc, OpBuilder &builder) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, rowMap, rowMap, rowMap,
                                      identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, qkTranspose.getType(),
      ValueRange{qkTranspose, currentMax, currentWeight, newSum}, output,
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[0], args[1]);
        Value result = b.create<math::ExpOp>(loc, diff);
        Value scaledResult = b.create<arith::MulFOp>(loc, result, args[2]);
        Value finalResult = b.create<arith::DivFOp>(loc, scaledResult, args[3]);
        b.create<linalg::YieldOp>(loc, finalResult);
      });
  return genericOp.getResult(0);
}

static Value scaleAccumulator(Value accumulator, Value oldSum, Value newSum,
                              Value oldWeight, Value output, Location loc,
                              OpBuilder &builder) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, rowMap, rowMap, rowMap,
                                      identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, accumulator.getType(),
      ValueRange{accumulator, oldSum, newSum, oldWeight}, output, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value prod = b.create<arith::MulFOp>(loc, args[1], args[3]);
        Value scaled = b.create<arith::MulFOp>(loc, args[0], prod);
        Value result = b.create<arith::DivFOp>(loc, scaled, args[2]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static Value computeQKTranspose(Value query, Value key, Value transposedOutput,
                                Value output, Value zero,
                                RankedTensorType tensorType, Location loc,
                                OpBuilder &builder) {
  SmallVector<int64_t> perm{1, 0};
  auto transposeOp =
      builder.create<linalg::TransposeOp>(loc, key, transposedOutput, perm);
  Value acc =
      builder.create<linalg::FillOp>(loc, ValueRange{zero}, output).result();
  auto matmulOp = builder.create<linalg::MatmulOp>(
      loc, tensorType, ValueRange{query, transposeOp.getResult()[0]}, acc);
  return matmulOp.getResult(0);
}

static std::tuple<Value, Value, Value, Value>
extractSlices(Value key, Value value, Value query, Value output,
              ArrayRef<int64_t> queryShape, ArrayRef<Value> ivs,
              Value sequenceTileLength, Type elementType, Location loc,
              OpBuilder &builder) {
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  auto headDimension = builder.getIndexAttr(queryShape.back());
  SmallVector<OpFoldResult> strides(queryShape.size(), one);
  SmallVector<OpFoldResult> sizes(queryShape.size(), one);
  SmallVector<OpFoldResult> offsets(queryShape.size(), zero);
  sizes[1] = sequenceTileLength;
  sizes[2] = headDimension;
  offsets[0] = ivs[0];
  offsets[1] = ivs[1];
  SmallVector<int64_t> tensorShape{ShapedType::kDynamic, queryShape.back()};
  auto tensorType = RankedTensorType::get(tensorShape, elementType);
  Value keySlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, key, offsets, sizes, strides);
  Value valueSlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, value, offsets, sizes, strides);

  offsets = SmallVector<OpFoldResult>(queryShape.size(), zero);
  offsets[0] = ivs[0];
  Value querySlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, query, offsets, sizes, strides);
  Value outputSlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, output, offsets, sizes, strides);

  return std::make_tuple(keySlice, valueSlice, querySlice, outputSlice);
}

static std::tuple<Value, Value, Value>
insertSlices(Value newResult, Value result, Value newMax, Value max,
             Value newSum, Value sum, ArrayRef<int64_t> queryShape,
             ArrayRef<Value> ivs, Value sequenceTileLength, Location loc,
             OpBuilder &builder) {
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  auto headDimension = builder.getIndexAttr(queryShape.back());
  SmallVector<OpFoldResult> strides(queryShape.size(), one);
  SmallVector<OpFoldResult> sizes(queryShape.size(), one);
  SmallVector<OpFoldResult> offsets(queryShape.size(), zero);
  sizes[1] = sequenceTileLength;
  sizes[2] = headDimension;
  offsets[0] = ivs[0];
  Value updatedAcc = builder.create<tensor::InsertSliceOp>(
      loc, newResult, result, offsets, sizes, strides);
  offsets = SmallVector<OpFoldResult>{zero};
  sizes = SmallVector<OpFoldResult>{sequenceTileLength};
  strides = SmallVector<OpFoldResult>{one};
  Value updatedMax = builder.create<tensor::InsertSliceOp>(
      loc, newMax, max, offsets, sizes, strides);
  Value updatedSum = builder.create<tensor::InsertSliceOp>(
      loc, newSum, sum, offsets, sizes, strides);
  return std::make_tuple(updatedAcc, updatedMax, updatedSum);
}

static scf::LoopNest createLoopNest(SmallVectorImpl<Value> &ivs, Value lb,
                                    Value step, Value ub, ValueRange args,
                                    Location loc, OpBuilder &builder) {
  SmallVector<Value> lbs{lb};
  SmallVector<Value> steps{step};
  SmallVector<Value> ubs{ub};
  scf::LoopNest loopNest = scf::buildLoopNest(
      builder, loc, lbs, ubs, steps, args,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
          ValueRange iterArgs) -> scf::ValueVector { return iterArgs; });
  for (scf::ForOp loop : loopNest.loops) {
    ivs.push_back(loop.getInductionVar());
  }
  return loopNest;
}

class ReifyFlashAttentionFwdTransform final
    : public OpRewritePattern<FlashAttentionFwdOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FlashAttentionFwdOp attnOp,
                                PatternRewriter &rewriter) const override {
    Location loc = attnOp.getLoc();
    rewriter.setInsertionPoint(attnOp);

    Value query = attnOp.query();
    ShapedType queryType = attnOp.getQueryType();
    Type elementType = queryType.getElementType();
    ArrayRef<int64_t> queryShape = queryType.getShape();
    SmallVector<OpFoldResult> queryDimValues =
        tensor::createDimValues(rewriter, loc, query);
    Value sequenceTileLength =
        getValueOrCreateConstantIndexOp(rewriter, loc, queryDimValues[1]);
    Value batchTileLength =
        getValueOrCreateConstantIndexOp(rewriter, loc, queryDimValues[0]);

    Value key = attnOp.key();
    Value value = attnOp.value();
    SmallVector<OpFoldResult> keyDimValues =
        tensor::createDimValues(rewriter, loc, key);
    Value sequenceLength =
        getValueOrCreateConstantIndexOp(rewriter, loc, keyDimValues[1]);

    // Construct first loop
    Value zeroValue = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneValue = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> ivs;
    Value output = attnOp.output();
    scf::LoopNest firstLoopNest =
        createLoopNest(ivs, zeroValue, oneValue, batchTileLength,
                       ValueRange({output}), loc, rewriter);
    Value iterArg = firstLoopNest.loops.back().getRegionIterArg(0);

    rewriter.setInsertionPointToStart(firstLoopNest.loops.back().getBody());

    // Create max and sum statistics
    SmallVector<OpFoldResult> dims{sequenceTileLength};
    Value zeroF32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value largeNegativeF32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, -100.0));
    Value max = rewriter.create<tensor::EmptyOp>(loc, dims, elementType);
    Value negativeMax =
        rewriter.create<linalg::FillOp>(loc, ValueRange{largeNegativeF32}, max)
            .result();
    Value sum = rewriter.create<tensor::EmptyOp>(loc, dims, elementType);
    Value zeroSum =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32}, sum).result();

    // Construct second loop
    scf::LoopNest secondLoopNest = createLoopNest(
        ivs, zeroValue, sequenceTileLength, sequenceLength,
        ValueRange({iterArg, negativeMax, zeroSum}), loc, rewriter);

    Value iterArgResult = secondLoopNest.loops.back().getRegionIterArg(0);
    Value iterArgMax = secondLoopNest.loops.back().getRegionIterArg(1);
    Value iterArgSum = secondLoopNest.loops.back().getRegionIterArg(2);

    rewriter.setInsertionPointToStart(secondLoopNest.loops.back().getBody());

    auto [keySlice, valueSlice, querySlice, outputSlice] =
        extractSlices(key, value, query, iterArgResult, queryShape, ivs,
                      sequenceTileLength, elementType, loc, rewriter);

    // Compute matmul(q, transpose(k))
    auto headDimension = rewriter.getIndexAttr(queryShape.back());
    SmallVector<OpFoldResult> transposedShape{headDimension,
                                              sequenceTileLength};
    Value empty =
        rewriter.create<tensor::EmptyOp>(loc, transposedShape, elementType);
    SmallVector<OpFoldResult> resultShape{sequenceTileLength,
                                          sequenceTileLength};
    Value emptySquare =
        rewriter.create<tensor::EmptyOp>(loc, resultShape, elementType);
    auto tensorType = RankedTensorType::get(
        SmallVector<int64_t>(2, ShapedType::kDynamic), elementType);
    Value qkTranspose =
        computeQKTranspose(querySlice, keySlice, empty, emptySquare, zeroF32,
                           tensorType, loc, rewriter);

    empty = rewriter.create<tensor::EmptyOp>(
        loc, SmallVector<OpFoldResult>{sequenceTileLength}, elementType);

    // Compute current statistics
    Value currentMax = computeRowwiseReduction<arith::MaxFOp>(
        qkTranspose, empty, loc, rewriter);
    Value currentSum = computeRowwiseReduction<arith::AddFOp>(
        qkTranspose, empty, loc, rewriter);

    // Update global statistics
    Value newMax = computeNewMax(iterArgMax, currentMax, empty, loc, rewriter);
    Value oldWeight =
        subtractAndExponentiate1D(iterArgMax, newMax, empty, loc, rewriter);
    Value currentWeight =
        subtractAndExponentiate1D(currentMax, newMax, empty, loc, rewriter);
    Value newSum = computeNewSum(iterArgSum, currentSum, oldWeight,
                                 currentWeight, empty, loc, rewriter);

    // Compute softmax
    Value softmax = computeSoftmax(qkTranspose, currentMax, currentWeight,
                                   newSum, emptySquare, loc, rewriter);

    // Update accumulator
    empty = rewriter.create<tensor::EmptyOp>(
        loc, SmallVector<OpFoldResult>{sequenceLength, headDimension},
        elementType);
    Value scaledAcc = scaleAccumulator(outputSlice, iterArgSum, newSum,
                                       oldWeight, empty, loc, rewriter);

    // Compute matmul(softmax, v)
    Value result = rewriter
                       .create<linalg::MatmulOp>(
                           loc, outputSlice.getType(),
                           ValueRange{softmax, valueSlice}, scaledAcc)
                       .getResult(0);

    // Insert slices
    auto [updatedAcc, updatedMax, updatedSum] =
        insertSlices(result, iterArgResult, newMax, max, newSum, sum,
                     queryShape, ivs, sequenceTileLength, loc, rewriter);

    if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
            secondLoopNest.loops.back().getBody()->getTerminator())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(
          yieldOp, ValueRange{updatedAcc, updatedMax, updatedSum});
    }

    if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
            firstLoopNest.loops.back().getBody()->getTerminator())) {
      rewriter.setInsertionPoint(yieldOp);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(
          yieldOp, ValueRange{secondLoopNest.results[0]});
    }

    attnOp.getResults()[0].replaceAllUsesWith(firstLoopNest.results[0]);
    return success();
  }
};

} // namespace

namespace {
struct TileAndDecomposeFlashAttentionTransformPass
    : public TileAndDecomposeFlashAttentionTransformBase<
          TileAndDecomposeFlashAttentionTransformPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void TileAndDecomposeFlashAttentionTransformPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(&getContext());
  patterns.insert<ReifyFlashAttentionFwdTransform>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createTileAndDecomposeFlashAttentionTransformPass() {
  return std::make_unique<TileAndDecomposeFlashAttentionTransformPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
