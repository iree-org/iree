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
  SmallVector<utils::IteratorType> iteratorTypes{
      utils::IteratorType::parallel, utils::IteratorType::reduction};
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

static Value computeNewSum(Value oldMax, Value newMax, Value oldSum,
                           Value currentSum, Value output, Location loc,
                           OpBuilder &builder) {
  SmallVector<utils::IteratorType> iteratorTypes{utils::IteratorType::parallel};
  auto identityMap = AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps(5, identityMap);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, currentSum.getType(), ValueRange{oldMax, newMax, oldSum, currentSum},
      output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[0], args[1]);
        Value weight = b.create<math::ExpOp>(loc, diff);
        Value first = b.create<arith::MulFOp>(loc, weight, args[2]);
        Value result = b.create<arith::AddFOp>(loc, first, args[3]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static Value computeSoftmax(Value qkTranspose, Value currentMax, Value output,
                            Location loc, OpBuilder &builder) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, qkTranspose.getType(), ValueRange{qkTranspose, currentMax}, output,
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[0], args[1]);
        Value result = b.create<math::ExpOp>(loc, diff);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static Value scaleSoftmax(Value softmax, Value newSum, Value output,
                          Location loc, OpBuilder &builder) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, softmax.getType(), ValueRange{softmax, newSum}, output, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::DivFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static Value scaleAccumulator(Value accumulator, Value oldSum, Value newSum,
                              Value output, Location loc, OpBuilder &builder) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, rowMap, rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, accumulator.getType(), ValueRange{accumulator, oldSum, newSum},
      output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value prod = b.create<arith::DivFOp>(loc, args[1], args[2]);
        Value result = b.create<arith::MulFOp>(loc, args[0], prod);
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
  for (scf::ForOp loop : loopNest.loops)
    ivs.push_back(loop.getInductionVar());
  return loopNest;
}

/// This is an implementation of flash attention which
/// is a tiled and fused implementation of the attention operator.
/// The attention operator computes:
/// matmul(softmax(matmul(Q, transpose(K))), V)
/// where: Q is the query matrix [B x N x d]
///        K is the key matrix   [B x N x d]
///        V is the value matrix [B x N x d]
///
/// The core algorithm is as follows:
/// For each element in B,
/// 1. Load a tile from the Q matrix of size T x d -> q
/// 2. Initialize statistics: running_sum, running_max
/// 3. for i = 0 to N with step T
///    a. Load a tile from the K matrix of size T x d -> k
///    a. Load a tile from the V matrix of size T x d -> v
///    b. Transpose(k) -> kT
///    c. Compute matmul(q, kT) -> qkT
///    d. Compute sum(qkT) along rows -> current_sum
///    e. Compute max(qkT) along rows -> current_max
///    f. Compute new max: max(current_max, running_max)
///    g. Compute new sum: alpha * running_sum + beta * current_sum
///    h. Compute curent estimate of softmax: exp(qKT - current_max) -> s
///    i. Scale softmax estimate and current value of result by
///       appropriate factors
///    j. Compute matmul(s, v) and add to accumulator
///
///
LogicalResult reifyAttentionTransform(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  funcOp.walk([&](IREE::LinalgExt::AttentionOp attnOp) {
    Location loc = attnOp.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(attnOp);

    Value query = attnOp.getQuery();
    ShapedType queryType = attnOp.getQueryType();
    Type elementType = queryType.getElementType();
    ArrayRef<int64_t> queryShape = queryType.getShape();
    SmallVector<OpFoldResult> queryDimValues =
        tensor::createDimValues(rewriter, loc, query);
    Value sequenceTileLength =
        getValueOrCreateConstantIndexOp(rewriter, loc, queryDimValues[1]);
    Value batchTileLength =
        getValueOrCreateConstantIndexOp(rewriter, loc, queryDimValues[0]);

    Value key = attnOp.getKey();
    Value value = attnOp.getValue();
    SmallVector<OpFoldResult> keyDimValues =
        tensor::createDimValues(rewriter, loc, key);
    Value sequenceLength =
        getValueOrCreateConstantIndexOp(rewriter, loc, keyDimValues[1]);

    // Construct first loop
    Value zeroValue = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneValue = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> ivs;
    Value output = attnOp.getOutput();
    scf::LoopNest firstLoopNest =
        createLoopNest(ivs, zeroValue, oneValue, batchTileLength,
                       ValueRange({output}), loc, rewriter);
    Value iterArg = firstLoopNest.loops.back().getRegionIterArg(0);

    OpBuilder::InsertionGuard guardFirstLoop(rewriter);
    rewriter.setInsertionPointToStart(firstLoopNest.loops.back().getBody());

    // Create max and sum statistics
    SmallVector<OpFoldResult> dims{sequenceTileLength};
    Value zeroF32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value largeNegativeF32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, -1e30));
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

    OpBuilder::InsertionGuard guardSecondLoop(rewriter);
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
    Value newMax = computeRowwiseReduction<arith::MaxFOp>(
        qkTranspose, iterArgMax, loc, rewriter);
    Value softmax =
        computeSoftmax(qkTranspose, newMax, emptySquare, loc, rewriter);
    Value currentSum =
        computeRowwiseReduction<arith::AddFOp>(softmax, zeroSum, loc, rewriter);

    Value newSum = computeNewSum(iterArgMax, newMax, iterArgSum, currentSum,
                                 empty, loc, rewriter);

    // Scale softmax
    Value scaledSoftmax =
        scaleSoftmax(softmax, newSum, emptySquare, loc, rewriter);

    //// Update accumulator
    empty = rewriter.create<tensor::EmptyOp>(
        loc, SmallVector<OpFoldResult>{sequenceLength, headDimension},
        elementType);
    Value scaledAcc =
        scaleAccumulator(outputSlice, iterArgSum, newSum, empty, loc, rewriter);

    // Compute matmul(softmax, v)
    Value result = rewriter
                       .create<linalg::MatmulOp>(
                           loc, outputSlice.getType(),
                           ValueRange{scaledSoftmax, valueSlice}, scaledAcc)
                       .getResult(0);

    // Insert slices
    auto [updatedAcc, updatedMax, updatedSum] = insertSlices(
        result, iterArgResult, newMax, iterArgMax, newSum, iterArgSum,
        queryShape, ivs, sequenceTileLength, loc, rewriter);

    if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
            secondLoopNest.loops.back().getBody()->getTerminator())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(
          yieldOp, ValueRange{updatedAcc, updatedMax, updatedSum});
    }

    if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
            firstLoopNest.loops.back().getBody()->getTerminator())) {
      OpBuilder::InsertionGuard yieldGuard(rewriter);
      rewriter.setInsertionPoint(yieldOp);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(
          yieldOp, ValueRange{secondLoopNest.results[0]});
    }

    attnOp.getResults()[0].replaceAllUsesWith(firstLoopNest.results[0]);
    return WalkResult::advance();
  });
  return success();
}

} // namespace

namespace {
struct TileAndDecomposeAttentionPass
    : public TileAndDecomposeAttentionBase<TileAndDecomposeAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void TileAndDecomposeAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  if (failed(reifyAttentionTransform(getOperation())))
    return signalPassFailure();
}

std::unique_ptr<Pass> createTileAndDecomposeAttentionPass() {
  return std::make_unique<TileAndDecomposeAttentionPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
