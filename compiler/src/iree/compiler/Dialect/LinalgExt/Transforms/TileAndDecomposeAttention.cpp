// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

namespace {

// Computes a reduction along the rows of a 2d tensor of shape MxN
// to produce a tensor of shape M
template <typename T>
static Value computeRowwiseReduction(Value a, Value output, Location loc,
                                     OpBuilder &builder,
                                     SmallVectorImpl<Operation *> &ops) {
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
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value computePartialSoftmax(Value qkTranspose, Value currentMax,
                                   Location loc, OpBuilder &builder,
                                   SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, qkTranspose.getType(), ValueRange{currentMax}, qkTranspose,
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[1], args[0]);
        Value result = b.create<math::Exp2Op>(loc, diff);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

/// Return the scale factor for the new softmax maximum and add the generic to
/// the provided list of operations.
static Value computeScaleFactor(Value oldMax, Value newMax, Location loc,
                                OpBuilder &builder,
                                SmallVectorImpl<Operation *> &ops) {
  SmallVector<utils::IteratorType> iteratorTypes(1,
                                                 utils::IteratorType::parallel);
  auto identityMap = AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps(2, identityMap);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, oldMax.getType(), newMax, oldMax, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[1], args[0]);
        Value weight = b.create<math::Exp2Op>(loc, diff);
        b.create<linalg::YieldOp>(loc, weight);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value updateAndScale(Value scaleFactor, Value oldSum, Location loc,
                            OpBuilder &builder,
                            SmallVectorImpl<Operation *> &ops) {
  SmallVector<utils::IteratorType> iteratorTypes(1,
                                                 utils::IteratorType::parallel);
  auto identityMap = AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps(2, identityMap);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, oldSum.getType(), scaleFactor, oldSum, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value scaledOldSum = b.create<arith::MulFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, scaledOldSum);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value scalePartialSoftmax(Value softmax, Value inverseNewSum,
                                 Location loc, OpBuilder &builder,
                                 SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, softmax.getType(), ValueRange{inverseNewSum}, softmax, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::MulFOp>(loc, args[1], args[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value applyFinalScaling(Value result, Value newSum, Location loc,
                               OpBuilder &builder,
                               SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps = {rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, result.getType(), newSum, result, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value one = b.create<arith::ConstantOp>(
            loc, b.getFloatAttr(args[0].getType(), 1.0));
        Value reciprocal = b.create<arith::DivFOp>(loc, one, args[0]);
        Value result = b.create<arith::MulFOp>(loc, reciprocal, args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value scaleAccumulator(Value accumulator, Value scaleFactor,
                              Location loc, OpBuilder &builder,
                              SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, accumulator.getType(), scaleFactor, accumulator, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::MulFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value computeQKTranspose(Value query, Value key, Value output,
                                Value zero, Location loc, OpBuilder &builder,
                                SmallVectorImpl<Operation *> &ops) {
  auto fillOp = builder.create<linalg::FillOp>(loc, ValueRange{zero}, output);
  ops.push_back(fillOp);
  Value acc = fillOp.result();
  auto matmulOp = builder.create<linalg::MatmulTransposeBOp>(
      loc, output.getType(), ValueRange{query, key}, acc);
  ops.push_back(matmulOp);
  return matmulOp.getResult(0);
}

static Value extractSlice(Value key, ArrayRef<int64_t> keyShape,
                          ArrayRef<Value> ivs, OpFoldResult keyValueTileLength,
                          OpFoldResult headDimension, Type elementType,
                          Location loc, OpBuilder &builder,
                          bool swapLastTwoDims = false) {
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> strides(keyShape.size(), one);
  SmallVector<OpFoldResult> sizes(keyShape.size(), one);
  SmallVector<OpFoldResult> offsets(keyShape.size(), zero);
  sizes[1] = keyValueTileLength;
  sizes[2] = headDimension;
  if (!ivs.empty()) {
    offsets[1] = ivs[0];
  }
  SmallVector<int64_t> tensorShape{keyShape[1], keyShape[2]};
  if (swapLastTwoDims) {
    std::swap(sizes[1], sizes[2]);
    std::swap(offsets[1], offsets[2]);
    std::swap(tensorShape[0], tensorShape[1]);
  }
  auto tensorType = RankedTensorType::get(tensorShape, elementType);
  Value keySlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, key, offsets, sizes, strides);
  return keySlice;
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

static Value truncateToF16(Value input, Value output,
                           SmallVectorImpl<Operation *> &ops,
                           OpBuilder &builder, Location loc) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{input}, output, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::TruncFOp>(loc, b.getF16Type(), args[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static std::tuple<Value, Value, Value>
createAttentionBody(Value keySlice, Value valueSlice, Value querySlice,
                    Value outputSlice, Value maxSlice, Value sumSlice,
                    OpFoldResult sequenceTileLength,
                    OpFoldResult keyValueTileLength, OpFoldResult headDimension,
                    Type elementType, SmallVectorImpl<Operation *> &ops,
                    bool transposeV, Location loc, OpBuilder &builder) {

  Type f32Type = builder.getF32Type();
  // Compute matmul(q, transpose(k))
  Value zero =
      builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(f32Type));
  SmallVector<OpFoldResult> resultShape{sequenceTileLength, keyValueTileLength};
  Value emptySquare =
      builder.create<tensor::EmptyOp>(loc, resultShape, f32Type);
  Value qkTranspose = computeQKTranspose(querySlice, keySlice, emptySquare,
                                         zero, loc, builder, ops);

  // Compute current statistics
  Value newMax = computeRowwiseReduction<arith::MaximumFOp>(
      qkTranspose, maxSlice, loc, builder, ops);
  Value partialSoftmax =
      computePartialSoftmax(qkTranspose, newMax, loc, builder, ops);
  Value scaleFactor = computeScaleFactor(maxSlice, newMax, loc, builder, ops);
  Value scaledOldSum = updateAndScale(scaleFactor, sumSlice, loc, builder, ops);
  Value newSum = computeRowwiseReduction<arith::AddFOp>(
      partialSoftmax, scaledOldSum, loc, builder, ops);
  if (elementType.isF16()) {
    Value empty =
        builder.create<tensor::EmptyOp>(loc, resultShape, builder.getF16Type());
    partialSoftmax = truncateToF16(partialSoftmax, empty, ops, builder, loc);
  }

  // Update accumulator
  Value scaledAcc =
      scaleAccumulator(outputSlice, scaleFactor, loc, builder, ops);

  // Compute matmul(softmax, v)
  Operation *matmulOp;
  if (transposeV) {
    matmulOp = builder.create<linalg::MatmulTransposeBOp>(
        loc, scaledAcc.getType(), ValueRange{partialSoftmax, valueSlice},
        scaledAcc);
  } else {
    matmulOp = builder.create<linalg::MatmulOp>(
        loc, scaledAcc.getType(), ValueRange{partialSoftmax, valueSlice},
        scaledAcc);
  }
  ops.push_back(matmulOp);
  Value result = matmulOp->getResult(0);
  return std::make_tuple(result, newMax, newSum);
}

static Value extractOrInsertOutputSlice(Value src, Value dst,
                                        ArrayRef<int64_t> queryShape,
                                        OpFoldResult sequenceTileLength,
                                        OpFoldResult headDimension,
                                        Location loc, OpBuilder &builder) {
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> strides(3, one);
  SmallVector<OpFoldResult> sizes = {one, sequenceTileLength, headDimension};
  SmallVector<OpFoldResult> offsets(3, zero);
  Value slice;
  if (!dst) {
    SmallVector<int64_t> accShape{queryShape[1], queryShape[2]};
    Type elementType = src.getType().cast<ShapedType>().getElementType();
    auto tensorType = RankedTensorType::get(accShape, elementType);
    slice = builder.create<tensor::ExtractSliceOp>(loc, tensorType, src,
                                                   offsets, sizes, strides);
  } else {
    slice = builder.create<tensor::InsertSliceOp>(loc, src, dst, offsets, sizes,
                                                  strides);
  }
  return slice;
}

static Value extractOutputSlice(Value src, ArrayRef<int64_t> queryShape,
                                OpFoldResult sequenceTileLength,
                                OpFoldResult headDimension, Location loc,
                                OpBuilder &builder) {
  return extractOrInsertOutputSlice(src, {}, queryShape, sequenceTileLength,
                                    headDimension, loc, builder);
}

static Value insertOutputSlice(Value src, Value dst,
                               OpFoldResult sequenceTileLength,
                               OpFoldResult headDimension, Location loc,
                               OpBuilder &builder) {
  return extractOrInsertOutputSlice(src, dst, {}, sequenceTileLength,
                                    headDimension, loc, builder);
}

} // namespace

/// Tile iree_linalg_ext.attention.
/// TODO: Adopt getTiledImplementation with this.
IREE::LinalgExt::AttentionOp tileAttention(IREE::LinalgExt::AttentionOp attnOp,
                                           SmallVectorImpl<Operation *> &ops,
                                           RewriterBase &rewriter,
                                           std::optional<uint64_t> tileSize) {
  Location loc = attnOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(attnOp);

  Value query = attnOp.getQuery();
  ShapedType queryType = attnOp.getQueryType();
  Type elementType = queryType.getElementType();
  ArrayRef<int64_t> queryShape = queryType.getShape();
  SmallVector<OpFoldResult> queryDimValues =
      tensor::getMixedSizes(rewriter, loc, query);
  OpFoldResult headDimension = queryDimValues[2];
  OpFoldResult sequenceTileLength = queryDimValues[1];
  OpFoldResult keyValueTileLength = sequenceTileLength;
  SmallVector<int64_t> keyShape{queryShape};
  if (tileSize) {
    keyValueTileLength = rewriter.getIndexAttr(tileSize.value());
    for (auto [idx, val] : llvm::enumerate(attnOp.getKeyType().getShape())) {
      keyShape[idx] = idx == 1 ? tileSize.value() : val;
    }
  }

  Value key = attnOp.getKey();
  Value value = attnOp.getValue();
  SmallVector<OpFoldResult> keyDimValues =
      tensor::getMixedSizes(rewriter, loc, key);
  OpFoldResult sequenceLength = keyDimValues[1];

  // Create output accumulator
  Value output = attnOp.getOutput();
  Type f32Type = rewriter.getF32Type();
  SmallVector<OpFoldResult> accShape{queryDimValues[1], queryDimValues[2]};
  Value accumulatorF32 =
      rewriter.create<tensor::EmptyOp>(loc, accShape, f32Type);

  // Create accumulator, max and sum statistics
  Value outputSlice = extractOutputSlice(output, queryShape, sequenceTileLength,
                                         headDimension, loc, rewriter);
  Value zeroF32 =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(f32Type));
  auto accumulatorFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32}, accumulatorF32);
  accumulatorF32 = accumulatorFill.result();
  ops.push_back(accumulatorFill);

  Value largeNegativeF32 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(f32Type, -1.0e+30));
  SmallVector<OpFoldResult> dims{sequenceTileLength};
  Value max = rewriter.create<tensor::EmptyOp>(loc, dims, f32Type);
  auto maxFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{largeNegativeF32}, max);
  Value negativeMax = maxFill.result();
  ops.push_back(maxFill);
  Value sum = rewriter.create<tensor::EmptyOp>(loc, dims, f32Type);
  auto sumFill = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32}, sum);
  Value zeroSum = sumFill.result();
  ops.push_back(sumFill);

  // Construct sequential loop
  SmallVector<Value> ivs;
  Value zeroValue = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  scf::LoopNest loopNest = createLoopNest(
      ivs, zeroValue,
      getValueOrCreateConstantIndexOp(rewriter, loc, keyValueTileLength),
      getValueOrCreateConstantIndexOp(rewriter, loc, sequenceLength),
      ValueRange({accumulatorF32, negativeMax, zeroSum}), loc, rewriter);
  ops.push_back(loopNest.loops.back());

  Value iterArgResult = loopNest.loops.back().getRegionIterArg(0);
  Value iterArgMax = loopNest.loops.back().getRegionIterArg(1);
  Value iterArgSum = loopNest.loops.back().getRegionIterArg(2);

  OpBuilder::InsertionGuard guardSecondLoop(rewriter);
  rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());

  // Extract slices
  Value keySlice = extractSlice(key, keyShape, ivs, keyValueTileLength,
                                headDimension, elementType, loc, rewriter);
  Value valueSlice =
      extractSlice(value, keyShape, ivs, keyValueTileLength, headDimension,
                   elementType, loc, rewriter, attnOp.getTransposeV());
  Value querySlice = extractSlice(query, queryShape, {}, sequenceTileLength,
                                  headDimension, elementType, loc, rewriter);

  auto tiledAttentionOp = rewriter.create<IREE::LinalgExt::AttentionOp>(
      attnOp.getLoc(),
      SmallVector<Type>{accumulatorF32.getType(), sum.getType(), max.getType()},
      SmallVector<Value>{querySlice, keySlice, valueSlice},
      SmallVector<Value>{iterArgResult, iterArgMax, iterArgSum});

  if (attnOp.getTransposeV())
    tiledAttentionOp.setTransposeVAttr(attnOp.getTransposeVAttr());

  Value tiledResult = tiledAttentionOp.getResult(0);
  Value newMax = tiledAttentionOp.getResult(1);
  Value newSum = tiledAttentionOp.getResult(2);

  if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
          loopNest.loops.back().getBody()->getTerminator())) {
    OpBuilder::InsertionGuard yieldGuard(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(
        yieldOp, ValueRange{tiledResult, newMax, newSum});
  }

  OpBuilder::InsertionGuard yieldGuard(rewriter);
  rewriter.setInsertionPointAfter(loopNest.loops.back());

  loopNest.results[0] = applyFinalScaling(
      loopNest.results[0], loopNest.results[2], loc, rewriter, ops);

  if (elementType.isF16()) {
    loopNest.results[0] =
        truncateToF16(loopNest.results[0], outputSlice, ops, rewriter, loc);
  }
  loopNest.results[0] =
      insertOutputSlice(loopNest.results[0], output, sequenceTileLength,
                        headDimension, loc, rewriter);

  rewriter.replaceOp(attnOp, loopNest.results[0]);
  ops.push_back(tiledAttentionOp);

  return tiledAttentionOp;
}

/// Decompose tiled iree_linalg_ext.attention op.
/// TODO: Adopt decomposeOperation with this.
void decomposeTiledAttention(IREE::LinalgExt::AttentionOp tiledAttnOp,
                             SmallVectorImpl<Operation *> &ops,
                             RewriterBase &rewriter,
                             std::optional<uint64_t> tileSize,
                             bool useSCFMaxIter) {
  Location loc = tiledAttnOp.getLoc();
  Value keySlice = tiledAttnOp.getKey();
  Value valueSlice = tiledAttnOp.getValue();
  Value querySlice = tiledAttnOp.getQuery();
  Value tiledResult = tiledAttnOp.getOutput();
  Value max = *tiledAttnOp.getMax();
  Value sum = *tiledAttnOp.getSum();

  if (!useSCFMaxIter) {
    max = ops[1]->getResult(0);
  }

  assert(max && "expected max statistic operand to be present");
  assert(sum && "expected sum statistic operand to be present");

  OpBuilder::InsertionGuard withinScfLoop(rewriter);
  rewriter.setInsertionPointAfter(tiledAttnOp);
  SmallVector<OpFoldResult> queryDimValues =
      tensor::getMixedSizes(rewriter, loc, querySlice);
  OpFoldResult headDimension = queryDimValues[1];
  OpFoldResult sequenceTileLength = queryDimValues[0];
  OpFoldResult keyValueTileLength =
      tileSize ? rewriter.getIndexAttr(tileSize.value()) : sequenceTileLength;

  Type elementType = tiledAttnOp.getQueryType().getElementType();
  auto [result, newMax, newSum] = createAttentionBody(
      keySlice, valueSlice, querySlice, tiledResult, max, sum,
      sequenceTileLength, keyValueTileLength, headDimension, elementType, ops,
      tiledAttnOp.getTransposeV(), loc, rewriter);

  rewriter.replaceOp(tiledAttnOp, ValueRange{result, newMax, newSum});
}

/// Utility function which tiles and then decomposes attention op via
/// FlashAttention algorithm.
void tileAndDecomposeAttention(IREE::LinalgExt::AttentionOp attnOp,
                               SmallVectorImpl<Operation *> &ops,
                               RewriterBase &rewriter, bool onlyTile,
                               std::optional<uint64_t> tileSize,
                               bool useSCFMaxIter) {
  IREE::LinalgExt::AttentionOp tiledAttentionOp =
      tileAttention(attnOp, ops, rewriter, tileSize);
  if (onlyTile) {
    return;
  }
  decomposeTiledAttention(tiledAttentionOp, ops, rewriter, tileSize,
                          useSCFMaxIter);
}

namespace {

/// This is an implementation of flash attention which
/// is a tiled and fused implementation of the attention operator.
/// The attention operator computes:
/// matmul(softmax(matmul(Q, transpose(K))), V)
/// where: Q is the query matrix [B x N x d]
///        K is the key matrix   [B x S x d]
///        V is the value matrix [B x S x d]
///
/// The core algorithm is as follows:
/// For each element in B,
/// 1. Load a tile from the Q matrix of size T x d -> q
/// 2. Initialize statistics: running_sum, running_max
/// 3. for i = 0 to S with step T
///    a. Load a tile from the K matrix of size T x d -> k
///    b. Load a tile from the V matrix of size T x d -> v
///    c. Compute matmul_transpose_b(q, k) -> qkT
///    d. Compute max(max(qkT) along rows, old_max) -> new_max
///    e. Compute curent estimate of softmax: exp(qKT - current_max) -> s
///    f. Compute product of fixup and old_sum -> fsum
///    g. Compute sum(sum(qkT) along rows, fsum) -> new_sum
///    h. Compute 1.0 / new_sum -> inv_new_sum
///    i. Compute softmax = softmax * inv_new_sum
///    j. Truncate softmax to fp16
///    k. Compute fsum  * inv_new_sum * accumulator -> new_accumulator
///    j. Compute matmul(s, v) and add new_accumulator
///
///
LogicalResult reifyAttentionTransform(mlir::FunctionOpInterface funcOp,
                                      bool onlyTile,
                                      std::optional<uint64_t> tileSize,
                                      bool useSCFIterMax) {
  IRRewriter rewriter(funcOp.getContext());
  funcOp.walk([&](IREE::LinalgExt::AttentionOp attnOp) {
    SmallVector<Operation *> ops;
    tileAndDecomposeAttention(attnOp, ops, rewriter, onlyTile, tileSize,
                              useSCFIterMax);
    return WalkResult::advance();
  });
  return success();
}

} // namespace

namespace {
struct TileAndDecomposeAttentionPass
    : public TileAndDecomposeAttentionBase<TileAndDecomposeAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }
  TileAndDecomposeAttentionPass() = default;
  TileAndDecomposeAttentionPass(bool useSCFIterMax) {
    this->useSCFIterMax = useSCFIterMax;
  }
  TileAndDecomposeAttentionPass(bool useSCFIterMax, bool onlyTile,
                                uint64_t tileSize) {
    this->useSCFIterMax = useSCFIterMax;
    this->onlyTile = onlyTile;
    this->tileSize = tileSize;
  }
  TileAndDecomposeAttentionPass(const TileAndDecomposeAttentionPass &pass) {
    useSCFIterMax = pass.useSCFIterMax;
    onlyTile = pass.onlyTile;
    tileSize = pass.tileSize;
  }
  void runOnOperation() override;
};
} // namespace

void TileAndDecomposeAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  std::optional<uint64_t> optionalTileSize{std::nullopt};
  if (tileSize.hasValue()) {
    optionalTileSize = tileSize.getValue();
  }
  if (failed(reifyAttentionTransform(getOperation(), onlyTile, optionalTileSize,
                                     useSCFIterMax))) {
    return signalPassFailure();
  }
  // Run patterns to remove unused iter in scf::ForOp loop, if we intend to not
  // use iterator for "max".
  if (!useSCFIterMax) {
    RewritePatternSet patterns(context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
}

std::unique_ptr<Pass> createTileAndDecomposeAttentionPass(bool useSCFIterMax) {
  return std::make_unique<TileAndDecomposeAttentionPass>(useSCFIterMax);
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
