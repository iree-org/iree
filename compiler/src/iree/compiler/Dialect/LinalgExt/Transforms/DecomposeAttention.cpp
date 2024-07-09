// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

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

static Value applyMasking(Value qkSlice, Value mask, OpBuilder &builder) {
  ShapedType qkType = cast<ShapedType>(qkSlice.getType());
  Location loc = qkSlice.getLoc();

  // Create a fill op for scale.
  SmallVector<OpFoldResult> qkDims =
      tensor::getMixedSizes(builder, loc, qkSlice);

  // Attention_mask is 1.0 for positions we want to attend and 0.0 for
  // masked positions. this operation will create a tensor which is 0.0 for
  // positions we want to attend and -10000.0 for masked positions
  Value c0 = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(qkType.getElementType()));

  Value cLargeNeg = builder.create<arith::ConstantOp>(
      loc, builder.getFloatAttr(qkType.getElementType(), -1e6));

  Value empty =
      builder.create<tensor::EmptyOp>(loc, qkDims, qkType.getElementType());
  // Create a generic op to multiply the query by the scale.
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto identityMap = AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  SmallVector<AffineMap> indexingMaps(3, identityMap);
  auto applyMaskOp = builder.create<linalg::GenericOp>(
      loc, TypeRange{empty.getType()}, ValueRange{qkSlice, mask},
      ValueRange{empty}, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value boolMask = args[1];
        if (!boolMask.getType().isInteger(1)) {
          boolMask =
              b.create<arith::TruncIOp>(loc, builder.getI1Type(), args[1]);
        }
        Value masking = b.create<arith::SelectOp>(loc, boolMask, c0, cLargeNeg);
        Value result = b.create<arith::AddFOp>(loc, args[0], masking);
        b.create<linalg::YieldOp>(loc, result);
      });
  return applyMaskOp.getResult(0);
}

static std::tuple<Value, Value, Value>
createAttentionBody(Value keySlice, Value valueSlice, Value querySlice,
                    std::optional<Value> maskSlice, Value outputSlice,
                    Value maxSlice, Value sumSlice,
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

  // Apply masking if mask is specified.
  if (maskSlice.has_value()) {
    qkTranspose = applyMasking(qkTranspose, maskSlice.value(), builder);
  }

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

static Value scaleQuery(Value querySlice, Value scale, RewriterBase &rewriter) {
  ShapedType queryType = cast<ShapedType>(querySlice.getType());
  Location loc = querySlice.getLoc();

  // Create a fill op for scale.
  SmallVector<OpFoldResult> queryDims =
      tensor::getMixedSizes(rewriter, loc, querySlice);
  Value empty = rewriter.create<tensor::EmptyOp>(loc, queryDims,
                                                 queryType.getElementType());
  auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{scale}, empty)
                    .getResult(0);

  // Create a generic op to multiply the query by the scale.
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto identityMap =
      AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
  SmallVector<AffineMap> indexingMaps(2, identityMap);
  auto scaleOp = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{fillOp.getType()}, ValueRange{querySlice},
      ValueRange{fillOp}, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::MulFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return scaleOp.getResult(0);
}

} // namespace

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
/// Decompose tiled iree_linalg_ext.attention op.
/// TODO: Adopt decomposeOperation with this.
void decomposeTiledAttention(IREE::LinalgExt::AttentionOp tiledAttnOp,
                             SmallVectorImpl<Operation *> &ops,
                             RewriterBase &rewriter,
                             std::optional<uint64_t> tileSize) {
  Location loc = tiledAttnOp.getLoc();
  Value keySlice = tiledAttnOp.getKey();
  Value valueSlice = tiledAttnOp.getValue();
  Value querySlice = tiledAttnOp.getQuery();
  Value tiledResult = tiledAttnOp.getOutput();
  Value max = *tiledAttnOp.getMax();
  Value sum = *tiledAttnOp.getSum();

  OpBuilder::InsertionGuard withinScfLoop(rewriter);
  rewriter.setInsertionPointAfter(tiledAttnOp);
  SmallVector<OpFoldResult> queryDimValues =
      tensor::getMixedSizes(rewriter, loc, querySlice);
  OpFoldResult headDimension = queryDimValues[1];
  OpFoldResult sequenceTileLength = queryDimValues[0];
  OpFoldResult keyValueTileLength =
      tileSize ? rewriter.getIndexAttr(tileSize.value()) : sequenceTileLength;

  Type elementType = tiledAttnOp.getQueryType().getElementType();

  // Since we use exp2 for attention instead of the original exp, we have to
  // multiply the scale by log2(e). We use exp2 instead of exp as most GPUs
  // have better support for exp2.
  Value scale = tiledAttnOp.getScale();
  Value log2e = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(elementType, M_LOG2E));
  scale = rewriter.create<arith::MulFOp>(loc, scale, log2e);

  // In the original algorithm, the scaling is done after the softmax:
  //        softmax(Q @ K.T * scale) @ V
  //
  // But, it is mathematically equivalent to do it on Q first and then multiply
  // it by K.T. This just allows us to do the scaling once, instead of each
  // iteration of the loop.
  querySlice = scaleQuery(querySlice, scale, rewriter);
  ops.push_back(querySlice.getDefiningOp());
  std::optional<Value> maybeMask = tiledAttnOp.getMask();
  auto [result, newMax, newSum] = createAttentionBody(
      keySlice, valueSlice, querySlice, maybeMask, tiledResult, max, sum,
      sequenceTileLength, keyValueTileLength, headDimension, elementType, ops,
      tiledAttnOp.getTransposeV(), loc, rewriter);

  rewriter.replaceOp(tiledAttnOp, ValueRange{result, newMax, newSum});
}

namespace {
struct DecomposeAttentionPass
    : public DecomposeAttentionBase<DecomposeAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }
  DecomposeAttentionPass() = default;
  DecomposeAttentionPass(bool onlyTile, uint64_t tileSize) {
    this->tileSize = tileSize;
  }
  DecomposeAttentionPass(const DecomposeAttentionPass &pass) {
    tileSize = pass.tileSize;
  }
  void runOnOperation() override;
};
} // namespace

void DecomposeAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  std::optional<uint64_t> optionalTileSize{std::nullopt};
  if (tileSize.hasValue()) {
    optionalTileSize = tileSize.getValue();
  }
  getOperation().walk([&](AttentionOp attnOp) {
    SmallVector<Operation *> ops;
    decomposeTiledAttention(attnOp, ops, rewriter, optionalTileSize);
  });
  getOperation().walk([&](OnlineAttentionOp onlineAtt) {
    rewriter.setInsertionPoint(onlineAtt);
    FailureOr<SmallVector<Value>> results =
        onlineAtt.decomposeOperation(rewriter);
    if (failed(results)) {
      onlineAtt->emitOpError("Could not decompose online attention");
      return signalPassFailure();
    }
    rewriter.replaceOp(onlineAtt, results.value());
  });
}

std::unique_ptr<Pass> createDecomposeAttentionPass() {
  return std::make_unique<DecomposeAttentionPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
