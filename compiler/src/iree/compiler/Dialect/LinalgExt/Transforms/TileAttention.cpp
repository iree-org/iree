// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

namespace {

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
    Type elementType = cast<ShapedType>(src.getType()).getElementType();
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

struct TileAttentionPass : public TileAttentionBase<TileAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }
  TileAttentionPass() = default;
  TileAttentionPass(bool onlyTile, uint64_t tileSize) {
    this->tileSize = tileSize;
  }
  TileAttentionPass(const TileAttentionPass &pass) { tileSize = pass.tileSize; }
  void runOnOperation() override;
};

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

  Value scale = attnOp.getScale();

  auto tiledAttentionOp = rewriter.create<IREE::LinalgExt::AttentionOp>(
      attnOp.getLoc(),
      SmallVector<Type>{accumulatorF32.getType(), sum.getType(), max.getType()},
      SmallVector<Value>{querySlice, keySlice, valueSlice, scale},
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

void convertToOnlineAttention(IREE::LinalgExt::AttentionOp attnOp,
                              SmallVectorImpl<Operation *> &ops,
                              RewriterBase &rewriter) {
  Location loc = attnOp.getLoc();
  MLIRContext *ctx = attnOp.getContext();

  FailureOr<AttentionOpDetail> maybeOpInfo =
      AttentionOpDetail::get(attnOp.getIndexingMapsArray());
  assert(succeeded(maybeOpInfo) && "Invalid attention indexing maps");
  AttentionOpDetail opInfo = maybeOpInfo.value();

  // Create standard maps for max and sum: (batch, m)
  int64_t rank = opInfo.getDomainRank();
  AffineMap maxMap = AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, ctx);
  for (auto dim :
       llvm::concat<const int64_t>(opInfo.getBatchDims(), opInfo.getMDims())) {
    maxMap = maxMap.insertResult(rewriter.getAffineDimExpr(dim),
                                 maxMap.getNumResults());
  }
  AffineMap sumMap = maxMap;

  SmallVector<Range> sizes = attnOp.getIterationDomain(rewriter);

  // Create fill for acc, max and sum.
  // TODO: Acc should not need a fill. The attention op should get a filled
  // input instead of an empty input.
  Value zeroAcc = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(attnOp.getOutputType().getElementType()));
  Value accFill =
      rewriter
          .create<linalg::FillOp>(loc, ValueRange{zeroAcc}, attnOp.getOutput())
          .getResult(0);

  SmallVector<OpFoldResult> rowRedSize =
      llvm::map_to_vector(sizes, [](Range x) { return x.size; });
  rowRedSize = applyPermutationMap<OpFoldResult>(maxMap, rowRedSize);

  Type f32Type = rewriter.getF32Type();
  Value rowRedEmpty =
      rewriter.create<tensor::EmptyOp>(loc, rowRedSize, f32Type);

  Value maxInit =
      arith::getIdentityValue(arith::AtomicRMWKind::maximumf, f32Type, rewriter,
                              loc, /*useOnlyFiniteValue=*/true);
  Value sumInit = arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type,
                                          rewriter, loc);

  Value maxFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{maxInit}, rowRedEmpty)
          .getResult(0);
  Value sumFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{sumInit}, rowRedEmpty)
          .getResult(0);

  // Create online attention op.
  SmallVector<AffineMap> indexingMaps = attnOp.getIndexingMapsArray();
  indexingMaps.push_back(maxMap);
  indexingMaps.push_back(sumMap);
  OnlineAttentionOp onlineAttn = rewriter.create<OnlineAttentionOp>(
      loc, TypeRange{accFill.getType(), maxFill.getType(), sumFill.getType()},
      attnOp.getQuery(), attnOp.getKey(), attnOp.getValue(), attnOp.getScale(),
      accFill, maxFill, sumFill, rewriter.getAffineMapArrayAttr(indexingMaps));
  ops.push_back(onlineAttn);

  Value x = onlineAttn.getResult(0);
  Value sum = onlineAttn.getResult(2);

  // Merge the outputs of online attention:
  //  x = (1 / sum) * x

  // Compress the indexing maps.
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{sumMap, attnOp.getOutputMap()});

  SmallVector<utils::IteratorType> iteratorTypes(compressedMaps[0].getNumDims(),
                                                 utils::IteratorType::parallel);

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, x.getType(), sum, x, compressedMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value one = b.create<arith::ConstantOp>(
            loc, b.getFloatAttr(args[0].getType(), 1.0));
        Value reciprocal = b.create<arith::DivFOp>(loc, one, args[0]);
        // Convert sum to the same datatype as x.
        reciprocal = convertScalarToDtype(b, loc, reciprocal, args[1].getType(),
                                          /*isUnsignedCast=*/false);
        Value result = b.create<arith::MulFOp>(loc, reciprocal, args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);

  rewriter.replaceOp(attnOp, genericOp);
}

void TileAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  std::optional<uint64_t> optionalTileSize{std::nullopt};
  if (tileSize.hasValue()) {
    optionalTileSize = tileSize.getValue();
  }
  getOperation().walk([&](AttentionOp attnOp) {
    SmallVector<Operation *> ops;
    tileAttention(attnOp, ops, rewriter, optionalTileSize);
  });
}

std::unique_ptr<Pass> createTileAttentionPass() {
  return std::make_unique<TileAttentionPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
