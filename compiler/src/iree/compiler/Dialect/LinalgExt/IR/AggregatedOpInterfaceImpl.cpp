// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Command line options used purely for development purposes. Not to be relied
/// on in any way.
static llvm::cl::opt<float> clAttentionSoftmaxMax(
    "iree-linalgext-attention-softmax-max",
    llvm::cl::desc("maximum expected value from attention softmax"),
    llvm::cl::init(1.0));

template <typename T>
static Value elementwiseValueInPlace(OpBuilder &builder, Location loc,
                                     AffineMap inputMap, AffineMap scaleMap,
                                     Value value, Value scale) {
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{inputMap, scaleMap});
  inputMap = compressedMaps[0];
  scaleMap = compressedMaps[1];

  SmallVector<utils::IteratorType> iteratorTypes(inputMap.getNumDims(),
                                                 utils::IteratorType::parallel);

  auto genericOp = builder.create<linalg::GenericOp>(
      loc, value.getType(), scale, value,
      SmallVector<AffineMap>{scaleMap, inputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Convert scale to the same datatype as input.
        Value scale = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                           /*isUnsignedCast=*/false);
        Value result = b.create<T>(loc, scale, args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static Value reciprocalValue(OpBuilder &b, Location loc, Value input,
                             Value output) {
  int64_t rank = cast<ShapedType>(input.getType()).getRank();
  SmallVector<AffineMap> maps = {b.getMultiDimIdentityMap(rank),
                                 b.getMultiDimIdentityMap(rank)};

  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  auto genericOp = b.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{input}, output, maps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value in = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                        /*isUnsignedCast=*/false);
        // Convert scale to the same datatype as input.
        Value one =
            b.create<arith::ConstantOp>(loc, b.getFloatAttr(in.getType(), 1.0));
        Value result = b.create<arith::DivFOp>(loc, one, in);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static Value truncateFloat(OpBuilder &builder, Location loc, AffineMap inputMap,
                           AffineMap outputMap, Value value, Value output,
                           bool clampToFPRange) {
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{inputMap, outputMap});
  inputMap = compressedMaps[0];
  outputMap = compressedMaps[1];

  SmallVector<utils::IteratorType> iteratorTypes(inputMap.getNumDims(),
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), value, output,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        auto srcTy = cast<FloatType>(args[0].getType());
        auto dstTy = cast<FloatType>(args[1].getType());

        Value input = args[0];

        if (clampToFPRange) {
          double mxDbl =
              APFloat::getLargest(dstTy.getFloatSemantics(), /*Negative=*/false)
                  .convertToDouble();

          // Clamp input to dstTy(usually `fp8`) MAX value to prevent NaNs.
          // We do not clamp for `-MAX` because this function meant to only be
          // used by attention's exp2 who's value is always > 0.
          Value mx = builder.create<arith::ConstantOp>(
              loc, builder.getFloatAttr(srcTy, mxDbl));
          input = b.create<arith::MinimumFOp>(loc, mx, input);
        }

        // Convert scale to the same datatype as input.
        Value trunc = convertScalarToDtype(b, loc, input, dstTy,
                                           /*isUnsignedCast=*/false);
        b.create<linalg::YieldOp>(loc, trunc);
      });
  return genericOp.getResult(0);
}

template <typename T>
static Value reduce(OpBuilder &builder, Location loc, AffineMap inputMap,
                    AffineMap outputMap, Value input, Value output) {
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{inputMap, outputMap});
  inputMap = compressedMaps[0];
  outputMap = compressedMaps[1];

  // Dims not present in outputMap are reductionDims.
  SmallVector<utils::IteratorType> iteratorTypes(
      inputMap.getNumDims(), utils::IteratorType::reduction);
  for (AffineExpr dim : outputMap.getResults()) {
    int pos = cast<AffineDimExpr>(dim).getPosition();
    iteratorTypes[pos] = utils::IteratorType::parallel;
  }

  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), input, output,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Convert input to the same datatype as acc.
        Value in = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                        /*isUnsignedCast=*/false);
        Value result = b.create<T>(loc, in, args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });

  return genericOp.getResult(0);
}

static Value computeMatmul(OpBuilder &builder, Location loc, AffineMap lhsMap,
                           AffineMap rhsMap, AffineMap accMap, Value lhs,
                           Value rhs, Value acc) {

  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{lhsMap, rhsMap, accMap});
  lhsMap = compressedMaps[0];
  rhsMap = compressedMaps[1];
  accMap = compressedMaps[2];

  // Dims not present in accMap are reduction dims.
  SmallVector<utils::IteratorType> iteratorTypes(
      accMap.getNumDims(), utils::IteratorType::reduction);
  for (AffineExpr dim : accMap.getResults()) {
    int pos = cast<AffineDimExpr>(dim).getPosition();
    iteratorTypes[pos] = utils::IteratorType::parallel;
  }

  auto genericOp = builder.create<linalg::GenericOp>(
      loc, acc.getType(), SmallVector<Value>{lhs, rhs}, acc,
      SmallVector<AffineMap>{lhsMap, rhsMap, accMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Cast inputs to match output datatype.
        Value lhs = convertScalarToDtype(b, loc, args[0], args[2].getType(),
                                         /*isUnsignedCast=*/false);
        Value rhs = convertScalarToDtype(b, loc, args[1], args[2].getType(),
                                         /*isUnsignedCast=*/false);
        Value mul = b.create<arith::MulFOp>(loc, lhs, rhs);
        Value add = b.create<arith::AddFOp>(loc, mul, args[2]);
        b.create<linalg::YieldOp>(loc, add);
      });

  return genericOp.getResult(0);
}

static Value applyPostQKMatmulElementwise(OpBuilder &builder, Location loc,
                                          Region &region, Value value) {
  auto rank = cast<RankedTensorType>(value.getType()).getRank();
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, value.getType(), ValueRange{}, value, indexingMaps, iteratorTypes);
  auto &dstRegion = genericOp.getRegion();
  builder.cloneRegionBefore(region, dstRegion, dstRegion.end());
  {
    OpBuilder::InsertionGuard withinRegion(builder);
    builder.setInsertionPoint(dstRegion.back().getTerminator());
    builder.create<linalg::YieldOp>(
        loc, dstRegion.back().getTerminator()->getOperands());
    dstRegion.back().getTerminator()->erase();
  }
  return genericOp.getResult(0);
}

static Value applyMask(OpBuilder &builder, Location loc, AffineMap qkMap,
                       AffineMap maskMap, Value qk, Value mask) {

  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{qkMap, maskMap});
  qkMap = compressedMaps[0];
  maskMap = compressedMaps[1];

  SmallVector<utils::IteratorType> iteratorTypes(qkMap.getNumDims(),
                                                 utils::IteratorType::parallel);

  Value zero = builder.create<arith::ConstantOp>(
      loc, builder.getFloatAttr(getElementTypeOrSelf(qk.getType()), 0.0));
  Value negInf = builder.create<arith::ConstantOp>(
      loc, builder.getFloatAttr(getElementTypeOrSelf(qk.getType()),
                                -std::numeric_limits<double>::infinity()));
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, qk.getType(), SmallVector<Value>{mask}, qk,
      SmallVector<AffineMap>{maskMap, qkMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value qkVal = args[1];
        Value maskVal = args[0];

        // TODO: Replace bool mask condition once treated as i1 (instead of i8)
        auto maskValType = maskVal.getType();
        if (maskValType.isInteger()) {
          if (maskValType.getIntOrFloatBitWidth() != 1) {
            maskVal =
                b.create<arith::TruncIOp>(loc, builder.getI1Type(), maskVal);
          }
          maskVal = b.create<arith::SelectOp>(loc, maskVal, zero, negInf);
        } else {
          maskVal = convertScalarToDtype(b, loc, maskVal, qkVal.getType(),
                                         /*isUnsignedCast=*/false);
          // Scaling to compensate for base-2 softmax
          Value log2e = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(qkVal.getType(), M_LOG2E));
          maskVal = b.create<arith::MulFOp>(loc, maskVal, log2e);
        }
        // Finally, set the returned value to the qk element plus the mask
        // element (or 0/-infinity if bool mask). We opt for a AddFOp (instead
        // of a SelectFOp to stay consistent with the additive definition of
        // attention masking)
        Value add = b.create<arith::AddFOp>(loc, qkVal, maskVal);
        b.create<linalg::YieldOp>(loc, add);
      });

  return genericOp.getResult(0);
}

// Compute output = exp2(output - input)
static Value computeSubAndExp2(OpBuilder &builder, Location loc,
                               AffineMap inputMap, AffineMap outputMap,
                               Value input, Value output) {
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{inputMap, outputMap});
  inputMap = compressedMaps[0];
  outputMap = compressedMaps[1];

  SmallVector<utils::IteratorType> iteratorTypes(inputMap.getNumDims(),
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), input, output,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Convert input to the same datatype as output.
        Value in = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                        /*isUnsignedCast=*/false);
        Value diff = b.create<arith::SubFOp>(loc, args[1], in);
        Value weight = b.create<math::Exp2Op>(loc, diff);
        b.create<linalg::YieldOp>(loc, weight);
      });
  return genericOp.getResult(0);
}

// Helper method to check if a slice will be contiguous given the offset,
// slice size. This checks that `inputSize` and `offset` are both evenly
// divisible by `tileSize`.
static bool willBeContiguousSlice(OpFoldResult inputSize, OpFoldResult tileSize,
                                  OpFoldResult offset) {
  auto constInputSize = getConstantIntValue(inputSize);
  auto constTileSize = getConstantIntValue(tileSize);
  if (!constTileSize.has_value() || !constInputSize.has_value() ||
      constInputSize.value() % constTileSize.value() != 0) {
    return false;
  }
  auto constOffset = getConstantIntValue(offset);
  if (constOffset.has_value() &&
      constOffset.value() % constTileSize.value() == 0) {
    return true;
  }
  auto affineOp = cast<Value>(offset).getDefiningOp<affine::AffineApplyOp>();
  return affineOp &&
         affineOp.getMap().getResult(0).isMultipleOf(constTileSize.value());
}

//===----------------------------------------------------------------------===//
// Attention Helpers
//===----------------------------------------------------------------------===//

Value computeQKAndElementwise(Location loc, OpBuilder &b, Value query,
                              Value key, Value scale, std::optional<Value> mask,
                              AffineMap qMap, AffineMap kMap, AffineMap sMap,
                              std::optional<AffineMap> maskMap,
                              SmallVector<OpFoldResult> iterationDomain,
                              Type sElementType, Region &elementwiseRegion,
                              DictionaryAttr qkAttrs, bool lowPrecision) {
  MLIRContext *ctx = b.getContext();
  // Since we use exp2 for attention instead of the original exp, we have to
  // multiply the scale by log2(e). We use exp2 instead of exp as most platforms
  // have better support for exp2 (we verified that we gain some speedup on
  // some GPUs).
  Value log2e = b.create<arith::ConstantOp>(
      loc, b.getFloatAttr(scale.getType(), M_LOG2E));
  scale = b.create<arith::MulFOp>(loc, scale, log2e);

  auto qETy = getElementTypeOrSelf(query.getType());

  AffineMap scaleMap = AffineMap::get(/*dimCount=*/qMap.getNumInputs(),
                                      /*symbolCount=*/0, ctx);

  // In the original algorithm, the scaling is done after the softmax:
  //        softmax(Q @ K.T * scale) @ V
  //
  // But, it is mathematically equivalent to do it on Q first and then multiply
  // it by K.T. This just allows us to do the scaling once, instead of each
  // iteration of the loop. This is only valid for f16 or f32 types as f8
  // is extremely limited on its dynamic range therefore this would
  // significantly affect numerics.
  if (!lowPrecision) {
    query = elementwiseValueInPlace<arith::MulFOp>(b, loc, qMap, scaleMap,
                                                   query, scale);
  }

  // ---- QK Matmul ----

  // Get sizes for S.
  SmallVector<OpFoldResult> sSizes;
  for (AffineExpr dimExpr : sMap.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    sSizes.push_back(iterationDomain[dim]);
  }

  // S = Q @ K
  // SMap = QMap @ KMap
  Value emptyS = b.create<tensor::EmptyOp>(loc, sSizes, sElementType);
  Value sZero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(sElementType));
  Value s = b.create<linalg::FillOp>(loc, sZero, emptyS).getResult(0);

  s = computeMatmul(b, loc, qMap, kMap, sMap, query, key, s);
  if (qkAttrs) {
    s.getDefiningOp()->setAttrs(qkAttrs);
  }

  s = applyPostQKMatmulElementwise(b, loc, elementwiseRegion, s);

  if (lowPrecision) {
    // For low bit-depth types we perform post Q @ K scaling. This is to avoid
    // losing numerical precision due to the low dynamic range of fp8 types when
    // pre applying the sclaing.
    AffineMap sMap = b.getMultiDimIdentityMap(sSizes.size());
    AffineMap scaleMap = AffineMap::get(/*dimCount=*/sMap.getNumInputs(),
                                        /*symbolCount=*/0, ctx);
    s = elementwiseValueInPlace<arith::MulFOp>(b, loc, sMap, scaleMap, s,
                                               scale);

    // If we need to truncate to fp8 post softmax we apply a scaling to use the
    // full fp8 range. We can do this with a offset as post `exp2` this equates
    // to multiplying by a static value. We are able to do this as `max` and
    // `sum` are scaled by the same value so the end result is the same.
    auto fpTy = cast<FloatType>(qETy);
    double mx =
        APFloat::getLargest(fpTy.getFloatSemantics(), /*Negative=*/false)
            .convertToDouble();
    Value offset = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(sElementType, clAttentionSoftmaxMax / mx));
    s = elementwiseValueInPlace<arith::AddFOp>(b, loc, sMap, scaleMap, s,
                                               offset);
  }

  // S += mask
  if (mask != nullptr) {
    s = applyMask(b, loc, sMap, *maskMap, s, mask.value());
  }

  return s;
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> AttentionOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();
  Value query = getQuery();
  Value key = getKey();
  Value value = getValue();
  std::optional<Value> mask = getMask();
  DictionaryAttr config = getDecompositionConfigAttr();

  DictionaryAttr qkAttrs, pvAttrs;
  if (config) {
    qkAttrs = config.getAs<DictionaryAttr>(getQKAttrStr());
    pvAttrs = config.getAs<DictionaryAttr>(getPVAttrStr());
  }
  Value output = getOutput();

  FailureOr<AttentionOpDetail> maybeOpInfo = AttentionOpDetail::get(
      getQueryMap(), getKeyMap(), getValueMap(), getOutputMap());
  assert(succeeded(maybeOpInfo) && "Invalid attention indexing maps");
  AttentionOpDetail opInfo = maybeOpInfo.value();

  SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
      getIterationDomain(b), [](Range x) { return x.size; });

  AffineMap qMap = getQueryMap();
  AffineMap kMap = getKeyMap();
  AffineMap sMap = opInfo.getSMap();

  auto qETy = getElementTypeOrSelf(query.getType());
  bool lowPrecision = qETy.getIntOrFloatBitWidth() <= 8;

  // We compute output of first matmul in f32.
  Type f32Type = b.getF32Type();

  // ---- QK Matmul + elementwise math ----
  Value s = computeQKAndElementwise(loc, b, query, key, getScale(), mask, qMap,
                                    kMap, sMap, getMaskMap(), sizes, f32Type,
                                    getRegion(), qkAttrs, lowPrecision);

  // ---- Softmax ----

  AffineMap accMap = getOutputMap();

  llvm::SmallBitVector projectedK2Dims(opInfo.getDomainRank(), false);
  for (auto dim : opInfo.getK2Dims()) {
    projectedK2Dims.set(dim);
  }

  AffineMap maxMap = projectDims(sMap, projectedK2Dims).dropZeroResults();
  AffineMap sumMap = maxMap;

  SmallVector<OpFoldResult> rowRedSize =
      applyPermutationMap<OpFoldResult>(maxMap, sizes);

  Value rowRedEmpty = b.create<tensor::EmptyOp>(loc, rowRedSize, f32Type);

  Value accInit = arith::getIdentityValue(arith::AtomicRMWKind::addf,
                                          getElementTypeOrSelf(output), b, loc,
                                          /*useOnlyFiniteValue=*/true);
  Value maxInit =
      arith::getIdentityValue(arith::AtomicRMWKind::maximumf, f32Type, b, loc,
                              /*useOnlyFiniteValue=*/true);
  Value sumInit =
      arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type, b, loc);

  Value accFill =
      b.create<linalg::FillOp>(loc, ValueRange{accInit}, output).getResult(0);
  Value maxFill =
      b.create<linalg::FillOp>(loc, ValueRange{maxInit}, rowRedEmpty)
          .getResult(0);
  Value sumFill =
      b.create<linalg::FillOp>(loc, ValueRange{sumInit}, rowRedEmpty)
          .getResult(0);

  // max = rowMax(S)
  Value max = reduce<arith::MaximumFOp>(b, loc, sMap, maxMap, s, maxFill);

  // P = exp2(S - max)
  AffineMap pMap = sMap;
  Value p = computeSubAndExp2(b, loc, maxMap, sMap, max, s);

  // sum = rowSum(P)
  Value sum = reduce<arith::AddFOp>(b, loc, pMap, sumMap, p, sumFill);

  // P = P / sum
  p = elementwiseValueInPlace<arith::DivFOp>(b, loc, pMap, sumMap, p, sum);

  // ---- Scale and truncate LHS to match RHS ----
  SmallVector<OpFoldResult> sSizes;
  for (AffineExpr dimExpr : sMap.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    sSizes.push_back(sizes[dim]);
  }

  auto pETy = getElementTypeOrSelf(p.getType());
  auto vETy = getElementTypeOrSelf(value.getType());
  if (pETy != vETy && isa<FloatType>(vETy)) {
    Value convertP = b.create<tensor::EmptyOp>(loc, sSizes, vETy);
    p = truncateFloat(b, loc, pMap, pMap, p, convertP, lowPrecision);
  }

  // result = P @ V + acc
  Value result =
      computeMatmul(b, loc, pMap, getValueMap(), accMap, p, value, accFill);
  if (pvAttrs) {
    result.getDefiningOp()->setAttrs(pvAttrs);
  }

  return SmallVector<Value>{result};
}

//===----------------------------------------------------------------------===//
// OnlineAttentionOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>>
OnlineAttentionOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();
  Value query = getQuery();
  Value key = getKey();
  Value value = getValue();
  std::optional<Value> mask = getMask();
  Value oldAcc = getOutput();
  Value oldMax = getMax();
  Value oldSum = getSum();
  Type elementType = getElementTypeOrSelf(getOutput().getType());
  DictionaryAttr config = getDecompositionConfigAttr();

  DictionaryAttr qkAttrs, pvAttrs;
  if (config) {
    qkAttrs = config.getAs<DictionaryAttr>(getQKAttrStr());
    pvAttrs = config.getAs<DictionaryAttr>(getPVAttrStr());
  }

  FailureOr<AttentionOpDetail> maybeOpInfo = AttentionOpDetail::get(
      getQueryMap(), getKeyMap(), getValueMap(), getOutputMap());
  assert(succeeded(maybeOpInfo) && "Invalid attention indexing maps");
  AttentionOpDetail opInfo = maybeOpInfo.value();

  SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
      getIterationDomain(b), [](Range x) { return x.size; });

  AffineMap qMap = getQueryMap();
  AffineMap kMap = getKeyMap();
  AffineMap sMap = opInfo.getSMap();

  auto qETy = getElementTypeOrSelf(query.getType());
  bool lowPrecision = qETy.getIntOrFloatBitWidth() <= 8;

  // ---- QK Matmul + elementwise math ----
  Value s = computeQKAndElementwise(
      loc, b, query, key, getScale(), mask, qMap, kMap, sMap, getMaskMap(),
      sizes, elementType, getRegion(), qkAttrs, lowPrecision);

  // TODO: This decomposition should be in a seperate op called
  // "online softmax".
  // ---- Online Softmax ----

  // newMax = max(oldMax, rowMax(S))
  AffineMap maxMap = getMaxMap();
  Value newMax = reduce<arith::MaximumFOp>(b, loc, sMap, maxMap, s, oldMax);

  // norm = exp2(oldMax - newMax)
  // normMap = maxMap
  AffineMap normMap = getMaxMap();
  Value norm = computeSubAndExp2(b, loc, maxMap, normMap, newMax, oldMax);

  // normSum = norm * oldSum
  AffineMap sumMap = getSumMap();
  Value normSum = elementwiseValueInPlace<arith::MulFOp>(b, loc, sumMap,
                                                         normMap, oldSum, norm);

  // P = exp2(S - newMax)
  // PMap = SMap
  AffineMap pMap = sMap;
  Value p = computeSubAndExp2(b, loc, maxMap, sMap, newMax, s);

  // newSum = normSum + rowSum(P)
  Value newSum = reduce<arith::AddFOp>(b, loc, pMap, sumMap, p, normSum);

  // newAcc = norm * oldAcc
  AffineMap accMap = getOutputMap();

  // ---- Scale and truncate LHS to match RHS ----
  SmallVector<OpFoldResult> sSizes;
  for (AffineExpr dimExpr : sMap.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    sSizes.push_back(sizes[dim]);
  }

  auto pETy = getElementTypeOrSelf(p.getType());
  auto vETy = getElementTypeOrSelf(value.getType());
  if (pETy != vETy && isa<FloatType>(vETy)) {
    Value convertP = b.create<tensor::EmptyOp>(loc, sSizes, vETy);
    p = truncateFloat(b, loc, pMap, pMap, p, convertP, lowPrecision);
  }

  Value newAcc = elementwiseValueInPlace<arith::MulFOp>(b, loc, accMap, normMap,
                                                        oldAcc, norm);

  // ---- Matmul 2 ----

  // newAcc = P @ V + newAcc
  newAcc = computeMatmul(b, loc, pMap, getValueMap(), accMap, p, value, newAcc);
  if (pvAttrs) {
    newAcc.getDefiningOp()->setDiscardableAttrs(pvAttrs);
  }

  return SmallVector<Value>{newAcc, newMax, newSum};
}

//===----------------------------------------------------------------------===//
// Im2colOp
//===----------------------------------------------------------------------===//

/// Decomposition implementation for iree_linalg_ext.im2col op.
/// The im2col op is decomposed into serial loops of `insert->extract->copy`.
/// The `batch` and `M` dimensions of the operation iteration space are always
/// tiled to 1, and the `K` dimension is left un-tiled if possible. When the
/// full `K` dimension is a contiguous slice of the input tensor, the K dim
/// can be left un-tiled so it can be vectorized. Otherwise, it will be tiled
/// to 1 along with the `batch` and `M` dimensions.
/// TODO(Max191): Fallback to larger tile sizes instead of immediately tiling K
///               dimension to 1 when non-contiguous.
///
/// The simple decomposition (with K tiled to 1) will look like:
/// ```
///   %im2col = iree_linalg_ext.im2col
///       strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
///       m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
///       batch_pos = [0] m_pos = [1, 2] k_pos = [3]
///       ins(%in : tensor<2x34x34x640xf32>)
///       outs(%out : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
/// ```
/// Decomposes to:
/// ```
/// scf.for %B = %c0 to %c2 step %c1
///   scf.for %M = %c0 to %c4 step %c1
///     scf.for %K = %c0 to %c8 step %c1
///       %slice = tensor.extract_slice %in[%B, %h, %w, %k] ... to tensor<1xf32>
///       %copy = linalg.copy ins(%slice) outs(%out)
///       %insert = tensor.insert_slice %copy into %loop_arg
/// ```
/// Where the offsets are computed as:
///   `%h` = `(%m_off + %M) / 32 + ((%k_off + %K) / 640) / 3`
///   `%w` = `(%m_off + %M) mod 32 + ((%k_off + %K) / 640) mod 3`
///   `%k` = `(%k_off + %K) mod 640`
///
FailureOr<SmallVector<Value>> Im2colOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();
  Value inputSlice = getInput();

  // This is part of the im2col verifier, but check here in case this changes.
  assert(getConstantIntValue(getMixedMStrides().back()).value() == 1 &&
         getConstantIntValue(getMixedKStrides().back()).value() == 1 &&
         "Expected inner m_offset and k_offset to be 1");

  // Get the linearized mOffset and kOffset.
  auto linearizeIndex = [&](ArrayRef<OpFoldResult> inds,
                            ArrayRef<OpFoldResult> basis) {
    MLIRContext *ctx = b.getContext();
    SmallVector<AffineExpr> dims(inds.size()), symbols(basis.size());
    bindDimsList<AffineExpr>(ctx, dims);
    bindSymbolsList<AffineExpr>(ctx, symbols);
    AffineExpr linearExpr = mlir::linearize(ctx, dims, symbols);
    SmallVector<OpFoldResult> mapOperands(inds);
    mapOperands.append(basis.begin(), basis.end());
    auto linearMap = AffineMap::get(
        /*dimCount=*/inds.size(), /*symbolCount=*/basis.size(), linearExpr);
    OpFoldResult linearIdx =
        affine::makeComposedFoldedAffineApply(b, loc, linearMap, mapOperands);
    return linearIdx;
  };
  OpFoldResult mOffset = linearizeIndex(getMixedMOffset(), getMixedMStrides());
  OpFoldResult kOffset = linearizeIndex(getMixedKOffset(), getMixedKStrides());

  // Step 1: Tile the im2col op to loops with contiguous slices in the
  // innermost loop.
  //
  // If the `kOffset` will index to a full contiguous slice of the K dim of
  // the input tensor, then don't tile the K loop of the im2col op and
  // maintain a larger contiguous slice.
  SmallVector<Range> iterationDomain(getIterationDomain(b));
  OpFoldResult kTileSize = iterationDomain.back().size;
  auto constKTileSize = getConstantIntValue(kTileSize);
  if (constKTileSize) {
    kTileSize = b.getIndexAttr(constKTileSize.value());
  }
  SmallVector<OpFoldResult> inputSizes =
      tensor::getMixedSizes(b, loc, getInput());
  // Find the innermost non-batch dimension. This dimension is the fastest
  // changing dimension with the K dimension of the im2col iteration domain.
  // This means it is the innermost dimension of the extract_slice on the
  // input tensor, and the slice wants to be contiguous along this dimension.
  SetVector<int64_t> batchPosSet(getBatchPos().begin(), getBatchPos().end());
  OpFoldResult innerSliceSize;
  for (int idx = inputSizes.size() - 1; idx >= 0; --idx) {
    if (batchPosSet.contains(idx)) {
      continue;
    }
    innerSliceSize = inputSizes[idx];
    // If the innermost non-batch dimension is an m_pos dimension, then use the
    // corresponding kernel_size instead of the input tensor size. This is
    // because the slice will be of size `kernel_size` at some offset
    // `i * kernel_size` in this case.
    for (auto [mPos, kernelSize] :
         llvm::zip_equal(getMPos(), getMixedKernelSize())) {
      if (mPos == idx) {
        innerSliceSize = kernelSize;
      }
    }
    break;
  }
  bool vectorizeInnerKLoop =
      getKPos().back() == getInputRank() - 1 &&
      willBeContiguousSlice(innerSliceSize, kTileSize, kOffset);
  if (vectorizeInnerKLoop) {
    iterationDomain.pop_back();
  } else {
    kTileSize = b.getIndexAttr(1);
  }

  // Build loop nest.
  SmallVector<Value> lbs, ubs, steps;
  for (auto range : iterationDomain) {
    lbs.push_back(getValueOrCreateConstantIndexOp(b, loc, range.offset));
    ubs.push_back(getValueOrCreateConstantIndexOp(b, loc, range.size));
    steps.push_back(getValueOrCreateConstantIndexOp(b, loc, range.stride));
  }
  scf::LoopNest loopNest = scf::buildLoopNest(
      b, loc, lbs, ubs, steps, getOutput(),
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
          ValueRange iterArgs) -> scf::ValueVector { return iterArgs; });
  SmallVector<Value> ivs;
  for (scf::ForOp loop : loopNest.loops) {
    ivs.push_back(loop.getInductionVar());
  }

  // Step 2: Compute indices into the input tensor for extract_slice.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loopNest.loops.front());
  SetVector<int64_t> mPosSet(getMPos().begin(), getMPos().end());

  // Compute the basis for the iteration space of the convolution window
  // (i.e., the H and W dims of the convolution output).
  SmallVector<Value> mBasis;
  ArrayRef<int64_t> strides = getStrides();
  ArrayRef<int64_t> dilations = getDilations();
  SmallVector<OpFoldResult> kernelSize = getMixedKernelSize();
  for (auto [idx, pos] : llvm::enumerate(getMPos())) {
    AffineExpr x, k;
    bindDims(getContext(), x, k);
    AffineExpr mapExpr =
        (x - 1 - (k - 1) * dilations[idx]).floorDiv(strides[idx]) + 1;
    OpFoldResult size = affine::makeComposedFoldedAffineApply(
        b, loc, AffineMap::get(2, 0, {mapExpr}, getContext()),
        {inputSizes[pos], kernelSize[idx]});
    mBasis.push_back(getValueOrCreateConstantIndexOp(b, loc, size));
  }

  // Delinearize the k_offset into an offset into the convolution window and
  // any reduced channels. For an NHWC conv2d, the basis for delinearization
  // would be [P, Q, C] for a PxQ kernel with C channels.
  Location nestedLoc =
      loopNest.loops.back().getBody()->getTerminator()->getLoc();
  b.setInsertionPointToStart(loopNest.loops.back().getBody());

  SmallVector<OpFoldResult> kBasis;
  SmallVector<int64_t> mKernelIdx(getInputRank(), -1);
  for (auto [idx, mPos] : enumerate(getMPos())) {
    mKernelIdx[mPos] = idx;
  }
  for (auto [idx, size] : enumerate(inputSizes)) {
    if (batchPosSet.contains(idx))
      continue;
    if (mPosSet.contains(idx)) {
      kBasis.push_back(kernelSize[mKernelIdx[idx]]);
      continue;
    }
    kBasis.push_back(size);
  }
  OpFoldResult kIndex = kOffset;
  for (auto [i, ivIdx, stride] :
       llvm::enumerate(getKOutputDims(), getMixedKStrides())) {
    if (vectorizeInnerKLoop && i == getMixedKOffset().size() - 1) {
      break;
    }
    OpFoldResult ivOffset = mulOfrs(b, nestedLoc, stride, ivs[ivIdx]);
    kIndex = addOfrs(b, nestedLoc, kIndex, ivOffset);
  }
  ValueRange delinKOffset =
      b.create<affine::AffineDelinearizeIndexOp>(
           nestedLoc, getValueOrCreateConstantIndexOp(b, loc, kIndex), kBasis,
           /*hasOuterBound=*/true)
          .getResults();
  // Split the delinearized offsets into the window offsets (for M offsets)
  // and the K offsets for the input tensor.
  SmallVector<Value> windowOffset, inputKOffset;
  int delinKIdx = 0;
  for (int i = 0; i < getInputRank(); ++i) {
    if (batchPosSet.contains(i))
      continue;
    if (mPosSet.contains(i)) {
      windowOffset.push_back(delinKOffset[delinKIdx++]);
      continue;
    }
    inputKOffset.push_back(delinKOffset[delinKIdx++]);
  }

  // Compute offsets for extract. The linearized im2col result M offset is
  // computed as the m_offset * m_strides inner product plus the linearized
  // offset from the tiled m loops. The M offsets into the im2col input are then
  // computed as the delinearized im2col result M offset (in the convolution
  // result iteration space), plus the convolutional window offsets computed
  // above.
  SmallVector<int64_t> mOutDims = getMOutputDims();
  SmallVector<OpFoldResult> mIvs, mOutStrides(getMixedMStrides());
  for (auto [idx, dim] : llvm::enumerate(getMOutputDims())) {
    mIvs.push_back(ivs[dim]);
  }
  OpFoldResult linearMIv = linearizeIndex(mIvs, mOutStrides);
  OpFoldResult linearMOffset = addOfrs(b, nestedLoc, linearMIv, mOffset);
  // Delinearize the m_offset * m_strides into the convolution output space.
  // `mBasis` contains the basis for the iteration space of result of the
  // convolution op (i.e., basis for result H and W dims).
  ValueRange delinMOffset =
      b.create<affine::AffineDelinearizeIndexOp>(
           nestedLoc, getValueOrCreateConstantIndexOp(b, loc, linearMOffset),
           mBasis,
           /*hasOuterBound=*/true)
          .getResults();

  // Compute the final offsets into the input tensor.
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);
  SmallVector<OpFoldResult> sliceOffsets(getInputRank(), zero);
  SmallVector<OpFoldResult> sliceStrides(getInputRank(), one);
  SmallVector<OpFoldResult> sliceSizes(getInputRank(), one);
  // Add the offset into the convolution window, and account for strides and
  // dilations.
  AffineExpr mOff, wOff;
  bindDims(b.getContext(), mOff, wOff);
  for (auto [idx, mPos] : llvm::enumerate(getMPos())) {
    auto map =
        AffineMap::get(2, 0, {mOff * strides[idx] + wOff * dilations[idx]});
    OpFoldResult offset = affine::makeComposedFoldedAffineApply(
        b, nestedLoc, map, {delinMOffset[idx], windowOffset[idx]});
    sliceOffsets[mPos] = offset;
    sliceSizes[mPos] = one;
  }
  // Set the K offset and size for the input tensor.
  const int64_t kPos = getKPos().front();
  sliceOffsets[kPos] = inputKOffset.front();
  sliceSizes[kPos] = kTileSize;

  // Set the batch offsets for the input tensor.
  int ivIdx = 0;
  for (auto bPos : getBatchPos()) {
    sliceOffsets[bPos] = ivs[ivIdx++];
  }

  // Step 3. Decompose the im2col op into:
  // ```
  // %extract = tensor.extract_slice %input
  // %copy = linalg.copy ins(%extract) outs(%out_slice)
  // %insert = tensor.insert_slice %copy into %loop_arg
  // ```
  //
  // Extract a slice from the input tensor.

  ShapedType outputType = getOutputType();
  SmallVector<OpFoldResult> kTileSizes(
      std::min<int64_t>(getOutputRank(), getInputRank()), b.getIndexAttr(1));
  kTileSizes.back() = kTileSize;

  SmallVector<int64_t> kTileSizeStatic;
  SmallVector<Value> kTileSizeDynamic;
  dispatchIndexOpFoldResults(kTileSizes, kTileSizeDynamic, kTileSizeStatic);
  auto extractType = cast<RankedTensorType>(outputType.clone(kTileSizeStatic));
  auto extract =
      b.create<tensor::ExtractSliceOp>(nestedLoc, extractType, inputSlice,
                                       sliceOffsets, sliceSizes, sliceStrides);

  // Insert the slice into the destination tensor.
  sliceOffsets = SmallVector<OpFoldResult>(getOutputRank(), zero);
  sliceSizes = SmallVector<OpFoldResult>(getOutputRank(), one);
  sliceStrides = SmallVector<OpFoldResult>(getOutputRank(), one);
  sliceSizes.back() = kTileSize;
  for (auto [idx, iv] : llvm::enumerate(ivs)) {
    sliceOffsets[idx] = iv;
  }
  // Insert a `linalg.copy` so there is something to vectorize in the
  // decomposition. Without this copy, the extract and insert slice ops
  // do not get vectorized, and the sequence becomes a scalar memref.copy.
  // This memref.copy could be vectorized after bufferization, but it is
  // probably better to vectorize during generic vectorization.
  Value copyDest = b.create<tensor::ExtractSliceOp>(
      nestedLoc, extractType, loopNest.loops.back().getRegionIterArg(0),
      sliceOffsets, sliceSizes, sliceStrides);
  auto copiedSlice =
      b.create<linalg::CopyOp>(nestedLoc, extract.getResult(), copyDest);
  auto insert =
      b.create<tensor::InsertSliceOp>(nestedLoc, copiedSlice.getResult(0),
                                      loopNest.loops.back().getRegionIterArg(0),
                                      sliceOffsets, sliceSizes, sliceStrides);
  auto yieldOp =
      cast<scf::YieldOp>(loopNest.loops.back().getBody()->getTerminator());
  yieldOp->getOpOperands().front().assign(insert.getResult());
  return SmallVector<Value>({loopNest.results[0]});
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> CustomOp::decomposeOperation(OpBuilder &builder) {
  CustomOp customOp = *this;

  IRRewriter rewriter(builder);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(customOp);
  // Inline the body of the operation using the ins/outs as the arguments.
  SmallVector<Value> argReplacements;
  Location loc = getLoc();
  Block *body = customOp.getBody();
  for (auto [operand, argument] :
       llvm::zip_equal(customOp->getOperands(), body->getArguments())) {
    if (operand.getType() != argument.getType()) {
      assert(isa<RankedTensorType>(operand.getType()) &&
             isa<RankedTensorType>(argument.getType()) &&
             "expected operand and arguments to be `RankedTensorType`");
      Value cast =
          builder.create<tensor::CastOp>(loc, argument.getType(), operand);
      argReplacements.push_back(cast);
    } else {
      argReplacements.push_back(operand);
    }
  }

  Block *oldBlock = customOp->getBlock();
  Block *newBlock = rewriter.splitBlock(oldBlock, Block::iterator(customOp));
  rewriter.mergeBlocks(body, oldBlock, argReplacements);

  // Get the operands of the `iree_linalg_ext.yield` which is the terminator of
  // `oldBlock` right now.
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(oldBlock->getTerminator());
  rewriter.setInsertionPointToEnd(oldBlock);
  SmallVector<Value> customOpReplacements;
  for (auto [yieldedVal, result] :
       llvm::zip_equal(yieldOp->getOperands(), customOp->getResults())) {
    if (yieldedVal.getType() != result.getType()) {
      assert(isa<RankedTensorType>(yieldedVal.getType()) &&
             isa<RankedTensorType>(result.getType()) &&
             "expected yielded value and result to be `RankedTensorType`");
      Value cast =
          builder.create<tensor::CastOp>(loc, result.getType(), yieldedVal);
      customOpReplacements.push_back(cast);
    } else {
      customOpReplacements.push_back(yieldedVal);
    }
  }
  // Erase the yield op.
  rewriter.eraseOp(yieldOp);

  // Merge the block back.
  rewriter.mergeBlocks(newBlock, oldBlock);

  return customOpReplacements;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
