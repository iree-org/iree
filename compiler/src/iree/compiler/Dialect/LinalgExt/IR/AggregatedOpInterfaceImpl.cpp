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
#include "mlir/Analysis/SliceAnalysis.h"
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
#include "mlir/Interfaces/FunctionInterfaces.h"

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

  auto genericOp = linalg::GenericOp::create(
      builder, loc, value.getType(), scale, value,
      SmallVector<AffineMap>{scaleMap, inputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Convert scale to the same datatype as input.
        Value scale = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                           /*isUnsignedCast=*/false);
        Value result = T::create(b, loc, scale, args[1]);
        linalg::YieldOp::create(b, loc, result);
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
  auto genericOp = linalg::GenericOp::create(
      b, loc, output.getType(), ValueRange{input}, output, maps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value in = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                        /*isUnsignedCast=*/false);
        // Convert scale to the same datatype as input.
        Value one = arith::ConstantOp::create(
            b, loc, b.getFloatAttr(in.getType(), 1.0));
        Value result = arith::DivFOp::create(b, loc, one, in);
        linalg::YieldOp::create(b, loc, result);
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
  auto genericOp = linalg::GenericOp::create(
      builder, loc, output.getType(), value, output,
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
          Value mx = arith::ConstantOp::create(
              builder, loc, builder.getFloatAttr(srcTy, mxDbl));
          input = arith::MinimumFOp::create(b, loc, mx, input);
        }

        // Convert scale to the same datatype as input.
        Value trunc = convertScalarToDtype(b, loc, input, dstTy,
                                           /*isUnsignedCast=*/false);
        linalg::YieldOp::create(b, loc, trunc);
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

  auto genericOp = linalg::GenericOp::create(
      builder, loc, output.getType(), input, output,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Convert input to the same datatype as acc.
        Value in = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                        /*isUnsignedCast=*/false);
        Value result = T::create(b, loc, in, args[1]);
        linalg::YieldOp::create(b, loc, result);
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

  auto genericOp = linalg::GenericOp::create(
      builder, loc, acc.getType(), SmallVector<Value>{lhs, rhs}, acc,
      SmallVector<AffineMap>{lhsMap, rhsMap, accMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Cast inputs to match output datatype.
        Value lhs = convertScalarToDtype(b, loc, args[0], args[2].getType(),
                                         /*isUnsignedCast=*/false);
        Value rhs = convertScalarToDtype(b, loc, args[1], args[2].getType(),
                                         /*isUnsignedCast=*/false);
        Value mul = arith::MulFOp::create(b, loc, lhs, rhs);
        Value add = arith::AddFOp::create(b, loc, mul, args[2]);
        linalg::YieldOp::create(b, loc, add);
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
  auto genericOp =
      linalg::GenericOp::create(builder, loc, value.getType(), ValueRange{},
                                value, indexingMaps, iteratorTypes);
  auto &dstRegion = genericOp.getRegion();
  builder.cloneRegionBefore(region, dstRegion, dstRegion.end());
  {
    OpBuilder::InsertionGuard withinRegion(builder);
    builder.setInsertionPoint(dstRegion.back().getTerminator());
    linalg::YieldOp::create(builder, loc,
                            dstRegion.back().getTerminator()->getOperands());
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

  Value zero = arith::ConstantOp::create(
      builder, loc,
      builder.getFloatAttr(getElementTypeOrSelf(qk.getType()), 0.0));
  Value negInf = arith::ConstantOp::create(
      builder, loc,
      builder.getFloatAttr(getElementTypeOrSelf(qk.getType()),
                           -std::numeric_limits<double>::infinity()));
  auto genericOp = linalg::GenericOp::create(
      builder, loc, qk.getType(), SmallVector<Value>{mask}, qk,
      SmallVector<AffineMap>{maskMap, qkMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value qkVal = args[1];
        Value maskVal = args[0];

        // TODO: Replace bool mask condition once treated as i1 (instead of i8)
        auto maskValType = maskVal.getType();
        if (maskValType.isInteger()) {
          if (maskValType.getIntOrFloatBitWidth() != 1) {
            maskVal =
                arith::TruncIOp::create(b, loc, builder.getI1Type(), maskVal);
          }
          maskVal = arith::SelectOp::create(b, loc, maskVal, zero, negInf);
        } else {
          maskVal = convertScalarToDtype(b, loc, maskVal, qkVal.getType(),
                                         /*isUnsignedCast=*/false);
          // Scaling to compensate for base-2 softmax
          Value log2e = arith::ConstantOp::create(
              b, loc, b.getFloatAttr(qkVal.getType(), M_LOG2E));
          maskVal = arith::MulFOp::create(b, loc, maskVal, log2e);
        }
        // Finally, set the returned value to the qk element plus the mask
        // element (or 0/-infinity if bool mask). We opt for a AddFOp (instead
        // of a SelectFOp to stay consistent with the additive definition of
        // attention masking)
        Value add = arith::AddFOp::create(b, loc, qkVal, maskVal);
        linalg::YieldOp::create(b, loc, add);
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
  auto genericOp = linalg::GenericOp::create(
      builder, loc, output.getType(), input, output,
      SmallVector<AffineMap>{inputMap, outputMap}, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Convert input to the same datatype as output.
        Value in = convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                        /*isUnsignedCast=*/false);
        Value diff = arith::SubFOp::create(b, loc, args[1], in);
        Value weight = math::Exp2Op::create(b, loc, diff);
        linalg::YieldOp::create(b, loc, weight);
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
  Value log2e = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(scale.getType(), M_LOG2E));
  scale = arith::MulFOp::create(b, loc, scale, log2e);

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
  Value emptyS = tensor::EmptyOp::create(b, loc, sSizes, sElementType);
  Value sZero = arith::ConstantOp::create(b, loc, b.getZeroAttr(sElementType));
  Value s = linalg::FillOp::create(b, loc, sZero, emptyS).getResult(0);

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
    Value offset = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(sElementType, clAttentionSoftmaxMax / mx));
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

  Value rowRedEmpty = tensor::EmptyOp::create(b, loc, rowRedSize, f32Type);

  Value accInit = arith::getIdentityValue(arith::AtomicRMWKind::addf,
                                          getElementTypeOrSelf(output), b, loc,
                                          /*useOnlyFiniteValue=*/true);
  Value maxInit =
      arith::getIdentityValue(arith::AtomicRMWKind::maximumf, f32Type, b, loc,
                              /*useOnlyFiniteValue=*/true);
  Value sumInit =
      arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type, b, loc);

  Value accFill =
      linalg::FillOp::create(b, loc, ValueRange{accInit}, output).getResult(0);
  Value maxFill =
      linalg::FillOp::create(b, loc, ValueRange{maxInit}, rowRedEmpty)
          .getResult(0);
  Value sumFill =
      linalg::FillOp::create(b, loc, ValueRange{sumInit}, rowRedEmpty)
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
    Value convertP = tensor::EmptyOp::create(b, loc, sSizes, vETy);
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
    Value convertP = tensor::EmptyOp::create(b, loc, sSizes, vETy);
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

static std::optional<int64_t>
chooseDimToVectorize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                     SmallVector<Range> iterationDomain,
                     SmallVector<OpFoldResult> inputSizes,
                     OpFoldResult kOffset) {
  int64_t innerInputDim = im2colOp.getInputRank() - 1;
  SmallVector<SmallVector<int64_t>> vectorizationMap =
      im2colOp.getInputToOutputDimVectorizationMap();
  SmallVector<int64_t> vectorizableOutputDims = vectorizationMap[innerInputDim];
  if (vectorizableOutputDims.empty()) {
    return std::nullopt;
  }
  SetVector<int64_t> kDimSet(llvm::from_range, im2colOp.getKOutputDims());
  // There may be multiple output dims that we can vectorize, so prioritize the
  // innermost dims first.
  llvm::sort(vectorizableOutputDims);
  // Check each dim in order from innermost to outermost, and return the first
  // one that is vectorizable.
  while (!vectorizableOutputDims.empty()) {
    int64_t outputDimToVectorize = vectorizableOutputDims.pop_back_val();
    // If a K dim is being vectorized, then it is contiguous along either the
    // input channel dimension, or the filter kernel window. If it is contiguous
    // along the kernel window, then the actual inner slice size is equal to the
    // size of the corresponding kernel window dimension. Otherwise, the inner
    // slice size is just the size of the input tensor's inner dimension.
    OpFoldResult innerSliceSize = inputSizes[innerInputDim];
    if (kDimSet.contains(outputDimToVectorize)) {
      for (auto [kernelSize, mPos] :
           llvm::zip_equal(im2colOp.getMixedKernelSize(), im2colOp.getMPos())) {
        if (mPos == innerInputDim) {
          innerSliceSize = kernelSize;
        }
      }
    }

    // If the input slice is contiguous along the innermost dimension, then it
    // is vectorizable. If it is not, then move on to the next innermost dim.
    SetVector<int64_t> mDimSet(llvm::from_range, im2colOp.getMOutputDims());
    OpFoldResult offset = b.getIndexAttr(0);
    if (kDimSet.contains(outputDimToVectorize)) {
      offset = kOffset;
    } else if (mDimSet.contains(outputDimToVectorize)) {
      // TODO(Max191): Support vectorization along the M dimension.
      continue;
    }
    OpFoldResult outputDimSize = iterationDomain[outputDimToVectorize].size;
    if (!willBeContiguousSlice(innerSliceSize, outputDimSize, offset)) {
      continue;
    }
    return outputDimToVectorize;
  }
  return std::nullopt;
}

/// Decomposition implementation for iree_linalg_ext.im2col op.
/// The im2col op is decomposed into serial loops of `insert->extract->copy`.
/// The decomposition supports leaving either the `batch` or `K` dimension
/// untiled when the corresponding slice in the input tensor is contiguous.
/// If the entire `K` dimension maps to a contiguous slice, the loop over `K`
/// is left untiled to enable more efficient data transfer. Likewise, if the
/// `batch` dimension is contiguous, it is left untiled instead. All other
/// dimensions, including any non-contiguous `batch` or `K`, are tiled to 1.
/// TODO(Max191): Fallback to larger tile sizes instead of immediately tiling K
///               dimension to 1 when non-contiguous.
///
/// The simple decomposition (with K tiled to 1) will look like:
/// ```
///   %im2col = iree_linalg_ext.im2col
///       strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
///       m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
///       batch_pos = [0] m_pos = [1, 2] k_pos = [3]
///       input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
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
  // If the innermost dim of the input tensor contains a full contiguous slice,
  // then don't tile the corresponding loop of the im2col op and maintain a
  // larger contiguous slice. Note that if the im2col input tensor has the batch
  // dim at last, im2col output tensor has an implicit transpose to move the
  // batch dim in front, and tiling should be along the batch dim.
  SmallVector<Range> iterationDomain(getIterationDomain(b));
  SmallVector<OpFoldResult> inputSizes =
      tensor::getMixedSizes(b, loc, getInput());
  std::optional<unsigned> maybeOutputDimToVectorize =
      chooseDimToVectorize(b, loc, *this, iterationDomain, inputSizes, kOffset);

  OpFoldResult innerInputTileSize;
  if (maybeOutputDimToVectorize.has_value()) {
    unsigned outputDimToVectorize = maybeOutputDimToVectorize.value();
    innerInputTileSize = iterationDomain[outputDimToVectorize].size;
    iterationDomain.erase(iterationDomain.begin() + outputDimToVectorize);
  } else {
    innerInputTileSize = b.getIndexAttr(1);
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
  // The index computation below uses the induction variables as the offsets
  // into the output tensor, so we need an offset for each dim of the output.
  // For the dimension that is vectorized, the offset is zero, because we
  // take a full slice along that dimension.
  if (maybeOutputDimToVectorize.has_value()) {
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    ivs.insert(ivs.begin() + maybeOutputDimToVectorize.value(), zero);
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
  SetVector<int64_t> batchPosSet(getBatchPos().begin(), getBatchPos().end());
  for (auto [idx, size] : enumerate(inputSizes)) {
    if (batchPosSet.contains(idx))
      continue;
    if (mPosSet.contains(idx)) {
      kBasis.push_back(kernelSize[mKernelIdx[idx]]);
      continue;
    }
    kBasis.push_back(size);
  }

  // Transpose the order of (P, Q, C) according to `inputKPerm` encoded in
  // im2col metadata.
  ArrayRef<int64_t> inputKPerm = getInputKPerm();
  applyPermutationToVector(kBasis, inputKPerm);

  OpFoldResult kIndex = kOffset;
  for (auto [i, ivIdx, stride] :
       llvm::enumerate(getKOutputDims(), getMixedKStrides())) {
    if (isConstantIntValue(ivs[ivIdx], 0)) {
      continue;
    }
    OpFoldResult ivOffset = mulOfrs(b, nestedLoc, stride, ivs[ivIdx]);
    kIndex = addOfrs(b, nestedLoc, kIndex, ivOffset);
  }
  ValueRange delinKOffset =
      affine::AffineDelinearizeIndexOp::create(
          b, nestedLoc, getValueOrCreateConstantIndexOp(b, loc, kIndex), kBasis,
          /*hasOuterBound=*/true)
          .getResults();
  // Split the delinearized offsets into the window offsets (for M offsets)
  // and the K offsets for the input tensor based on the layout.
  SmallVector<Value> windowOffset, inputKOffset;
  int delinKIdx = 0;
  SmallVector<int64_t> invInputKPerm = invertPermutationVector(inputKPerm);
  for (int i = 0; i < getInputRank(); ++i) {
    if (batchPosSet.contains(i))
      continue;
    if (mPosSet.contains(i)) {
      windowOffset.push_back(delinKOffset[invInputKPerm[delinKIdx++]]);
      continue;
    }
    inputKOffset.push_back(delinKOffset[invInputKPerm[delinKIdx++]]);
  }

  // Compute offsets for extract. The linearized im2col result M offset is
  // computed as the m_offset * m_strides inner product plus the linearized
  // offset from the tiled m loops. The M offsets into the im2col input are then
  // computed as the delinearized im2col result M offset (in the convolution
  // result iteration space), plus the convolutional window offsets computed
  // above.
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
      affine::AffineDelinearizeIndexOp::create(
          b, nestedLoc, getValueOrCreateConstantIndexOp(b, loc, linearMOffset),
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

  sliceSizes.back() = innerInputTileSize;

  // Set the batch and K offsets for the input tensor.
  const int64_t kPos = getKPos().front();
  sliceOffsets[kPos] = inputKOffset.front();
  SmallVector<int64_t> inverseOutputPerm =
      invertPermutationVector(getOutputPerm());
  for (auto [ivIdx, bPos] : llvm::enumerate(getBatchPos())) {
    sliceOffsets[bPos] = ivs[inverseOutputPerm[ivIdx]];
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
  int64_t inputRank = getInputRank();
  int64_t outputRank = getOutputRank();

  // For now, only extract a 1D slice when the vectorized dim is not innermost
  // in the output, and the input and output ranks are different. Otherwise,
  // try to preserve the original rank to avoid rank reducing slices.
  int64_t sliceRank = std::min(inputRank, outputRank);
  auto inputToOutputSlicePerm =
      llvm::to_vector(llvm::seq<int64_t>(0, sliceRank));
  if (maybeOutputDimToVectorize.has_value()) {
    int64_t outputDimToVectorize = maybeOutputDimToVectorize.value();
    if (inputRank == outputRank) {
      inputToOutputSlicePerm[outputDimToVectorize] = outputRank - 1;
      inputToOutputSlicePerm[outputRank - 1] = outputDimToVectorize;
    } else if (outputDimToVectorize != outputRank - 1) {
      sliceRank = 1;
      inputToOutputSlicePerm = {0};
    }
  }
  SmallVector<OpFoldResult> inputTileSizes(sliceRank, b.getIndexAttr(1));
  inputTileSizes.back() = innerInputTileSize;
  SmallVector<int64_t> tileSizeStatic;
  std::tie(tileSizeStatic, std::ignore) = decomposeMixedValues(inputTileSizes);
  auto extractType = cast<RankedTensorType>(outputType.clone(tileSizeStatic));
  auto extract =
      tensor::ExtractSliceOp::create(b, nestedLoc, extractType, inputSlice,
                                     sliceOffsets, sliceSizes, sliceStrides);
  // Insert the slice into the destination tensor.
  sliceOffsets = SmallVector<OpFoldResult>(outputRank, zero);
  for (auto [idx, iv] : llvm::enumerate(ivs)) {
    sliceOffsets[idx] = iv;
  }
  sliceSizes = SmallVector<OpFoldResult>(outputRank, one);
  if (maybeOutputDimToVectorize.has_value()) {
    sliceSizes[maybeOutputDimToVectorize.value()] = innerInputTileSize;
  }
  sliceStrides = SmallVector<OpFoldResult>(outputRank, one);

  // Insert a `linalg.copy` so there is something to vectorize in the
  // decomposition. Without this copy, the extract and insert slice ops
  // do not get vectorized, and the sequence becomes a scalar memref.copy.
  // This memref.copy could be vectorized after bufferization, but it is
  // probably better to vectorize during generic vectorization.
  SmallVector<int64_t> outputSliceShape =
      applyPermutation(tileSizeStatic, inputToOutputSlicePerm);
  RankedTensorType outputSliceType = extractType.clone(outputSliceShape);
  Value copyDest = tensor::ExtractSliceOp::create(
      b, nestedLoc, outputSliceType, loopNest.loops.back().getRegionIterArg(0),
      sliceOffsets, sliceSizes, sliceStrides);
  Value copiedSlice =
      isIdentityPermutation(inputToOutputSlicePerm)
          ? linalg::CopyOp::create(b, nestedLoc, extract.getResult(), copyDest)
                .getResult(0)
          : linalg::TransposeOp::create(b, nestedLoc, extract.getResult(),
                                        copyDest, inputToOutputSlicePerm)
                ->getResult(0);
  auto insert = tensor::InsertSliceOp::create(
      b, nestedLoc, copiedSlice, loopNest.loops.back().getRegionIterArg(0),
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
          tensor::CastOp::create(builder, loc, argument.getType(), operand);
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
          tensor::CastOp::create(builder, loc, result.getType(), yieldedVal);
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

//===----------------------------------------------------------------------===//
// ExpReductionOp
//===----------------------------------------------------------------------===//

/// The return value of captureUsedOperationsAndBlockArguments
struct UsedOperationsAndBlockArguments {
  SetVector<int64_t> usedInputIndices;
  SetVector<Operation *> usedOperations;
};

/// For a given `resultNumber` in a linalg::GenericOp, this op scans the
/// GenericOp's body for the block arguments and operations that are involved
/// in its computation.
///
/// Block arguments used are returned as indices over the dpsInputs and
/// dpsInputs, to be used as:
/// ```
/// for (auto idx : usedInputIndices)
///   if (idx < getNumDpsInputs())
///     getDpsInputOperand(idx)
///   else
///     getDpsInitOperand(idx)
/// ```
/// As resultNumber is specified, if a dpsInit is used that is not resultNumber
/// failure is returned.
///
/// Operations are returned as generic operations.
static FailureOr<UsedOperationsAndBlockArguments>
captureUsedOperationsAndBlockArguments(linalg::GenericOp genericOp,
                                       int64_t resultNumber) {
  BackwardSliceOptions options;
  options.inclusive = true;
  options.filter = [&genericOp](Operation *op) -> bool {
    return op->getBlock() == genericOp.getBody();
  };
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
  Value result = yieldOp.getOperand(resultNumber);
  SetVector<Operation *> usedOperations;
  if (failed(getBackwardSlice(result, &usedOperations, options))) {
    return failure();
  }

  SetVector<int64_t> usedInputIndices;
  // Get all block arguments used by the operations. If any of the arguments
  // used is a dpsInit argument other than resultNumber, return failure.
  for (Operation *op : usedOperations) {
    for (Value operand : op->getOperands()) {
      auto blockArg = dyn_cast<BlockArgument>(operand);
      if (!blockArg) {
        continue;
      }
      if (blockArg.getOwner() != genericOp.getBlock()) {
        continue;
      }
      int64_t argNumber = blockArg.getArgNumber();
      if (argNumber < genericOp.getNumDpsInputs()) {
        usedInputIndices.insert(argNumber);
        continue;
      }
      if (argNumber - genericOp.getNumDpsInputs() != resultNumber) {
        return failure();
      }
    }
  }

  return UsedOperationsAndBlockArguments{usedInputIndices, usedOperations};
}

/// Returns a vector of GenericOps with only one output.
/// Each generic op in the vector corresponds to an output in the input
/// generic op. However, these resultant ops will only contain the:
///   1. Block Arguments that are involved in the result's computation and
///   2. The Operations involved in the result's computation
static FailureOr<SmallVector<linalg::GenericOp>>
decomposeMultipleResults(linalg::GenericOp genericOp, RewriterBase &rewriter) {
  if (genericOp.getNumResults() < 2) {
    return SmallVector<linalg::GenericOp>{genericOp};
  }

  IRRewriter::InsertionGuard g(rewriter);
  SmallVector<linalg::GenericOp> results;
  // Create num_results linalg.generics, each producing a single result (and
  // relying on canonicalizations to simplify).
  for (auto resultNumber : llvm::seq<int64_t>(genericOp.getNumResults())) {
    rewriter.setInsertionPoint(genericOp);
    auto yieldOp = cast<linalg::YieldOp>(genericOp.getBlock()->getTerminator());
    Value result = yieldOp.getOperand(resultNumber);
    // Get all operations required to produce this result.
    auto usedOperationsAndBlockArguments =
        captureUsedOperationsAndBlockArguments(genericOp, resultNumber);
    if (failed(usedOperationsAndBlockArguments)) {
      return failure();
    }
    // Create a new linalg.generic operation for this result.
    SmallVector<Value> inputs = llvm::map_to_vector(
        usedOperationsAndBlockArguments->usedInputIndices,
        [&](int64_t x) { return genericOp.getDpsInputOperand(x)->get(); });
    SmallVector<Value> inits = {
        genericOp.getDpsInitOperand(resultNumber)->get()};

    SmallVector<AffineMap> indexingMaps = llvm::map_to_vector(
        usedOperationsAndBlockArguments->usedInputIndices,
        [&](int64_t x) { return genericOp.getIndexingMapsArray()[x]; });
    indexingMaps.push_back(genericOp.getIndexingMapMatchingResult(
        genericOp->getOpResult(resultNumber)));
    llvm::SmallBitVector unusedDims = getUnusedDimsBitVector(indexingMaps);
    indexingMaps = compressUnusedDims(indexingMaps);
    SmallVector<utils::IteratorType> iteratorTypes;
    for (auto i : llvm::seq<int64_t>(genericOp.getNumLoops())) {
      if (!unusedDims.test(i)) {
        iteratorTypes.push_back(genericOp.getIteratorTypesArray()[i]);
      }
    }
    auto newOp = linalg::GenericOp::create(
        rewriter, genericOp.getLoc(), TypeRange(inits), inputs, inits,
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
          Block *oldBody = genericOp.getBody();
          usedOperationsAndBlockArguments->usedInputIndices.insert(
              resultNumber + genericOp.getNumDpsInputs());
          IRMapping regionMapping;
          for (auto [oldBlockArgNum, newBlockArg] : llvm::zip_equal(
                   usedOperationsAndBlockArguments->usedInputIndices,
                   blockArgs)) {
            regionMapping.map(oldBody->getArgument(oldBlockArgNum),
                              newBlockArg);
          }
          for (Operation *usedOperation :
               usedOperationsAndBlockArguments->usedOperations) {
            b.clone(*usedOperation, regionMapping);
          }
          linalg::YieldOp::create(b, loc, regionMapping.lookup(result));
        });
    rewriter.replaceAllUsesWith(genericOp.getResult(resultNumber),
                                newOp.getResult(0));

    results.push_back(newOp);
  }

  return results;
}

FailureOr<SmallVector<Value>> ExpReductionOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();
  IRRewriter rewriter(b);

  // Let the first dpsInputOperand be s
  // Split the op into:
  // curr_max = max(s, old_max)
  // ex = e^{x - curr_max}
  // norm = e^{curr_max - old_max}
  // for each outs in exp_reduction:
  //     norm_outs = outs * norm
  // linalg.generic ins(ex, ...) outs(norm_outs)

  const int reducingOpIndex = getReducingOpIndex();
  OpOperand *sValue = getDpsInputOperand(reducingOpIndex);
  OpOperand *prevMax = getDpsInitOperand(reducingOpIndex);
  AffineMap normValMap = getMatchingIndexingMap(sValue);
  AffineMap prevMaxMap = getMatchingIndexingMap(prevMax);

  // curr_max = max(sValue, prev_max)
  Value currMax = reduce<arith::MaximumFOp>(
      rewriter, loc, normValMap, prevMaxMap, sValue->get(), prevMax->get());
  // ex = e^{sValue - curr_max}
  Value ex = computeSubAndExp2(rewriter, loc, prevMaxMap, normValMap, currMax,
                               sValue->get());
  // norm = e^(prev_max - curr_max)
  Value norm = computeSubAndExp2(rewriter, loc, prevMaxMap, prevMaxMap, currMax,
                                 prevMax->get());

  SmallVector<Value> inputs = getDpsInputs();
  SmallVector<Value> normOuts(getNumDpsInits());
  inputs[reducingOpIndex] = ex;
  normOuts[reducingOpIndex] = currMax;
  for (int64_t index : getExpReducedOperands()) {
    OpOperand *oldOut = getDpsInitOperand(index);
    AffineMap oldOutMap = getMatchingIndexingMap(oldOut);
    Value normOut = elementwiseValueInPlace<arith::MulFOp>(
        rewriter, loc, oldOutMap, prevMaxMap, oldOut->get(), norm);
    normOuts[index] = normOut;
  }

  auto expRedGeneric = linalg::GenericOp::create(
      rewriter, loc, TypeRange(normOuts), inputs, normOuts,
      getIndexingMapsArray(), getLoopIteratorTypes());

  IRMapping mapper;
  getBodyRegion().cloneInto(&expRedGeneric.getBodyRegion(), mapper);
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(expRedGeneric.getBody()->getTerminator());
  auto yieldOp =
      cast<IREE::LinalgExt::YieldOp>(expRedGeneric.getBody()->getTerminator());
  rewriter.replaceOpWithNewOp<linalg::YieldOp>(yieldOp, yieldOp.getOperands());
  FailureOr<SmallVector<linalg::GenericOp>> decomposedResults =
      decomposeMultipleResults(expRedGeneric, rewriter);
  if (failed(decomposedResults)) {
    return failure();
  }

  return llvm::map_to_vector(
      decomposedResults.value(),
      [](linalg::GenericOp op) -> Value { return op->getResult(0); });
}

//===----------------------------------------------------------------------===//
// ArgCompareOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> ArgCompareOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();
  Value input = getInputValue();
  Value outVal = outputValue();
  Value outIdx = outputIndex();
  int64_t reductionDim = getDimension();

  ShapedType inputType = getInputType();
  ShapedType outValType = getOutputValueType();
  ShapedType outIdxType = getOutputIndexType();

  Type idxElemType = outIdxType.getElementType();
  int64_t rank = inputType.getRank();

  SmallVector<AffineExpr> inputExprs, outputExprs;
  for (int64_t i = 0; i < rank; ++i) {
    inputExprs.push_back(b.getAffineDimExpr(i));
    if (i != reductionDim) {
      outputExprs.push_back(b.getAffineDimExpr(i));
    }
  }

  MLIRContext *ctx = b.getContext();
  AffineMap inputMap = AffineMap::get(rank, 0, inputExprs, ctx);
  AffineMap outputMap = AffineMap::get(rank, 0, outputExprs, ctx);
  SmallVector<AffineMap> indexingMaps = {inputMap, outputMap, outputMap};

  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  iteratorTypes[reductionDim] = utils::IteratorType::reduction;

  Block &srcBlock = getRegion().front();
  auto genericOp = linalg::GenericOp::create(
      b, loc, TypeRange{outValType, outIdxType}, ValueRange{input},
      ValueRange{outVal, outIdx}, indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value inputElem = args[0];
        Value accVal = args[1];
        Value accIdx = args[2];

        Value idx =
            linalg::IndexOp::create(nestedBuilder, nestedLoc, reductionDim);
        // Add index_base offset if provided (index_base is of type index).
        if (Value indexBase = getIndexBase()) {
          idx = arith::AddIOp::create(nestedBuilder, nestedLoc, idx, indexBase);
        }
        Value currentIdx = idx;
        if (!isa<IndexType>(idxElemType)) {
          currentIdx = arith::IndexCastOp::create(nestedBuilder, nestedLoc,
                                                  idxElemType, idx);
        }

        // Inline the comparator region. The region takes (candidateValue,
        // currentBestValue) and yields an i1 indicating if the candidate
        // should be preferred.
        IRMapping regionMap;
        regionMap.map(srcBlock.getArgument(0), inputElem);
        regionMap.map(srcBlock.getArgument(1), accVal);
        for (Operation &op : srcBlock.without_terminator()) {
          nestedBuilder.clone(op, regionMap);
        }

        auto yieldOp = cast<IREE::LinalgExt::YieldOp>(srcBlock.getTerminator());
        Value cmp = regionMap.lookup(yieldOp.getOperand(0));

        Value newVal = arith::SelectOp::create(nestedBuilder, nestedLoc, cmp,
                                               inputElem, accVal);
        Value newIdx = arith::SelectOp::create(nestedBuilder, nestedLoc, cmp,
                                               currentIdx, accIdx);
        linalg::YieldOp::create(nestedBuilder, nestedLoc,
                                ValueRange{newVal, newIdx});
      });

  return SmallVector<Value>{genericOp.getResult(0), genericOp.getResult(1)};
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
