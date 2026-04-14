// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/Im2colUtils.h"
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
                       AffineMap maskMap, Value qk, Value mask, bool useExp2) {

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
          if (useExp2) {
            Value log2e = arith::ConstantOp::create(
                b, loc, b.getFloatAttr(qkVal.getType(), M_LOG2E));
            maskVal = arith::MulFOp::create(b, loc, maskVal, log2e);
          }
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

// Compute output = exp2/exp(output - input) depending on useExp2 flag.
static Value computeSubAndExp(OpBuilder &builder, Location loc,
                              AffineMap inputMap, AffineMap outputMap,
                              Value input, Value output, bool useExp2) {
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
        Value weight = useExp2 ? math::Exp2Op::create(b, loc, diff).getResult()
                               : math::ExpOp::create(b, loc, diff).getResult();
        linalg::YieldOp::create(b, loc, weight);
      });
  return genericOp.getResult(0);
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
                              DictionaryAttr qkAttrs, bool lowPrecision,
                              bool useExp2) {
  MLIRContext *ctx = b.getContext();
  // If using exp2 for attention instead of the original exp, we have to
  // multiply the scale by log2(e). We use exp2 instead of exp as most platforms
  // have better support for exp2 (we verified that we gain some speedup on
  // some GPUs).
  if (useExp2) {
    Value log2e = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(scale.getType(), M_LOG2E));
    scale = arith::MulFOp::create(b, loc, scale, log2e);
  }

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
    // pre applying the scaling.
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
    s = applyMask(b, loc, sMap, *maskMap, s, mask.value(), useExp2);
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
  Value s = computeQKAndElementwise(
      loc, b, query, key, getScale(), mask, qMap, kMap, sMap, getMaskMap(),
      sizes, f32Type, getRegion(), qkAttrs, lowPrecision, /*useExp2=*/true);

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
  Value p = computeSubAndExp(b, loc, maxMap, sMap, max, s, /*useExp2=*/true);

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
  bool useExp2 = true;
  if (config) {
    qkAttrs = config.getAs<DictionaryAttr>(getQKAttrStr());
    pvAttrs = config.getAs<DictionaryAttr>(getPVAttrStr());
    if (auto useExp2Attr = config.getAs<BoolAttr>(getUseExp2AttrStr())) {
      useExp2 = useExp2Attr.getValue();
    }
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
      sizes, elementType, getRegion(), qkAttrs, lowPrecision, useExp2);

  // TODO: This decomposition should be in a separate op called
  // "online softmax".
  // ---- Online Softmax ----

  // newMax = max(oldMax, rowMax(S))
  AffineMap maxMap = getMaxMap();
  Value newMax = reduce<arith::MaximumFOp>(b, loc, sMap, maxMap, s, oldMax);

  // norm = exp2(oldMax - newMax) or exp(oldMax - newMax) depending on useExp2
  // normMap = maxMap
  AffineMap normMap = getMaxMap();
  Value norm =
      computeSubAndExp(b, loc, maxMap, normMap, newMax, oldMax, useExp2);

  // normSum = norm * oldSum
  AffineMap sumMap = getSumMap();
  Value normSum = elementwiseValueInPlace<arith::MulFOp>(b, loc, sumMap,
                                                         normMap, oldSum, norm);

  // P = exp2(S - newMax) or exp(S - newMax) depending on useExp2
  // PMap = SMap
  AffineMap pMap = sMap;
  Value p = computeSubAndExp(b, loc, maxMap, sMap, newMax, s, useExp2);

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

/// Structural precondition check for async-copy decomposition mode.
static bool canDecomposeIm2colAsyncCopyImpl(Im2colOp im2colOp) {
  // Not padded.
  if (im2colOp.hasPadding()) {
    return false;
  }

  // Identity output_perm and input_k_perm.
  if (!isIdentityPermutation(im2colOp.getOutputPerm()) ||
      !isIdentityPermutation(im2colOp.getInputKPerm())) {
    return false;
  }

  // Output shape must be static.
  auto outputType = cast<RankedTensorType>(im2colOp.getOutputType());
  if (!outputType.hasStaticShape()) {
    return false;
  }

  // Input shape must be static too: decomposeOperationAsyncCopyImpl
  // linearizes per-input-dim offsets using inputShape as the basis,
  // and cannot emit correct IR when any outer input dim is dynamic.
  auto inputType = cast<RankedTensorType>(im2colOp.getInputType());
  if (!inputType.hasStaticShape()) {
    return false;
  }

  // Innermost k_pos channel size C must be static.
  ArrayRef<int64_t> kPos = im2colOp.getKPos();
  if (kPos.empty()) {
    return false;
  }
  int64_t C = inputType.getShape()[kPos.back()];
  if (ShapedType::isDynamic(C)) {
    return false;
  }

  // Compile-time constant k_off must be channel-aligned. This check is
  // done before chooseDimToVectorize because the latter's
  // willBeContiguousSlice helper does not tolerate a non-aligned constant
  // offset; keeping this guard first ensures we reject cleanly rather
  // than relying on chooseDimToVectorize's behavior.
  SmallVector<OpFoldResult> mixedOffsets = im2colOp.getMixedOffsets();
  int64_t numBatchDims = im2colOp.getBatchPos().size();
  int64_t numMDims = im2colOp.getNumMOutputDims();
  int64_t kCanonicalIdx = numBatchDims + numMDims;
  if (kCanonicalIdx < static_cast<int64_t>(mixedOffsets.size())) {
    OpFoldResult kOff = mixedOffsets[kCanonicalIdx];
    if (auto constVal = getConstantIntValue(kOff)) {
      if (*constVal % C != 0) {
        return false;
      }
    }
  }

  // Vectorizable dim must exist AND be the innermost output dim.
  OpBuilder b(im2colOp);
  SmallVector<Range> iterDomain(im2colOp.getIterationDomain(b));
  std::optional<int64_t> vecDim = chooseDimToVectorize(
      b, im2colOp.getLoc(), im2colOp, iterDomain, mixedOffsets);
  if (!vecDim.has_value()) {
    return false;
  }
  if (*vecDim != static_cast<int64_t>(outputType.getRank() - 1)) {
    return false;
  }

  // All K output dims other than the vectorized (innermost) one must
  // have size 1 — rules out expanded-K layouts.
  ArrayRef<int64_t> outputShape = outputType.getShape();
  for (int64_t kOutDim : im2colOp.getKOutputDims()) {
    if (kOutDim == *vecDim) {
      continue;
    }
    if (outputShape[kOutDim] != 1) {
      return false;
    }
  }

  return true;
}

bool Im2colOp::canDecomposeAsyncCopy() {
  return canDecomposeIm2colAsyncCopyImpl(*this);
}

/// Compute per-input-dim offsets from per-output-dim iteration positions.
/// Expects one Value per output dim; non-iterated dims should be zero.
static SmallVector<OpFoldResult>
computeIm2colInputOffsets(OpBuilder &b, Location nestedLoc, Im2colOp im2colOp,
                          ValueRange outputDimPositions) {
  assert(outputDimPositions.size() ==
             static_cast<size_t>(im2colOp.getOutputRank()) &&
         "one position Value per im2col output dim");

  SmallVector<OpFoldResult> mixedOffsets = im2colOp.getMixedOffsets();
  SetVector<int64_t> mPosSet(im2colOp.getMPos().begin(),
                             im2colOp.getMPos().end());
  SetVector<int64_t> batchPosSet(im2colOp.getBatchPos().begin(),
                                 im2colOp.getBatchPos().end());
  ArrayRef<int64_t> strides = im2colOp.getStrides();
  ArrayRef<int64_t> dilations = im2colOp.getDilations();
  ArrayRef<int64_t> inputKPerm = im2colOp.getInputKPerm();
  SmallVector<int64_t> invInputKPerm = invertPermutationVector(inputKPerm);

  SmallVector<SmallVector<OpFoldResult>> mixedOutputSizes =
      im2colOp.getMixedOutputSizes();
  SmallVector<int64_t> kOutputDims = im2colOp.getKOutputDims();
  int64_t batchSize = im2colOp.getBatchPos().size();
  int64_t numMOutputDims = im2colOp.getNumMOutputDims();

  // Delinearize each output dim independently using its output_sizes.
  auto delinearizeOutputDims =
      [&](ArrayRef<int64_t> outputDims,
          int64_t canonicalOffset) -> SmallVector<Value> {
    SmallVector<Value> results;
    for (auto [i, actualDim] : llvm::enumerate(outputDims)) {
      int64_t canonicalIdx = canonicalOffset + i;
      OpFoldResult pos = addOfrs(b, nestedLoc, mixedOffsets[canonicalIdx],
                                 outputDimPositions[actualDim]);
      const SmallVector<OpFoldResult> &innerSizes =
          mixedOutputSizes[canonicalIdx];
      if (innerSizes.size() == 1) {
        results.push_back(getValueOrCreateConstantIndexOp(b, nestedLoc, pos));
      } else {
        ValueRange components =
            affine::AffineDelinearizeIndexOp::create(
                b, nestedLoc,
                getValueOrCreateConstantIndexOp(b, nestedLoc, pos), innerSizes,
                /*hasOuterBound=*/true)
                .getResults();
        results.append(components.begin(), components.end());
      }
    }
    return results;
  };

  SmallVector<Value> delinKOffset =
      delinearizeOutputDims(kOutputDims, batchSize + numMOutputDims);

  SmallVector<Value> windowOffset, inputKOffset;
  int delinKIdx = 0;
  for (int i = 0; i < im2colOp.getInputRank(); ++i) {
    if (batchPosSet.contains(i)) {
      continue;
    }
    if (mPosSet.contains(i)) {
      windowOffset.push_back(delinKOffset[invInputKPerm[delinKIdx++]]);
      continue;
    }
    inputKOffset.push_back(delinKOffset[invInputKPerm[delinKIdx++]]);
  }

  SmallVector<int64_t> mOutputDims = im2colOp.getMOutputDims();
  SmallVector<Value> delinMOffset =
      delinearizeOutputDims(mOutputDims, batchSize);

  OpFoldResult zero = b.getIndexAttr(0);
  SmallVector<OpFoldResult> sliceOffsets(im2colOp.getInputRank(), zero);

  AffineExpr mOff, wOff;
  bindDims(b.getContext(), mOff, wOff);
  for (auto [idx, mPos] : llvm::enumerate(im2colOp.getMPos())) {
    auto map =
        AffineMap::get(2, 0, {mOff * strides[idx] + wOff * dilations[idx]});
    OpFoldResult offset = affine::makeComposedFoldedAffineApply(
        b, nestedLoc, map, {delinMOffset[idx], windowOffset[idx]});
    sliceOffsets[mPos] = offset;
  }

  for (auto [kPos, kOff] : llvm::zip_equal(im2colOp.getKPos(), inputKOffset)) {
    sliceOffsets[kPos] = kOff;
  }
  SmallVector<int64_t> inverseOutputPerm =
      invertPermutationVector(im2colOp.getOutputPerm());
  for (auto [ivIdx, bPos] : llvm::enumerate(im2colOp.getBatchPos())) {
    int64_t canonicalIdx = ivIdx;
    int64_t actualDim = inverseOutputPerm[canonicalIdx];
    sliceOffsets[bPos] = addOfrs(b, nestedLoc, mixedOffsets[canonicalIdx],
                                 outputDimPositions[actualDim]);
  }

  return sliceOffsets;
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
///       offsets = [0, %m_off, %k_off]
///       output_sizes = [[2], [32, 32], [3, 3, 640]]
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
static FailureOr<SmallVector<Value>>
decomposeOperationStreamCopy(Im2colOp im2colOp, OpBuilder &b) {
  Location loc = im2colOp.getLoc();
  Value inputSlice = im2colOp.getInput();
  SmallVector<OpFoldResult> mixedOffsets = im2colOp.getMixedOffsets();
  SmallVector<SmallVector<OpFoldResult>> mixedOutputSizes =
      im2colOp.getMixedOutputSizes();

  int64_t outputRank = im2colOp.getOutputRank();
  int64_t inputRank = im2colOp.getInputRank();

  // Step 1: Choose the vectorization dimension.
  SmallVector<Range> iterationDomain(im2colOp.getIterationDomain(b));
  std::optional<int64_t> maybeOutputDimToVectorize =
      chooseDimToVectorize(b, loc, im2colOp, iterationDomain, mixedOffsets);

  OpFoldResult innerInputTileSize;
  if (maybeOutputDimToVectorize.has_value()) {
    int64_t vecDim = maybeOutputDimToVectorize.value();
    innerInputTileSize = iterationDomain[vecDim].size;
    iterationDomain.erase(iterationDomain.begin() + vecDim);
  } else {
    innerInputTileSize = b.getIndexAttr(1);
  }

  // Build loop nest over all non-vectorized dimensions.
  SmallVector<Value> lbs, ubs, steps;
  for (auto range : iterationDomain) {
    lbs.push_back(getValueOrCreateConstantIndexOp(b, loc, range.offset));
    ubs.push_back(getValueOrCreateConstantIndexOp(b, loc, range.size));
    steps.push_back(getValueOrCreateConstantIndexOp(b, loc, range.stride));
  }
  scf::LoopNest loopNest = scf::buildLoopNest(
      b, loc, lbs, ubs, steps, im2colOp.getOutput(),
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
          ValueRange iterArgs) -> scf::ValueVector { return iterArgs; });
  SmallVector<Value> ivs;
  for (scf::ForOp loop : loopNest.loops) {
    ivs.push_back(loop.getInductionVar());
  }
  // Step 2: When vectorizing a dim, insert a zero IV for it (it spans the full
  // tile).
  if (maybeOutputDimToVectorize.has_value()) {
    Value zeroIV = arith::ConstantIndexOp::create(b, loc, 0);
    ivs.insert(ivs.begin() + maybeOutputDimToVectorize.value(), zeroIV);
  }

  // Step 3: Compute source indices.
  OpBuilder::InsertionGuard guard(b);
  Location nestedLoc =
      loopNest.loops.back().getBody()->getTerminator()->getLoc();
  b.setInsertionPointToStart(loopNest.loops.back().getBody());

  Im2colSourceIndices srcIndices = computeIm2colSourceIndices(
      b, nestedLoc, im2colOp, ivs, innerInputTileSize);

  // The slice is always 1D — just a flat slice along the vectorized input
  // dimension. With a 1D slice, no transpose is needed regardless of
  // which output dimension is being vectorized.
  ShapedType outputType = im2colOp.getOutputType();
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);
  int64_t vecInputDim = inputRank - 1;

  std::optional<int64_t> staticTileSize =
      getConstantIntValue(innerInputTileSize);
  int64_t paddedStaticSize =
      staticTileSize ? *staticTileSize : ShapedType::kDynamic;

  SmallVector<OpFoldResult> outOffsets(outputRank, zero);
  for (auto [idx, iv] : llvm::enumerate(ivs)) {
    outOffsets[idx] = iv;
  }
  SmallVector<OpFoldResult> outSizes(outputRank, one);
  if (maybeOutputDimToVectorize.has_value()) {
    outSizes[maybeOutputDimToVectorize.value()] = innerInputTileSize;
  }
  SmallVector<OpFoldResult> outStrides(outputRank, one);

  // Step 4: Compute read offsets and extract the input slice.
  // Subtract padLow from source offsets to get real input coordinates.
  // Clamp to [0, dimSize - 1] to avoid negative indices in extract_slice.
  // When out-of-bounds, validSize is 0 so extract_slice produces an empty
  // slice and tensor.pad fills the padding.
  SmallVector<OpFoldResult> readOffsets;
  SmallVector<OpFoldResult> extractSizes(inputRank, one);
  SmallVector<OpFoldResult> extractStrides(inputRank, one);
  Value sliceToInsert;

  SmallVector<OpFoldResult> padLow(inputRank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> inputPadLow = im2colOp.getMixedInputPadLow();
  if (!inputPadLow.empty()) {
    padLow = inputPadLow;
  }
  SmallVector<OpFoldResult> inputDimSizes =
      tensor::getMixedSizes(b, nestedLoc, im2colOp.getInput());
  MLIRContext *clampCtx = b.getContext();
  AffineExpr cd0 = getAffineDimExpr(0, clampCtx);
  AffineExpr cd1 = getAffineDimExpr(1, clampCtx);
  AffineMap maxZeroMap =
      AffineMap::get(1, 0, {cd0, getAffineConstantExpr(0, clampCtx)}, clampCtx);
  AffineMap clampHighMap = AffineMap::get(2, 0, {cd0, cd1 - 1}, clampCtx);
  for (int64_t d = 0; d < inputRank; ++d) {
    OpFoldResult adjusted =
        subOfrs(b, nestedLoc, srcIndices.sliceOffsets[d], padLow[d]);
    if (im2colOp.hasPadding()) {
      adjusted = affine::makeComposedFoldedAffineMax(b, nestedLoc, maxZeroMap,
                                                     {adjusted});
      adjusted = affine::makeComposedFoldedAffineMin(
          b, nestedLoc, clampHighMap, {adjusted, inputDimSizes[d]});
    }
    readOffsets.push_back(adjusted);
  }

  Value validSize;
  if (im2colOp.hasPadding()) {
    validSize = computeIm2colValidSize(b, nestedLoc, im2colOp, srcIndices,
                                       innerInputTileSize, ivs,
                                       maybeOutputDimToVectorize);
    extractSizes[vecInputDim] = validSize;
  } else {
    extractSizes[vecInputDim] = innerInputTileSize;
  }

  auto extractType = RankedTensorType::get(
      {im2colOp.hasPadding() ? ShapedType::kDynamic : paddedStaticSize},
      outputType.getElementType());
  auto extract =
      tensor::ExtractSliceOp::create(b, nestedLoc, extractType, inputSlice,
                                     readOffsets, extractSizes, extractStrides);

  // Branch only on the vectorizable payload:
  //  - No padding: linalg.copy (static type, concrete copy op)
  //  - Has padding: tensor.pad (dynamic extract padded to static size)
  if (!im2colOp.hasPadding()) {
    auto sliceType = cast<RankedTensorType>(extract.getType());
    auto destExtract = tensor::ExtractSliceOp::create(
        b, nestedLoc, sliceType, loopNest.loops.back().getRegionIterArg(0),
        outOffsets, outSizes, outStrides);
    auto copy = linalg::CopyOp::create(b, nestedLoc, extract.getResult(),
                                       destExtract.getResult());
    sliceToInsert = copy.getResult(0);
  } else {
    Value tileSize =
        getValueOrCreateConstantIndexOp(b, nestedLoc, innerInputTileSize);
    SmallVector<OpFoldResult> lowPad = {b.getIndexAttr(0)};
    Value highPadAmt = arith::SubIOp::create(b, nestedLoc, tileSize, validSize);
    SmallVector<OpFoldResult> highPad = {highPadAmt};

    auto paddedType =
        RankedTensorType::get({paddedStaticSize}, outputType.getElementType());
    auto paddedSlice = tensor::PadOp::create(
        b, nestedLoc, paddedType, extract.getResult(), lowPad, highPad,
        im2colOp.getPadValue(), /*nofold=*/false);
    sliceToInsert = paddedSlice.getResult();
  }

  auto insert = tensor::InsertSliceOp::create(
      b, nestedLoc, sliceToInsert, loopNest.loops.back().getRegionIterArg(0),
      outOffsets, outSizes, outStrides);
  auto yieldOp =
      cast<scf::YieldOp>(loopNest.loops.back().getBody()->getTerminator());
  yieldOp->getOpOperands().front().assign(insert.getResult());
  return SmallVector<Value>({loopNest.results[0]});
}

/// Async-copy decomposition: lower the im2col op to a gather over a
/// collapsed source. The source is reshaped to 2D [outer_size, C] where
/// outer_size is the product of all non-channel input dims. A
/// linalg.generic computes, for each non-innermost output position, the
/// linearized flat index into that collapsed source via the shared
/// computeIm2colInputOffsets helper. An iree_linalg_ext.gather reads the
/// contiguous channel slice for every output row; the result is expanded
/// back to the original output shape.
static FailureOr<SmallVector<Value>>
decomposeOperationAsyncCopyImpl(Im2colOp im2colOp, OpBuilder &b) {
  if (!canDecomposeIm2colAsyncCopyImpl(im2colOp)) {
    return im2colOp.emitOpError(
        "async_copy decomposition preconditions not satisfied "
        "(see canDecomposeAsyncCopy)");
  }

  Location loc = im2colOp.getLoc();
  auto inputType = cast<RankedTensorType>(im2colOp.getInputType());
  auto outputType = cast<RankedTensorType>(im2colOp.getOutputType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  int64_t inputRank = inputType.getRank();
  int64_t outputRank = outputType.getRank();
  int64_t batchSize = ShapedType::getNumElements(outputShape.drop_back());

  // Step 1: Collapse source to 2D: [[0..inputRank-2], [inputRank-1]].
  SmallVector<ReassociationIndices> srcReassoc = {
      llvm::to_vector(llvm::seq<int64_t>(0, inputRank - 1)), {inputRank - 1}};
  Value collapsedSource =
      tensor::CollapseShapeOp::create(b, loc, im2colOp.getInput(), srcReassoc);

  // Step 2: Build a 1D index tensor by running a linalg.generic with a
  // single parallel iterator over batchSize.
  Type indexType = b.getIndexType();
  Value indexEmpty =
      tensor::EmptyOp::create(b, loc, ArrayRef<int64_t>{batchSize}, indexType);
  AffineMap indexMap = b.getMultiDimIdentityMap(1);
  SmallVector<utils::IteratorType> iterTypes = {utils::IteratorType::parallel};

  auto indexGeneric = linalg::GenericOp::create(
      b, loc, indexEmpty.getType(), /*inputs=*/ValueRange{},
      /*outputs=*/ValueRange{indexEmpty},
      /*indexingMaps=*/ArrayRef<AffineMap>{indexMap}, iterTypes,
      [&](OpBuilder &nestedB, Location nestedLoc, ValueRange) {
        // Delinearize linalg.index 0 into per-output-dim positions,
        // covering every output dim except the innermost (vectorized)
        // one. Size-1 non-vectorized K output dims contribute zero to
        // the delinearization because their basis entry is 1.
        Value flatIdx = linalg::IndexOp::create(nestedB, nestedLoc, 0);
        SmallVector<OpFoldResult> iterBasis;
        for (int64_t d = 0; d < outputRank - 1; ++d) {
          iterBasis.push_back(nestedB.getIndexAttr(outputShape[d]));
        }

        // Delinearize to (outputRank - 1) positions.
        SmallVector<Value> nonVectorizedPositions;
        if (outputRank == 1) {
          // Rare: only the vectorized (and innermost) dim, no iteration
          // positions to delinearize. batchSize is 1 in this case.
        } else if (outputRank == 2) {
          nonVectorizedPositions.push_back(flatIdx);
        } else {
          auto delinearize = affine::AffineDelinearizeIndexOp::create(
              nestedB, nestedLoc, flatIdx, iterBasis,
              /*hasOuterBound=*/true);
          for (unsigned i = 0; i < delinearize.getNumResults(); ++i) {
            nonVectorizedPositions.push_back(delinearize.getResult(i));
          }
        }

        // Assemble outputDimPositions for the helper: actual-output-dim
        // order, length == outputRank, with a zero constant in the
        // vectorized slot.
        Value zero = arith::ConstantIndexOp::create(nestedB, nestedLoc, 0);
        SmallVector<Value> outputDimPositions;
        outputDimPositions.reserve(outputRank);
        for (int64_t d = 0; d < outputRank - 1; ++d) {
          outputDimPositions.push_back(nonVectorizedPositions[d]);
        }
        outputDimPositions.push_back(zero); // vectorized slot

        SmallVector<OpFoldResult> sliceOffsets = computeIm2colInputOffsets(
            nestedB, nestedLoc, im2colOp, outputDimPositions);

        // Linearize sliceOffsets[0..inputRank-2] using
        // inputShape[0..inputRank-2] to get the flat gather index into
        // the collapsed source.
        SmallVector<Value> outerCoords;
        SmallVector<OpFoldResult> outerBasis;
        for (int64_t i = 0; i < inputRank - 1; ++i) {
          outerCoords.push_back(getValueOrCreateConstantIndexOp(
              nestedB, nestedLoc, sliceOffsets[i]));
          outerBasis.push_back(nestedB.getIndexAttr(inputShape[i]));
        }
        Value flatGatherIdx = affine::AffineLinearizeIndexOp::create(
            nestedB, nestedLoc, outerCoords, outerBasis,
            /*disjoint=*/false);

        linalg::YieldOp::create(nestedB, nestedLoc, flatGatherIdx);
      });
  Value indices = indexGeneric.getResult(0);

  // Step 3: Collapse the im2col output to 2D [batchSize, C_per_window],
  // build the gather, then expand back.
  SmallVector<ReassociationIndices> outputReassoc = {
      llvm::to_vector(llvm::seq<int64_t>(0, outputRank - 1)), {outputRank - 1}};
  Value collapsedOutput = tensor::CollapseShapeOp::create(
      b, loc, im2colOp.getOutput(), outputReassoc);

  auto gatherOp = IREE::LinalgExt::GatherOp::create(
      b, loc, cast<RankedTensorType>(collapsedOutput.getType()),
      collapsedSource, indices, collapsedOutput, b.getDenseI64ArrayAttr({0}));

  // Propagate the lowering_config attribute by raw name copy. This keeps
  // LinalgExt free of any dependency on Codegen::IREECodegenAttrs.
  constexpr StringLiteral kLoweringConfigAttrName = "lowering_config";
  if (Attribute lcAttr = im2colOp->getAttr(kLoweringConfigAttrName)) {
    gatherOp->setAttr(kLoweringConfigAttrName, lcAttr);
  }

  Value result = tensor::ExpandShapeOp::create(
      b, loc, outputType, gatherOp.getResult(0), outputReassoc);

  return SmallVector<Value>{result};
}

FailureOr<SmallVector<Value>>
Im2colOp::decomposeOperationAsyncCopy(OpBuilder &b) {
  return decomposeOperationAsyncCopyImpl(*this, b);
}

FailureOr<SmallVector<Value>> Im2colOp::decomposeOperation(OpBuilder &b) {
  return decomposeOperationStreamCopy(*this, b);
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
  Value ex = computeSubAndExp(rewriter, loc, prevMaxMap, normValMap, currMax,
                              sValue->get(), /*useExp2=*/true);
  // norm = e^(prev_max - curr_max)
  Value norm = computeSubAndExp(rewriter, loc, prevMaxMap, prevMaxMap, currMax,
                                prevMax->get(), /*useExp2=*/true);

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

} // namespace mlir::iree_compiler::IREE::LinalgExt
