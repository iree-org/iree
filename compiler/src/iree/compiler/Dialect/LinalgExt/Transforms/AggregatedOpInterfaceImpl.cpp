// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

static Value scaleValueInPlace(OpBuilder &builder, Location loc,
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
        Value result = b.create<arith::MulFOp>(loc, scale, args[1]);
        b.create<linalg::YieldOp>(loc, result);
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

AffineMap dropResultsContainingDims(AffineMap map, ArrayRef<int64_t> dims) {
  SmallVector<int> resultsToDrop;
  for (auto [idx, result] : llvm::enumerate(map.getResults())) {
    for (int dim : dims) {
      if (result.isFunctionOfDim(dim)) {
        resultsToDrop.push_back(idx);
        break;
      }
    }
  }

  // Iterate in reverse to preserve indices to drop.
  for (int result : llvm::reverse(resultsToDrop)) {
    map = map.dropResult(result);
  }
  return map;
}

FailureOr<SmallVector<Value>>
OnlineAttentionOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();
  Value query = getQuery();
  Value key = getKey();
  Value value = getValue();
  Value oldAcc = getOutput();
  Value oldMax = getMax();
  Value oldSum = getSum();
  Type elementType = getQuery().getType().getElementType();

  FailureOr<AttentionOpDetail> maybeOpInfo =
      AttentionOpDetail::get(getIndexingMapsArray());
  assert(succeeded(maybeOpInfo) && "Invalid attention indexing maps");
  AttentionOpDetail opInfo = maybeOpInfo.value();

  SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
      getIterationDomain(b), [](Range x) { return x.size; });

  // Since we use exp2 for attention instead of the original exp, we have to
  // multiply the scale by log2(e). We use exp2 instead of exp as most GPUs
  // have better support for exp2.
  Value scale = getScale();
  Value log2e =
      b.create<arith::ConstantOp>(loc, b.getFloatAttr(elementType, M_LOG2E));
  scale = b.create<arith::MulFOp>(loc, scale, log2e);

  // In the original algorithm, the scaling is done after the softmax:
  //        softmax(Q @ K.T * scale) @ V
  //
  // But, it is mathematically equivalent to do it on Q first and then multiply
  // it by K.T. This just allows us to do the scaling once, instead of each
  // iteration of the loop.
  AffineMap qMap = getQueryMap();
  AffineMap scaleMap = AffineMap::get(/*dimCount=*/qMap.getNumInputs(),
                                      /*symbolCount=*/0, getContext());
  query = scaleValueInPlace(b, loc, qMap, scaleMap, query, scale);

  // ---- Matmul 1 ----

  // Get sizes for S.
  AffineMap sMap = opInfo.getSMap();
  SmallVector<OpFoldResult> sSizes;
  for (AffineExpr dimExpr : sMap.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    sSizes.push_back(sizes[dim]);
  }

  // S = Q @ K
  // SMap = QMap @ KMap
  Value emptyS = b.create<tensor::EmptyOp>(loc, sSizes, elementType);
  Value sZero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));
  Value s = b.create<linalg::FillOp>(loc, sZero, emptyS).getResult(0);
  s = computeMatmul(b, loc, getQueryMap(), getKeyMap(), sMap, query, key, s);

  // TODO: This decomposition should be in a seperate op called
  // "online softmax".
  // ---- Online Softmax ----

  // newMax = max(oldMax, rowMax(S))
  AffineMap maxMap = getMaxMap();
  Value newMax = reduce<arith::MaximumFOp>(b, loc, sMap, maxMap, s, oldMax);

  // P = exp2(S - newMax)
  // PMap = SMap
  AffineMap pMap = sMap;
  Value p = computeSubAndExp2(b, loc, maxMap, sMap, newMax, s);

  // norm = exp2(oldMax - newMax)
  // normMap = maxMap
  AffineMap normMap = getMaxMap();
  Value norm = computeSubAndExp2(b, loc, maxMap, normMap, newMax, oldMax);

  // normSum = norm * oldSum
  AffineMap sumMap = getSumMap();
  Value normSum = scaleValueInPlace(b, loc, sumMap, normMap, oldSum, norm);

  // newSum = normSum + rowMax(P)
  Value newSum = reduce<arith::AddFOp>(b, loc, pMap, sumMap, p, normSum);

  // newAcc = norm * oldAcc
  AffineMap accMap = getOutputMap();
  Value newAcc = scaleValueInPlace(b, loc, accMap, normMap, oldAcc, norm);

  // ---- Matmul 2 ----

  // newAcc = P @ V + newAcc
  newAcc = computeMatmul(b, loc, pMap, getValueMap(), accMap, p, value, newAcc);

  return SmallVector<Value>{newAcc, newMax, newSum};
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
