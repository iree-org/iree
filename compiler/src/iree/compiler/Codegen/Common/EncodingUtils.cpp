// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <numeric>

namespace mlir::iree_compiler {

using IREE::Encoding::EncodingAttr;
using IREE::Encoding::getEncodingAttr;
using IREE::Encoding::getEncodingContractionDims;

// If tensorType has the encoding of a matmul RESULT with narrow N, returns
// the transposed type. Otherwise, just returns tensorType.
static RankedTensorType transposeIfNarrowNResult(RankedTensorType tensorType) {
  auto encoding =
      llvm::dyn_cast_or_null<EncodingAttr>(tensorType.getEncoding());
  if (!encoding) {
    return tensorType;
  }
  if (!isNarrowNResult(encoding)) {
    return tensorType;
  }
  auto newIndex = encoding.getOperandIndex();
  TypeAttr originalTypeAttr = encoding.getOriginalType();
  RankedTensorType originalType = tensorType;
  if (originalTypeAttr) {
    originalType =
        llvm::dyn_cast<RankedTensorType>(originalTypeAttr.getValue());
  }
  SmallVector<int64_t> newOriginalShape(originalType.getShape());
  auto userIndexingMaps = encoding.getUserIndexingMaps();
  SmallVector<AffineMap> maps;
  for (auto a : userIndexingMaps) {
    maps.push_back(cast<AffineMapAttr>(a).getAffineMap());
  }
  auto cDims = linalg::inferContractionDims(maps);
  SmallVector<int64_t> newShape(tensorType.getShape());
  SmallVector<int64_t> permIndices(maps[0].getNumDims());
  std::iota(std::begin(permIndices), std::end(permIndices), 0);
  // Matrix case: there are both M and N dimensions. Transposing means swapping
  // them.
  if (cDims->m.size() == 1 && cDims->n.size() == 1) {
    int m = cDims->m[0];
    int n = cDims->n[0];
    std::swap(permIndices[m], permIndices[n]);
    int mDim = encoding.mapDimToOperandIndex(m);
    int nDim = encoding.mapDimToOperandIndex(n);
    std::swap(newShape[mDim], newShape[nDim]);
    std::swap(newOriginalShape[mDim], newOriginalShape[nDim]);
  }
  // Vector case: there is no N dimension to swap the M dimension with. We
  // swap the maps themselves.
  if (cDims->n.empty()) {
    std::swap(maps[0], maps[1]);
  }

  // auto newRoundDimsTo = encoding.getRoundDimsToArray();
  SmallVector<int64_t> newRoundDimsTo(encoding.getRoundDimsToArray());
  assert(newRoundDimsTo.size() == 0 || newRoundDimsTo.size() >= 3);
  if (newRoundDimsTo.size() != 0) {
    std::swap(newRoundDimsTo[newRoundDimsTo.size() - 3],
              newRoundDimsTo[newRoundDimsTo.size() - 2]);
  }
  auto context = tensorType.getContext();
  AffineMap permutation = AffineMap::getPermutationMap(permIndices, context);
  for (auto &map : maps) {
    map = map.compose(permutation);
  }
  SmallVector<Attribute> newMaps;
  for (auto map : maps) {
    newMaps.push_back(AffineMapAttr::get(map));
  }
  ArrayAttr newIndexingMaps = ArrayAttr::get(context, newMaps);
  auto elemType = tensorType.getElementType();
  OpBuilder builder(context);

  auto opTypeAttr = IREE::Encoding::EncodingOpTypeAttr::get(
      context, IREE::Encoding::EncodingOpType::matmul);
  auto newEncoding = IREE::Encoding::EncodingAttr::get(
      context, newIndex, opTypeAttr, encoding.getElementTypes(),
      TypeAttr::get(RankedTensorType::get(newOriginalShape, elemType)),
      encoding.getMatmulNarrow_N(), encoding.getMatmulNarrow_M(),
      newIndexingMaps, DenseI64ArrayAttr::get(context, newRoundDimsTo));
  return RankedTensorType::get(newShape, elemType, newEncoding);
}

/// For a given tensor type with an encoding, return the materialized
/// type to use for it. If no encoding is set, then return the tensor type
/// itself.
static RankedTensorType
getMaterializedType(RankedTensorType tensorType,
                    MaterializeEncodingFn materializeEncodingFn) {
  RankedTensorType maybeTransposedTensorType =
      transposeIfNarrowNResult(tensorType);
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(maybeTransposedTensorType);
  if (failed(materializeEncodingInfo)) {
    return dropEncoding(tensorType);
  }
  return cast<RankedTensorType>(tensor::PackOp::inferPackedType(
      getOriginalTypeWithEncoding(maybeTransposedTensorType)
          .clone(tensorType.getElementType()),
      materializeEncodingInfo->innerTileSizes,
      materializeEncodingInfo->innerDimsPos,
      materializeEncodingInfo->outerDimsPerm));
}

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    MaterializeEncodingFn materializeEncodingFn)
    : materializeEncodingFn(materializeEncodingFn) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion([=](RankedTensorType t) -> RankedTensorType {
    return getMaterializedType(t, materializeEncodingFn);
  });
}

MaterializeEncodingConversionTarget::MaterializeEncodingConversionTarget(
    MLIRContext &context)
    : ConversionTarget(context) {
  // Mark any operation that has operands/results with encoding as
  // illegal.
  markUnknownOpDynamicallyLegal([](Operation *op) {
    auto typeHasEncoding = [](Type t) -> bool {
      auto tensorType = dyn_cast<RankedTensorType>(t);
      return tensorType && tensorType.getEncoding();
    };
    auto valueHasEncoding = [=](Value v) -> bool {
      return typeHasEncoding(v.getType());
    };
    bool hasOperandOrResultsWithEncoding =
        llvm::any_of(op->getOperands(), valueHasEncoding) ||
        llvm::any_of(op->getResultTypes(), typeHasEncoding);
    return !hasOperandOrResultsWithEncoding;
  });
}

RankedTensorType getOriginalTypeWithEncoding(RankedTensorType type) {
  auto encoding = getEncodingAttr(type);
  if (!encoding) {
    return type;
  }
  RankedTensorType originalType = type;
  if (auto originalTypeAttr = encoding.getOriginalType()) {
    originalType = cast<RankedTensorType>(originalTypeAttr.getValue());
  }
  return RankedTensorType::get(originalType.getShape(),
                               originalType.getElementType(), encoding);
}

RankedTensorType dropEncoding(RankedTensorType type) {
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

int64_t getIntOrZero(IntegerAttr a) {
  return a == IntegerAttr() ? 0 : a.getInt();
}

MaterializeEncodingInfo getEncodingInfoForMatmul(EncodingAttr encoding,
                                                 int64_t rank,
                                                 TileMxNxK tileMxNxK) {
  auto index = encoding.getOperandIndex().getValue();
  MaterializeEncodingInfo encodingInfo;
  auto cDims = getEncodingContractionDims(encoding);
  // The following expects M, N, K, and Batch sizes of at most 1 for now
  assert(cDims->m.size() <= 1 && cDims->n.size() <= 1 && cDims->k.size() <= 1 &&
         cDims->batch.size() <= 1 &&
         "Expected at most one M, N, K, and Batch dimension");
  if (!cDims->batch.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        encoding.mapDimToOperandIndex(cDims->batch[0]));
  }
  if (index != IREE::Encoding::MATMUL_RHS && !cDims->m.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        encoding.mapDimToOperandIndex(cDims->m[0]));
    encodingInfo.innerDimsPos.push_back(
        encoding.mapDimToOperandIndex(cDims->m[0]));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.M);
  }
  if (index != IREE::Encoding::MATMUL_LHS && !cDims->n.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        encoding.mapDimToOperandIndex(cDims->n[0]));
    encodingInfo.innerDimsPos.push_back(
        encoding.mapDimToOperandIndex(cDims->n[0]));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.N);
  }
  if (index != IREE::Encoding::MATMUL_RESULT) {
    encodingInfo.outerDimsPerm.push_back(
        encoding.mapDimToOperandIndex(cDims->k[0]));
    encodingInfo.innerDimsPos.push_back(
        encoding.mapDimToOperandIndex(cDims->k[0]));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.K);
  }
  return encodingInfo;
}

bool isNarrowNResult(EncodingAttr encoding) {
  if (encoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RESULT) {
    return false;
  }
  IntegerAttr narrowM = encoding.getMatmulNarrow_M();
  IntegerAttr narrowN = encoding.getMatmulNarrow_N();
  return narrowN && (!narrowM || narrowM.getInt() > narrowN.getInt());
}

} // namespace mlir::iree_compiler
