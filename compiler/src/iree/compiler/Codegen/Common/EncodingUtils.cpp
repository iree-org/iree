// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
namespace mlir::iree_compiler {

using IREE::LinalgExt::EncodingAttr;
using IREE::LinalgExt::EncodingRole;

// If tensorType has the encoding of a matmul RESULT with narrow N, returns
// the transposed type. Otherwise, just returns tensorType.
static RankedTensorType transposeIfNarrowNResult(RankedTensorType tensorType) {
  EncodingAttr encoding =
      tensorType.getEncoding().dyn_cast_or_null<EncodingAttr>();
  if (!encoding || tensorType.getRank() <= 1) {
    return tensorType;
  }
  if (!isNarrowNResult(encoding)) {
    return tensorType;
  }
  if (tensorType.getRank() <= 1) {
    return tensorType;
  }
  auto newRole = encoding.getRole().getValue();
  TypeAttr originalTypeAttr = encoding.getOriginalType();
  RankedTensorType originalType;
  if (originalTypeAttr) {
    originalType = originalTypeAttr.getValue().dyn_cast<RankedTensorType>();
  }
  if (!originalType) {
    originalType = tensorType;
  }
  SmallVector<int64_t> newOriginalShape{originalType.getShape()};
  auto userIndexingMaps = encoding.getUserIndexingMaps();
  SmallVector<AffineMap> maps;
  for (auto a : userIndexingMaps) {
    maps.push_back(cast<AffineMapAttr>(a).getAffineMap());
  }
  auto cDims = linalg::inferContractionDims(maps);
  SmallVector<int64_t> newShape{tensorType.getShape()};
  SmallVector<int64_t> permIndices(maps[0].getNumDims());
  for (int i = 0; i < permIndices.size(); ++i) {
    permIndices[i] = i;
  }
  // Matrix case: there are both M and N dimensions. Transposing means swapping
  // them.
  if (cDims->m.size() == 1 && cDims->n.size() == 1) {
    int m = cDims->m[0];
    int n = cDims->n[0];
    std::swap(newShape[m], newShape[n]);
    std::swap(newOriginalShape[m], newOriginalShape[n]);
    std::swap(permIndices[m], permIndices[n]);
  }
  // Vector case: there is no N dimension to swap the M dimension with. We
  // swap the maps themselves.
  if (cDims->n.empty()) {
    std::swap(maps[0], maps[1]);
  }
  AffineMap permutation =
      AffineMap::getPermutationMap(permIndices, tensorType.getContext());
  for (auto &map : maps) {
    map = map.compose(permutation);
  }
  SmallVector<Attribute> newMaps;
  for (auto map : maps) {
    newMaps.push_back(AffineMapAttr::get(map));
  }
  ArrayAttr newIndexingMaps = ArrayAttr::get(tensorType.getContext(), newMaps);
  auto newEncoding = EncodingAttr::get(
      encoding.getContext(),
      IREE::LinalgExt::EncodingRoleAttr::get(encoding.getContext(), newRole),
      encoding.getElementTypes(),
      TypeAttr::get(
          RankedTensorType::get(newOriginalShape, tensorType.getElementType())),
      encoding.getMatmulNarrow_N(), encoding.getMatmulNarrow_M(),
      newIndexingMaps);
  auto r =
      RankedTensorType::get(newShape, tensorType.getElementType(), newEncoding);
  return r;
}

/// For a given tensor type with an encoding, return the materialized
/// type to use for it. If no encoding is set, then return the tensor type
/// itself.
static RankedTensorType
getMaterializedType(RankedTensorType tensorType,
                    MaterializeEncodingFn materializeEncodingFn,
                    bool transposeNarrowN) {
  if (transposeNarrowN) {
    tensorType = transposeIfNarrowNResult(tensorType);
  }
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(tensorType);

  if (failed(materializeEncodingInfo)) {
    return dropEncoding(tensorType);
  }
  auto resultType =
      tensor::PackOp::inferPackedType(getOriginalTypeWithEncoding(tensorType)
                                          .clone(tensorType.getElementType()),
                                      materializeEncodingInfo->innerTileSizes,
                                      materializeEncodingInfo->innerDimsPos,
                                      materializeEncodingInfo->outerDimsPerm)
          .cast<RankedTensorType>();
  return resultType;
}

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    MaterializeEncodingFn materializeEncodingFn, bool transposeNarrowN)
    : materializeEncodingFn(materializeEncodingFn),
      transposeNarrowN(transposeNarrowN) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion([=](RankedTensorType t) -> RankedTensorType {
    return getMaterializedType(t, materializeEncodingFn, transposeNarrowN);
  });
}

MaterializeEncodingConversionTarget::MaterializeEncodingConversionTarget(
    MLIRContext &context)
    : ConversionTarget(context) {
  // Mark any operation that has operands/results with encoding as
  // illegal.
  markUnknownOpDynamicallyLegal([](Operation *op) {
    auto typeHasEncoding = [](Type t) -> bool {
      auto tensorType = t.dyn_cast<RankedTensorType>();
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

EncodingAttr getEncodingAttr(RankedTensorType type) {
  return type.getEncoding().dyn_cast_or_null<EncodingAttr>();
}

static AffineMap getMapForRole(EncodingAttr encoding) {
  EncodingRole role = encoding.getRole().getValue();
  if (role == EncodingRole::LHS)
    return cast<AffineMapAttr>(encoding.getUserIndexingMaps()[0])
        .getAffineMap();
  else if (role == EncodingRole::RHS)
    return cast<AffineMapAttr>(encoding.getUserIndexingMaps()[1])
        .getAffineMap();
  else
    return cast<AffineMapAttr>(encoding.getUserIndexingMaps()[2])
        .getAffineMap();
}

FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(EncodingAttr encoding) {
  auto indexingMapsAttr = encoding.getUserIndexingMaps();
  SmallVector<AffineMap> indexingMaps = llvm::map_to_vector(
      indexingMapsAttr.getValue(), [](Attribute m) -> AffineMap {
        return cast<AffineMapAttr>(m).getAffineMap();
      });
  return linalg::inferContractionDims(indexingMaps);
}

/// Given the dim position of the encoding `user_indexing_maps`, return the
/// matching index of the given encoding's tensor
static unsigned mapDimToRoleIndex(int64_t dimPos, EncodingAttr encoding) {
  AffineMap map = getMapForRole(encoding);
  auto idx = map.getResultPosition(getAffineDimExpr(dimPos, map.getContext()));
  assert(idx.has_value());
  return idx.value();
}

RankedTensorType getOriginalTypeWithEncoding(RankedTensorType type) {
  auto encoding = getEncodingAttr(type);
  if (!encoding) {
    return type;
  }
  RankedTensorType originalType = type;
  if (auto originalTypeAttr = encoding.getOriginalType()) {
    originalType = originalTypeAttr.getValue().cast<RankedTensorType>();
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
  EncodingRole role = encoding.getRole().getValue();
  MaterializeEncodingInfo encodingInfo;
  auto cDims = getEncodingContractionDims(encoding);
  // The following expects M, N, K, and Batch sizes of at most 1 for now
  assert(cDims->m.size() <= 1 && cDims->n.size() <= 1 && cDims->k.size() <= 1 &&
         cDims->batch.size() <= 1 &&
         "Expected at most one M, N, K, and Batch dimension");
  if (!cDims->batch.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        mapDimToRoleIndex(cDims->batch[0], encoding));
  }
  if (role != EncodingRole::RHS && !cDims->m.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        mapDimToRoleIndex(cDims->m[0], encoding));
    encodingInfo.innerDimsPos.push_back(
        mapDimToRoleIndex(cDims->m[0], encoding));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.M);
  }
  if (role != EncodingRole::LHS && !cDims->n.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        mapDimToRoleIndex(cDims->n[0], encoding));
    encodingInfo.innerDimsPos.push_back(
        mapDimToRoleIndex(cDims->n[0], encoding));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.N);
  }
  if (role != EncodingRole::RESULT) {
    encodingInfo.outerDimsPerm.push_back(
        mapDimToRoleIndex(cDims->k[0], encoding));
    encodingInfo.innerDimsPos.push_back(
        mapDimToRoleIndex(cDims->k[0], encoding));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.K);
  }
  return encodingInfo;
}

bool isNarrowNResult(EncodingAttr encoding) {
  if (encoding.getRole().getValue() != EncodingRole::RESULT) {
    return false;
  }
  IntegerAttr narrowM = encoding.getMatmulNarrow_M();
  IntegerAttr narrowN = encoding.getMatmulNarrow_N();
  return narrowN && (!narrowM || narrowM.getInt() > narrowN.getInt());
}

} // namespace mlir::iree_compiler
