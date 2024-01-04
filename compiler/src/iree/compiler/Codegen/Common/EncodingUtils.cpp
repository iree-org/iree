// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"

namespace mlir::iree_compiler {

using IREE::LinalgExt::EncodingAttr;
using IREE::LinalgExt::EncodingRole;
using IREE::LinalgExt::EncodingUser;

/// For a given tensor type with an encoding, return the materialized
/// type to use for it. If no encoding is set, then return the tensor type
/// itself.
static RankedTensorType
getMaterializedType(RankedTensorType tensorType,
                    MaterializeEncodingFn materializeEncodingFn) {
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(tensorType);
  if (failed(materializeEncodingInfo)) {
    return dropEncoding(tensorType);
  }
  return tensor::PackOp::inferPackedType(
             getOriginalTypeWithEncoding(tensorType)
                 .clone(tensorType.getElementType()),
             materializeEncodingInfo->innerTileSizes,
             materializeEncodingInfo->innerDimsPos,
             materializeEncodingInfo->outerDimsPerm)
      .cast<RankedTensorType>();
}

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    MaterializeEncodingFn materializeEncodingFn)
    : materializeEncodingFn(materializeEncodingFn) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion(
      [materializeEncodingFn](RankedTensorType t) -> RankedTensorType {
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
    return cast<AffineMapAttr>(encoding.getIndexingMaps()[0]).getAffineMap();
  else if (role == EncodingRole::RHS)
    return cast<AffineMapAttr>(encoding.getIndexingMaps()[1]).getAffineMap();
  else
    return cast<AffineMapAttr>(encoding.getIndexingMaps()[2]).getAffineMap();
}

static FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(EncodingAttr encoding) {
  auto indexingMapsAttr = encoding.getIndexingMaps();
  SmallVector<AffineMap> indexingMaps = llvm::map_to_vector(
      indexingMapsAttr.getValue(), [](Attribute m) -> AffineMap {
        return cast<AffineMapAttr>(m).getAffineMap();
      });
  return linalg::inferContractionDims(indexingMaps);
}

std::optional<SmallVector<int64_t>>
getPermutationToCanonicalMatmulShape(EncodingAttr encoding) {
  FailureOr<linalg::ContractionDimensions> cDims =
      getEncodingContractionDims(encoding);
  if (failed(cDims)) {
    return std::nullopt;
  }
  SmallVector<int64_t> perm;
  EncodingRole role = encoding.getRole().getValue();
  EncodingUser user = encoding.getUser().getValue();
  AffineMap map = getMapForRole(encoding);
  auto getResultIndex = [map](int64_t idx) {
    return map.getResultPosition(getAffineDimExpr(idx, map.getContext()))
        .value();
  };
  // Add batch dim
  if (user == EncodingUser::BATCH_MATMUL) {
    perm.push_back(getResultIndex(cDims->batch[0]));
  }
  // Add M dim
  if (role != EncodingRole::RHS && cDims->m.size() == 1) {
    perm.push_back(getResultIndex(cDims->m[0]));
  }
  // Add K dim
  if (role != EncodingRole::RESULT) {
    perm.push_back(getResultIndex(cDims->k[0]));
  }
  // Add N dim
  if (role != EncodingRole::LHS && cDims->n.size() == 1) {
    perm.push_back(getResultIndex(cDims->n[0]));
  }
  return perm;
}

RankedTensorType getCanonicalMatmulTypeWithEncoding(RankedTensorType type) {
  auto encoding = getEncodingAttr(type);
  if (!encoding) {
    return type;
  }
  auto perm = getPermutationToCanonicalMatmulShape(encoding);
  if (!perm) {
    return type;
  }
  return RankedTensorType::get(applyPermutation(type.getShape(), perm.value()),
                               type.getElementType(), encoding);
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

bool isMatmulEncodingUser(EncodingUser user) {
  return user == EncodingUser::MATMUL;
}

bool isBatchMatmulEncodingUser(EncodingUser user) {
  return user == EncodingUser::BATCH_MATMUL;
}

int64_t getIntOrZero(IntegerAttr a) {
  return a == IntegerAttr() ? 0 : a.getInt();
}

bool isVecmatEncoding(EncodingAttr encoding) {
  auto cDims = getEncodingContractionDims(encoding);
  return !failed(cDims) && cDims->batch.size() == 0 && cDims->m.size() == 0 &&
         cDims->k.size() == 1 && cDims->n.size() == 1;
}

bool isMatvecEncoding(EncodingAttr encoding) {
  auto cDims = getEncodingContractionDims(encoding);
  return !failed(cDims) && cDims->batch.size() == 0 && cDims->m.size() == 1 &&
         cDims->k.size() == 1 && cDims->n.size() == 0;
}

bool isBatchVecmatEncoding(EncodingAttr encoding) {
  auto cDims = getEncodingContractionDims(encoding);
  return !failed(cDims) && cDims->batch.size() == 1 && cDims->m.size() == 0 &&
         cDims->k.size() == 1 && cDims->n.size() == 1;
}

bool isBatchMatvecEncoding(EncodingAttr encoding) {
  auto cDims = getEncodingContractionDims(encoding);
  return !failed(cDims) && cDims->batch.size() == 1 && cDims->m.size() == 1 &&
         cDims->k.size() == 1 && cDims->n.size() == 0;
}

bool isVectorEncoding(int64_t rank, EncodingUser user) {
  return rank == 1 || (isBatchMatmulEncodingUser(user) && rank == 2);
}

MaterializeEncodingInfo getEncodingInfoForMatmul(EncodingAttr encoding,
                                                 int64_t rank,
                                                 TileMxNxK tileMxNxK) {
  EncodingUser user = encoding.getUser().getValue();
  EncodingRole role = encoding.getRole().getValue();
  auto perm = getPermutationToCanonicalMatmulShape(encoding).value();
  bool isVector = isVectorEncoding(rank, user);
  bool isVecmatVector = (isVector && (isVecmatEncoding(encoding) ||
                                      isBatchVecmatEncoding(encoding)));
  bool isMatvecVector = (isVector && (isMatvecEncoding(encoding) ||
                                      isBatchMatvecEncoding(encoding)));
  // Start dim of the MxK (LHS), KxN (RHS), or MxN (RESULT) 2D matrix.
  int64_t matmulDimBase = isBatchMatmulEncodingUser(user) ? 1 : 0;
  MaterializeEncodingInfo encodingInfo;
  encodingInfo.innerDimsPos = {perm[matmulDimBase]};
  if (!isVector) {
    encodingInfo.innerDimsPos.push_back(perm[matmulDimBase + 1]);
    encodingInfo.outerDimsPerm =
        llvm::to_vector(llvm::seq<int64_t>(0, matmulDimBase));
    encodingInfo.outerDimsPerm.push_back(matmulDimBase);
    encodingInfo.outerDimsPerm.push_back(matmulDimBase + 1);
    encodingInfo.outerDimsPerm =
        applyPermutation(perm, encodingInfo.outerDimsPerm);
  } else if (matmulDimBase > 0) {
    encodingInfo.outerDimsPerm =
        llvm::to_vector(llvm::seq<int64_t>(0, matmulDimBase));
    encodingInfo.outerDimsPerm.push_back(matmulDimBase);
    encodingInfo.outerDimsPerm =
        applyPermutation(perm, encodingInfo.outerDimsPerm);
  }

  switch (role) {
  case (EncodingRole::LHS): {
    if (isVecmatVector) {
      encodingInfo.innerTileSizes = {tileMxNxK.K};
      break;
    }
    encodingInfo.innerTileSizes = {tileMxNxK.M, tileMxNxK.K};
    break;
  }
  case (EncodingRole::RHS): {
    if (isMatvecVector) {
      encodingInfo.innerTileSizes = {tileMxNxK.K};
      break;
    }
    encodingInfo.innerTileSizes = {tileMxNxK.N, tileMxNxK.K};
    encodingInfo.innerDimsPos = {perm[matmulDimBase + 1], perm[matmulDimBase]};
    encodingInfo.outerDimsPerm =
        llvm::to_vector(llvm::seq<int64_t>(0, matmulDimBase));
    encodingInfo.outerDimsPerm.push_back(matmulDimBase + 1);
    encodingInfo.outerDimsPerm.push_back(matmulDimBase);
    encodingInfo.outerDimsPerm =
        applyPermutation(perm, encodingInfo.outerDimsPerm);
    break;
  }
  case (EncodingRole::RESULT): {
    if (isVecmatVector) {
      encodingInfo.innerTileSizes = {tileMxNxK.N};
      break;
    }
    if (isMatvecVector) {
      encodingInfo.innerTileSizes = {tileMxNxK.M};
      break;
    }
    encodingInfo.innerTileSizes = {tileMxNxK.M, tileMxNxK.N};
    break;
  }
  default: {
    assert(false);
    return {};
  }
  }
  return encodingInfo;
}

} // namespace mlir::iree_compiler
