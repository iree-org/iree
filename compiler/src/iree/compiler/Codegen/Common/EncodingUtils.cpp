// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"

namespace mlir::iree_compiler {

using IREE::LinalgExt::EncodingAttr;
using IREE::LinalgExt::EncodingRole;
using IREE::LinalgExt::getEncodingAttr;
using IREE::LinalgExt::getEncodingContractionDims;

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
        encoding.mapDimToRoleIndex(cDims->batch[0]));
  }
  if (role != EncodingRole::RHS && !cDims->m.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        encoding.mapDimToRoleIndex(cDims->m[0]));
    encodingInfo.innerDimsPos.push_back(
        encoding.mapDimToRoleIndex(cDims->m[0]));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.M);
  }
  if (role != EncodingRole::LHS && !cDims->n.empty()) {
    encodingInfo.outerDimsPerm.push_back(
        encoding.mapDimToRoleIndex(cDims->n[0]));
    encodingInfo.innerDimsPos.push_back(
        encoding.mapDimToRoleIndex(cDims->n[0]));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.N);
  }
  if (role != EncodingRole::RESULT) {
    encodingInfo.outerDimsPerm.push_back(
        encoding.mapDimToRoleIndex(cDims->k[0]));
    encodingInfo.innerDimsPos.push_back(
        encoding.mapDimToRoleIndex(cDims->k[0]));
    encodingInfo.innerTileSizes.push_back(tileMxNxK.K);
  }
  return encodingInfo;
}

} // namespace mlir::iree_compiler
