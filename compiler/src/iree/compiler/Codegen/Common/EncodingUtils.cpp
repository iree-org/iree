// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace iree_compiler {

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

MaterializeEncodingInfo getEncodingInfoForMatmul(EncodingUser user,
                                                 EncodingRole role,
                                                 TileMxNxK tileMxNxK) {
  // Start dim of the MxK (LHS), KxN (RHS), or MxN (RESULT) 2D matrix.
  int64_t matmulDimBase = isBatchMatmulEncodingUser(user) ? 1 : 0;

  MaterializeEncodingInfo encodingInfo;
  encodingInfo.innerDimsPos = {matmulDimBase, matmulDimBase + 1};
  switch (role) {
  case (EncodingRole::LHS): {
    encodingInfo.innerTileSizes = {tileMxNxK.M, tileMxNxK.K};
    break;
  }
  case (EncodingRole::RHS): {
    encodingInfo.innerTileSizes = {tileMxNxK.N, tileMxNxK.K};
    encodingInfo.innerDimsPos = {matmulDimBase + 1, matmulDimBase};
    encodingInfo.outerDimsPerm =
        llvm::to_vector(llvm::seq<int64_t>(0, matmulDimBase));
    encodingInfo.outerDimsPerm.push_back(matmulDimBase + 1);
    encodingInfo.outerDimsPerm.push_back(matmulDimBase);
    break;
  }
  case (EncodingRole::RESULT): {
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

} // namespace iree_compiler
} // namespace mlir
