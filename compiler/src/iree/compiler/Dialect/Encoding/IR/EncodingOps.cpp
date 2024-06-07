// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"

namespace mlir::iree_compiler::IREE::Encoding {

//===----------------------------------------------------------------------===//
// encoding.set_encoding
//===----------------------------------------------------------------------===//

LogicalResult SetEncodingOp::verify() {
  // Source and the result have the same rank.
  if (getSourceType().getEncoding()) {
    return emitOpError(
        "source of set_encoding op cannot have a tensor encoding");
  }
  if (!isa_and_nonnull<EncodingAttr>(getResultType().getEncoding())) {
    return emitOpError(
        "result of set_encoding op expected to have a valid tensor encoding");
  }
  // The source and result must have the same rank.
  if (getResultType().getRank() != getSourceType().getRank()) {
    return emitOpError("cannot change the rank of the tensor");
  }
  if (failed(verifyCompatibleShape(getResultType(), getSourceType()))) {
    return emitOpError("expected to preserve the logical shape of the tensor");
  }
  return success();
}

LogicalResult SetEncodingOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] =
      tensor::getMixedSizes(builder, getLoc(), getSource());
  return success();
}

//===----------------------------------------------------------------------===//
// encoding.unset_encoding
//===----------------------------------------------------------------------===//

LogicalResult UnsetEncodingOp::verify() {
  if (getResultType().getEncoding()) {
    return emitOpError(
        "result of unset_encoding op cannot have a tensor encoding");
  }
  if (!isa_and_nonnull<EncodingAttr>(getSourceType().getEncoding())) {
    return emitOpError(
        "source of unset_encoding op expected to have a valid tensor encoding");
  }
  // The source and result must have the same rank.
  if (getResultType().getRank() != getSourceType().getRank()) {
    return emitOpError("cannot change the rank of the tensor");
  }
  if (failed(verifyCompatibleShape(getResultType(), getSourceType()))) {
    return emitOpError("expected to preserve the logical shape of the tensor");
  }
  return success();
}

LogicalResult UnsetEncodingOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] =
      tensor::getMixedSizes(builder, getLoc(), getSource());
  return success();
}

//===----------------------------------------------------------------------===//
// encoding.encoding
//===----------------------------------------------------------------------===//

EncodingAttr EncodingAttr::get(MLIRContext *ctx, EncodingRole role,
                               ArrayRef<Type> elemTypes, Type origType,
                               std::optional<int64_t> matmulNarrowM,
                               std::optional<int64_t> matmulNarrowN,
                               ArrayRef<AffineMap> maps,
                               std::optional<int64_t> maxPadding) {
  Builder b(ctx);
  auto optionalToAttr = [&](std::optional<int64_t> x) {
    return x ? b.getIndexAttr(*x) : IntegerAttr();
  };
  auto roleAttr = EncodingRoleAttr::get(ctx, role);
  auto origTypeAttr = origType ? TypeAttr::get(origType) : TypeAttr();
  return get(ctx, roleAttr, b.getTypeArrayAttr(elemTypes), origTypeAttr,
             optionalToAttr(matmulNarrowM), optionalToAttr(matmulNarrowN),
             b.getAffineMapArrayAttr(maps), optionalToAttr(maxPadding));
}

AffineMap EncodingAttr::getMapForRole() {
  EncodingRole role = getRole().getValue();
  switch (role) {
  case EncodingRole::LHS:
    return llvm::cast<AffineMapAttr>(getUserIndexingMaps()[0]).getAffineMap();
  case EncodingRole::RHS:
    return llvm::cast<AffineMapAttr>(getUserIndexingMaps()[1]).getAffineMap();
  case EncodingRole::RESULT:
    return llvm::cast<AffineMapAttr>(getUserIndexingMaps()[2]).getAffineMap();
  default:
    return AffineMap();
  }
}

unsigned EncodingAttr::mapDimToRoleIndex(int64_t dimPos) {
  AffineMap map = getMapForRole();
  auto idx = map.getResultPosition(getAffineDimExpr(dimPos, getContext()));
  assert(idx.has_value());
  return idx.value();
}

//===---------------------------------------------------------------------===//
// Encoding Dialect Helpers
//===---------------------------------------------------------------------===//

EncodingAttr getEncodingAttr(RankedTensorType type) {
  return dyn_cast_or_null<EncodingAttr>(type.getEncoding());
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

} // namespace mlir::iree_compiler::IREE::Encoding

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.cpp.inc" // IWYU pragma: keep
// clang-format: on
