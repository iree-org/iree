// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingPatterns.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

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
  auto encoding =
      dyn_cast_or_null<SerializableAttr>(getResultType().getEncoding());
  if (!encoding) {
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
  // Verify encoding_dims count matches what the encoding expects.
  std::optional<int64_t> expectedDims = encoding.getNumDynamicEncodingDims();
  if (expectedDims.has_value() &&
      static_cast<int64_t>(getEncodingDims().size()) != expectedDims.value()) {
    return emitOpError() << "encoding expects " << expectedDims.value()
                         << " dynamic encoding dimension(s), but "
                         << getEncodingDims().size() << " provided";
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

FailureOr<Value> SetEncodingOp::reifyEncodingDim(OpBuilder &builder,
                                                 unsigned resultIndex,
                                                 unsigned dimIndex) {
  // SetEncodingOp has a single result, so resultIndex must be 0.
  assert(resultIndex == 0 && "SetEncodingOp has only one result");

  ValueRange encodingDims = getEncodingDims();
  if (dimIndex >= encodingDims.size()) {
    return failure();
  }

  return encodingDims[dimIndex];
}

//===----------------------------------------------------------------------===//
// encoding.unset_encoding
//===----------------------------------------------------------------------===//

LogicalResult UnsetEncodingOp::verify() {
  if (getResultType().getEncoding()) {
    return emitOpError(
        "result of unset_encoding op cannot have a tensor encoding");
  }
  auto encoding =
      dyn_cast_or_null<SerializableAttr>(getSourceType().getEncoding());
  if (!encoding) {
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
  unsigned requiredDynCount = getResultType().getNumDynamicDims();
  if (getResultDims().size() != requiredDynCount) {
    return emitOpError() << "result type set has " << requiredDynCount
                         << " dynamic dimensions but only "
                         << getResultDims().size()
                         << " dimension values are attached";
  }
  // Verify encoding_dims count matches what the encoding expects.
  std::optional<int64_t> expectedDims = encoding.getNumDynamicEncodingDims();
  if (expectedDims.has_value() &&
      static_cast<int64_t>(getEncodingDims().size()) != expectedDims.value()) {
    return emitOpError() << "encoding expects " << expectedDims.value()
                         << " dynamic encoding dimension(s), but "
                         << getEncodingDims().size() << " provided";
  }
  return success();
}

LogicalResult UnsetEncodingOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] =
      getMixedValues(getResultType().getShape(), getResultDims(), builder);
  return success();
}

//===----------------------------------------------------------------------===//
// encoding.dim
//===----------------------------------------------------------------------===//

LogicalResult DimOp::verify() {
  auto sourceType = cast<RankedTensorType>(getSource().getType());
  Attribute encoding = sourceType.getEncoding();

  if (!encoding) {
    return emitOpError() << "source tensor " << getSource()
                         << " must have an encoding";
  }

  auto serializableAttr = dyn_cast<SerializableAttr>(encoding);
  if (!serializableAttr) {
    return emitOpError() << "source tensor encoding " << encoding
                         << " must implement SerializableAttr";
  }

  // Check that the index is valid if we can determine the number of dims.
  std::optional<int64_t> numEncodingDims =
      serializableAttr.getNumDynamicEncodingDims();
  if (numEncodingDims) {
    int64_t index = getConstantIndex();
    if (index < 0 || index >= *numEncodingDims) {
      return emitOpError() << "encoding dimension index " << index
                           << " is out of bounds for encoding with "
                           << *numEncodingDims << " dimensions";
    }
  }
  return success();
}

void DimOp::build(OpBuilder &builder, OperationState &result, Value source,
                  int64_t index) {
  build(builder, result, builder.getIndexType(), source,
        builder.getIndexAttr(index));
}

void DimOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "enc_dim");
}

void DimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  populateEncodingDimReificationPatterns(results);
}

} // namespace mlir::iree_compiler::IREE::Encoding

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.cpp.inc" // IWYU pragma: keep
