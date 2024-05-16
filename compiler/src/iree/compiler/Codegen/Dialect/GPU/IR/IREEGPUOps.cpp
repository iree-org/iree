// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::GPU {

//===----------------------------------------------------------------------===//
// ShuffleTensorOp
//===----------------------------------------------------------------------===//

LogicalResult ShuffleTensorOp::verify() {
  // Get the equivalent tensor type for the alloc to verify against.
  RankedTensorType destType = getDestType();
  Type allocElementType = destType.getElementType();
  RankedTensorType allocTensorType =
      RankedTensorType::get(destType.getShape(), allocElementType);

  // Verify source type against inferred type. Slice insertion and extraction
  // use the same verification logic.
  RankedTensorType expectedType = tensor::ExtractSliceOp::inferResultType(
      allocTensorType, getMixedOffsets(), getMixedSizes(), getMixedStrides());
  SliceVerificationResult result =
      isRankReducedType(expectedType, getSourceType());
  if (result != SliceVerificationResult::Success) {
    return emitError("Invalid source slice type");
  }

  if (allocElementType != getSourceType().getElementType() ||
      allocElementType != getType().getElementType()) {
    return emitError("Element type mismatch between source and destination");
  }
  return success();
}

LogicalResult ShuffleTensorOp::verifyRegions() {
  auto &region = getRegion();
  Block &block = region.front();
  if (block.getNumArguments() != 1) {
    return emitError("expected the block to have a single argument");
  }

  if (block.getArgumentTypes()[0] != getDestType()) {
    return emitError("expected block to have single argument type of")
           << getDestType();
  }

  // Ensure that the region yields an element of the right type.
  auto yieldOp = llvm::cast<GPU::YieldOp>(block.getTerminator());
  if (yieldOp.getValue().getType() != getResult().getType()) {
    return emitOpError("expected yield type to match result type");
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::GPU
