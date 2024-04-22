// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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
  Type allocElementType = getSharedAllocType().getElementType();
  RankedTensorType allocTensorType = getSharedAllocType();

  // Verify source type against inferred type. Slice insertion and extraction
  // use the same verification logic.
  RankedTensorType expectedType = tensor::ExtractSliceOp::inferResultType(
      allocTensorType, getMixedSourceOffsets(), getMixedSourceSizes(),
      getMixedSourceStrides());
  SliceVerificationResult result =
      isRankReducedType(expectedType, getSourceType());
  if (result != SliceVerificationResult::Success) {
    return emitError("Invalid source slice type");
  }

  // Do the same for the resulting tensor type
  expectedType = tensor::ExtractSliceOp::inferResultType(
      allocTensorType, getMixedResultOffsets(), getMixedResultSizes(),
      getMixedResultStrides());
  result = isRankReducedType(expectedType, getType());
  if (result != SliceVerificationResult::Success) {
    return emitError("Invalid result slice type");
  }

  if (allocElementType != getSourceType().getElementType() ||
      allocElementType != getType().getElementType()) {
    return emitError(
        "Element type mismatch between source, allocation, and result");
  }

  // TODO: Verification of the allocation size in the static case.
  return success();
}

} // namespace mlir::iree_compiler::IREE::GPU
