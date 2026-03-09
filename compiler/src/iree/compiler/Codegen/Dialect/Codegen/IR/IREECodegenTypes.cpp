// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/DialectImplementation.h"

// clang-format off
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.cpp.inc" // IWYU pragma: export
// clang-format on

namespace mlir::iree_compiler::IREE::Codegen {

int64_t TileSwizzle::getExpandedSize() const {
  int64_t totalExpandedDims = 0;
  for (const ExpandShapeDimVectorType &expandDims : expandShape) {
    totalExpandedDims += static_cast<int64_t>(expandDims.size());
  }
  return totalExpandedDims;
}

LogicalResult
TileSwizzle::verify(function_ref<InFlightDiagnostic()> emitError) const {
  int64_t totalExpandedDims = getExpandedSize();

  // The permutation size must match the total expanded dimensions.
  if (static_cast<int64_t>(permutation.size()) != totalExpandedDims) {
    return emitError() << "swizzle permutation size (" << permutation.size()
                       << ") does not match total expanded dimensions ("
                       << totalExpandedDims << ")";
  }

  // Check that permutation is valid.
  if (!isPermutationVector(permutation)) {
    return emitError() << "swizzle permutation is not a valid permutation";
  }

  return success();
}

// Returns the index of the first destination dimension corresponding to the
// given source dimension `srcIdx`.
static size_t expandedDimIdx(const TileSwizzle::ExpandShapeType &expandShape,
                             size_t srcIdx) {
  size_t dstIdx = 0;
  for (size_t i = 0; i < srcIdx; ++i) {
    dstIdx += expandShape[i].size();
  }
  return dstIdx;
}

void expand(TileSwizzle &swizzle, size_t srcIdx, TileSwizzle::Dim dim) {
  int64_t dstIdx = expandedDimIdx(swizzle.expandShape, srcIdx);
  swizzle.expandShape[srcIdx].insert(swizzle.expandShape[srcIdx].begin(), dim);
  for (int64_t &p : swizzle.permutation) {
    p += (p >= dstIdx);
  }
  swizzle.permutation.insert(swizzle.permutation.begin(), dstIdx);
}

SmallVector<int64_t>
sliceSwizzledShape(const TileSwizzle &swizzle,
                   llvm::function_ref<bool(TileSwizzle::Dim)> predicate) {
  SmallVector<int64_t> shape;
  for (TileSwizzle::ExpandShapeDimVectorType e : swizzle.expandShape) {
    for (TileSwizzle::Dim d : e) {
      shape.push_back(predicate(d) ? d.size : 1);
    }
  }
  applyPermutationToVector(shape, swizzle.permutation);
  return shape;
}

} // namespace mlir::iree_compiler::IREE::Codegen
