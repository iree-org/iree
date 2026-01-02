// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtAttrs.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtAttrs.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::TensorExt {

void IREETensorExtDialect::initializeAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

SparseIterationDimsAttr getSparseIterationDimsAttr(Operation *op) {
  static constexpr StringLiteral kAttrName =
      "iree_tensor_ext.sparse_iteration_dims";
  if (auto attr = op->getAttrOfType<SparseIterationDimsAttr>(kAttrName)) {
    return attr;
  }
  return SparseIterationDimsAttr();
}

//===---------------------------------------------------------------------===//
// iree_tensor_ext.ragged_shape
//===---------------------------------------------------------------------===//

SmallVector<int64_t> RaggedShapeAttr::getSparseDimensions() const {
  int64_t raggedRow = getRaggedRow();
  return {raggedRow, raggedRow + 1};
}

/// MemRefLayoutAttrInterface methods.

AffineMap RaggedShapeAttr::getAffineMap() const { return AffineMap(); }

bool RaggedShapeAttr::isIdentity() const { return false; }

LogicalResult RaggedShapeAttr::verifyLayout(
    ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError) const {
  SmallVector<int64_t> sparseDimensions = getSparseDimensions();
  if (llvm::any_of(sparseDimensions,
                   [&](int64_t dim) { return dim >= shape.size(); })) {
    return emitError() << "sparse dimensions specified are greater than the "
                          "rank of the shaped type";
  }
  return success();
}

static SmallVector<int64_t>
getDefaultStrides(ArrayRef<int64_t> shape,
                  std::optional<int64_t> forceDynamicFrom = std::nullopt) {
  SmallVector<int64_t> strides;
  strides.resize(shape.size(), 1);
  if (shape.size() <= 1) {
    return strides;
  }
  for (int index = shape.size() - 2; index >= 0; --index) {
    if ((forceDynamicFrom && index < forceDynamicFrom.value()) ||
        ShapedType::isDynamic(shape[index + 1]) ||
        ShapedType::isDynamic(strides[index + 1])) {
      strides[index] = ShapedType::kDynamic;
      continue;
    }
    strides[index] = shape[index + 1] * strides[index + 1];
  }
  return strides;
}

LogicalResult
RaggedShapeAttr::getStridesAndOffset(ArrayRef<int64_t> shape,
                                     SmallVectorImpl<int64_t> &strides,
                                     int64_t &offset) const {
  // This should only be required on the first "subview" of the ragged tensor.
  // So we can assume that there are no strides to begin with and set all
  // strides starting from the least significant sparse dimension to dynamic.
  strides.clear();
  SmallVector<int64_t> sparseDims = getSparseDimensions();
  std::optional<int64_t> forceDynamicFrom = std::nullopt;
  for (auto dim : sparseDims) {
    if (!forceDynamicFrom) {
      forceDynamicFrom = dim;
      continue;
    }
    forceDynamicFrom = std::max(forceDynamicFrom.value(), dim);
  }
  strides = getDefaultStrides(shape, forceDynamicFrom);
  offset = 0;
  return success();
}

} // namespace mlir::iree_compiler::IREE::TensorExt
