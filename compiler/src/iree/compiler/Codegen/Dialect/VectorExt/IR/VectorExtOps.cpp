// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

using VectorValue = TypedValue<VectorType>;

//===----------------------------------------------------------------------===//
// LayoutConflictResolutionOp
//===----------------------------------------------------------------------===//

// Validate that the layout has the same shape as the input.
LogicalResult ToLayoutOp::verify() {
  return getLayout().isValidLayout(getInput().getType(), getLoc());
}

// to_simd -> to_simt
OpFoldResult ToSIMDOp::fold(FoldAdaptor) {
  if (auto simtOp = getOperand().getDefiningOp<ToSIMTOp>()) {
    return simtOp.getOperand();
  }
  return {};
}

// to_simt -> to_simd
OpFoldResult ToSIMTOp::fold(FoldAdaptor) {
  if (auto simdOp = getOperand().getDefiningOp<ToSIMDOp>()) {
    return simdOp.getOperand();
  }
  return {};
}

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format on
