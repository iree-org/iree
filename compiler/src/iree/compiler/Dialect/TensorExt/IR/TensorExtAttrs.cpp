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

} // namespace mlir::iree_compiler::IREE::TensorExt
