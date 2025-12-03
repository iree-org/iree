// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.cpp.inc" // IWYU pragma: keep

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void PCFDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::PCF
