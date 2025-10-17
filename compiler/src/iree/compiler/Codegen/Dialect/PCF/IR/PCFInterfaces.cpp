// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// Interface extra declarations
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrInterfaces.cpp.inc" // IWYU pragma: keep
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOpInterfaces.cpp.inc" // IWYU pragma: keep

} // namespace mlir::iree_compiler::IREE::PCF
