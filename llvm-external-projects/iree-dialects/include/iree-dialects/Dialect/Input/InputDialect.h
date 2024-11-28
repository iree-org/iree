// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_INPUT_DIALECT_H
#define IREE_DIALECTS_DIALECT_INPUT_DIALECT_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// Include generated dialect code (this comment blocks clang-format from
// clobbering order).
#include "iree-dialects/Dialect/Input/InputDialect.h.inc"

// Include generated enums code (this comment blocks clang-format from
// clobbering order).
#include "iree-dialects/Dialect/Input/InputEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/Input/InputAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "iree-dialects/Dialect/Input/InputTypes.h.inc"

//===----------------------------------------------------------------------===//
// IREE ABI helpers for constructing buffer views
//===----------------------------------------------------------------------===//

namespace mlir::iree_compiler::IREE::Input {

// Returns a stable identifier for the MLIR element type or nullopt if the
// type is unsupported in the ABI.
std::optional<int32_t> getElementTypeValue(Type type);

// Returns a stable identifier for the MLIR encoding type or empty optional
// (opaque) if the type is unsupported in the ABI.
std::optional<int32_t> getEncodingTypeValue(Attribute attr);

} // namespace mlir::iree_compiler::IREE::Input

#endif // IREE_DIALECTS_DIALECT_INPUT_DIALECT_H
