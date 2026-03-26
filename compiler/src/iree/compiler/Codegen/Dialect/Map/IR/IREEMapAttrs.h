// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_MAP_IR_IREEMAPATTRS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_MAP_IR_IREEMAPATTRS_H_

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapDialect.h"
#include "mlir/IR/BuiltinAttributes.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h.inc"
// clang-format on

#endif // IREE_COMPILER_CODEGEN_DIALECT_MAP_IR_IREEMAPATTRS_H_
