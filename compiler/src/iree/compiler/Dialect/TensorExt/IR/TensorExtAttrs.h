// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTATTRS_H_
#define IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTATTRS_H_

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtAttrInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtEnums.h.inc"

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtAttrs.h.inc" // IWYU pragma: keep
// clang-format on

#endif // IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTATTRS_H_
