// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
#define IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h.inc" // IWYU pragma: export
// #undef GET_OP_CLASSES
// clang-format on

#endif // IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
