// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_
#define IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtInterfaces.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// clang-format off

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtEnums.h.inc" // IWYU pragma: export

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h.inc" // IWYU pragma: export

// clang-format on

#endif // IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_
