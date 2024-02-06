// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_INDEXING_IR_INDEXINGOPS_H_
#define IREE_COMPILER_DIALECT_INDEXING_IR_INDEXINGOPS_H_

#include "iree/compiler/Dialect/Indexing/IR/IndexingInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::iree_compiler {} // namespace mlir::iree_compiler

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Indexing/IR/IndexingOps.h.inc" // IWYU pragma: export

#endif // IREE_COMPILER_DIALECT_INDEXING_IR_INDEXINGOPS_H_
