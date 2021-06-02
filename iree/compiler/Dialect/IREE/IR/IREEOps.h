// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_IR_IREEOPS_H_
#define IREE_COMPILER_DIALECT_IREE_IR_IREEOPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h.inc"  // IWYU pragma: export

#endif  // IREE_COMPILER_DIALECT_IREE_IR_IREEOPS_H_
