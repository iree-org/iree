// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGPLUS_IR_LINALGPLUSOPS_H_
#define IREE_COMPILER_DIALECT_LINALGPLUS_IR_LINALGPLUSOPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgPlus/IR/LinalgPlusOps.h.inc"  // IWYU pragma: export

#endif  // IREE_COMPILER_DIALECT_LINALGPLUS_IR_LINALGPLUSOPS_H_
