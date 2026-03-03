// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_MIPS_IR_MIPSOPS_H_
#define IREE_COMPILER_DIALECT_MIPS_IR_MIPSOPS_H_

#include "iree/compiler/Dialect/MIPS/IR/MIPSDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/MIPS/IR/MIPSOps.h.inc" // IWYU pragma: export

// clang-format on

#endif // IREE_COMPILER_DIALECT_MIPS_IR_MIPSOPS_H_
