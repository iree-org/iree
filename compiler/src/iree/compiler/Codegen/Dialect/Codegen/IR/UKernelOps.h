// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_UKERNELOPS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_UKERNELOPS_H_

#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h.inc" // IWYU pragma: export
// clang-format on

#endif // #ifndef IREE_COMPILER_CODEGEN_DIALECT_UKERNELOPS_H_
