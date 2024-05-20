// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_IREEGPUOPS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_IREEGPUOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h.inc" // IWYU pragma: export
// clang-format on

#endif // #ifndef IREE_COMPILER_CODEGEN_DIALECT_IREEGPUOPS_H_
