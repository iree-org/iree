// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFTYPES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFTYPES_H_

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFDialect.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h.inc" // IWYU pragma: keep

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFTYPES_H_
