// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFATTRS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFATTRS_H_

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFDialect.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h.inc" // IWYU pragma: keep

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFATTRS_H_
