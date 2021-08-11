// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREE_IREE_OPS_H
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREE_IREE_OPS_H

#include "iree-dialects/Dialect/IREE/IREEDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/IREE/IREEOps.h.inc"

#endif  // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREE_IREE_OPS_H
