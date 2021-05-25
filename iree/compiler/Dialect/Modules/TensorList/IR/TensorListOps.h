// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLISTOPS_H_
#define IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLISTOPS_H_

#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.h.inc"

#endif  // IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLISTOPS_H_
