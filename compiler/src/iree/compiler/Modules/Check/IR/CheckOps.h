// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_CHECK_IR_CHECK_OPS_H_
#define IREE_COMPILER_MODULES_CHECK_IR_CHECK_OPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "iree/compiler/Modules/Check/IR/CheckOps.h.inc" // IWYU pragma: export

#endif // IREE_COMPILER_MODULES_CHECK_IR_CHECK_OPS_H_
