// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE ops for working with buffers and buffer views.
// These are used by common transforms between the sequencer and interpreter and
// allow us to share some of the common lowering passes from other dialects.

#ifndef INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_OPS_H_
#define INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_OPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CallInterfaces.h"

#define GET_OP_CLASSES
#include "iree_tf_compiler/dialect/tf_strings/ir/ops.h.inc"
#undef GET_OP_CLASSES

#endif  // INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_OPS_H_
