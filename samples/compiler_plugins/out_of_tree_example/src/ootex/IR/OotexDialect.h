// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OOTEX_IR_OOTEXDIALECT_H_
#define OOTEX_IR_OOTEXDIALECT_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "ootex/IR/OotexOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "ootex/IR/OotexOps.h.inc"
// clang-format on

#endif  // OOTEX_IR_OOTEXDIALECT_H_
