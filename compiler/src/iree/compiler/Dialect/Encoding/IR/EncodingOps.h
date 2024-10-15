// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGOPS_H_
#define IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGOPS_H_

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h.inc" // IWYU pragma: export
#undef GET_OP_CLASSES
// clang-format on

#endif // IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGOPS_H_
