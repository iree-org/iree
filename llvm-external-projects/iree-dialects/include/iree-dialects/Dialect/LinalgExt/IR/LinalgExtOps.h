// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
// #include "mlir/Interfaces/AggregatedOpInterface.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

// clang-format off

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtEnums.h.inc" // IWYU pragma: export

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtAttrs.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h.inc" // IWYU pragma: export

// clang-format on

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H_
