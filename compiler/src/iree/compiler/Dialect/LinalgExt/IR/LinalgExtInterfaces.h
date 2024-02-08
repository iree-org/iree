// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACES_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACES_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {
class LinalgExtOp;

namespace detail {
LogicalResult verifyLinalgExtOpInterface(Operation *op);
}

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h.inc" // IWYU pragma: export

/// Include the generated interface declarations.
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.h.inc" // IWYU pragma: export

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACES_H_
