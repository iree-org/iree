// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGINTERFACES_H_
#define IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGINTERFACES_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::Encoding {
class EncodingOp;

namespace detail {
LogicalResult verifyEncodingOpInterface(Operation *op);
}

/// Include the generated interface declarations.
#include "iree/compiler/Dialect/Encoding/IR/EncodingInterfaces.h.inc" // IWYU pragma: export

} // namespace mlir::iree_compiler::IREE::Encoding

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h.inc" // IWYU pragma: export

#endif // IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGINTERFACES_H_
