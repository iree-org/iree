// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGOPS_H_
#define IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGOPS_H_

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

#include "iree/compiler/Dialect/Encoding/IR/EncodingEnums.h.inc" // IWYU pragma: export

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Encoding/IR/EncodingAttrs.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h.inc" // IWYU pragma: export

// clang-format on

//===---------------------------------------------------------------------===//
// Encoding Dialect Helpers
//===---------------------------------------------------------------------===//

namespace mlir::iree_compiler::IREE::Encoding {

/// Returns the encoding attribute from the type if there is an encoding.
/// Otherwise, returns null.
EncodingAttr getEncodingAttr(RankedTensorType type);

/// Returns the ContractionDimensions for the encoding user_indexing_maps.
FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(EncodingAttr encoding);

// Assign a name to operand indices for clarity
const int64_t MATMUL_LHS = 0;
const int64_t MATMUL_RHS = 1;
const int64_t MATMUL_RESULT = 2;
/// Convert operand index to strings for printing
std::string stringifyOperandIndex(IntegerAttr);

} // namespace mlir::iree_compiler::IREE::Encoding

#endif // IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGOPS_H_
