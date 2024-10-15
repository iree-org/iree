// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGTYPES_H_
#define IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGTYPES_H_

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
#undef GET_ATTRDEF_CLASSES
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h.inc" // IWYU pragma: export
#undef GET_TYPEDEF_CLASSES
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

/// Assign a name to operand indices for clarity
const int64_t MATMUL_LHS = 0;
const int64_t MATMUL_RHS = 1;
const int64_t MATMUL_RESULT = 2;

/// Convert operand index to strings for printing
std::string stringifyOperandIndex(IntegerAttr);

/// Designates a dimension in a matmul (either the M or the N dimension) as
/// being "narrow", i.e. small enough that we bother lowering the amount of
/// padding along that dimension compared to how padding we apply to
/// sufficiently large dimensions.
struct MatmulNarrowDim {
  // Enumerates dimensions of a matmul that may be labelled as narrow.
  enum class Dim {
    None,
    M,
    N,
  };
  Dim dim = Dim::None; // Which dimension is designated by *this.
  int64_t size = 0;    // Size of the designated dimension, or kDynamic.

  explicit operator bool() const { return dim != Dim::None; }
  bool isM() const { return dim == Dim::M; }
  bool isN() const { return dim == Dim::N; }
};

/// Returns the narrow dim in a given `linalgOp`, with respect to the given
/// `narrowThreshold` below which a dimension is eligible to be considered
/// narrow. If both M and N are narrow, M is returned. If neither M nor N are
/// narrow, this returns a default-constructed falsish value.
MatmulNarrowDim getMatmulNarrowDim(linalg::LinalgOp linalgOp,
                                   int narrowThreshold);

/// Returns the narrow dim in a given `encoding`. This works by inspecting
/// the `round_dims_to` array attribute in the `encoding`. If the
/// `round_dims_to` of one dimension (M or N) is smaller than the other, then
/// that's the narrow dimension, because the only way it would have been set
/// to be smaller in the first place, is if we previously flagged that dimension
/// as narrow. If the `round_dims_to` of the M and N dimensions agree, then
/// neither is a narrow dimension and this returns a default-constructed falsish
/// value.
MatmulNarrowDim getMatmulNarrowDim(EncodingAttr encoding);

} // namespace mlir::iree_compiler::IREE::Encoding

#endif // IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGTYPES_H_
