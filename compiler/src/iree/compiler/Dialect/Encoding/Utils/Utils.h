// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_ENCODING_UTILS_UTILS_H_
#define IREE_COMPILER_DIALECT_ENCODING_UTILS_UTILS_H_

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::Encoding {

/// Returns the encoding attribute from the type if there is an encoding that
/// implements SerializableAttr. Otherwise, returns null.
SerializableAttr getSerializableAttr(RankedTensorType type);

/// Returns the encoding attribute from the type if there is an encoding.
/// Otherwise, returns null.
EncodingAttr getEncodingAttr(RankedTensorType type);

/// Returns true if the type contains packed_storage attribute.
bool hasPackedStorageAttr(RankedTensorType type);

/// Returns the ContractionDimensions for the encoding user_indexing_maps.
FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(EncodingAttr encoding);

/// Returns the narrow dim in a given `encoding`, ceiled to a power of two. This
/// works by inspecting the `iteration_sizes` array attribute in the `encoding`.
/// If the `iteration_sizes` of one dimension (M or N) is smaller than the
/// other, then that's the narrow dimension, because the only way it would have
/// been set to be smaller in the first place, is if we previously flagged that
/// dimension as narrow. If the `iteration_sizes` of the M and N dimensions
/// agree, then neither is a narrow dimension and this returns a
/// default-constructed falsish value.
MatmulNarrowDim getPo2MatmulNarrowDim(EncodingAttr encoding);

/// Returns true if `encoding` represents a narrow-N matmul RESULT, e.g. the
/// result of a matvec.
bool isNarrowNResult(EncodingAttr encoding);

} // namespace mlir::iree_compiler::IREE::Encoding

#endif // IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGTYPES_H_
