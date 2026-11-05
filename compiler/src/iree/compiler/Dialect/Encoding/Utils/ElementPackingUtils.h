// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_ENCODING_UTILS_ELEMENTPACKINGUTILS_H_
#define IREE_COMPILER_DIALECT_ENCODING_UTILS_ELEMENTPACKINGUTILS_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::iree_compiler {

/// Returns true if the given |shapedType|, if appearing at runtime-kernel
/// interface, has sub-byte element types that should be tightly packed
/// together.
bool needToPackSubByteElements(RankedTensorType shapedType);

/// Legalizes the underlying element type of |shapedType| for storage, taking
/// into account the encoding attributes of |shapedType|, if present.
///
/// In IREE, if compiling from the same source model, we control both the
/// runtime and kernel. For such cases, we perform tight packing for supported
/// sub-byte elements, and expand to the next power-of-two bit width for other
/// cases.
Type legalizeStorageElementType(RankedTensorType shapedType);

/// Emits IR with the given |builder| to calculate the total number of bytes
/// required for the given |shapedType| in storage. Returns the value for the
/// final count on success; returns nullptr on failure. Dynamic dimensions in
/// |shapedType| have corresponding values in |dynamicDims|.
Value calculateStorageElementCountInBytes(Location loc,
                                          RankedTensorType shapedType,
                                          ValueRange dynamicDims,
                                          OpBuilder &builder);

/// Maps logical element indices to byte offsets, using the default dense layout
/// for unencoded tensors. Returns null for unsupported encodings.
AffineMap getStorageElementOffsetMap(RankedTensorType type);

/// Emits the byte offset of the element at `indices`. `dynamicDims` supplies
/// the dynamic extents in tensor dimension order. Returns null without emitting
/// IR if the storage layout is unsupported.
Value calculateStorageElementOffsetInBytes(Location loc, RankedTensorType type,
                                           ValueRange dynamicDims,
                                           ValueRange indices,
                                           OpBuilder &builder);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_ENCODING_UTILS_ELEMENTPACKINGUTILS_H_
