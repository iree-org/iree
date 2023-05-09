// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_TYPEUTILS_H_
#define IREE_COMPILER_UTILS_TYPEUTILS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {

/// Returns true if the given |bitwidth|, if appearing at runtime-kernel
/// interface, is less than a byte that should be tightly packed together.
bool needToPackSubByteElementBitWidth(unsigned bitwidth);
/// Returns true if the given |shapedType|, if appearing at runtime-kernel
/// interface, has sub-byte element types that should be tightly packed
/// together.
bool needToPackSubByteElements(RankedTensorType shapedType);

/// Legalizes the given |elementType| for storage.
///
/// In IREE, if compiling from the same source model, we control both the
/// runtime and kernel. For such cases, we perform tight packing for supported
/// sub-byte elements, and expand to the next power-of-two bitwidth for other
/// cases.
Type legalizeStorageElementType(Type elementType);

/// Emits IR with the given |builder| to calculate the total number of bytes
/// required for the given |shapedType| in storage. Returns the value for the
/// final count on success; returns nullptr on failure. Dynamic dimensions in
/// |shapedType| have corresponding values in |dynamicDims|.
Value calculateStorageElementCountInBytes(Location loc,
                                          RankedTensorType shapedType,
                                          ValueRange dynamicDims,
                                          OpBuilder &builder);

/// Emits IR with the given |builder| to calculate the byte offset for the
/// element at the given |linearizedIndex|.
Value calculateStorageElementOffsetInBytes(Location loc,
                                           RankedTensorType originalType,
                                           Value linearizedIndex,
                                           OpBuilder &builder);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_TYPEUTILS_H_
