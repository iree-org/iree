// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_IR_UTILS_H_
#define IREE_COMPILER_DIALECT_HAL_IR_UTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"

namespace mlir::iree_compiler {

// Maps a resource type to the corresponding HAL memory types and buffer usage.
// This will fail if the resource type is not directly mappable to HAL bits.
// The bits set here are those that must be set for the buffer to be used as the
// buffer within the program with its defined resource lifetime.
LogicalResult
deriveRequiredResourceBufferBits(Location loc, IREE::Stream::Lifetime lifetime,
                                 IREE::HAL::MemoryTypeBitfield &memoryTypes,
                                 IREE::HAL::BufferUsageBitfield &bufferUsage);

// Maps a resource type to the corresponding HAL memory types and buffer usage.
// This will fail if the resource type is not directly mappable to HAL bits.
// The bits set here represent the superset of required and allowed bits and
// are useful for providing buffers back to users via the ABI that may need to
// be used for more than just what the internal program requires.
LogicalResult
deriveAllowedResourceBufferBits(Location loc, IREE::Stream::Lifetime lifetime,
                                IREE::HAL::MemoryTypeBitfield &memoryTypes,
                                IREE::HAL::BufferUsageBitfield &bufferUsage);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_HAL_IR_UTILS_H_
