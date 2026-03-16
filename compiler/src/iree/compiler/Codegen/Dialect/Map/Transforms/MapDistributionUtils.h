// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_MAP_TRANSFORMS_MAPDISTRIBUTIONUTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_MAP_TRANSFORMS_MAPDISTRIBUTIONUTILS_H_

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

/// Maps a distributed vector dim to its original layout dim and leaf stride.
struct LeafDimInfo {
  int64_t origDim;
  int64_t dataStride;
};

/// Build the distributed dim -> (origDim, dataStride) mapping.
SmallVector<LeafDimInfo> getLeafDimMap(IREE::Map::PackLayoutAttr layout);

/// Compute the inverse of the layout's thread mapping, returning per-mode
/// data-space offsets for the given thread ID, i.e. the base offsets from
/// which the value mapping starts (see PackLayoutAttr for thread/value
/// mapping definitions).
///
/// The thread mapping is injective, so its inverse is well-defined: given a
/// thread ID, there is a unique set of thread coordinates that produced it.
///
/// Example: layout <((4, 2), (4, 8)) : ((1, 0), (0, 4))>
///   Thread map: (4, 8) : (1, 4)
///   Value map:  (2, 4) : (1, 8)
///   Result: offset_0 = (tid % 4) * 2,  offset_1 = (tid / 4) % 8
///
///   Each thread owns coordinates:
///     dim 0: offset_0 + v  for v in {0, 1}         (value map stride 1)
///     dim 1: offset_1 + v  for v in {0, 8, 16, 24} (value map stride 8)
SmallVector<Value> buildThreadOffsets(OpBuilder &b, Location loc,
                                      IREE::Map::PackLayoutAttr layout,
                                      Value threadId);

} // namespace mlir::iree_compiler
#endif // IREE_COMPILER_CODEGEN_DIALECT_MAP_TRANSFORMS_MAPDISTRIBUTIONUTILS_H_
