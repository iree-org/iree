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

/// Maps a distributed vector dim to its original TV layout dim and leaf stride.
struct LeafDimInfo {
  int64_t origDim;
  int64_t dataStride;
};

/// Build the distributed dim -> (origDim, dataStride) mapping.
SmallVector<LeafDimInfo> getLeafDimMap(IREE::Map::PackLayoutAttr layout);

/// Compute per-original-dim thread offsets as dynamic Values.
///
/// For each dim, extracts thread leaves (stride > 0) and computes:
///   offset_d = sum_i ((tid / threadStride_i) % size_i) * dataStride_i
SmallVector<Value> buildThreadOffsets(OpBuilder &b, Location loc,
                                      IREE::Map::PackLayoutAttr layout,
                                      Value threadId);

} // namespace mlir::iree_compiler
#endif // IREE_COMPILER_CODEGEN_DIALECT_MAP_TRANSFORMS_MAPDISTRIBUTIONUTILS_H_
