// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- MarkerUtils.h - Methods for manipulating markers on Linalg op ------===//
//
// Method that set markers on Linalg operations that determine which processor
// heirarchy to use for partitioning
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_CODEGENUTILS_MARKERUTILS_H_
#define IREE_COMPILER_CODEGEN_CODEGENUTILS_MARKERUTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler {

/// Marker to denote that a linalg operation has been partitioned to
/// workgroups and tiled along reduction dimennsions.
StringRef getWorkgroupKTiledMarker();

/// Marker to denote that a linalg operation has been partitioned to
/// workgroups and operands promoted to scratchspace memory.
StringRef getWorkgroupMemoryMarker();

/// Marker to denote that a linalg operation on workgoups has been partitioned
/// to workgroups L1 tiles.
StringRef getWorkgroupL1TileMarker();

/// Marker for copy operations that are moving data from StorageClass to
/// Workgroup memory.
StringRef getCopyToWorkgroupMemoryMarker();

/// Marker for tiling linalg reduction dimensions.
StringRef getTileReductionMarker();

/// Marker for operations that are going to be vectorized.
StringRef getVectorizeMarker();

/// Marker for tagging an operation for deletion. Tile and fuse pattern does
/// not delete the original operation to not invalidate the
/// `linalg::LinalgDependenceGraph` data structure. Instead it is marked with
/// a marker that can be used later to delete these operations.
StringRef getDeleteMarker();

/// Returns the marker set on an operation, or "" if no marker is set.
StringRef getMarkerOrNull(Operation *op);

/// Returns true if an operation has the specified `marker`. When `marker` is
/// empty, returns true if the operation has any marker.
bool hasMarker(Operation *, ArrayRef<StringRef> markers = {});

/// Sets a given marker on an operation.
void setMarker(Operation *, StringRef);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_CODEGENUTILS_MARKERUTILS_H_
