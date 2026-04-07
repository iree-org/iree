// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_SLICEUTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_SLICEUTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"

namespace mlir::iree_compiler {

/// Tracks iteration dimension equivalence across a set of operations.
///
/// Given a set of operations, assigns a "global dimension index" to each loop
/// dimension, then unifies indices that are linked through producer-consumer
/// SSA relationships and indexing maps. Dimensions sharing the same global
/// index are considered equivalent and must be tiled together.
class IterationDimTracker {
public:
  explicit IterationDimTracker(ArrayRef<Operation *> operations);

  /// Returns true if the given global dimension index is present across all
  /// operations.
  bool presentInAllOps(int64_t globalDimIdx) const;

  /// Returns all global dimension indices associated with the given operation.
  ArrayRef<int64_t> getAllGlobalDimIdx(Operation *op) const;

  /// Returns the global dimension index corresponding to the given local loop
  /// dimension `pos` for the specified operation.
  int64_t getGlobalDimIdx(Operation *op, int64_t pos) const;

  /// Returns the total number of unique global dimension indices.
  int64_t getTotalLoopNum() const { return totalLoopNum; }

private:
  /// Builds and unifies dimension index mappings for all operations,
  /// using producer–consumer SSA value relationships.
  void buildDimMapping();

  /// Propagation helpers for buildDimMapping.
  void propagateOnIndexingMapOp(
      IndexingMapOpInterface indexingMapOp,
      llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
      llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueToGlobalDimMaps);
  void propagateOnPackUnpackOp(
      Operation *op, llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
      llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueToGlobalDimMaps,
      int64_t numLoops);
  void propagateOnUnknownOp(
      Operation *op, llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
      llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueToGlobalDimMaps,
      int64_t numLoops);

  SmallVector<Operation *> operations;
  // Tracks the total number of unique loop dimensions among the given set of
  // operations.
  int64_t totalLoopNum = 0;
  // For each compute operation, maps its local loop dimension index to the
  // global index. Operation -> (local dim index -> global dim
  // index)
  llvm::SmallDenseMap<Operation *, SmallVector<int64_t>>
      operationToGlobalDimMaps;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_SLICEUTILS_H_
