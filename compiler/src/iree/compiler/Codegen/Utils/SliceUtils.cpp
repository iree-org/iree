// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/SliceUtils.h"

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// IterationDimTracker
//===----------------------------------------------------------------------===//

IterationDimTracker::IterationDimTracker(ArrayRef<Operation *> operations)
    : operations(operations.begin(), operations.end()) {
  // Ensure operations are processed in topological order.
  mlir::computeTopologicalSorting(this->operations);
  buildDimMapping();
}

bool IterationDimTracker::presentInAllOps(int64_t globalDimIdx) const {
  for ([[maybe_unused]] auto &[_, dims] : operationToGlobalDimMaps) {
    if (!llvm::is_contained(dims, globalDimIdx)) {
      return false;
    }
  }
  return true;
}

ArrayRef<int64_t> IterationDimTracker::getAllGlobalDimIdx(Operation *op) const {
  auto it = operationToGlobalDimMaps.find(op);
  assert(it != operationToGlobalDimMaps.end() &&
         "Operation not found in DimTracker");
  return it->second;
}

int64_t IterationDimTracker::getGlobalDimIdx(Operation *op, int64_t pos) const {
  ArrayRef<int64_t> globalDims = getAllGlobalDimIdx(op);
  return globalDims[pos];
}

//===----------------------------------------------------------------------===//
// IterationDimTracker propagation helpers
//===----------------------------------------------------------------------===//

/// Ties loop dimensions together based on the operation's indexing maps,
/// considering only simple result dimension expressions (`AffineDimExpr`).
///
/// Complex expressions (e.g., `affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2,
/// d1 * 3 + d3)>`) are ignored because they fall outside the "loop dimension"
/// concept. Such expressions describe how indices are computed within the
/// innermost loop body, but they do not directly identify which loop
/// dimensions correspond or should be tied.
void IterationDimTracker::propagateOnIndexingMapOp(
    IndexingMapOpInterface indexingMapOp,
    llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
    llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueToGlobalDimMaps) {
  Operation *op = indexingMapOp.getOperation();
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();
    // Skip operands that have no known mapping from their producers.
    if (!valueToGlobalDimMaps.contains(value)) {
      continue;
    }
    AffineMap map = indexingMapOp.getMatchingIndexingMap(&operand);
    for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
      // Stop if the current dimension exceeds the number of mapped ones.
      if (dim >= valueToGlobalDimMaps[value].size()) {
        break;
      }
      // Skip on complex expressions.
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        continue;
      }
      int64_t pos = dimExpr.getPosition();
      // Unify the dimension index between the producer and the current op.
      indicesEquivalence.unionSets(valueToGlobalDimMaps[value][dim],
                                   operationToGlobalDimMaps[op][pos]);
    }
  }
  // Propagate to results.
  auto dsOp = cast<DestinationStyleOpInterface>(op);
  for (OpResult result : op->getResults()) {
    OpOperand *operand = dsOp.getTiedOpOperand(result);
    AffineMap map = indexingMapOp.getMatchingIndexingMap(operand);
    for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
      // Skip on complex expressions.
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        continue;
      }
      int64_t pos = dimExpr.getPosition();
      valueToGlobalDimMaps[result].push_back(operationToGlobalDimMaps[op][pos]);
    }
  }
}

/// Ties the dimensions of pack and unpack operations with their operands in
/// the outer (unpacked) dimensions.
void IterationDimTracker::propagateOnPackUnpackOp(
    Operation *op, llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
    llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueToGlobalDimMaps,
    int64_t numLoops) {
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();
    if (!valueToGlobalDimMaps.contains(value)) {
      continue;
    }
    int64_t rank = cast<ShapedType>(value.getType()).getRank();
    int64_t outDimSize = std::min(rank, numLoops);
    for (int64_t i = 0; i < outDimSize; ++i) {
      indicesEquivalence.unionSets(valueToGlobalDimMaps[value][i],
                                   operationToGlobalDimMaps[op][i]);
    }
  }
  // Propagate to results.
  for (Value result : op->getResults()) {
    valueToGlobalDimMaps[result] = operationToGlobalDimMaps[op];
  }
}

/// Ties the dimensions of operations with their operands, if the operand rank
/// matches the operation's loop count.
void IterationDimTracker::propagateOnUnknownOp(
    Operation *op, llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
    llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueToGlobalDimMaps,
    int64_t numLoops) {
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();
    if (!valueToGlobalDimMaps.contains(value) ||
        numLoops != cast<ShapedType>(value.getType()).getRank()) {
      continue;
    }
    for (int64_t i = 0; i < numLoops; ++i) {
      indicesEquivalence.unionSets(valueToGlobalDimMaps[value][i],
                                   operationToGlobalDimMaps[op][i]);
    }
  }
  // Propagate to results.
  for (Value result : op->getResults()) {
    if (numLoops == cast<ShapedType>(result.getType()).getRank()) {
      valueToGlobalDimMaps[result] = operationToGlobalDimMaps[op];
    }
  }
}

//===----------------------------------------------------------------------===//
// IterationDimTracker::buildDimMapping
//===----------------------------------------------------------------------===//

void IterationDimTracker::buildDimMapping() {
  // Tracks equivalent global dimension indices.
  llvm::EquivalenceClasses<int64_t> indicesEquivalence;
  // For each SSA value, maps its local dimension index to a global index.
  // Value -> (local dim index -> global dim index)
  llvm::SmallDenseMap<Value, SmallVector<int64_t>> valueToGlobalDimMaps;

  for (Operation *op : operations) {
    auto tilingOp = cast<TilingInterface>(op);
    int64_t numLoops = tilingOp.getLoopIteratorTypes().size();
    // Unconditionally assign new global indices, to be unified later.
    for (int64_t i = 0; i < numLoops; ++i) {
      int64_t globalIndex = totalLoopNum++;
      indicesEquivalence.insert(globalIndex);
      operationToGlobalDimMaps[op].push_back(globalIndex);
    }
    // The assigned global dimension indices are now unified based on
    // producer–consumer SSA value relationships:
    // - For operations implementing `IndexingMapOpInterface`, unify
    // dimensions by iterating over their indexing maps.
    // - For pack/unpack operations, use an identity mapping, since tiling
    // applies to the outer (unpacked) dimensions.
    // - For all other (unknown) operations, assume an identity mapping for
    // any value whose rank matches the operation's loop count.
    TypeSwitch<Operation *>(op)
        .Case([&](IndexingMapOpInterface op) {
          propagateOnIndexingMapOp(op, indicesEquivalence,
                                   valueToGlobalDimMaps);
        })
        .Case<linalg::PackOp, linalg::UnPackOp>([&](auto op) {
          propagateOnPackUnpackOp(op, indicesEquivalence, valueToGlobalDimMaps,
                                  numLoops);
        })
        .Default([&](auto op) {
          propagateOnUnknownOp(op, indicesEquivalence, valueToGlobalDimMaps,
                               numLoops);
        });
  }

  // Remap the global dimension indices in two steps:
  // 1. Assign the same temporary index to all equivalent dimensions.
  // 2. Convert these temporary indices to a compact, zero-based range.
  auto applyReplaceMap = [&](llvm::SmallDenseMap<int64_t, int64_t> &map) {
    for (auto &opEntry : operationToGlobalDimMaps) {
      for (auto &dim : opEntry.second) {
        dim = map.lookup(dim);
      }
    }
  };
  llvm::SmallDenseMap<int64_t, int64_t> replaceMap0, replaceMap1;
  int64_t tempDimIndex = totalLoopNum;
  totalLoopNum = 0;
  for (auto it = indicesEquivalence.begin(); it != indicesEquivalence.end();
       ++it) {
    if (!(*it)->isLeader()) {
      continue;
    }
    for (auto mit = indicesEquivalence.member_begin(**it);
         mit != indicesEquivalence.member_end(); ++mit) {
      replaceMap0[*mit] = tempDimIndex;
    }
    replaceMap1[tempDimIndex] = totalLoopNum;
    tempDimIndex++;
    totalLoopNum++;
  }
  applyReplaceMap(replaceMap0);
  applyReplaceMap(replaceMap1);
}

} // namespace mlir::iree_compiler
