// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_ANALYSIS_LINEARSCAN_LIVEINTERVALS_H_
#define IREE_COMPILER_DIALECT_VM_ANALYSIS_LINEARSCAN_LIVEINTERVALS_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler {

// Forward declaration for internal use.
class ValueLiveness;

// Represents the live interval of an SSA value.
// The interval [start, end] covers all instruction indices where the value
// is live, from its definition to its last use (inclusive).
struct LiveInterval {
  Value value;          // SSA value this interval represents.
  uint32_t start = 0;   // First instruction index (definition point).
  uint32_t end = 0;     // Last instruction index (last use, inclusive).
  bool isRef = false;   // true = ref bank, false = i32 bank.
  size_t byteWidth = 4; // 4 for i32, 8 for i64, 0 for ref.

  // Returns true if this interval overlaps with another.
  bool overlaps(const LiveInterval &other) const {
    return !(end < other.start || other.end < start);
  }

  // Returns true if this interval contains the given instruction index.
  bool contains(uint32_t index) const { return index >= start && index <= end; }

  // Returns the length of the interval.
  uint32_t length() const { return end - start + 1; }
};

// Computes live intervals for all values in a VM function.
// This analysis linearizes the function's blocks in reverse post-order,
// numbers all instructions, and computes [start, end] intervals for each value.
class LiveIntervals {
public:
  // Annotates the IR with live interval information for testing.
  // Adds attributes to operations showing their instruction indices and
  // result intervals.
  static LogicalResult annotateIR(IREE::VM::FuncOp funcOp);

  LiveIntervals() = default;
  LiveIntervals(LiveIntervals &&) = default;
  LiveIntervals &operator=(LiveIntervals &&) = default;
  LiveIntervals(const LiveIntervals &) = delete;
  LiveIntervals &operator=(const LiveIntervals &) = delete;

  // Builds live intervals for all values in the function.
  // Requires a pre-computed ValueLiveness analysis.
  LogicalResult build(IREE::VM::FuncOp funcOp);

  // Returns the interval for a value, or nullptr if not found.
  const LiveInterval *getInterval(Value value) const;

  // Returns all intervals (unsorted).
  ArrayRef<LiveInterval> getIntervals() const { return intervals_; }

  // Returns intervals sorted by start position (for linear scan).
  ArrayRef<const LiveInterval *> getSortedByStart() const {
    return sortedByStart_;
  }

  // Returns the instruction index for an operation.
  uint32_t getInstructionIndex(Operation *op) const;

  // Returns the linearized block order (reverse post-order).
  ArrayRef<Block *> getBlockOrder() const { return blockOrder_; }

  // Returns the total number of instructions.
  uint32_t getInstructionCount() const { return instructionCount_; }

  // Debug printing.
  void dump() const;

private:
  // Sorts blocks in dominance order (reverse post-order).
  // Ensures definitions come before uses in the linear ordering.
  void sortBlocksInDominanceOrder(IREE::VM::FuncOp funcOp);

  // Numbers all instructions in the linearized block order.
  void numberInstructions();

  // Builds intervals from the ValueLiveness analysis.
  void buildIntervals(ValueLiveness &liveness);

  // Finds the last use instruction index for a value.
  uint32_t findLastUse(Value value, ValueLiveness &liveness);

  SmallVector<LiveInterval> intervals_;
  SmallVector<const LiveInterval *> sortedByStart_;
  DenseMap<Value, size_t> valueToInterval_;
  DenseMap<Operation *, uint32_t> opToIndex_;
  SmallVector<Block *> blockOrder_;
  uint32_t instructionCount_ = 0;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_ANALYSIS_LINEARSCAN_LIVEINTERVALS_H_
