// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_ANALYSIS_VALUELIVENESS_H_
#define IREE_COMPILER_DIALECT_VM_ANALYSIS_VALUELIVENESS_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler {

// SSA value liveness analysis.
// Used to compute live ranges of values over all ops within a function CFG.
// These live ranges can be queried for information such as whether two values
// interfere or when a value is no longer live.
class ValueLiveness {
public:
  // Annotates the IR with the liveness information. This is only required if
  // the liveness information (block in/out, intervals, etc) are interesting to
  // persist beyond just encoding, such as in tests where we want to compare
  // values.
  static LogicalResult annotateIR(IREE::VM::FuncOp funcOp);

  ValueLiveness() = default;
  explicit ValueLiveness(Operation *op) {
    (void)recalculate(cast<IREE::VM::FuncOp>(op));
  }
  ValueLiveness(ValueLiveness &&) = default;
  ValueLiveness &operator=(ValueLiveness &&) = default;
  ValueLiveness(const ValueLiveness &) = delete;
  ValueLiveness &operator=(const ValueLiveness &) = delete;

  // Recalculates the liveness information for the given function.
  LogicalResult recalculate(IREE::VM::FuncOp funcOp);

  // Returns an unordered list of values live on block entry.
  ArrayRef<Value> getBlockLiveIns(Block *block);

  // Returns true if |useOp| has the last use of |value|.
  bool isLastValueUse(Value value, Operation *useOp);
  // Returns true if |useOp|'s operand at |operandIndex| is the last use of the
  // value.
  bool isLastValueUse(Value value, Operation *useOp, int operandIndex);

private:
  // Produces an op ordering for the entire function.
  // The ordering is only useful for computing bitmap ordinals as the CFG is not
  // sorted in any defined order (don't rely on op A < op B meaning that A is
  // executed before B).
  void calculateOpOrdering(IREE::VM::FuncOp funcOp);

  // Computes the initial liveness sets for blocks based entirely on information
  // local to each block.
  LogicalResult computeInitialLivenessSets(IREE::VM::FuncOp funcOp);

  // Computes the blockLiveness_ liveness sets for each block using cross-block
  // information.
  LogicalResult computeLivenessSets(IREE::VM::FuncOp funcOp);

  // Computes the liveRanges_ for the function with a bit for each operation a
  // value is live during (including its last usage).
  LogicalResult computeLiveIntervals(IREE::VM::FuncOp funcOp);

  // All operations in the function indexed by their unique ordinal (the same
  // as used in opOrdering_).
  std::vector<Operation *> opsInOrder_;

  // All operations within the function mapped to a unique integer index.
  // This index is used when computing BitVectors across all of the operations.
  DenseMap<Operation *, int> opOrdering_;

  // For a Block defines the values that are defined or live within/across.
  struct BlockSets {
    // All values defined within the block (either by ops or block args).
    llvm::SmallSetVector<Value, 8> defined;
    // All values used within the block that are not defined there.
    llvm::SmallSetVector<Value, 8> live;
    // Values live on block entry (used in the block or successors).
    llvm::SmallSetVector<Value, 8> liveIn;
    // Values live on block exit (used in successors).
    llvm::SmallSetVector<Value, 8> liveOut;
  };
  DenseMap<Block *, BlockSets> blockLiveness_;

  // Liveness ranges indicating for which operations the value is live.
  // Each bit in the BitVector corresponds to an operation with the matching
  // ordinal in opOrdering_.
  DenseMap<Value, llvm::BitVector> liveRanges_;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_ANALYSIS_VALUELIVENESS_H_
