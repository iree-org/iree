// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_ANALYSIS_VALUELIVENESS_H_
#define IREE_COMPILER_DIALECT_VM_ANALYSIS_VALUELIVENESS_H_

#include <optional>

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler {

// SSA value liveness analysis.
// A thin VM-specific wrapper around the upstream mlir::Liveness analysis.
// The upstream analysis provides the block live-in/live-out sets and
// last-use queries; this wrapper adds the ref-aware "last real use" query
// used by register allocation to place MOVE bits (ignoring vm.discard_refs
// cleanup markers).
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

  // Returns true if |value| is live on entry to |block|.
  bool isLiveIn(Block *block, Value value);

  // Returns true if |value| is live on exit from |block|.
  bool isLiveOut(Block *block, Value value);

  // Returns true if |useOp| has the last use of |value|.
  bool isLastValueUse(Value value, Operation *useOp);
  // Returns true if |useOp|'s operand at |operandIndex| is the last use of the
  // value.
  bool isLastValueUse(Value value, Operation *useOp, int operandIndex);

  // Returns true if |useOp|'s operand at |operandIndex| is the last "real" use
  // of the value, ignoring DiscardRefsOp uses. This is used by register
  // allocation to set the MOVE bit on the last "real" use, allowing subsequent
  // discards to be elided when the MOVE has already released the ref.
  bool isLastRealValueUse(Value value, Operation *useOp, int operandIndex);

private:
  // Upstream liveness analysis providing block live-in/live-out sets and
  // per-op last-use information. Reconstructed by recalculate().
  std::optional<Liveness> liveness_;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_ANALYSIS_VALUELIVENESS_H_
