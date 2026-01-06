// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_ANALYSIS_REGISTERALLOCATION_H_
#define IREE_COMPILER_DIALECT_VM_ANALYSIS_REGISTERALLOCATION_H_

#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler {

struct LiveInterval;
class LiveIntervals;

namespace IREE::VM {
class RegisterBank;
} // namespace IREE::VM

// Represents a register value at a particular usage.
//
// The VM contains multiple register banks:
// - 32-bit integer registers
//   - may be aliased as 64/128/etc-bit registers
// - ref registers
//
// Registers are represented in bytecode as an N-bit integer with the high bit
// indicating whether it is from the general (0b0) or ref bank (0b1).
//
// ref register ordinals also include a bit denoting whether the register
// reference has move semantics. When set the VM can assume that the value is
// no longer used in the calling code and that ownership can be transferred to
// the receiving op/call. This allows reference count increment elision, though
// the VM is free to ignore this if it so chooses.
class Register {
public:
  static constexpr int kInt32RegisterCount = 0x7FFF;
  static constexpr int kRefRegisterCount = 0x3FFF;
  static constexpr uint16_t kRefTypeBit = 0x8000;
  static constexpr uint16_t kRefMoveBit = 0x4000;

  // Returns a ref-type register with an optional move bit set.
  static Register getRef(Type type, int ordinal, bool isMove = false) {
    return {/*isRef=*/true, isMove, /*byteWidth=*/0, ordinal};
  }

  // Returns a value-type register with a bit-width derived from |type|.
  static Register getValue(Type type, int ordinal) {
    assert(type.isIntOrFloat() && "require int/float (no index)");
    assert(type.getIntOrFloatBitWidth() % 8 == 0 &&
           "require 8-bit aligned value types");
    assert(ordinal < kInt32RegisterCount);
    size_t byteWidth = IREE::Util::getRoundedElementByteWidth(type);
    return {/*isRef=*/false, /*isMove=*/false, byteWidth, ordinal};
  }

  // Returns a register with the same type as |other| but with the new
  // |ordinal|.
  static Register getWithSameType(Register other, int ordinal) {
    return {other.isRef(), /*isMove=*/false, other.byteWidth(), ordinal};
  }

  static Register getEmptyKey() { return Register(); }
  static Register getTombstoneKey() {
    auto reg = Register();
    reg.null_ = 0;
    reg.tombstone_ = 1;
    return reg;
  }

  Register()
      : null_(1), tombstone_(0), isRef_(0), isMove_(0), byteWidth_(0),
        reserved_(0), ordinal_(0) {}

  // Returns the register without any usage-specific bits set (such as move).
  Register asBaseRegister() const {
    return {isRef(), false, byteWidth(), ordinal()};
  }

  constexpr bool isRef() const { return isRef_; }
  constexpr bool isMove() const { return isMove_; }
  void setMove(bool isMove) { isMove_ = isMove ? 1 : 0; }

  constexpr bool isValue() const { return !isRef_; }
  constexpr size_t byteWidth() const { return byteWidth_; }

  // 0-N ordinal within the register bank.
  constexpr uint16_t ordinal() const { return ordinal_; }

  // Encodes the register into a uint16_t value used by the runtime VM.
  uint16_t encode() const {
    assert(!null_ && !tombstone_ && "cannot encode a sentinel register");
    if (isRef()) {
      return kRefTypeBit | (isMove() ? kRefMoveBit : 0) | ordinal();
    } else {
      return ordinal();
    }
  }

  // Encodes the hi register of the lo:hi pair into a uint16_t value used by the
  // runtime VM. Only valid if the register is split (byteWidth() == 8).
  uint16_t encodeHi() const {
    assert(!null_ && !tombstone_ && "cannot encode a sentinel register");
    assert(!isRef() && byteWidth() == 8 && "only valid with 64-bit values");
    return ordinal() + 1;
  }

  std::string toString() const {
    if (null_ || tombstone_) {
      return "<invalid>";
    } else if (isRef()) {
      std::string result = isMove() ? "R" : "r";
      return result + std::to_string(ordinal());
    } else if (byteWidth() == 8) {
      return std::string("i") + std::to_string(ordinal()) + "+" +
             std::to_string(ordinal() + 1);
    } else if (byteWidth() == 4) {
      return std::string("i") + std::to_string(ordinal());
    } else {
      return "<unknown>";
    }
  }

  unsigned getHashValue() const { return hashValue_; }

  // Compares two registers excluding the move bit.
  bool operator==(const Register &other) const {
    if (null_ != other.null_ || tombstone_ != other.tombstone_) {
      return false;
    }
    if (isRef() != other.isRef() || ordinal() != other.ordinal()) {
      return false;
    } else if (isRef()) {
      return true;
    } else {
      return byteWidth() == other.byteWidth();
    }
  }
  bool operator!=(const Register &other) const { return !(*this == other); }

private:
  Register(bool isRef, bool isMove, size_t byteWidth, int ordinal)
      : null_(0), tombstone_(0), isRef_(isRef), isMove_(isMove),
        byteWidth_(byteWidth), ordinal_(ordinal) {}

  union {
    struct {
      uint16_t null_ : 1;      // 1 if the register is indicating an empty value
      uint16_t tombstone_ : 1; // 1 if a DenseMap tombstone value
      uint16_t isRef_ : 1;
      uint16_t isMove_ : 1;
      uint16_t byteWidth_ : 8;
      uint16_t reserved_ : 4;
      uint16_t ordinal_ : 16;
    };
    uint32_t hashValue_;
  };
};

// Analysis that performs VM register allocation on the given function op and
// its children. Once calculated value usages can be mapped to VM register
// reference bytes.
class RegisterAllocation : public VMRegisterAllocation {
public:
  // Annotates the IR with the register mappings. This is only required if the
  // register mappings are interesting to persist beyond just encoding, such as
  // in tests where we want to compare values.
  static LogicalResult annotateIR(IREE::VM::FuncOp funcOp);

  RegisterAllocation() = default;
  explicit RegisterAllocation(Operation *op) {
    (void)recalculate(cast<IREE::VM::FuncOp>(op));
  }
  RegisterAllocation(RegisterAllocation &&) = default;
  RegisterAllocation &operator=(RegisterAllocation &&) = default;
  RegisterAllocation(const RegisterAllocation &) = delete;
  RegisterAllocation &operator=(const RegisterAllocation &) = delete;

  // Recalculates the register allocation using linear scan.
  LogicalResult recalculate(IREE::VM::FuncOp funcOp);

  // Maximum allocated register ordinals.
  // May be -1 if no registers of the specific type were allocated.
  int getMaxI32RegisterOrdinal() {
    return maxI32RegisterOrdinal_ + scratchI32RegisterCount_;
  }
  int getMaxRefRegisterOrdinal() {
    return maxRefRegisterOrdinal_ + scratchRefRegisterCount_;
  }

  // Maps a |value| to a register with no move bit set.
  // Prefer mapUseToRegister when a move is desired.
  Register mapToRegister(Value value) const;

  // Maps a |value| to a register as calculated during allocation. The returned
  // register will have the proper type and move bits set.
  Register mapUseToRegister(Value value, Operation *useOp, int operandIndex);

  // Remaps branch successor operands to the target block argument registers.
  // Returns a list of source to target register mappings. Source ref registers
  // may have their move bit set based on liveness analysis (MOVE indicates
  // ownership transfer on the last use of a ref).
  SmallVector<std::pair<Register, Register>, 8>
  remapSuccessorRegisters(Operation *op, int successorIndex);

  // Legacy overload without branch op context - does not set MOVE bits.
  SmallVector<std::pair<Register, Register>, 8>
  remapSuccessorRegisters(Location loc, Block *targetBlock,
                          OperandRange targetOperands);

private:
  // Internal implementation that handles MOVE bit computation.
  SmallVector<std::pair<Register, Register>, 8>
  remapSuccessorRegisters(Operation *branchOp, Block *targetBlock,
                          OperandRange targetOperands,
                          unsigned baseOperandIndex);

public:
  // Returns true if the given discard op can be fully elided because all its
  // operands have already been released via MOVE on preceding operations.
  bool isDiscardElidable(Operation *op) const {
    auto it = discardOperandElidability_.find(op);
    if (it == discardOperandElidability_.end())
      return false;
    return llvm::all_of(it->second, [](bool b) { return b; });
  }

  // Returns true if the operand at |operandIndex| of the given discard op can
  // be elided because it was already released via MOVE on a preceding
  // operation.
  bool isDiscardOperandElidable(Operation *op,
                                unsigned operandIndex) const override {
    auto it = discardOperandElidability_.find(op);
    if (it == discardOperandElidability_.end())
      return false;
    if (operandIndex >= it->second.size())
      return false;
    return it->second[operandIndex];
  }

  int mapValueToRegisterOrdinal(Value value) const override {
    return mapToRegister(value).ordinal();
  }

private:
  int maxI32RegisterOrdinal_ = -1;
  int maxRefRegisterOrdinal_ = -1;
  int scratchI32RegisterCount_ = 0;
  int scratchRefRegisterCount_ = 0;

  // Identifies discard ops that can be elided because all operands have MOVE
  // on their last real use.
  void computeElidableDiscards(IREE::VM::FuncOp funcOp);

  // Runs linear scan allocation for a single register bank.
  LogicalResult
  runLinearScan(IREE::VM::FuncOp funcOp, const LiveIntervals &liveIntervals,
                IREE::VM::RegisterBank &bank,
                SmallVectorImpl<const LiveInterval *> &active,
                const llvm::DenseMap<Value, Value> &coalesceSource,
                bool isRefBank, int &maxOrdinal);

  // Cached liveness information.
  ValueLiveness liveness_;

  // Cached dominance information for back-edge detection.
  std::optional<DominanceInfo> dominanceInfo_;

  // Mapping from all values within the operation to registers.
  llvm::DenseMap<Value, Register> map_;

  // Per-operand elidability for discard ops. Each entry maps a discard op to
  // a vector of booleans indicating whether each operand can be elided (true
  // means the operand was already released via MOVE on a preceding operation).
  llvm::DenseMap<Operation *, SmallVector<bool>> discardOperandElidability_;

  // Allocation metadata for live intervals (separate from immutable interval
  // data to keep LiveIntervals const-correct).
  struct IntervalAllocation {
    int32_t assigned = -1;   // Assigned register ordinal (-1 = unassigned).
    int32_t preference = -1; // Coalescing hint (-1 = no preference).
  };
  llvm::DenseMap<const LiveInterval *, IntervalAllocation> intervalAllocations_;
};

} // namespace mlir::iree_compiler

namespace llvm {
template <>
struct DenseMapInfo<mlir::iree_compiler::Register> {
  using Register = mlir::iree_compiler::Register;

  static inline Register getEmptyKey() { return Register::getEmptyKey(); }

  static inline Register getTombstoneKey() {
    return Register::getTombstoneKey();
  }

  static unsigned getHashValue(const Register &val) {
    return val.getHashValue();
  }

  static bool isEqual(const Register &lhs, const Register &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // IREE_COMPILER_DIALECT_VM_ANALYSIS_REGISTERALLOCATION_H_
