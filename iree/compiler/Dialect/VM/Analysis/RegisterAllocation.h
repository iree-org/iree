// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_COMPILER_DIALECT_VM_ANALYSIS_REGISTERALLOCATION_H_
#define IREE_COMPILER_DIALECT_VM_ANALYSIS_REGISTERALLOCATION_H_

#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {

// The VM contains multiple register banks:
// - 128 32-bit integer registers
//   - may be aliased as 32 128-bit registers
// - 64 ref_ptr registers
//
// Registers are represented in bytecode as an 8-bit integer with the high bit
// indicating whether it is from the integer (0b0) or ref_ptr bank (0b1).
//
// ref_ptr register bytes also include a bit denoting whether the register
// reference has move semantics. When set the VM can assume that the value is
// no longer used in the calling code and that ownership can be transferred to
// the receiving op/call. This allows reference count increment elision, though
// the VM is free to ignore this if it so chooses.

constexpr int kIntRegisterCount = 0x7FFF;
constexpr int kRefRegisterCount = 0x3FFF;
constexpr uint16_t kRefRegisterTypeBit = 0x8000;
constexpr uint16_t kRefRegisterMoveBit = 0x4000;

// Returns true if |reg| is a register in the ref_ptr bank.
constexpr bool isRefRegister(uint16_t reg) {
  return (reg & kRefRegisterTypeBit) == kRefRegisterTypeBit;
}

// Returns true if the ref_ptr |reg| denotes a move operation.
constexpr bool isRefMove(uint16_t reg) {
  return (reg & kRefRegisterMoveBit) == kRefRegisterMoveBit;
}

// Returns the register 0-based ordinal within its set.
constexpr uint16_t getRegisterOrdinal(uint16_t reg) {
  return isRefRegister(reg)
             ? (reg & ~(kRefRegisterTypeBit | kRefRegisterMoveBit))
             : reg;
}

// Returns the register ID without the move bit.
constexpr uint16_t getBaseRegister(uint16_t reg) {
  return isRefRegister(reg) ? (reg & ~kRefRegisterMoveBit) : reg;
}

// Compares whether two register bytes are equal to each other, ignoring any
// move semantics on ref_ptr registers.
constexpr bool compareRegistersEqual(uint16_t a, uint16_t b) {
  return getBaseRegister(a) == getBaseRegister(b);
}

// Analysis that performs VM register allocation on the given function op and
// its children. Once calculated value usages can be mapped to VM register
// reference bytes.
class RegisterAllocation {
 public:
  // Annotates the IR with the register mappings. This is only required if the
  // register mappings are interesting to persist beyond just encoding, such as
  // in tests where we want to compare values.
  static LogicalResult annotateIR(IREE::VM::FuncOp funcOp);

  RegisterAllocation() = default;
  explicit RegisterAllocation(Operation *op) {
    recalculate(cast<IREE::VM::FuncOp>(op));
  }
  RegisterAllocation(RegisterAllocation &&) = default;
  RegisterAllocation &operator=(RegisterAllocation &&) = default;
  RegisterAllocation(const RegisterAllocation &) = delete;
  RegisterAllocation &operator=(const RegisterAllocation &) = delete;

  // Recalculates the register allocation.
  LogicalResult recalculate(IREE::VM::FuncOp funcOp);

  // Maximum allocated register ordinals.
  // May be -1 if no registers of the specific type were allocated.
  uint16_t getMaxI32RegisterOrdinal() { return maxI32RegisterOrdinal_; }
  uint16_t getMaxRefRegisterOrdinal() { return maxRefRegisterOrdinal_; }

  // Maps a |value| to a register with no move bit set.
  // Prefer mapUseToRegister when a move is desired.
  uint16_t mapToRegister(Value value);

  // Maps a |value| to a register as calculated during allocation. The returned
  // register will have the proper type and move bits set.
  uint16_t mapUseToRegister(Value value, Operation *useOp, int operandIndex);

  // Remaps branch successor operands to the target block argument registers.
  // Returns a list of source to target register mappings. Source ref registers
  // may have their move bit set.
  SmallVector<std::pair<uint16_t, uint16_t>, 8> remapSuccessorRegisters(
      Operation *op, int successorIndex);

 private:
  int maxI32RegisterOrdinal_ = -1;
  int maxRefRegisterOrdinal_ = -1;

  // Cached liveness information.
  ValueLiveness liveness_;

  // Mapping from all values within the operation to registers.
  llvm::DenseMap<Value, uint16_t> map_;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_ANALYSIS_REGISTERALLOCATION_H_
