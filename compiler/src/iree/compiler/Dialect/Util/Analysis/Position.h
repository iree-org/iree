// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_ANALYSIS_POSITION_H_
#define IREE_COMPILER_DIALECT_UTIL_ANALYSIS_POSITION_H_

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler {

// Represents a position in the IR.
// This may directly reference an MLIR instance such as a Value or Operation or
// reference an abstract location such as a returned value from a callable.
//
// This is the MLIR equivalent to the IRPosition used in LLVM (see
// llvm/Transforms/IPO/Attributor.h).
class Position {
public:
  static const Position EmptyKey;
  static const Position TombstoneKey;

  Position()
      : Position(ENC_BLOCK, llvm::DenseMapInfo<void *>::getEmptyKey(), 0) {}

  static const Position forValue(Value value) {
    return Position(ENC_VALUE, value.getAsOpaquePointer(), 0);
  }
  bool isValue() const { return enc.getInt() == ENC_VALUE; }
  Value getValue() const {
    assert(isValue());
    return Value::getFromOpaquePointer(enc.getPointer());
  }

  static const Position forReturnedValue(Operation *op, unsigned resultIdx) {
    return Position(ENC_RETURNED_VALUE, op, resultIdx);
  }
  bool isReturnedValue() const { return enc.getInt() == ENC_RETURNED_VALUE; }
  std::pair<Operation *, unsigned> getReturnedValue() const {
    assert(isReturnedValue());
    return std::make_pair(reinterpret_cast<Operation *>(enc.getPointer()),
                          ordinal);
  }

  static const Position forOperation(Operation *op) {
    return Position(ENC_OPERATION, op, 0);
  }
  bool isOperation() const { return enc.getInt() == ENC_OPERATION; }
  Operation &getOperation() const {
    assert(isOperation());
    return *reinterpret_cast<Operation *>(enc.getPointer());
  }

  static const Position forBlock(Block *block) {
    return Position(ENC_BLOCK, block, 0);
  }
  bool isBlock() const { return enc.getInt() == ENC_BLOCK; }
  Block &getBlock() const {
    assert(isBlock());
    return *reinterpret_cast<Block *>(enc.getPointer());
  }

  bool operator==(const Position &RHS) const {
    return enc == RHS.enc && RHS.ordinal == ordinal;
  }
  bool operator!=(const Position &RHS) const { return !(*this == RHS); }

  // Conversion into a void * to allow reuse of pointer hashing.
  operator void *() const { return enc.getOpaqueValue(); }

  void print(llvm::raw_ostream &os) const;
  void print(llvm::raw_ostream &os, AsmState &asmState) const;

private:
  template <typename T, typename Enable>
  friend struct llvm::DenseMapInfo;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Position pos);

  explicit Position(char encoding, void *ptr, unsigned ordinal)
      : enc(ptr, encoding), ordinal(ordinal) {}

  enum {
    ENC_VALUE = 0b00,
    ENC_RETURNED_VALUE = 0b01,
    ENC_OPERATION = 0b10,
    ENC_BLOCK = 0b11,
  };
  static constexpr int NumEncodingBits =
      llvm::PointerLikeTypeTraits<void *>::NumLowBitsAvailable;
  static_assert(NumEncodingBits >= 2, "At least two bits are required!");
  llvm::PointerIntPair<void *, NumEncodingBits, char> enc;
  unsigned ordinal; // used only with ENC_RETURNED_VALUE
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Position pos);

// Maps each input Value into a Position.
static inline auto getPositions(ValueRange valueRange) {
  return llvm::map_range(valueRange, [](Value value) -> Position {
    return Position::forValue(value);
  });
}

// Maps each entry argument of the given |region| into a Position.
static inline auto getArgumentPositions(Region &region) {
  return llvm::map_range(
      region.getArguments(),
      [](BlockArgument &arg) -> Position { return Position::forValue(arg); });
}

// Maps each entry argument of the given |block| into a Position.
static inline auto getArgumentPositions(Block &block) {
  return llvm::map_range(
      block.getArguments(),
      [](BlockArgument &arg) -> Position { return Position::forValue(arg); });
}

// TODO(benvanik): wade through the hell that is STL/ADT iterator goo to figure
// out how we can have this return a mapped_iterator even if the callback
// differs. Since these functions are called a tremendous number of times during
// fixed point iteration we want them to avoid allocations.

// Returns a position for each external return value of the given region.
// If the op returns the region results then this is equivalent to taking the
// positions of the results of the op. If the op is FunctionLike then the
// positions are references to the combined results of the region.
SmallVector<Position> getReturnedValuePositions(Region &region);

} // namespace mlir::iree_compiler

namespace llvm {

using mlir::iree_compiler::Position;

// Helper that allows Position as a key in a DenseMap.
template <>
struct DenseMapInfo<Position> {
  static inline Position getEmptyKey() { return Position::EmptyKey; }
  static inline Position getTombstoneKey() { return Position::TombstoneKey; }
  static unsigned getHashValue(const Position &pos) {
    return (DenseMapInfo<void *>::getHashValue(pos) << 4) ^
           (DenseMapInfo<unsigned>::getHashValue(pos.ordinal));
  }

  static bool isEqual(const Position &a, const Position &b) { return a == b; }
};

} // end namespace llvm

#endif // IREE_COMPILER_DIALECT_UTIL_ANALYSIS_POSITION_H_
