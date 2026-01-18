// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_ANALYSIS_LINEARSCAN_REGISTERBANK_H_
#define IREE_COMPILER_DIALECT_VM_ANALYSIS_LINEARSCAN_REGISTERBANK_H_

#include <optional>
#include "llvm/ADT/BitVector.h"

namespace mlir::iree_compiler::IREE::VM {

// Manages allocation for a single register bank.
// Tracks which registers are in use and provides allocation/deallocation.
// Used for both integer (i32/i64) and reference register banks.
class RegisterBank {
public:
  // Creates a bank with the given capacity (number of registers).
  explicit RegisterBank(int capacity);

  // Allocates a register span of the given byte width (4 for i32, 8 for i64).
  // Returns ordinal of first register, or nullopt if full.
  // If fromEnd=true, allocates from high end (for entry block arguments).
  //
  // For i64 values (byteWidth=8), the allocation is aligned to even ordinals
  // and occupies two consecutive registers (ordinal, ordinal+1).
  std::optional<int> allocate(size_t byteWidth, bool fromEnd = false);

  // Marks register(s) as used without allocating (for live-in values).
  // Does not update maxUsed_ - used for values already allocated elsewhere.
  void reserve(int ordinal, size_t byteWidth = 4);

  // Releases register(s) back to available pool.
  void release(int ordinal, size_t byteWidth = 4);

  // Query if ordinal is currently in use.
  bool isUsed(int ordinal) const;

  // Allocates at a specific ordinal (for coalescing hints).
  // Returns true if successful, false if already in use.
  // Updates maxUsed_ like allocate().
  bool allocateAt(int ordinal, size_t byteWidth = 4);

  // Get max ordinal ever allocated (for register count tracking).
  int getMaxUsed() const { return maxUsed_; }

  // Get capacity.
  int getCapacity() const { return capacity_; }

private:
  // Finds first unset ordinal span with proper alignment.
  std::optional<int> findFirstUnsetSpan(size_t byteWidth);

  // Finds last unset ordinal span for monotonic allocation.
  std::optional<int> findLastUnsetSpan(size_t byteWidth);

  // Marks a span as used and updates maxUsed_.
  void markUsed(int ordinal, size_t byteWidth);

  int capacity_;
  llvm::BitVector used_;
  int maxUsed_ = -1;
};

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_ANALYSIS_LINEARSCAN_REGISTERBANK_H_
