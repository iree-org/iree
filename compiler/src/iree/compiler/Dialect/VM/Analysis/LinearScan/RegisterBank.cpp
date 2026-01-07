// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Analysis/LinearScan/RegisterBank.h"

#include "llvm/Support/MathExtras.h"

namespace mlir::iree_compiler::IREE::VM {

RegisterBank::RegisterBank(int capacity)
    : capacity_(capacity), used_(capacity) {}

std::optional<int> RegisterBank::allocate(size_t byteWidth, bool fromEnd) {
  // For ref registers (byteWidth=0), treat as single slot.
  if (byteWidth == 0) {
    byteWidth = 4;
  }
  std::optional<int> ordinal;
  if (fromEnd) {
    ordinal = findLastUnsetSpan(byteWidth);
  } else {
    ordinal = findFirstUnsetSpan(byteWidth);
  }
  if (ordinal) {
    markUsed(*ordinal, byteWidth);
  }
  return ordinal;
}

void RegisterBank::reserve(int ordinal, size_t byteWidth) {
  // For ref registers (byteWidth=0), treat as single slot.
  if (byteWidth == 0) {
    byteWidth = 4;
  }
  // Set bits without updating maxUsed_ - used for live-in values that were
  // already allocated in a previous block.
  unsigned ordinalEnd = ordinal + (byteWidth / 4) - 1;
  assert(ordinal >= 0 && ordinalEnd < static_cast<unsigned>(capacity_) &&
         "reserve ordinal out of bounds");
  used_.set(ordinal, ordinalEnd + 1);
}

void RegisterBank::release(int ordinal, size_t byteWidth) {
  // For ref registers (byteWidth=0), treat as single slot.
  if (byteWidth == 0) {
    byteWidth = 4;
  }
  unsigned ordinalEnd = ordinal + (byteWidth / 4) - 1;
  assert(ordinal >= 0 && ordinalEnd < static_cast<unsigned>(capacity_) &&
         "release ordinal out of bounds");
  used_.reset(ordinal, ordinalEnd + 1);
}

bool RegisterBank::isUsed(int ordinal) const { return used_.test(ordinal); }

bool RegisterBank::allocateAt(int ordinal, size_t byteWidth) {
  // For ref registers (byteWidth=0), treat as single slot.
  if (byteWidth == 0) {
    byteWidth = 4;
  }
  unsigned ordinalEnd = ordinal + (byteWidth / 4) - 1;
  if (ordinal < 0 || ordinalEnd >= static_cast<unsigned>(capacity_)) {
    return false;
  }
  // Check entire span is available.
  for (unsigned i = ordinal; i <= ordinalEnd; ++i) {
    if (used_.test(i)) {
      return false;
    }
  }
  markUsed(ordinal, byteWidth);
  return true;
}

std::optional<int> RegisterBank::findFirstUnsetSpan(size_t byteWidth) {
  unsigned requiredAlignment = byteWidth / 4;
  unsigned ordinalStart = used_.find_first_unset();
  while (ordinalStart != static_cast<unsigned>(-1)) {
    // Check alignment (i64 must be on even ordinals).
    if ((ordinalStart % requiredAlignment) != 0) {
      ordinalStart = used_.find_next_unset(ordinalStart);
      continue;
    }
    unsigned ordinalEnd = ordinalStart + (byteWidth / 4) - 1;
    if (ordinalEnd >= static_cast<unsigned>(capacity_)) {
      return std::nullopt;
    }
    // Check entire span is available.
    bool rangeAvailable = true;
    for (unsigned ordinal = ordinalStart + 1; ordinal <= ordinalEnd;
         ++ordinal) {
      rangeAvailable &= !used_.test(ordinal);
    }
    if (rangeAvailable) {
      return static_cast<int>(ordinalStart);
    }
    ordinalStart = used_.find_next_unset(ordinalEnd);
  }
  return std::nullopt;
}

std::optional<int> RegisterBank::findLastUnsetSpan(size_t byteWidth) {
  // Allocate from the high end for monotonic allocation (entry block args).
  // Find the first position after all currently used registers.
  unsigned requiredAlignment = byteWidth / 4;
  int lastUsed = used_.find_last();
  unsigned ordinalStart =
      llvm::alignTo(static_cast<unsigned>(lastUsed + 1), requiredAlignment);
  unsigned ordinalEnd = ordinalStart + (byteWidth / 4) - 1;
  if (ordinalEnd >= static_cast<unsigned>(capacity_)) {
    return std::nullopt;
  }
  return static_cast<int>(ordinalStart);
}

void RegisterBank::markUsed(int ordinal, size_t byteWidth) {
  unsigned ordinalEnd = ordinal + (byteWidth / 4) - 1;
  used_.set(ordinal, ordinalEnd + 1);
  maxUsed_ = std::max(static_cast<int>(ordinalEnd), maxUsed_);
}

} // namespace mlir::iree_compiler::IREE::VM
