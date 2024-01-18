// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cmath>
#include <string>

#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"

namespace mlir::iree_compiler::IREE::Util {

//===----------------------------------------------------------------------===//
// Floating point range and statistics analysis
//===----------------------------------------------------------------------===//

// Tracks some floating point statistics and flags which can be derived from
// actual values and operations in the program:
//   - minValue: Lowest possible value
//   - maxValue: Greatest possible value
//   - truncationFlag: Whether it can be assumed that all values are consistent
//     with having had |trunc()| called on them.
// In its default state, the |valid| bit is false.
//
// Note that regardless of the program type, this tracks statistics using
// doubles. For the level of analysis envisioned, this is sufficient.
struct FloatRangeStats {
  enum TruncationFlag {
    TRUNC = 0,
    TRUNC_UNKNOWN = 1,
  };

  double minValue = NAN;
  double maxValue = NAN;
  bool valid = false;
  TruncationFlag truncationFlag = TRUNC_UNKNOWN;

  FloatRangeStats() {}
  FloatRangeStats(double minValue, double maxValue)
      : minValue(minValue), maxValue(maxValue), valid(true) {}

  static FloatRangeStats getInvalid() { return FloatRangeStats(); }
  static FloatRangeStats getWidest() {
    return FloatRangeStats(-INFINITY, INFINITY);
  }

  static TruncationFlag unionTruncationFlag(TruncationFlag lhs,
                                            TruncationFlag rhs) {
    return static_cast<TruncationFlag>(std::max(lhs, rhs));
  }

  // Whether the range is known to only contain values that have been
  // truncated to exclude fractional bits.
  bool isTruncated() { return truncationFlag == TRUNC; }

  bool isFinite() { return std::isfinite(minValue) && std::isfinite(maxValue); }

  // Reset to initial state.
  void reset() { *this = FloatRangeStats(); }

  // Adds a value to the range for the case when all possible values are
  // known. Call reset() prior to iteration to enter the open-range state.
  void addDomainValue(double value);

  bool isInvalid() const { return !valid; }

  bool operator==(const FloatRangeStats &other) const {
    return valid == other.valid && minValue == other.minValue &&
           maxValue == other.maxValue;
  }

  bool operator!=(const FloatRangeStats &other) const {
    return !(*this == other);
  }

  // Makes this state the union of information known by both.
  void operator+=(const FloatRangeStats &rhs) {
    // If invalid, just accept the rhs as-is.
    if (!valid) {
      *this = rhs;
      return;
    }
    if (rhs.valid) {
      minValue = std::min(minValue, rhs.minValue);
      maxValue = std::max(maxValue, rhs.maxValue);
      truncationFlag = unionTruncationFlag(truncationFlag, rhs.truncationFlag);
    }
  }

  std::string getAsStr(AsmState &asmState) const;
};

// State that tracks floating point ranges and flags.
struct FloatRangeState : public DFX::AbstractState {
  // Returns the worst possible representable state.
  static FloatRangeStats getWorstState() {
    return FloatRangeStats::getWidest();
  }
  static FloatRangeStats getWorstState(const FloatRangeState &) {
    return getWorstState();
  }

  bool isValidState() const override { return !assumed.isInvalid(); }
  bool isAtFixpoint() const override { return assumed == known; }

  ChangeStatus indicateOptimisticFixpoint() override {
    known = assumed;
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    assumed = known;
    return ChangeStatus::CHANGED;
  }

  FloatRangeStats getAssumed() const { return assumed; }
  FloatRangeStats getKnown() const { return known; }

  // Resets the assumed value to the given value. This does no unioning and
  // assumes it is a proper fixpoint minimum.
  void setAssumed(FloatRangeStats newAssumed) { assumed = newAssumed; }

  // Apply stats derived from operands of common math operations to this.
  void applyMinf(const FloatRangeStats &lhs, const FloatRangeStats &rhs);
  void applyMaxf(const FloatRangeStats &lhs, const FloatRangeStats &rhs);
  void applyFloor(const FloatRangeStats &operand);

  // "Clamps" this state with |rhs|. The assumed value will contain the union
  // of information assumed by both states.
  void operator^=(const FloatRangeState &rhs) {
    FloatRangeStats rhsAssumed = rhs.getAssumed();
    assumed += rhsAssumed;
  }

private:
  FloatRangeStats assumed = FloatRangeStats::getInvalid();
  FloatRangeStats known = FloatRangeStats::getWidest();
};

// Attribute known floating point range and flags to an IR Value.
class FloatRangeValueElement
    : public DFX::StateWrapper<FloatRangeState, DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<FloatRangeState, DFX::ValueElement>;
  using BaseType::BaseType;

  static FloatRangeValueElement &createForPosition(const Position &pos,
                                                   DFX::Solver &solver) {
    return *(new (solver.getAllocator()) FloatRangeValueElement(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override {
    return "FloatRangeValueElement";
  }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  const std::string getAsStr(AsmState &asmState) const override;

private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
};

} // namespace mlir::iree_compiler::IREE::Util
