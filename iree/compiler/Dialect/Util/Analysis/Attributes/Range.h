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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

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
struct FpRangeStats {
  enum TruncationFlag {
    TRUNC = 0,
    TRUNC_UNKNOWN = 1,
  };

  double minValue = NAN;
  double maxValue = NAN;
  bool valid = false;
  TruncationFlag truncationFlag = TRUNC_UNKNOWN;

  FpRangeStats() {}
  FpRangeStats(double minValue, double maxValue)
      : minValue(minValue), maxValue(maxValue), valid(true) {}

  static FpRangeStats getInvalid() { return FpRangeStats(); }
  static FpRangeStats getWidest() { return FpRangeStats(-INFINITY, INFINITY); }

  static TruncationFlag unionTruncationFlag(TruncationFlag lhs,
                                            TruncationFlag rhs) {
    return static_cast<TruncationFlag>(std::max(lhs, rhs));
  }

  // Reset to initial state.
  void reset() { *this = FpRangeStats(); }

  // Adds a value to the range for the case when all possible values are
  // known. Call reset() prior to iteration to enter the open-range state.
  void addDomainValue(double value);

  bool isInvalid() const { return !valid; }

  bool operator==(const FpRangeStats &other) const {
    return valid == other.valid && minValue == other.minValue &&
           maxValue == other.maxValue;
  }

  bool operator!=(const FpRangeStats &other) const { return !(*this == other); }

  // Makes this state the union of information known by both.
  void operator+=(const FpRangeStats &rhs) {
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

  std::string getAsStr() const;
};

// State that tracks floating point ranges and flags.
struct FpRangeState : public DFX::AbstractState {
  // Returns the worst possible representable state.
  static FpRangeStats getWorstState() { return FpRangeStats::getWidest(); }
  static FpRangeStats getWorstState(const FpRangeState &) {
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

  FpRangeStats getAssumed() const { return assumed; }
  FpRangeStats getKnown() const { return known; }

  // Resets the assumed value to the given value. This does no unioning and
  // assumes it is a proper fixpoint minimum.
  void setAssumed(FpRangeStats newAssumed) { assumed = newAssumed; }

  // Apply stats derived from operands of common math operations to this.
  void applyMinf(const FpRangeStats &lhs, const FpRangeStats &rhs);
  void applyMaxf(const FpRangeStats &lhs, const FpRangeStats &rhs);
  void applyFloor(const FpRangeStats &operand);

  // "Clamps" this state with |rhs|. The assumed value will contain the union
  // of information assumed by both states.
  void operator^=(const FpRangeState &rhs) {
    FpRangeStats rhsAssumed = rhs.getAssumed();
    assumed += rhsAssumed;
  }

 private:
  FpRangeStats assumed = FpRangeStats::getInvalid();
  FpRangeStats known = FpRangeStats::getWidest();
};

// Attribute known floating point range and flags to an IR Value.
class FpRangeValueElement
    : public DFX::StateWrapper<FpRangeState, DFX::ValueElement> {
 public:
  using BaseType = DFX::StateWrapper<FpRangeState, DFX::ValueElement>;
  using BaseType::BaseType;

  static FpRangeValueElement &createForPosition(const Position &pos,
                                                DFX::Solver &solver) {
    return *(new (solver.getAllocator()) FpRangeValueElement(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override { return "FpRangeValueElement"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  const std::string getAsStr() const override;

 private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
};

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
