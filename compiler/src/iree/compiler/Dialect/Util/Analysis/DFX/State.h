// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_STATE_H_
#define IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_STATE_H_

#include "iree/compiler/Dialect/Util/Analysis/Position.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// ChangeStatus
//===----------------------------------------------------------------------===//

// TODO(benvanik): reuse mlir::ChangeResult?

// A result type used to indicate if a change happened. Boolean operations on
// ChangeStatus behave as though `CHANGED` is truthy.
enum class ChangeStatus {
  UNCHANGED,
  CHANGED,
};

inline ChangeStatus operator|(ChangeStatus lhs, ChangeStatus rhs) {
  return lhs == ChangeStatus::CHANGED ? lhs : rhs;
}
inline ChangeStatus &operator|=(ChangeStatus &lhs, ChangeStatus rhs) {
  lhs = lhs | rhs;
  return lhs;
}
inline ChangeStatus operator&(ChangeStatus lhs, ChangeStatus rhs) {
  return lhs == ChangeStatus::UNCHANGED ? lhs : rhs;
}

//===----------------------------------------------------------------------===//
// AbstractState
//===----------------------------------------------------------------------===//

namespace DFX {

// Base state representing assumed and known information.
struct AbstractState {
  virtual ~AbstractState() = default;

  // Returns true if in a valid state.
  // When false no information provided should be used.
  virtual bool isValidState() const = 0;

  // Returns true if the state is fixed and thus does not need to be updated
  // if information changes.
  virtual bool isAtFixpoint() const = 0;

  // Indicates that the abstract state should converge to the optimistic state.
  // This will usually make the optimistically assumed state the known to be
  // true state.
  //
  // Returns UNCHANGED as the assumed value does not change.
  virtual ChangeStatus indicateOptimisticFixpoint() = 0;

  // Indicates that the abstract state should converge to the pessimistic state.
  // This will usually revert the optimistically assumed state to the known to
  // be true state.
  //
  // Returns CHANGED as the assumed value may change.
  virtual ChangeStatus indicatePessimisticFixpoint() = 0;
};

//===----------------------------------------------------------------------===//
// IntegerStateBase
//===----------------------------------------------------------------------===//

template <typename BaseTy, BaseTy BestState, BaseTy WorstState>
struct IntegerStateBase : public AbstractState {
  using base_t = BaseTy;

  IntegerStateBase() = default;
  IntegerStateBase(base_t assumed) : assumed(assumed) {}

  // Returns the best possible representable state.
  static constexpr base_t getBestState() { return BestState; }
  static constexpr base_t getBestState(const IntegerStateBase &) {
    return getBestState();
  }

  // Returns the worst possible representable state.
  static constexpr base_t getWorstState() { return WorstState; }
  static constexpr base_t getWorstState(const IntegerStateBase &) {
    return getWorstState();
  }

  // NOTE: For now we simply pretend that the worst possible state is invalid.
  bool isValidState() const override { return assumed != getWorstState(); }

  bool isAtFixpoint() const override { return assumed == known; }

  ChangeStatus indicateOptimisticFixpoint() override {
    known = assumed;
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    assumed = known;
    return ChangeStatus::CHANGED;
  }

  // Returns the known state encoding.
  base_t getKnown() const { return known; }

  // Returns the assumed state encoding.
  base_t getAssumed() const { return assumed; }

  bool
  operator==(const IntegerStateBase<base_t, BestState, WorstState> &rhs) const {
    return this->getAssumed() == rhs.getAssumed() &&
           this->getKnown() == rhs.getKnown();
  }
  bool
  operator!=(const IntegerStateBase<base_t, BestState, WorstState> &rhs) const {
    return !(*this == rhs);
  }

  // "Clamps" this state with |rhs|. The result is subtype dependent but it is
  // intended that only information assumed in both states will be assumed in
  // this one afterwards.
  void operator^=(const IntegerStateBase<base_t, BestState, WorstState> &rhs) {
    handleNewAssumedValue(rhs.getAssumed());
  }

  // "Clamps" this state with |rhs|. The result is subtype dependent but it is
  // intended that information known in either state will be known in
  // this one afterwards.
  void operator+=(const IntegerStateBase<base_t, BestState, WorstState> &rhs) {
    handleNewKnownValue(rhs.getKnown());
  }

  void operator|=(const IntegerStateBase<base_t, BestState, WorstState> &rhs) {
    joinOR(rhs.getAssumed(), rhs.getKnown());
  }

  void operator&=(const IntegerStateBase<base_t, BestState, WorstState> &rhs) {
    joinAND(rhs.getAssumed(), rhs.getKnown());
  }

protected:
  // Handles a new known value |value|. Subtype dependent.
  virtual void handleNewKnownValue(base_t value) = 0;

  // Handles a new assumed value |value|. Subtype dependent.
  virtual void handleNewAssumedValue(base_t value) = 0;

  // Handles a value |value|. Subtype dependent.
  virtual void joinOR(base_t assumedValue, base_t knownValue) = 0;

  // Handles a new assumed value |value|. Subtype dependent.
  virtual void joinAND(base_t assumedValue, base_t knownValue) = 0;

  // The known state encoding in an integer of type base_t.
  base_t known = getWorstState();
  // The assumed state encoding in an integer of type base_t.
  base_t assumed = getBestState();
};

//===----------------------------------------------------------------------===//
// BooleanState
//===----------------------------------------------------------------------===//

// Specialization of the integer state for single-bit values.
struct BooleanState : public IntegerStateBase<bool, 1, 0> {
  using super = IntegerStateBase<bool, 1, 0>;
  using base_t = IntegerStateBase::base_t;

  BooleanState() : super() {}
  BooleanState(base_t assumed) : super(assumed) {}

  // Returns true if the state is known to hold.
  bool isKnown() const { return getKnown(); }

  // Sets the known and asssumed value to |value|.
  void setKnown(bool value) {
    known |= value;
    assumed |= value;
  }

  // Returns true if the state is assumed to hold.
  bool isAssumed() const { return getAssumed(); }

  // Sets the assumed value to |value| but never below the known one.
  void setAssumed(bool value) { assumed &= (known | value); }

private:
  void handleNewKnownValue(base_t value) override {
    if (value)
      known = (assumed = value);
  }
  void handleNewAssumedValue(base_t value) override {
    if (!value)
      assumed = known;
  }

  void joinOR(base_t assumedValue, base_t knownValue) override {
    known |= knownValue;
    assumed |= assumedValue;
  }

  void joinAND(base_t assumedValue, base_t knownValue) override {
    known &= knownValue;
    assumed &= assumedValue;
  }
};

//===----------------------------------------------------------------------===//
// BitIntegerState
//===----------------------------------------------------------------------===//

// Specialization of the integer state for a bitwise encoding.
template <typename BaseTy = uint32_t, BaseTy BestState = ~BaseTy(0),
          BaseTy WorstState = 0>
struct BitIntegerState
    : public IntegerStateBase<BaseTy, BestState, WorstState> {
  using base_t = BaseTy;

  // Returns true if the bits set in |BitsEncoding| are "known bits".
  bool isKnown(base_t BitsEncoding) const {
    return (this->known & BitsEncoding) == BitsEncoding;
  }

  // Returns true if the bits set in |BitsEncoding| are "assumed bits".
  bool isAssumed(base_t BitsEncoding) const {
    return (this->assumed & BitsEncoding) == BitsEncoding;
  }

  // Adds the bits in |BitsEncoding| to the "known bits".
  BitIntegerState &addKnownBits(base_t Bits) {
    // Make sure we never miss any "known bits".
    this->assumed |= Bits;
    this->known |= Bits;
    return *this;
  }

  // Removes the bits in |BitsEncoding| from the "known bits".
  BitIntegerState &removeKnownBits(base_t BitsEncoding) {
    this->known = (this->known & ~BitsEncoding);
    return *this;
  }

  // Keeps only "assumed bits" also set in |BitsEncoding| but all known ones.
  BitIntegerState &intersectAssumedBits(base_t BitsEncoding) {
    // Make sure we never loose any "known bits".
    this->assumed = (this->assumed & BitsEncoding) | this->known;
    return *this;
  }

  // Removes the bits in |BitsEncoding| from the "assumed bits" if not known.
  BitIntegerState &removeAssumedBits(base_t BitsEncoding) {
    return intersectAssumedBits(~BitsEncoding);
  }

private:
  void handleNewKnownValue(base_t value) override { addKnownBits(value); }
  void handleNewAssumedValue(base_t value) override {
    intersectAssumedBits(value);
  }

  void joinOR(base_t assumedValue, base_t knownValue) override {
    this->known |= knownValue;
    this->assumed |= assumedValue;
  }

  void joinAND(base_t assumedValue, base_t knownValue) override {
    this->known &= knownValue;
    this->assumed &= assumedValue;
  }
};

//===----------------------------------------------------------------------===//
// IncIntegerState
//===----------------------------------------------------------------------===//

// Specialization of the integer state for an increasing value, hence ~0u is
// the best state and 0 the worst.
template <typename BaseTy = uint32_t, BaseTy BestState = ~BaseTy(0),
          BaseTy WorstState = 0>
struct IncIntegerState
    : public IntegerStateBase<BaseTy, BestState, WorstState> {
  using super = IntegerStateBase<BaseTy, BestState, WorstState>;
  using base_t = BaseTy;

  IncIntegerState() : super() {}
  IncIntegerState(base_t assumed) : super(assumed) {}

  // Returns the best possible representable state.
  static constexpr base_t getBestState() { return BestState; }
  static constexpr base_t
  getBestState(const IncIntegerState<BaseTy, BestState, WorstState> &) {
    return getBestState();
  }

  // Takes maximum of known and |value|.
  IncIntegerState &takeKnownMaximum(base_t value) {
    // Make sure we never loose "known value".
    this->assumed = std::max(value, this->assumed);
    this->known = std::max(value, this->known);
    return *this;
  }

  // Takes minimum of assumed and |value|.
  IncIntegerState &takeAssumedMinimum(base_t value) {
    // Make sure we never loose "known value".
    this->assumed = std::max(std::min(this->assumed, value), this->known);
    return *this;
  }

private:
  void handleNewKnownValue(base_t value) override { takeKnownMaximum(value); }
  void handleNewAssumedValue(base_t value) override {
    takeAssumedMinimum(value);
  }

  void joinOR(base_t assumedValue, base_t knownValue) override {
    this->known = std::max(this->known, knownValue);
    this->assumed = std::max(this->assumed, assumedValue);
  }

  void joinAND(base_t assumedValue, base_t knownValue) override {
    this->known = std::min(this->known, knownValue);
    this->assumed = std::min(this->assumed, assumedValue);
  }
};

//===----------------------------------------------------------------------===//
// DecIntegerState
//===----------------------------------------------------------------------===//

// Specialization of the integer state for a decreasing value, hence 0 is the
// best state and ~0u the worst.
template <typename BaseTy = uint32_t>
struct DecIntegerState : public IntegerStateBase<BaseTy, 0, ~BaseTy(0)> {
  using base_t = BaseTy;

  // Takes minimum of known and |value|.
  DecIntegerState &takeKnownMinimum(base_t value) {
    // Make sure we never loose "known value".
    this->assumed = std::min(value, this->assumed);
    this->known = std::min(value, this->known);
    return *this;
  }

  // Takes maximum of assumed and |value|.
  DecIntegerState &takeAssumedMaximum(base_t value) {
    // Make sure we never loose "known value".
    this->assumed = std::min(std::max(this->assumed, value), this->known);
    return *this;
  }

private:
  void handleNewKnownValue(base_t value) override { takeKnownMinimum(value); }
  void handleNewAssumedValue(base_t value) override {
    takeAssumedMaximum(value);
  }

  void joinOR(base_t assumedValue, base_t knownValue) override {
    this->assumed = std::min(this->assumed, knownValue);
    this->assumed = std::min(this->assumed, assumedValue);
  }

  void joinAND(base_t assumedValue, base_t knownValue) override {
    this->assumed = std::max(this->assumed, knownValue);
    this->assumed = std::max(this->assumed, assumedValue);
  }
};

//===----------------------------------------------------------------------===//
// PotentialValuesState<T>
//===----------------------------------------------------------------------===//

// A class for a set state.
// The assumed boolean state indicates whether the corresponding set is full
// set or not. If the assumed state is false this is the worst state. The
// worst state (invalid state) of a set of potential values is when the set
// contains every possible value (i.e. we cannot in any way limit the value
// that the target position can take) but that never happens naturally and we
// only ever force it.
template <typename MemberTy, typename KeyInfo = DenseMapInfo<MemberTy>>
struct PotentialValuesState : AbstractState {
  using SetTy = DenseSet<MemberTy, KeyInfo>;

  PotentialValuesState() : validState(true) {}
  explicit PotentialValuesState(bool isValid) : validState(isValid) {}

  bool isValidState() const override { return validState.isValidState(); }

  bool isAtFixpoint() const override { return validState.isAtFixpoint(); }

  ChangeStatus indicatePessimisticFixpoint() override {
    return validState.indicatePessimisticFixpoint();
  }

  ChangeStatus indicateOptimisticFixpoint() override {
    return validState.indicateOptimisticFixpoint();
  }

  // Returns the assumed state.
  PotentialValuesState &getAssumed() { return *this; }
  const PotentialValuesState &getAssumed() const { return *this; }

  // Returns this set. We should check whether this set is valid or not by
  // isValidState() before calling this function.
  const SetTy &getAssumedSet() const {
    assert(isValidState() && "This set should not be used when it is invalid!");
    return set;
  }

  // Returns whether this state contains an undef value or not.
  bool isUndefContained() const {
    assert(isValidState() &&
           "This flag should not be used when it is invalid!");
    return undefIsContained;
  }

  bool operator==(const PotentialValuesState &rhs) const {
    if (isValidState() != rhs.isValidState())
      return false;
    if (!isValidState() && !rhs.isValidState())
      return true;
    if (isUndefContained() != rhs.isUndefContained())
      return false;
    return set == rhs.getAssumedSet();
  }

  // Maximum number of potential values to be tracked.
  static constexpr unsigned maxPotentialValues = 256;

  // Returns empty set as the best state of potential values.
  static PotentialValuesState getBestState() {
    return PotentialValuesState(true);
  }
  static PotentialValuesState getBestState(PotentialValuesState &state) {
    return getBestState();
  }

  // Returns full set as the worst state of potential values.
  static PotentialValuesState getWorstState() {
    return PotentialValuesState(false);
  }

  // Unions assumed set with the passed value.
  void unionAssumed(const MemberTy &c) { insert(c); }
  // Unions assumed set with assumed set of the passed state |rhs|.
  void unionAssumed(const PotentialValuesState &rhs) { unionWith(rhs); }
  // Unions assumed set with an undef value.
  void unionAssumedWithUndef() { unionWithUndef(); }

  // Intersects assumed set with assumed set of the passed state |rhs|.
  void intersectAssumed(const PotentialValuesState &rhs) { intersectWith(rhs); }

  // "Clamps" this state with |rhs|.
  PotentialValuesState operator^=(const PotentialValuesState &rhs) {
    validState ^= rhs.validState;
    unionAssumed(rhs);
    return *this;
  }
  PotentialValuesState operator&=(const PotentialValuesState &rhs) {
    validState &= rhs.validState;
    unionAssumed(rhs);
    return *this;
  }

private:
  // Checks the size of this set and invalidates when the size exceeds the
  // specified maxPotentialValues threshold.
  void checkAndInvalidate() {
    if (set.size() >= maxPotentialValues) {
      indicatePessimisticFixpoint();
    } else {
      reduceUndefValue();
    }
  }

  // If this state contains both undef and not undef we can reduce
  // undef to the not undef value.
  void reduceUndefValue() { undefIsContained = undefIsContained & set.empty(); }

  // Inserts an element into this set.
  void insert(const MemberTy &c) {
    if (!isValidState())
      return;
    set.insert(c);
    checkAndInvalidate();
  }

  // Takes union with |rhs|.
  void unionWith(const PotentialValuesState &rhs) {
    // If this is a full set, do nothing.
    if (!isValidState())
      return;
    // If rhs is full set, change L to a full set.
    if (!rhs.isValidState()) {
      indicatePessimisticFixpoint();
      return;
    }
    for (const MemberTy &c : rhs.set)
      set.insert(c);
    undefIsContained |= rhs.isUndefContained();
    checkAndInvalidate();
  }

  // Takes union with an undef value.
  void unionWithUndef() {
    undefIsContained = true;
    reduceUndefValue();
  }

  // Takes intersection with |rhs|.
  void intersectWith(const PotentialValuesState &rhs) {
    // If rhs is a full set, do nothing.
    if (!rhs.isValidState())
      return;
    // If this is a full set, change this to rhs.
    if (!isValidState()) {
      *this = rhs;
      return;
    }
    SetTy intersectSet;
    for (const MemberTy &c : set) {
      if (rhs.set.count(c))
        intersectSet.insert(c);
    }
    set = intersectSet;
    undefIsContained &= rhs.isUndefContained();
    reduceUndefValue();
  }

  // A helper state which indicate whether this state is valid or not.
  BooleanState validState;
  // Container for potential values.
  SetTy set;
  // Flag for undef value.
  bool undefIsContained = false;
};

using PotentialConstantIntValuesState = PotentialValuesState<APInt>;

//===----------------------------------------------------------------------===//
// PotentialCountsState<T>
//===----------------------------------------------------------------------===//

/// Idiomatic saturated integer-like type to represent saturated arithmetic.
/// Note that the bounds for saturation is determined by the user.
struct SaturatedInteger {
  bool operator==(const SaturatedInteger other) const {
    return (saturated && other.saturated) ||
           (!saturated && !other.saturated && v == other.v);
  }
  bool operator!=(const SaturatedInteger other) const {
    return !(*this == other);
  }
  bool operator==(const int64_t other) const {
    return !saturated && (v == other);
  }
  bool operator!=(const int64_t other) const { return !(*this == other); }
  SaturatedInteger operator+(const SaturatedInteger other) const {
    if (saturated || other.saturated)
      return SaturatedInteger{true, 0};
    return SaturatedInteger{false, other.v + v};
  }
  SaturatedInteger operator*(const SaturatedInteger other) const {
    if (saturated || other.saturated)
      return SaturatedInteger{true, 0};
    return SaturatedInteger{false, other.v * v};
  }
  SaturatedInteger operator-(const SaturatedInteger other) const {
    if (saturated || other.saturated)
      return SaturatedInteger{true, 0};
    return SaturatedInteger{false, v - other.v};
  }
  SaturatedInteger operator-(const int64_t other) const {
    return *this - SaturatedInteger{false, other};
  }

  bool saturated = true;
  int64_t v = 0;
};

// A class for a multiset state.
// The assumed boolean state indicates whether the corresponding multiset is
// a full multiset or not. If the assumed state is false this is the worst
// state. The worst state (invalid state) of a set of potential values is when
// the multiset contains infinite copies of every possible value (i.e. we cannot
// in any way limit the value that the target position can take) but that never
// happens naturally and we only ever force it.
template <typename MemberTy, typename KeyInfo = DenseMapInfo<MemberTy>>
struct PotentialCountsState : AbstractState {
  using MapTy = DenseMap<MemberTy, SaturatedInteger, KeyInfo>;

  PotentialCountsState() : validState(true) {}
  explicit PotentialCountsState(bool isValid) : validState(isValid) {}

  bool isValidState() const override { return validState.isValidState(); }

  bool isAtFixpoint() const override { return validState.isAtFixpoint(); }

  ChangeStatus indicatePessimisticFixpoint() override {
    return validState.indicatePessimisticFixpoint();
  }

  ChangeStatus indicateOptimisticFixpoint() override {
    return validState.indicateOptimisticFixpoint();
  }

  // Returns the assumed state.
  PotentialCountsState &getAssumed() { return *this; }
  const PotentialCountsState &getAssumed() const { return *this; }

  // Returns this set. We should check whether this set is valid or not by
  // isValidState() before calling this function.
  const MapTy &getAssumedMultiSet() const {
    assert(isValidState() &&
           "This multiset should not be used when it is invalid!");
    return multiset;
  }

  // Returns whether this state contains an undef value or not.
  bool isUndefContained() const {
    assert(isValidState() &&
           "This flag should not be used when it is invalid!");
    return undefIsContained;
  }

  bool operator==(const PotentialCountsState &rhs) const {
    if (isValidState() != rhs.isValidState())
      return false;
    if (!isValidState() && !rhs.isValidState())
      return true;
    if (isUndefContained() != rhs.isUndefContained())
      return false;
    return multiset == rhs.getAssumedMultiSet();
  }

  // Maximum number of potential values to be tracked.
  static constexpr unsigned maxPotentialValues = 256;

  // Returns empty set as the best state of potential values.
  static PotentialCountsState getBestState() {
    return PotentialCountsState(true);
  }
  static PotentialCountsState getBestState(PotentialCountsState &state) {
    return getBestState();
  }

  // Returns full set as the worst state of potential values.
  static PotentialCountsState getWorstState() {
    return PotentialCountsState(false);
  }

  // Unions assumed multiset with the passed value assuming an infinite count.
  void unionAssumed(const MemberTy &c) {
    insert(c, SaturatedInteger{/*saturated=*/true, 0});
  }
  // Unions assumed multiset with the passed value and count.
  void unionAssumed(const MemberTy &c, int64_t s) {
    insert(c, SaturatedInteger{/*saturated=*/false, s});
  }
  // Unions assumed multiset with assumed set of the passed state |rhs|.
  void unionAssumed(const PotentialCountsState &rhs) { unionWith(rhs); }
  // Unions assumed multiset with an undef value.
  void unionAssumedWithUndef() { unionWithUndef(); }

  // "Clamps" this state with |rhs|.
  PotentialCountsState operator^=(const PotentialCountsState &rhs) {
    validState ^= rhs.validState;
    unionAssumed(rhs);
    return *this;
  }
  PotentialCountsState operator&=(const PotentialCountsState &rhs) {
    validState &= rhs.validState;
    unionAssumed(rhs);
    return *this;
  }

private:
  // Checks the size of this set and invalidates when the size exceeds the
  // specified maxPotentialValues threshold.
  void checkAndInvalidate() {
    if (multiset.size() >= maxPotentialValues) {
      indicatePessimisticFixpoint();
    } else {
      reduceUndefValue();
    }
  }

  // If this state contains both undef and not undef we can reduce
  // undef to the not undef value.
  void reduceUndefValue() {
    undefIsContained = undefIsContained & multiset.empty();
  }

  // Inserts an element into this set.
  void insert(const MemberTy &c, SaturatedInteger s) {
    if (!isValidState())
      return;
    multiset[c] = s;
    checkAndInvalidate();
  }

  // Takes union with |rhs|.
  void unionWith(const PotentialCountsState &rhs) {
    // If this is a full set, do nothing.
    if (!isValidState())
      return;
    // If rhs is full set, change L to a full set.
    if (!rhs.isValidState()) {
      indicatePessimisticFixpoint();
      return;
    }
    for (auto &[c, count] : rhs.multiset) {
      if (multiset.contains(c)) {
        multiset[c] = multiset[c] + count;
      } else {
        multiset[c] = count;
      }
    }
    undefIsContained |= rhs.isUndefContained();
    checkAndInvalidate();
  }

  // Takes union with an undef value.
  void unionWithUndef() {
    undefIsContained = true;
    reduceUndefValue();
  }

  // A helper state which indicate whether this state is valid or not.
  BooleanState validState;
  // Container for potential values.
  MapTy multiset;
  // Flag for undef value.
  bool undefIsContained = false;
};

//===----------------------------------------------------------------------===//
// State utilities
//===----------------------------------------------------------------------===//

// Clamps |state| with information in |resultState| and returns an indication
// as to whether |state| changed (as in an update is required to be run again).
template <typename StateType>
ChangeStatus clampStateAndIndicateChange(StateType &state,
                                         const StateType &resultState) {
  auto assumed = state.getAssumed();
  state ^= resultState;
  return assumed == state.getAssumed() ? ChangeStatus::UNCHANGED
                                       : ChangeStatus::CHANGED;
}

// Replaces |state| with information in |resultState| and returns an indication
// as to whether |state| changed (as in an update is required to be run again).
template <typename StateType>
ChangeStatus inheritStateAndIndicateChange(StateType &state,
                                           const StateType &resultState) {
  auto assumed = state.getAssumed();
  state = resultState;
  return assumed == state.getAssumed() ? ChangeStatus::UNCHANGED
                                       : ChangeStatus::CHANGED;
}

// Helper to tie a abstract state implementation to an abstract element.
//
// Usage:
//  struct MyElement : public StateWrapper<IntegerRangeState, AbstractElement> {
//    ...
//  };
template <typename StateTy, typename BaseType, class... Ts>
struct StateWrapper : public BaseType, public StateTy {
  // Provide static access to the type of the state.
  using StateType = StateTy;

  StateWrapper(const Position &pos, Ts... Args)
      : BaseType(pos), StateTy(Args...) {}

  StateType &getState() override { return *this; }
  const StateType &getState() const override { return *this; }
};

} // namespace DFX

//===----------------------------------------------------------------------===//
// Debugging utilities
//===----------------------------------------------------------------------===//

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ChangeStatus status);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const DFX::AbstractState &state);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, DFX::SaturatedInteger s);

template <typename base_ty, base_ty BestState, base_ty WorstState>
llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const DFX::IntegerStateBase<base_ty, BestState, WorstState> &state) {
  return os << "(" << state.getKnown() << "-" << state.getAssumed() << ")"
            << static_cast<const DFX::AbstractState &>(state);
}

llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const DFX::PotentialConstantIntValuesState &state);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_STATE_H_
