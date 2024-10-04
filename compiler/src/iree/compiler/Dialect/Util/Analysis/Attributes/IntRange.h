// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_ATTRIBUTES_INTRANGE_H_
#define IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_ATTRIBUTES_INTRANGE_H_

#include <algorithm>
#include <cmath>
#include <string>

#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"

namespace mlir::iree_compiler::IREE::Util {

// Represents an integer value with a maximum representable range of +/- the
// range of an unsigned int64 quantity. Has special bits for invalid and out
// of range (hi or low based on sign). Representing it like this lets us have
// the maximum possible machine representation with floating point like
// semantics for NAN and +/- Inf. This is useful for tracking min/max ranges.
class IntegerValueInfo {
public:
  static IntegerValueInfo getInvalid() {
    return IntegerValueInfo(0, true, false, false);
  }
  static IntegerValueInfo getOutOfRangePositive() {
    return IntegerValueInfo(0, false, false, true);
  }
  static IntegerValueInfo getOutOfRangeNegative() {
    return IntegerValueInfo(0, false, true, true);
  }
  static IntegerValueInfo getSigned(int64_t value) {
    return IntegerValueInfo(std::abs(value), false, value < 0, false);
  }
  static IntegerValueInfo getUnsigned(uint64_t value) {
    return IntegerValueInfo(value, false, false, false);
  }

  bool isInvalid() const { return invalid; }

  // Returns the max of this instance and another value.
  IntegerValueInfo max(IntegerValueInfo other) const {
    if (invalid)
      return *this;
    if (*this <= other)
      return other;
    else
      return *this;
  }

  // Returns the min of this instance and another value.
  IntegerValueInfo min(IntegerValueInfo other) const {
    if (invalid)
      return *this;
    if (*this <= other)
      return *this;
    else
      return other;
  }

  bool operator<=(const IntegerValueInfo rhs) const;
  bool operator==(const IntegerValueInfo rhs) const {
    return magnitude == rhs.magnitude && invalid == rhs.invalid &&
           negative == rhs.negative && out_of_range == rhs.out_of_range;
  }

  std::string to_s() const;

private:
  IntegerValueInfo(uint64_t magnitude, bool invalid, bool negative,
                   bool out_of_range)
      : magnitude(magnitude), invalid(invalid), negative(negative),
        out_of_range(out_of_range) {}
  uint64_t magnitude;
  bool invalid : 1;
  bool negative : 1;
  bool out_of_range : 1;
};

// State that tracks the possible minimum value of an integer.
class IntegerMinState : public DFX::AbstractState {
public:
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

  IntegerValueInfo getAssumed() const { return assumed; }
  IntegerValueInfo getKnown() const { return known; }

  void setAssumed(IntegerValueInfo newAssumed) { assumed = newAssumed; }

  void operator^=(const IntegerMinState &rhs) {
    assumed = assumed.min(rhs.getAssumed());
  }

private:
  IntegerValueInfo assumed = IntegerValueInfo::getInvalid();
  IntegerValueInfo known = IntegerValueInfo::getOutOfRangePositive();
};

class IntegerMinValueElement
    : public DFX::StateWrapper<IntegerMinState, DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<IntegerMinState, DFX::ValueElement>;
  using BaseType::BaseType;

  static IntegerMinValueElement &createForPosition(const Position &pos,
                                                   DFX::Solver &solver) {
    return *(new (solver.getAllocator()) IntegerMinValueElement(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override {
    return "IntegerMinValueElement";
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

// State that tracks the possible minimum value of an integer.
class IntegerMaxState : public DFX::AbstractState {
public:
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

  IntegerValueInfo getAssumed() const { return assumed; }
  IntegerValueInfo getKnown() const { return known; }

  void setAssumed(IntegerValueInfo newAssumed) { assumed = newAssumed; }

  void operator^=(const IntegerMaxState &rhs) {
    assumed = assumed.max(rhs.getAssumed());
  }

private:
  IntegerValueInfo assumed = IntegerValueInfo::getInvalid();
  IntegerValueInfo known = IntegerValueInfo::getOutOfRangeNegative();
};

class IntegerMaxValueElement
    : public DFX::StateWrapper<IntegerMaxState, DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<IntegerMaxState, DFX::ValueElement>;
  using BaseType::BaseType;

  static IntegerMaxValueElement &createForPosition(const Position &pos,
                                                   DFX::Solver &solver) {
    return *(new (solver.getAllocator()) IntegerMaxValueElement(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override {
    return "IntegerMaxValueElement";
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

// State that tracks the maximum integer divisor.
class IntegerMaxDivisorState : public DFX::AbstractState {
public:
  bool isValidState() const override { return (bool)assumed; }
  bool isAtFixpoint() const override { return assumed == known; }

  ChangeStatus indicateOptimisticFixpoint() override {
    known = assumed;
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    assumed = known;
    return ChangeStatus::CHANGED;
  }

  std::optional<uint64_t> getAssumed() const { return assumed; }
  std::optional<uint64_t> getKnown() const { return known; }

  void setAssumed(std::optional<uint64_t> newAssumed) { assumed = newAssumed; }

  void operator^=(const IntegerMaxDivisorState &rhs);

private:
  std::optional<uint64_t> known = 1;
  std::optional<uint64_t> assumed;
};

class IntegerMaxDivisorElement
    : public DFX::StateWrapper<IntegerMaxDivisorState, DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<IntegerMaxDivisorState, DFX::ValueElement>;
  using BaseType::BaseType;

  static IntegerMaxDivisorElement &createForPosition(const Position &pos,
                                                     DFX::Solver &solver) {
    return *(new (solver.getAllocator()) IntegerMaxDivisorElement(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override {
    return "IntegerMaxDivisorElement";
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

#endif // IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_ATTRIBUTES_INTRANGE_H_
