// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/compiler/Dialect/Indexing/IR/IndexingInterfaces.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

namespace mlir::iree_compiler::IREE::Indexing {

//===----------------------------------------------------------------------===//
// Index Range State
//===----------------------------------------------------------------------===//

// State that tracks floating point ranges and flags.
struct IndexRangeState : public DFX::AbstractState {
  bool isValidState() const override { return valid; }
  bool isAtFixpoint() const override { return isAtFixedPoint; }

  ChangeStatus indicateOptimisticFixpoint() override {
    valid = true;
    isAtFixedPoint = true;
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    valid = false;
    isAtFixedPoint = true;
    return ChangeStatus::CHANGED;
  }

  void setWidestIndexRange() {
    assumed = SaturatedIndexRange();
    valid = true;
  }

  const SaturatedIndexRange &getAssumed() const { return assumed; }

  void setAssumed(SaturatedIndexRange &newAssumed) {
    assumed = newAssumed;
    valid = true;
  }

  // "Clamps" this state with |rhs|. The assumed value will contain the
  // intersection of information assumed by both states (i.e. smallest
  // possible range).
  void operator+=(const IndexRangeState &rhs) {
    // If one is invalid, take the valid one.
    if (!valid) {
      return;
    }
    if (!rhs.valid) {
      *this = rhs;
      return;
    }
    assumed = assumed.getIntersection(rhs.assumed);
  }

  // "Clamps" this state with |rhs|. The assumed value will contain the
  // union of information assumed by both states (i.e. largest
  // possible range).
  void operator^=(const IndexRangeState &rhs) {
    // If one is invalid, take the valid one.
    if (!rhs.valid) {
      return;
    }
    if (!valid) {
      *this = rhs;
      return;
    }
    assumed = assumed.getUnion(rhs.assumed);
  }

  const std::string getAsStr(AsmState &asmState) const;

private:
  SaturatedIndexRange assumed = SaturatedIndexRange();
  bool valid = false;
  bool isAtFixedPoint = false;
};

//===----------------------------------------------------------------------===//
// Index Range Element
//===----------------------------------------------------------------------===//

// Attribute known ranges of integer or index values.
class IndexRangeValueElement
    : public DFX::StateWrapper<IndexRangeState, DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<IndexRangeState, DFX::ValueElement>;
  using BaseType::BaseType;

  static IndexRangeValueElement &createForPosition(const Position &pos,
                                                   DFX::Solver &solver) {
    return *(new (solver.getAllocator()) IndexRangeValueElement(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override {
    return "IndexRangeValueElement";
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

//===----------------------------------------------------------------------===//
// Shaped Dims Range State
//===----------------------------------------------------------------------===//

// State that tracks floating point ranges and flags.
struct ShapedDimsRangeState : public DFX::AbstractState {
  bool isValidState() const override { return valid; }
  bool isAtFixpoint() const override { return isAtFixedPoint; }

  ChangeStatus indicateOptimisticFixpoint() override {
    valid = true;
    isAtFixedPoint = true;
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    valid = false;
    isAtFixedPoint = true;
    return ChangeStatus::CHANGED;
  }

  void setWidestAssumedDynamicDims(ShapedType type) {
    assumed.clear();
    for (auto size : type.getShape()) {
      if (ShapedType::isDynamic(size)) {
        assumed.push_back(SaturatedIndexRange(false, true, 0, 0, 1));
      }
    }
    valid = true;
  }

  void setWidestIndexRange() {
    assumed.clear();
    assumed.push_back(SaturatedIndexRange());
    valid = true;
  }

  const SaturatedIndexRangeList &getAssumed() const { return assumed; }

  void setAssumed(SaturatedIndexRangeList &newAssumed) {
    valid = true;
    assumed = newAssumed;
  }
  void setAssumed(SaturatedIndexRange &newAssumed) {
    assumed.clear();
    assumed.push_back(newAssumed);
    valid = true;
  }

  // "Clamps" this state with |rhs|. The assumed value will contain the
  // intersection of information assumed by both states (i.e. largest
  // possible range).
  void operator+=(const ShapedDimsRangeState &rhs) {
    // If one is invalid, take the valid one.
    if (!valid) {
      return;
    }
    if (!rhs.valid) {
      *this = rhs;
      return;
    }
    assert(assumed.size() == rhs.assumed.size() && "Invalid dim range union");
    for (int i = 0, e = assumed.size(); i < e; ++i) {
      assumed[i] = assumed[i].getIntersection(rhs.assumed[i]);
    }
  }

  // "Clamps" this state with |rhs|. The assumed value will contain the
  // union of information assumed by both states (i.e. smallest
  // possible range).
  void operator^=(const ShapedDimsRangeState &rhs) {
    // If one is invalid, take the valid one.
    if (!rhs.valid) {
      return;
    }
    if (!valid) {
      *this = rhs;
      return;
    }
    assert(assumed.size() == rhs.assumed.size() &&
           "Invalid dim range intersection");
    for (int i = 0, e = assumed.size(); i < e; ++i) {
      assumed[i] = assumed[i].getUnion(rhs.assumed[i]);
    }
  }

  const std::string getAsStr(AsmState &asmState) const;

private:
  SaturatedIndexRangeList assumed = {};
  bool valid = false;
  bool isAtFixedPoint = false;
};

//===----------------------------------------------------------------------===//
// Shaped Dims Range Element
//===----------------------------------------------------------------------===//

// Attribute known ranges of dynamic dimensions of shaped values.
class ShapedDimsRangeValueElement
    : public DFX::StateWrapper<ShapedDimsRangeState, DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<ShapedDimsRangeState, DFX::ValueElement>;
  using BaseType::BaseType;

  static ShapedDimsRangeValueElement &createForPosition(const Position &pos,
                                                        DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ShapedDimsRangeValueElement(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override {
    return "ShapedDimsRangeValueElement";
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

} // namespace mlir::iree_compiler::IREE::Indexing
