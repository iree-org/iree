// Copyright 2024 The IREE Authors
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
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::iree_compiler::IREE::Util {

//===----------------------------------------------------------------------===//
// Integer range constraints
//===----------------------------------------------------------------------===//

// Structure to represent the upper bounds and lower bounds of the results of
// a function in terms of the arguments. There are three valid states for
// each bound list |lbs| and |ubs|.
//
// 1. Bound list is empty, indicating no bound.
// 2. Bound list is non-empty with size equal to the number of function results.
//   a. An entry is the empty map ( () -> () ), meaning that result is unbounded
//      for that bound type.
//   b. An entry is non-empty, and has a number of dimension inputs equal to the
//      number of arguments to the function ( (d0, ..., dn) -> (10) ).
struct FunctionalRangeMaps {
  SmallVector<AffineMap> lbs;
  SmallVector<AffineMap> ubs;
  bool valid = false;

  FunctionalRangeMaps() {}
  FunctionalRangeMaps(ArrayRef<AffineMap> l, ArrayRef<AffineMap> u)
      : lbs(l), ubs(u), valid(true) {}

  static FunctionalRangeMaps getInvalid() { return FunctionalRangeMaps(); }
  static FunctionalRangeMaps getUnbounded() {
    return FunctionalRangeMaps({}, {});
  }

  // Reset to initial state.
  void reset() { *this = FunctionalRangeMaps(); }

  bool isInvalid() const { return !valid; }
  bool isUnbounded() const { return lbs.empty() || ubs.empty(); }
  bool isConstant() const {
    return !isUnbounded() && lbs == ubs && llvm::all_of(lbs, [](AffineMap lb) {
      return lb && lb.isSingleConstant();
    });
  }

  bool operator==(const FunctionalRangeMaps &other) const {
    return valid == other.valid && lbs == other.lbs && ubs == other.ubs;
  }

  std::string getAsStr(AsmState &asmState) const;
};

// State that tracks the valid ranges of integer output values and dimension
// sizes of shaped results of a function with respect to the function arguments.
struct FunctionalRangeState : public DFX::AbstractState {
  // Returns the worst possible representable state.
  static FunctionalRangeMaps getWorstState() {
    return FunctionalRangeMaps::getUnbounded();
  }
  static FunctionalRangeMaps getWorstState(const FunctionalRangeState &) {
    return getWorstState();
  }

  bool isValidState() const override { return !assumed.isInvalid(); }
  bool isAtFixpoint() const override { return assumed.isConstant(); }

  ChangeStatus indicateOptimisticFixpoint() override {
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    return ChangeStatus::UNCHANGED;
  }

  FunctionalRangeMaps getAssumed() const { return assumed; }

  // "Clamps" this state with |rhs|. The assumed value will be replaced by the
  // information in the new state.
  void operator^=(const FunctionalRangeState &rhs) {
    FunctionalRangeMaps rhsAssumed = rhs.getAssumed();
    assumed = rhsAssumed;
  }

  // Resets the assumed value to the given value. This does no unioning and
  // assumes it is a proper fixpoint minimum.
  void setAssumed(FunctionalRangeMaps newAssumed) { assumed = newAssumed; }

private:
  FunctionalRangeMaps assumed = FunctionalRangeMaps::getInvalid();
};

//===----------------------------------------------------------------------===//
// Constraint construction helpers
//===----------------------------------------------------------------------===//

// Wrapper around ValueBoundsConstraintSet for building contraints on values
// contained within a particular function.
class FunctionConstraintSet : ValueBoundsConstraintSet {
public:
  // Public constructor for the constraint set.
  FunctionConstraintSet(MLIRContext *context)
      : ValueBoundsConstraintSet(context){};

  // Constructs a ValueBoundsConstraintSet of all of the values in the given
  // function.
  static std::unique_ptr<FunctionConstraintSet>
  constructConstraintSet(Explorer &explorer, FunctionOpInterface func);

  // Add the bounds specified by |range| to the arguments and results of |op| in
  // this constraint set. Assumes the dimension counts of range line up with the
  // operand/result counts of the given operation.
  void incorporateFunctionalRangeBounds(Operation *op,
                                        FunctionalRangeMaps range);

  SmallVector<AffineExpr> getDimensionList(ValueDimList &operands);

  // Gets the bound maps of the given ValueDim in terms of the list of values
  // specified by |operands|.
  std::pair<AffineMap, AffineMap>
  getBounds(ValueDimList &operands, Value operand, std::optional<long> dim);

  // Returns the bounds of the given ValueDim without any dimension replacement.
  // The maps returned by this function have no practical use outside of basic
  // constant bound detection and limited GCD computation.
  std::pair<AffineMap, AffineMap> getUnspecifiedBounds(Value operand,
                                                       std::optional<long> dim);

  // Gets the constant bounds of the given ValueDim, if present, as <lb, ub>.
  std::pair<std::optional<int64_t>, std::optional<int64_t>>
  getConstantBounds(Value operand, std::optional<int64_t>);

private:
  void canonicalize();
};

//===----------------------------------------------------------------------===//
// Operation element
//===----------------------------------------------------------------------===//

// Operation element to track the constraint map for a specific function.
// Maintains an internal FunctionConstraintSet that is not compared as a part
// of the state. The constraint set is updated when another function called
// by this function has their state updated.
class FunctionRangeOperationElement
    : public DFX::StateWrapper<FunctionalRangeState, DFX::TypedOperationElement<
                                                         FunctionOpInterface>> {
public:
  using BaseType =
      DFX::StateWrapper<FunctionalRangeState,
                        DFX::TypedOperationElement<FunctionOpInterface>>;
  using BaseType::BaseType;

  static FunctionRangeOperationElement &createForPosition(const Position &pos,
                                                          DFX::Solver &solver) {
    return *(new (solver.getAllocator()) FunctionRangeOperationElement(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override {
    return "FunctionRangeOperationElement";
  }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  const std::string getAsStr(AsmState &asmState) const override;

  // Analysis querying helpers.
  const std::pair<AffineMap, AffineMap>
  getBounds(Value operand, std::optional<long> dim) const {
    assert(constraints && "Getting value bounds without constraints");
    return constraints->getUnspecifiedBounds(operand, dim);
  }

  const std::pair<std::optional<int64_t>, std::optional<int64_t>>
  getConstantBounds(Value operand, std::optional<long> dim) const {
    assert(constraints && "Getting constant bounds without constraints");
    return constraints->getConstantBounds(operand, dim);
  }

private:
  void initializeOperation(FunctionOpInterface func,
                           DFX::Solver &solver) override;
  ChangeStatus updateOperation(FunctionOpInterface func,
                               DFX::Solver &solver) override;

  // Constraint set for tracking the set of internal constraints to the
  // associated function. Constructed on element initialization.
  std::unique_ptr<FunctionConstraintSet> constraints = nullptr;
};

//===----------------------------------------------------------------------===//
// Index range analysis
//===----------------------------------------------------------------------===//

class IndexRangeAnalysis {
public:
  explicit IndexRangeAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
        TraversalAction::RECURSE);
    // Ignore the contents of executables (linalg goo, etc).
    explorer.initialize();

    assert(rootOp->getNumRegions() == 1 && "expected module-like root op");
    topLevelOps = llvm::to_vector(
        rootOp->getRegions().front().getOps<mlir::FunctionOpInterface>());
  }

  // Runs analysis and populates the state cache with value constraint
  // information.
  LogicalResult run();

  std::pair<AffineMap, AffineMap> getBounds(Value operand,
                                            std::optional<long> dim);

  std::pair<std::optional<int64_t>, std::optional<int64_t>>
  getConstantBounds(Value operand, std::optional<long> dim);

  int64_t getStaticGCD(Value operand, std::optional<long> dim);

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
  SmallVector<mlir::FunctionOpInterface> topLevelOps;
};

} // namespace mlir::iree_compiler::IREE::Util
