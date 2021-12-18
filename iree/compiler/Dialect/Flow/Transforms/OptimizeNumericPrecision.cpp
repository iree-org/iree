// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <limits>

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-flow-optimize-numeric-precision"

using llvm::dbgs;
using llvm::Optional;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Returns whether the given type is a valid scalar fp type or a shaped type
// of fp types.
bool isFpType(Type type) {
  if (type.isa<FloatType>()) return true;
  if (auto st = type.dyn_cast<ShapedType>()) {
    return st.getElementType().isa<FloatType>();
  }
  return false;
}

using PotentialAttributeValuesState = DFX::PotentialValuesState<Attribute>;
// Potential value set of Attribute instances representing constant values.
class ConstantAttributePVS
    : public DFX::StateWrapper<PotentialAttributeValuesState,
                               DFX::ValueElement> {
 public:
  using BaseType =
      DFX::StateWrapper<PotentialAttributeValuesState, DFX::ValueElement>;
  using BaseType::BaseType;

  static ConstantAttributePVS &createForPosition(const Position &pos,
                                                 DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ConstantAttributePVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "ConstantAttributePVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr() const override {
    std::string str;
    llvm::raw_string_ostream sstream(str);
    sstream << "pvs: ";
    if (isValidState()) {
      sstream << "[";
      if (isUndefContained()) {
        sstream << "undef, ";
      }
      llvm::interleaveComma(getAssumedSet(), sstream,
                            [&](Attribute value) { value.print(sstream); });
      sstream << "]";
    } else {
      sstream << "(invalid)";
    }
    sstream.flush();
    return str;
  }

 private:
  void initializeValue(Value value, DFX::Solver &solver) override {
    Attribute staticValue;
    if (matchPattern(value, m_Constant(&staticValue))) {
      dbgs() << "ConstantAttributePVS: Match constant\n";
      unionAssumed(staticValue);
      indicateOptimisticFixpoint();
    }
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    StateType newState;
    // dbgs() << "-- UPDATE VALUE: " << value << "\n";
    if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
          // dbgs() << "  UPDATE DEFINING OP: " << result << "\n";
          Attribute staticValue;
          if (matchPattern(value, m_Constant(&staticValue))) {
            // dbgs() << "    ConstantAttributePVS: Match constant\n";
            unionAssumed(staticValue);
            return WalkResult::advance();
          }

          // TODO: Walk joins.

          // Dynamic ops we can't see through.
          newState.unionAssumedWithUndef();
          newState.indicatePessimisticFixpoint();
          return WalkResult::advance();
        }) == TraversalResult::INCOMPLETE) {
      // Incomplete traversal.
      newState.unionAssumedWithUndef();
      newState.indicatePessimisticFixpoint();
    }

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }
};

const char ConstantAttributePVS::ID = 0;

// Structure for representing floating point bounds.
struct FpRange {
  enum TruncationFlag {
    TRUNC = 0,
    TRUNC_UNKNOWN = 1,
  };

  double minValue = NAN;
  double maxValue = NAN;
  bool valid = false;
  TruncationFlag truncationFlag = TRUNC_UNKNOWN;

  FpRange() {}
  FpRange(double minValue, double maxValue)
      : minValue(minValue), maxValue(maxValue), valid(true) {}

  static FpRange getInvalid() { return FpRange(); }
  static FpRange getWidest() { return FpRange(-INFINITY, INFINITY); }

  static TruncationFlag determineTruncationFlag(double value) {
    double truncValue = trunc(value);
    // TODO: I'm sure I need to be doing some ULP comparison.
    return truncValue == value ? TRUNC : TRUNC_UNKNOWN;
  }

  static TruncationFlag unionTruncationFlag(TruncationFlag lhs,
                                            TruncationFlag rhs) {
    return static_cast<TruncationFlag>(std::max(lhs, rhs));
  }

  // Reset to initial state.
  void reset() { *this = FpRange(); }

  // Adds a value to the range for the case when all possible values are
  // known. Call reset() prior to iteration to enter the open-range state.
  void addDomainValue(double value) {
    dbgs() << "ADD DOMAIN VALUE: " << value << " (FROM " << getAsStr() << ")\n";
    if (!valid) {
      minValue = value;
      maxValue = value;
      truncationFlag = determineTruncationFlag(value);
      valid = true;
    } else {
      minValue = std::min(minValue, value);
      maxValue = std::max(maxValue, value);
      truncationFlag =
          unionTruncationFlag(determineTruncationFlag(value), truncationFlag);
    }
  }

  bool isInvalid() const { return !valid; }

  bool operator==(const FpRange &other) const {
    return valid == other.valid && minValue == other.minValue &&
           maxValue == other.maxValue;
  }

  bool operator!=(const FpRange &other) const { return !(*this == other); }

  // Makes this state the union of information known by both.
  void operator+=(const FpRange &rhs) {
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

  std::string getAsStr() const {
    if (!valid) return std::string("<<INVALID>>");
    std::string s("[");
    s += std::to_string(minValue);
    s += ", ";
    s += std::to_string(maxValue);
    s += ", ";
    switch (truncationFlag) {
      case TRUNC_UNKNOWN:
        s += "!trunc";
        break;
      case TRUNC:
        s += "TRUNC";
        break;
    }
    s += "]";
    return s;
  }
};

// State that tracks the floating point range.
// This uses doubles to track the values. It may be possible
// to make it polymorphic at a future point.
struct FpRangeState : public DFX::AbstractState {
  // Returns the best possible representable state.
  // static FpRange getBestState() { return FpRange::getZero(); }
  // static FpRange getBestState(const FpRangeState &) { return getBestState();
  // }

  // Returns the worst possible representable state.
  static FpRange getWorstState() { return FpRange::getWidest(); }
  static FpRange getWorstState(const FpRangeState &) { return getWorstState(); }

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

  FpRange getAssumed() const { return assumed; }
  FpRange getKnown() const { return known; }

  // Resets the assumed value to the given value. This does no unioning and
  // assumes it is a proper minimum.
  void setAssumed(FpRange newAssumed) { assumed = newAssumed; }

  void handleMinf(const FpRange &lhs, const FpRange &rhs) {
    assumed += lhs;
    assumed += rhs;
    // Narrow the upper bound.
    if (assumed.valid) {
      assumed.maxValue = std::min(lhs.maxValue, rhs.maxValue);
    }
  }

  void handleMaxf(const FpRange &lhs, const FpRange &rhs) {
    assumed += lhs;
    assumed += rhs;
    // Narrow the lower bound.
    if (assumed.valid) {
      assumed.minValue = std::max(lhs.minValue, rhs.minValue);
    }
  }

  void handleFloor(const FpRange &operand) {
    assumed += operand;
    // Apply floor to the bounds.
    if (assumed.valid) {
      assumed.minValue = std::floor(assumed.minValue);
      assumed.maxValue = std::floor(assumed.maxValue);
      assumed.truncationFlag = FpRange::TRUNC;
    }
  }

  // "Clamps" this state with |rhs|. The assumed value will contain the union
  // of information assumed by both states.
  void operator^=(const FpRangeState &rhs) {
    FpRange rhsAssumed = rhs.getAssumed();
    assumed += rhsAssumed;
  }

 private:
  FpRange assumed = FpRange::getInvalid();
  FpRange known = FpRange::getWidest();
};

class ValueFpRange : public DFX::StateWrapper<FpRangeState, DFX::ValueElement> {
 public:
  using BaseType = DFX::StateWrapper<FpRangeState, DFX::ValueElement>;
  using BaseType::BaseType;

  static ValueFpRange &createForPosition(const Position &pos,
                                         DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueFpRange(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override { return "ValueFpRange"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  const std::string getAsStr() const override {
    auto range = getAssumed();
    std::string s("fp-range: ");
    s += range.getAsStr();
    return s;
  }

 private:
  void initializeValue(Value value, DFX::Solver &solver) override {
    if (!isFpType(value.getType())) {
      indicatePessimisticFixpoint();
      return;
    }
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    FpRangeState newState = getState();

    // If this is a block argument to a supported parent operation, then
    // remap the value we are working on to the correct input from the
    // parent. This works because the statistics we are accumulating flow
    // layer-wise, and we don't need to pay particular attention to specific
    // loop structures.
    if (auto valueBlockArg = value.dyn_cast<BlockArgument>()) {
      Block *ownerBlock = valueBlockArg.getOwner();
      if (auto linalgParent = llvm::dyn_cast_or_null<linalg::LinalgOp>(
              ownerBlock->getParentOp())) {
        value = linalgParent->getOperand(valueBlockArg.getArgNumber());
        dbgs() << "  ++ REMAP LINALG BLOCK ARG TO: " << value << "\n";
      }
    }

    // If we can get a full potential value set, then we can derive from that.
    auto pvs = solver.getElementFor<ConstantAttributePVS>(
        *this, Position::forValue(value), DFX::Resolution::OPTIONAL);
    if (pvs.isValidState() && !pvs.isUndefContained()) {
      for (Attribute constValue : pvs.getAssumedSet()) {
        if (auto scalarValue = constValue.dyn_cast<FloatAttr>()) {
          FpRange stats;
          stats.reset();
          stats.addDomainValue(scalarValue.getValueAsDouble());
          newState.setAssumed(stats);
          newState.indicateOptimisticFixpoint();
        } else if (auto elements = constValue.dyn_cast<ElementsAttr>()) {
          FpRange stats;
          stats.reset();
          for (APFloat elementValue : elements.getValues<APFloat>()) {
            stats.addDomainValue(elementValue.convertToDouble());
          }
          newState.setAssumed(stats);
          dbgs() << "*** COMPUTED KNOWN RANGE: " << stats.getAsStr() << "\n";
          newState.indicateOptimisticFixpoint();
        } else {
          // Unknown.
          // TODO
          dbgs() << "UNKNOWN ATTRIBUTE: " << constValue << "\n";
          newState.indicatePessimisticFixpoint();
        }
      }
    }

    if (!newState.isAtFixpoint()) {
      if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
            dbgs() << "  WALK: " << result << "\n";
            Operation *definingOp = result.getDefiningOp();
            if (auto linalgOp = dyn_cast<linalg::LinalgOp>(definingOp)) {
              // Because we are working on per-layer statistics, we get to
              // ignore the entire loop structure of the linalg op and just
              // chase the stats up through the terminator (which will have
              // values that match the results).
              Block *loopBody = linalgOp.getBlock();
              assert(!loopBody->empty());
              Operation &terminator = loopBody->back();
              Value loopBodyValue =
                  terminator.getOperand(result.getResultNumber());
              auto inner = solver.getElementFor<ValueFpRange>(
                  *this, Position::forValue(loopBodyValue),
                  DFX::Resolution::REQUIRED);

              newState ^= inner;
              return WalkResult::advance();
            } else if (auto minfOp = dyn_cast<arith::MinFOp>(definingOp)) {
              auto lhs = solver.getElementFor<ValueFpRange>(
                  *this, Position::forValue(minfOp.getLhs()),
                  DFX::Resolution::REQUIRED);
              auto rhs = solver.getElementFor<ValueFpRange>(
                  *this, Position::forValue(minfOp.getRhs()),
                  DFX::Resolution::REQUIRED);

              newState.handleMinf(lhs.getAssumed(), rhs.getAssumed());
              dbgs() << "VISITING minf: lhs = " << lhs.getAsStr()
                     << ", rhs = " << rhs.getAsStr() << " -> "
                     << newState.getAssumed().getAsStr() << "\n";
              return WalkResult::advance();
            } else if (auto maxfOp = dyn_cast<arith::MaxFOp>(definingOp)) {
              auto lhs = solver.getElementFor<ValueFpRange>(
                  *this, Position::forValue(maxfOp.getLhs()),
                  DFX::Resolution::REQUIRED);
              auto rhs = solver.getElementFor<ValueFpRange>(
                  *this, Position::forValue(maxfOp.getRhs()),
                  DFX::Resolution::REQUIRED);

              newState.handleMaxf(lhs.getAssumed(), rhs.getAssumed());
              dbgs() << "VISITING maxf: lhs = " << lhs.getAsStr()
                     << ", rhs = " << rhs.getAsStr() << " -> "
                     << newState.getAssumed().getAsStr() << "\n";
              return WalkResult::advance();
            } else if (auto floorOp = dyn_cast<math::FloorOp>(definingOp)) {
              auto operand = solver.getElementFor<ValueFpRange>(
                  *this, Position::forValue(floorOp.getOperand()),
                  DFX::Resolution::REQUIRED);
              newState.handleFloor(operand.getAssumed());
              dbgs() << "VISITING floor: " << operand.getAsStr() << " -> "
                     << newState.getAssumed().getAsStr() << "\n";
              return WalkResult::advance();
            }

            // Unrecognized op.
            dbgs() << "UNRECOGNIZED OP: " << *definingOp
                   << " (signalling pessimistic fixpoint for " << value
                   << ")\n";
            newState.indicatePessimisticFixpoint();
            return WalkResult::advance();
          }) == TraversalResult::INCOMPLETE) {
        newState.indicatePessimisticFixpoint();
      }
    }

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }
};

const char ValueFpRange::ID = 0;

class PrecisionAnalysis {
 public:
  explicit PrecisionAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    // TODO: FunctionLike.
    explorer.setOpAction<mlir::FuncOp>(TraversalAction::RECURSE);
  }

  LogicalResult run() {
    // Seed operands from math ops we may wish to optimize.
    explorer.getRootOp()->walk([&](linalg::MatmulOp op) {
      for (auto input : op.inputs()) {
        solver.getOrCreateElementFor<ValueFpRange>(Position::forValue(input));
      }
    });
    return solver.run();
  }

  DFX::Solver &getSolver() { return solver; }

 private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
};

class OptimizeNumericPrecisionPass
    : public OptimizeNumericPrecisionBase<OptimizeNumericPrecisionPass> {
 public:
  void runOnOperation() override {
    FpRangeState s;
    (void)s;
    PrecisionAnalysis analysis(getOperation());
    if (failed(analysis.run())) {
      return signalPassFailure();
    }

    // Report
    dbgs() << "*** REPORT ***\n";
    getOperation()->walk([&](linalg::MatmulOp op) {
      dbgs() << "OP: " << op << ":\n";
      for (auto input : op.inputs()) {
        auto &state = analysis.getSolver().getOrCreateElementFor<ValueFpRange>(
            Position::forValue(input));
        dbgs() << "  + " << state.getAsStr() << "\n";
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createOptimizeNumericPrecisionPass() {
  return std::make_unique<OptimizeNumericPrecisionPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
