// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Attributes/PotentialValues.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "iree-util-index"

using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Util {
namespace {

//===----------------------------------------------------------------------===//
// APIntStateBase
//===----------------------------------------------------------------------===//

class APIntStateBase : public DFX::AbstractState {
public:
  APIntStateBase(unsigned numBits, APInt known, APInt assumed)
      : numBits(numBits), known(known), assumed(assumed) {}

  static APInt getUndefValue() {
    // Zero bit APInt.
    return APInt::getZeroWidth();
  }
  static bool isUndefValue(APInt value) { return value.getBitWidth() == 0; }

  bool isValidState() const override { return !isUndefValue(assumed); }
  bool isAtFixpoint() const override { return assumed == known; }

  ChangeStatus indicateOptimisticFixpoint() override {
    known = assumed;
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    assumed = known;
    return ChangeStatus::CHANGED;
  }

  APInt getKnown() const { return known; }
  APInt getAssumed() const { return assumed; }

protected:
  unsigned numBits;
  APInt known;
  APInt assumed;
};

//===----------------------------------------------------------------------===//
// IncAPIntState
// APIntStateBase specialization for increasing values where larger quantities
// are better.
//===----------------------------------------------------------------------===//

class IncAPIntState : public APIntStateBase {
public:
  using APIntStateBase::APIntStateBase;

  // "Clamps" this state with |rhs|. The result is subtype dependent but it is
  // intended that only information assumed in both states will be assumed in
  // this one afterwards.
  void operator^=(const IncAPIntState &rhs) {
    takeAssumedMinimum(rhs.getAssumed());
  }

  IncAPIntState &takeAssumedMinimum(APInt assumed) {
    // The undef value is effectively -inf (lowest possible value).
    if (isUndefValue(assumed) || isUndefValue(this->assumed)) {
      this->assumed = this->known;
      return *this;
    }

    APInt assumedMin = llvm::APIntOps::umin(this->assumed, assumed);
    if (!isUndefValue(this->known)) {
      // Don't lose any known value.
      assumedMin = llvm::APIntOps::umax(assumedMin, this->known);
    }
    this->assumed = assumedMin;

    return *this;
  }
};

//===----------------------------------------------------------------------===//
// DecAPIntState
// APIntStateBase specialization for decreasing values where smaller quantities
// are better.
//===----------------------------------------------------------------------===//

class DecAPIntState : public APIntStateBase {
public:
  using APIntStateBase::APIntStateBase;

  // "Clamps" this state with |rhs|. The result is subtype dependent but it is
  // intended that only information assumed in both states will be assumed in
  // this one afterwards.
  void operator^=(const DecAPIntState &rhs) {
    takeAssumedMaximum(rhs.getAssumed());
  }

  DecAPIntState &takeAssumedMaximum(APInt assumed) {
    // The undef value is effectively +inf (largest possible value).
    if (isUndefValue(assumed) || isUndefValue(this->assumed)) {
      this->assumed = this->known;
      return *this;
    }

    APInt assumedMax = llvm::APIntOps::umax(this->assumed, assumed);
    if (!isUndefValue(this->known)) {
      // Don't lose any known value.
      assumedMax = llvm::APIntOps::umin(assumedMax, this->known);
    }
    this->assumed = assumedMax;

    return *this;
  }
};

//===----------------------------------------------------------------------===//
// ValueAPIntMax
// ValueElement which tracks the maximum possible value of a bit-vector.
//===----------------------------------------------------------------------===//

class ValueAPIntMax : public DFX::StateWrapper<IncAPIntState, DFX::ValueElement,
                                               IncAPIntState> {
public:
  using BaseType =
      DFX::StateWrapper<IncAPIntState, DFX::ValueElement, IncAPIntState>;

  static ValueAPIntMax &createForPosition(const Position &pos,
                                          DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueAPIntMax(pos));
  }

  const std::string getName() const override { return "ValueAPIntMax"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override;

private:
  static IncAPIntState defaultState(const Position &pos) {
    if (pos.isValue()) {
      Value value = pos.getValue();
      Type type = value.getType();
      if (auto intType = dyn_cast<IntegerType>(type)) {
        unsigned bitWidth = intType.getWidth();
        return IncAPIntState(bitWidth, /*known=*/APInt::getZero(bitWidth),
                             /*assumed=*/APInt::getMaxValue(bitWidth));
      } else if (auto indexType = dyn_cast<IndexType>(type)) {
        // Assume that index is 64 bits for the purpose of this accounting.
        return IncAPIntState(64, /*known=*/APInt::getZero(64),
                             /*assumed=*/APInt::getMaxValue(64));
      }
    }
    return IncAPIntState(0, IncAPIntState::getUndefValue(),
                         IncAPIntState::getUndefValue());
  }

  ValueAPIntMax(const Position &pos) : BaseType(pos, defaultState(pos)) {}
  void initializeValue(Value value, DFX::Solver &solver) override {
    if (!value.getType().isIndex() && !value.getType().isInteger()) {
      indicatePessimisticFixpoint();
      return;
    }
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    StateType newState = getState();

    // If we can get a full PVS, then we can derive the min from that.
    auto pvs = solver.getElementFor<IntValuePVS>(
        *this, Position::forValue(value), DFX::Resolution::OPTIONAL);
    if (pvs.isValidState() && !pvs.isUndefContained()) {
      LLVM_DEBUG(dbgs() << "ValueAPIntMax::updateValue: PVS VALID\n");
      auto &set = pvs.getAssumedSet();
      if (!set.empty()) {
        for (auto value : set) {
          LLVM_DEBUG(dbgs()
                     << "ValueAPIntMax::updateValue: VALUE =" << value << "\n");
          newState.takeAssumedMinimum(value);
        }
        newState.indicateOptimisticFixpoint();
      }
    } else {
      LLVM_DEBUG(dbgs() << "ValueAPIntMin::updateValue: PVS INVALID\n");
    }

    if (!newState.isAtFixpoint()) {
      // Scan IR to infer more information.
      if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
            Operation *definingOp = result.getDefiningOp();
            if (auto assumeRangeOp =
                    dyn_cast<IREE::Util::AssumeRangeOp>(definingOp)) {
              // Clamp the max range.
              APInt maxValue = assumeRangeOp.getMaxValue();
              newState.takeAssumedMinimum(maxValue);
            } else if (auto indexCastOp =
                           dyn_cast<arith::IndexCastOp>(definingOp)) {
              // arith.index_cast (note that this is a signed cast to/from
              // index).
              // TODO: Verify bitwidth is the same or trunc/extend.
              auto max = solver.getElementFor<ValueAPIntMax>(
                  *this, Position::forValue(indexCastOp.getOperand()),
                  DFX::Resolution::REQUIRED);
              newState ^= max;
            }
            return WalkResult::advance();
          }) == TraversalResult::INCOMPLETE) {
        newState.indicatePessimisticFixpoint();
      }
    }

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }
};

const char ValueAPIntMax::ID = 0;

const std::string ValueAPIntMax::getAsStr(AsmState &asmState) const {
  std::string s;
  llvm::raw_string_ostream out(s);
  out << "apint-max: ";
  APInt assumed = getAssumed();
  if (IncAPIntState::isUndefValue(assumed)) {
    out << "<UNDEF-VALUE>";
  } else if (assumed == APInt::getMaxValue(assumed.getBitWidth())) {
    out << "<MAX-VALUE>";
  } else {
    assumed.print(out, /*signed=*/false);
  }
  return s;
}

//===----------------------------------------------------------------------===//
// ValueAPIntMin
// ValueElement which tracks the minimum possible value of a bit-vector.
//===----------------------------------------------------------------------===//

class ValueAPIntMin : public DFX::StateWrapper<DecAPIntState, DFX::ValueElement,
                                               DecAPIntState> {
public:
  using BaseType =
      DFX::StateWrapper<DecAPIntState, DFX::ValueElement, DecAPIntState>;

  static ValueAPIntMin &createForPosition(const Position &pos,
                                          DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueAPIntMin(pos));
  }

  const std::string getName() const override { return "ValueAPIntMin"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override;

private:
  static DecAPIntState defaultState(const Position &pos) {
    if (pos.isValue()) {
      Value value = pos.getValue();
      Type type = value.getType();
      if (auto intType = dyn_cast<IntegerType>(type)) {
        unsigned bitWidth = intType.getWidth();
        return DecAPIntState(bitWidth, /*known=*/APInt::getMaxValue(bitWidth),
                             /*assumed=*/APInt::getZero(bitWidth));
      } else if (auto indexType = dyn_cast<IndexType>(type)) {
        // Assume that index is 64 bits for the purpose of this accounting.
        return DecAPIntState(64, /*known=*/APInt::getMaxValue(64),
                             /*assumed=*/APInt::getZero(64));
      }
    }
    return DecAPIntState(0, DecAPIntState::getUndefValue(),
                         DecAPIntState::getUndefValue());
  }

  ValueAPIntMin(const Position &pos) : BaseType(pos, defaultState(pos)) {}
  void initializeValue(Value value, DFX::Solver &solver) override {
    if (!value.getType().isIndex() && !value.getType().isInteger()) {
      indicatePessimisticFixpoint();
      return;
    }
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    StateType newState = getState();
    LLVM_DEBUG(dbgs() << "ValueAPIntMin::updateValue: " << value << "\n");

    // If we can get a full PVS, then we can derive the min from that.
    auto pvs = solver.getElementFor<IntValuePVS>(
        *this, Position::forValue(value), DFX::Resolution::OPTIONAL);
    if (pvs.isValidState() && !pvs.isUndefContained()) {
      LLVM_DEBUG(dbgs() << "ValueAPIntMin::updateValue: PVS VALID\n");
      auto &set = pvs.getAssumedSet();
      if (!set.empty()) {
        for (auto value : set) {
          LLVM_DEBUG(dbgs()
                     << "ValueAPIntMin::updateValue: VALUE =" << value << "\n");
          newState.takeAssumedMaximum(value);
        }
        newState.indicateOptimisticFixpoint();
      }
    } else {
      LLVM_DEBUG(dbgs() << "ValueAPIntMin::updateValue: PVS INVALID\n");
    }

    if (!newState.isAtFixpoint()) {
      // Scan IR to infer more information.
      if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
            Operation *definingOp = result.getDefiningOp();
            if (auto assumeRangeOp =
                    dyn_cast<IREE::Util::AssumeRangeOp>(definingOp)) {
              // Clamp the max range.
              APInt minValue = assumeRangeOp.getMinValue();
              newState.takeAssumedMaximum(minValue);
            } else if (auto indexCastOp =
                           dyn_cast<arith::IndexCastOp>(definingOp)) {
              // arith.index_cast (note that this is a signed cast to/from
              // index).
              // TODO: Verify bitwidth is the same or trunc/extend.
              auto max = solver.getElementFor<ValueAPIntMin>(
                  *this, Position::forValue(indexCastOp.getOperand()),
                  DFX::Resolution::REQUIRED);
              newState ^= max;
            }
            return WalkResult::advance();
          }) == TraversalResult::INCOMPLETE) {
        newState.indicatePessimisticFixpoint();
      }
    }

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }
};

const char ValueAPIntMin::ID = 0;

const std::string ValueAPIntMin::getAsStr(AsmState &asmState) const {
  std::string s;
  llvm::raw_string_ostream out(s);
  out << "apint-min: ";
  APInt assumed = getAssumed();
  if (IncAPIntState::isUndefValue(assumed)) {
    out << "<UNDEF-VALUE>";
  } else {
    assumed.print(out, /*signed=*/false);
  }
  return s;
}

//===----------------------------------------------------------------------===//
// IndexArithmeticAnalysis
//===----------------------------------------------------------------------===//

class IndexArithmeticAnalysis {
public:
  enum class OpAction {
    SIMPLIFY_CMPIOP,
  };
  explicit IndexArithmeticAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    rootOp->walk([&](arith::CmpIOp cmpOp) { addOp(cmpOp); });
  }

  const SmallVector<std::pair<OpAction, Operation *>> &getOpActions() const {
    return opActions;
  }

  void addOp(arith::CmpIOp cmpOp) {
    opActions.emplace_back(OpAction::SIMPLIFY_CMPIOP, cmpOp);
    for (Value v : {cmpOp.getLhs(), cmpOp.getRhs()}) {
      auto pos = Position::forValue(v);
      solver.getOrCreateElementFor<IntValuePVS>(pos);
      solver.getOrCreateElementFor<ValueAPIntMax>(pos);
      solver.getOrCreateElementFor<ValueAPIntMin>(pos);
    }
  }

  LogicalResult run() { return solver.run(); }

  const ValueAPIntMin &getMinElement(Value value) {
    return solver.getOrCreateElementFor<ValueAPIntMin>(
        Position::forValue(value));
  }
  const ValueAPIntMax &getMaxElement(Value value) {
    return solver.getOrCreateElementFor<ValueAPIntMax>(
        Position::forValue(value));
  }

  std::optional<APInt> getKnownMinValue(Value value) {
    auto &elt = getMinElement(value);
    if (!elt.isAtFixpoint() || !elt.isValidState())
      return {};
    return elt.getKnown();
  }

  std::optional<APInt> getKnownMaxValue(Value value) {
    auto &elt = getMaxElement(value);
    if (!elt.isAtFixpoint() || !elt.isValidState())
      return {};
    return elt.getKnown();
  }

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
  SmallVector<std::pair<OpAction, Operation *>> opActions;
};

bool simplifyCmpIOp(IndexArithmeticAnalysis &analysis, arith::CmpIOp cmpOp) {
  auto lhsMin = analysis.getKnownMinValue(cmpOp.getLhs());
  auto lhsMax = analysis.getKnownMaxValue(cmpOp.getLhs());
  auto rhsMin = analysis.getKnownMinValue(cmpOp.getRhs());
  auto rhsMax = analysis.getKnownMaxValue(cmpOp.getRhs());

  auto reportValue = [](const char *label, std::optional<APInt> intValue) {
    if (intValue) {
      dbgs() << "  " << label << ": " << *intValue << "\n";
    } else {
      dbgs() << "  " << label << ": INVALID\n";
    }
  };
  LLVM_DEBUG(dbgs() << "Attempt CmpIOp simplification:\n";
             reportValue("lhsMin", lhsMin); reportValue("lhsMax", lhsMax);
             reportValue("rhsMin", rhsMin); reportValue("rhsMax", rhsMax););

  // For signed comparisons, we have to be conservative and ensure that the
  // ranges are completely disjoint. We could be more liberal if we checked
  // additional facets of the bit patterns to work around signed order and
  // overflow possibilities (i.e. whether it is safe to promote a signed to
  // an unsigned compare).
  if (!lhsMin || !lhsMax || !rhsMin || !rhsMax) {
    LLVM_DEBUG(dbgs() << "Cannot simplify " << cmpOp << "\n");
    return false;
  }

  auto predicate = cmpOp.getPredicate();
  auto compare = [&](APInt lhs, APInt rhs) -> std::optional<bool> {
    switch (predicate) {
    case arith::CmpIPredicate::slt:
      return lhs.slt(rhs);
      break;

    default:
      return {};
    }
  };

  int trueCount = 0;
  int falseCount = 0;
  for (auto result : {
           compare(*lhsMin, *rhsMin),
           compare(*lhsMin, *rhsMax),
           compare(*lhsMax, *rhsMin),
           compare(*lhsMax, *rhsMax),
       }) {
    if (!result)
      return false; // Has an undef value.
    if (*result)
      ++trueCount;
    else
      ++falseCount;
  }

  OpBuilder builder(cmpOp);
  auto rewrite = [&](bool resultValue) {
    LLVM_DEBUG(dbgs() << "Rewrite " << cmpOp << " -> " << resultValue << "\n");
    Value newValue = builder.create<arith::ConstantOp>(
        cmpOp.getLoc(), builder.getBoolAttr(resultValue));
    cmpOp.getResult().replaceAllUsesWith(newValue);
  };
  if (trueCount && !falseCount) {
    // Replace with true.
    rewrite(true);
    return true;
  } else if (!trueCount && falseCount) {
    // Replace with false.
    rewrite(false);
    return true;
  }

  return false;
}

class SimplifyIndexArithmeticPass
    : public SimplifyIndexArithmeticBase<SimplifyIndexArithmeticPass> {
public:
  LogicalResult runRangeSimplifications() {
    IndexArithmeticAnalysis analysis(getOperation());
    if (failed(analysis.run())) {
      return failure();
    }

    int changeCount = 0;
    for (auto [action, op] : analysis.getOpActions()) {
      switch (action) {
      case IndexArithmeticAnalysis::OpAction::SIMPLIFY_CMPIOP:
        if (simplifyCmpIOp(analysis, cast<arith::CmpIOp>(op))) {
          changeCount += 1;
        }
        break;
      }
    }

    LLVM_DEBUG(dbgs() << "Simplification made " << changeCount << " changes\n");

    (void)changeCount;
    return success();
  }

  void runOnOperation() override {
    if (failed(runRangeSimplifications())) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<void>> createSimplifyIndexArithmeticPass() {
  return std::make_unique<SimplifyIndexArithmeticPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
