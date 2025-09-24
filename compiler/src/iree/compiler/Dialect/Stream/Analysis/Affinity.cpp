// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

#define DEBUG_TYPE "iree-stream-affinity-analysis"

namespace mlir::iree_compiler::IREE::Stream {

static llvm::cl::opt<int> clSolverMaxIterations(
    "iree-stream-affinity-solver-max-iterations",
    llvm::cl::desc("Maximum affinity analysis solver iteration count before it "
                   "gives up. Roughly equivalent to operation chain depth."),
    llvm::cl::init(128));

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static const std::string getAffinitySetAsStr(
    const DFX::PotentialValuesState<IREE::Stream::AffinityAttr> &state,
    AsmState &asmState) {
  std::string str;
  llvm::raw_string_ostream sstream(str);
  sstream << "pvs: ";
  if (state.isValidState()) {
    sstream << "[";
    if (state.isUndefContained()) {
      sstream << "undef, ";
    }
    llvm::interleaveComma(state.getAssumedSet(), sstream,
                          [&](IREE::Stream::AffinityAttr value) {
                            cast<Attribute>(value).print(sstream);
                          });
    sstream << "]";
  } else {
    sstream << "(invalid)";
  }
  sstream.flush();
  return str;
}

//===----------------------------------------------------------------------===//
// Precomputed IR properties
//===----------------------------------------------------------------------===//

// TODO(benvanik): template this as a helper and move into common... though
// hopefully it's rarely needed.
class SidebandElement
    : public DFX::StateWrapper<DFX::BooleanState, DFX::OperationElement> {
public:
  using BaseType = DFX::StateWrapper<DFX::BooleanState, DFX::OperationElement>;
  SidebandElement(const Position &pos,
                  AffinityAnalysis::PrecomputedQueries *precomputedQueries)
      : BaseType(pos), precomputedQueries(precomputedQueries) {}

  static SidebandElement &
  createForPosition(const Position &pos, DFX::Solver &solver,
                    AffinityAnalysis::PrecomputedQueries *precomputedQueries) {
    return *(new (solver.getAllocator())
                 SidebandElement(pos, precomputedQueries));
  }

  // Identity definitions.
  const std::string getName() const override { return "SidebandElement"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return "sideband";
  }

  const AffinityAnalysis::PrecomputedQueries &get() const {
    return *precomputedQueries;
  }

private:
  void initializeOperation(Operation *op, DFX::Solver &solver) override {
    getState().setKnown(true);
    indicateOptimisticFixpoint();
  }
  ChangeStatus updateOperation(Operation *op, DFX::Solver &solver) override {
    return ChangeStatus::UNCHANGED;
  }

  AffinityAnalysis::PrecomputedQueries *precomputedQueries;
};
const char SidebandElement::ID = 0;

// Populates value affinities for all operands and results of the affinity op
// that pins affinity transitively (through tied ops).
static void populatePinnedAffinities(
    IREE::Stream::AffinityOpInterface affinityOp, Explorer &explorer,
    AffinityAnalysis::PrecomputedQueries &precomputedQueries) {
  assert(affinityOp.pinsValueAffinity() && "must pin affinity");
  auto &pinnedAffinities = precomputedQueries.pinnedAffinities;
  auto markPinned = [&](Value value) {
    auto it = pinnedAffinities.find(value);
    if (it == pinnedAffinities.end()) {
      // First time this value has been pinned.
      pinnedAffinities[value].insert(affinityOp);
      return true; // continue searching
    } else if (it->second.contains(affinityOp)) {
      // Already pinned to this affinity.
      return false; // stop searching
    } else {
      // Adding a new affinity.
      it->second.insert(affinityOp);
      return true; // continue searching
    }
  };
  for (Value result : affinityOp->getResults()) {
    if (!isa<IREE::Stream::AffinityTypeInterface>(result.getType())) {
      continue;
    }
    explorer.walkTransitiveUses(result, [&](OpOperand &operand) {
      if (!isa<IREE::Stream::AffinityTypeInterface>(operand.get().getType())) {
        // Type changed; stop tracking affinity.
        return WalkResult::skip();
      } else if (auto affinityOp =
                     dyn_cast_if_present<IREE::Stream::AffinityOpInterface>(
                         operand.getOwner())) {
        // If the user _also_ pins affinity we need to stop. The overall
        // population will eventually reach it and start pinning from there.
        if (affinityOp.pinsValueAffinity()) {
          return WalkResult::skip();
        }
      }
      return markPinned(operand.get()) ? WalkResult::advance()
                                       : WalkResult::skip();
    });
  }
  for (Value operand : affinityOp->getOperands()) {
    if (!isa<IREE::Stream::AffinityTypeInterface>(operand.getType())) {
      continue;
    }
    explorer.walkDefiningOps(operand, [&](OpResult result) -> WalkResult {
      if (result.getOwner() == affinityOp) {
        return WalkResult::advance(); // starting point
      }
      if (!isa<IREE::Stream::AffinityTypeInterface>(result.getType())) {
        // Type changed; stop tracking affinity.
        return WalkResult::skip();
      } else if (auto affinityOp =
                     dyn_cast_if_present<IREE::Stream::AffinityOpInterface>(
                         result.getOwner())) {
        // If the defining op _also_ pins affinity we need to stop. The overall
        // population will eventually reach it and start pinning from there.
        if (affinityOp.pinsValueAffinity()) {
          return WalkResult::skip();
        }
      }
      return markPinned(result) ? WalkResult::advance() : WalkResult::skip();
    });
  }
}

void AffinityAnalysis::PrecomputedQueries::compute(Explorer &explorer) {
  explorer.forEachFunctionLikeOp([&](FunctionOpInterface funcOp) {
    funcOp.walk([&](Operation *op) {
      if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
        if (affinityOp.pinsValueAffinity()) {
          populatePinnedAffinities(affinityOp, explorer, *this);
        }
      }
    });
  });
}

void AffinityAnalysis::PrecomputedQueries::inject(DFX::Solver &solver) {
  solver.registerElement(SidebandElement::createForPosition(
      Position::forOperation(solver.getExplorer().getRootOp()), solver, this));
}

// static
const AffinityAnalysis::PrecomputedQueries &
AffinityAnalysis::PrecomputedQueries::get(DFX::Solver &solver) {
  auto *sidebandElement = solver.lookupElementFor<SidebandElement>(
      Position::forOperation(solver.getExplorer().getRootOp()));
  assert(sidebandElement && "sideband element must have been registered before "
                            "attempting to query it");
  return sidebandElement->get();
}

void AffinityAnalysis::PrecomputedQueries::print(llvm::raw_ostream &os,
                                                 AsmState &asmState) {
  os << "[PrecomputedQueries] pinned affinities:\n";
  for (auto it : pinnedAffinities) {
    os << "  ";
    it.first.printAsOperand(os, asmState);
    os << ":\n";
    llvm::interleave(
        it.second, os,
        [&](auto affinityOp) {
          os << "    ";
          affinityOp->print(llvm::dbgs(), asmState);
        },
        ",\n");
    os << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Analysis elements
//===----------------------------------------------------------------------===//

class ValueProducerAffinityPVS
    : public DFX::StateWrapper<
          DFX::PotentialValuesState<IREE::Stream::AffinityAttr>,
          DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<IREE::Stream::AffinityAttr>,
                        DFX::ValueElement>;
  using BaseType::BaseType;

  static ValueProducerAffinityPVS &createForPosition(const Position &pos,
                                                     DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueProducerAffinityPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override {
    return "ValueProducerAffinityPVS";
  }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getAffinitySetAsStr(getState(), asmState);
  }

private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
  void updateFromUse(Value value, OpOperand &operand, StateType &newState,
                     DFX::Solver &solver);
};
const char ValueProducerAffinityPVS::ID = 0;

class GlobalAffinityPVS
    : public DFX::StateWrapper<
          DFX::PotentialValuesState<IREE::Stream::AffinityAttr>,
          DFX::TypedOperationElement<IREE::Util::GlobalOpInterface>> {
public:
  using BaseType = DFX::StateWrapper<
      DFX::PotentialValuesState<IREE::Stream::AffinityAttr>,
      DFX::TypedOperationElement<IREE::Util::GlobalOpInterface>>;
  using BaseType::BaseType;

  static GlobalAffinityPVS &createForPosition(const Position &pos,
                                              DFX::Solver &solver) {
    return *(new (solver.getAllocator()) GlobalAffinityPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "GlobalAffinityPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getAffinitySetAsStr(getState(), asmState);
  }

private:
  void initializeOperation(IREE::Util::GlobalOpInterface globalOp,
                           DFX::Solver &solver) override;
  ChangeStatus updateOperation(IREE::Util::GlobalOpInterface globalOp,
                               DFX::Solver &solver) override;
};
const char GlobalAffinityPVS::ID = 0;

class OpAffinityPVS : public DFX::StateWrapper<
                          DFX::PotentialValuesState<IREE::Stream::AffinityAttr>,
                          DFX::OperationElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<IREE::Stream::AffinityAttr>,
                        DFX::OperationElement>;
  using BaseType::BaseType;

  static OpAffinityPVS &createForPosition(const Position &pos,
                                          DFX::Solver &solver) {
    return *(new (solver.getAllocator()) OpAffinityPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override { return "OpAffinityPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getAffinitySetAsStr(getState(), asmState);
  }

private:
  void initializeOperation(Operation *op, DFX::Solver &solver) override;
  ChangeStatus updateOperation(Operation *op, DFX::Solver &solver) override;
};
const char OpAffinityPVS::ID = 0;

//===----------------------------------------------------------------------===//
// ValueConsumerAffinityPVS
//===----------------------------------------------------------------------===//

class ValueConsumerAffinityPVS
    : public DFX::StateWrapper<
          DFX::PotentialValuesState<IREE::Stream::AffinityAttr>,
          DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<IREE::Stream::AffinityAttr>,
                        DFX::ValueElement>;
  using BaseType::BaseType;

  static ValueConsumerAffinityPVS &createForPosition(const Position &pos,
                                                     DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueConsumerAffinityPVS(pos));
  }

  // Identity definitions.
  const std::string getName() const override {
    return "ValueConsumerAffinityPVS";
  }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getAffinitySetAsStr(getState(), asmState);
  }

private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
  TraversalResult updateFromUse(Value value, OpOperand &operand,
                                StateType &newState, DFX::Solver &solver);
};
const char ValueConsumerAffinityPVS::ID = 0;

void ValueConsumerAffinityPVS::initializeValue(Value value,
                                               DFX::Solver &solver) {}

ChangeStatus ValueConsumerAffinityPVS::updateValue(Value value,
                                                   DFX::Solver &solver) {
  StateType newState;
  auto traversalResult = TraversalResult::COMPLETE;

  // Walk into all consumers of the SSA value.
  // Note that we may end up at multiple global stores of different globals
  // by walking down through calls/branches/etc.
  traversalResult |= solver.getExplorer().walkTransitiveUses(
      value,
      [&](OpOperand &operand) {
        traversalResult |= updateFromUse(value, operand, newState, solver);
        return WalkResult::advance();
      },
      (TraversalBehavior::DEFAULT | TraversalBehavior::DONT_WALK_TIED_VALUES));

  if (traversalResult == TraversalResult::INCOMPLETE) {
    // Incomplete traversal because of external call graph edges or pointers.
    newState.unionAssumedWithUndef();
    newState.indicatePessimisticFixpoint();
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

TraversalResult ValueConsumerAffinityPVS::updateFromUse(Value value,
                                                        OpOperand &operand,
                                                        StateType &newState,
                                                        DFX::Solver &solver) {
  // If the value is consumed by an affinity-aware op then we can directly use
  // the affinity specified on the op. A majority of the values we care about at
  // the stream level are consumed by affinity-aware ops and earlier in the
  // pipeline dialects may have transfer ops that define affinities we can
  // anchor on.
  if (auto affinityOp =
          dyn_cast<IREE::Stream::AffinityOpInterface>(operand.getOwner())) {
    auto opPVS = solver.getElementFor<OpAffinityPVS>(
        *this, Position::forOperation(operand.getOwner()),
        DFX::Resolution::REQUIRED);
    LLVM_DEBUG({
      llvm::dbgs() << "[ValueConsumerAffinityPVS] value ";
      value.printAsOperand(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << " affinity using consumer affinity from ";
      opPVS.print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    newState ^= opPVS;
  }

  // If the consumer op has the operand tied to one or more results then we walk
  // through to track the transitive consumers. When this analysis runs we are
  // usually still prior to baking out copy-on-write behavior so it's possible
  // that the results of the tied operation end up in different places.
  if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(operand.getOwner())) {
    auto tiedResults = tiedOp.getOperandTiedResults(operand.getOperandNumber());
    for (auto tiedResult : tiedResults) {
      auto resultPVS = solver.getElementFor<ValueConsumerAffinityPVS>(
          *this, Position::forValue(tiedResult), DFX::Resolution::REQUIRED);
      LLVM_DEBUG({
        llvm::dbgs() << "[ValueConsumerAffinityPVS] value ";
        value.printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << " affinity referencing tied operand ";
        operand.get().printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << " result ";
        tiedResult.printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << " as ";
        resultPVS.print(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      newState ^= resultPVS;
    }
  }

  // Handle consumers that are not affinity aware - this should have any control
  // flow ops so that we can track values that flow through the program.
  return TypeSwitch<Operation *, TraversalResult>(operand.getOwner())
      .Case([&](IREE::Stream::AsyncTransferOp op) {
        if (auto targetAffinityAttr = op.getResultAffinityAttr()) {
          LLVM_DEBUG({
            llvm::dbgs() << "[ValueConsumerAffinityPVS] value ";
            value.printAsOperand(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << " affinity unioning with transfer target "
                         << "affinity as " << targetAffinityAttr << "\n";
          });
          newState.unionAssumed(targetAffinityAttr);
        }
        return TraversalResult::COMPLETE;
      })
      .Case([&](mlir::arith::SelectOp op) {
        auto &resultPVS = solver.getElementFor<ValueConsumerAffinityPVS>(
            *this, Position::forValue(op.getResult()),
            DFX::Resolution::REQUIRED);
        newState ^= resultPVS.getState();
        return TraversalResult::COMPLETE;
      })
      .Case([&](mlir::BranchOpInterface op) {
        return solver.getExplorer().walkOutgoingBranchOperandArguments(
            op, operand.getOperandNumber(),
            [&](Block *targetBlock, BlockArgument arg) {
              auto &argUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
                  *this, Position::forValue(arg), DFX::Resolution::OPTIONAL);
              newState ^= argUsage;
              return WalkResult::advance();
            });
      })
      .Case([&](mlir::scf::ForOp op) {
        if (operand.getOperandNumber() >= op.getNumControlOperands()) {
          int64_t blockIdx =
              operand.getOperandNumber() - op.getNumControlOperands();
          auto &beforeUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
              *this, Position::forValue(op.getRegionIterArg(blockIdx)),
              DFX::Resolution::REQUIRED);
          newState ^= beforeUsage.getState();
        }
        return TraversalResult::COMPLETE;
      })
      .Case([&](mlir::scf::WhileOp op) {
        auto &beforeUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
            *this,
            Position::forValue(
                op.getBeforeBody()->getArgument(operand.getOperandNumber())),
            DFX::Resolution::REQUIRED);
        newState ^= beforeUsage.getState();
        return TraversalResult::COMPLETE;
      })
      .Case([&](mlir::scf::ConditionOp op) {
        auto &parentUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
            *this,
            Position::forValue(
                op->getParentOp()->getResult(operand.getOperandNumber() - 1)),
            DFX::Resolution::REQUIRED);
        newState ^= parentUsage.getState();
        if (auto whileOp =
                dyn_cast_or_null<mlir::scf::WhileOp>(op->getParentOp())) {
          auto value = Position::forValue(
              whileOp.getAfter().getArgument(operand.getOperandNumber() - 1));
          auto &valueUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
              *this, value, DFX::Resolution::REQUIRED);
          newState ^= valueUsage.getState();
        }
        return TraversalResult::COMPLETE;
      })
      .Case([&](mlir::scf::YieldOp op) {
        if (isa<mlir::scf::IfOp>(op->getParentOp())) {
          auto &operandUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
              *this,
              Position::forValue(op->getOperand(operand.getOperandNumber())),
              DFX::Resolution::REQUIRED);
          newState ^= operandUsage.getState();
          auto &parentUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
              *this,
              Position::forValue(
                  op->getParentOp()->getResult(operand.getOperandNumber())),
              DFX::Resolution::REQUIRED);
          newState ^= parentUsage.getState();
          return TraversalResult::COMPLETE;
        } else if (auto whileOp =
                       dyn_cast<mlir::scf::WhileOp>(op->getParentOp())) {
          auto value = Position::forValue(
              whileOp.getBefore().getArgument(operand.getOperandNumber()));
          auto &valueUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
              *this, value, DFX::Resolution::REQUIRED);
          newState ^= valueUsage.getState();
          auto &parentUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
              *this,
              Position::forValue(
                  whileOp->getResult(operand.getOperandNumber())),
              DFX::Resolution::REQUIRED);
          newState ^= parentUsage.getState();
          return TraversalResult::COMPLETE;
        } else if (auto forOp = dyn_cast<mlir::scf::ForOp>(op->getParentOp())) {
          auto value = Position::forValue(
              forOp.getRegionIterArg(operand.getOperandNumber()));
          auto &valueUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
              *this, value, DFX::Resolution::REQUIRED);
          newState ^= valueUsage.getState();
          auto &parentUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
              *this,
              Position::forValue(forOp->getResult(operand.getOperandNumber())),
              DFX::Resolution::REQUIRED);
          newState ^= parentUsage.getState();
          return TraversalResult::COMPLETE;
        } else {
          assert(false && "unhandled scf yield parent");
          return TraversalResult::INCOMPLETE;
        }
      })
      .Case([&](IREE::Stream::YieldOp op) {
        auto parentOp = op->getParentOp();
        auto &resultPVS = solver.getElementFor<ValueConsumerAffinityPVS>(
            *this,
            Position::forValue(parentOp->getResult(operand.getOperandNumber())),
            DFX::Resolution::REQUIRED);
        newState ^= resultPVS.getState();
        return TraversalResult::COMPLETE;
      })
      .Case([&](IREE::Util::ReturnOp op) {
        return solver.getExplorer().walkIncomingCalls(
            op->getParentOfType<mlir::CallableOpInterface>(),
            [&](mlir::CallOpInterface callOp) {
              auto &argUsage = solver.getElementFor<ValueConsumerAffinityPVS>(
                  *this,
                  Position::forValue(
                      callOp->getResult(operand.getOperandNumber())),
                  DFX::Resolution::OPTIONAL);
              getState() ^= argUsage;
              return WalkResult::advance();
            });
      })
      .Case([&](IREE::Util::OptimizationBarrierOp op) {
        auto &resultPVS = solver.getElementFor<ValueConsumerAffinityPVS>(
            *this, Position::forValue(op.getResult(operand.getOperandNumber())),
            DFX::Resolution::REQUIRED);
        newState ^= resultPVS.getState();
        return TraversalResult::COMPLETE;
      })
      .Case([&](IREE::Util::GlobalStoreOpInterface op) {
        auto *globalInfo =
            solver.getExplorer().queryGlobalInfoFrom(op.getGlobalName(), op);
        auto &globalPVS = solver.getElementFor<GlobalAffinityPVS>(
            *this, Position::forOperation(globalInfo->op),
            DFX::Resolution::REQUIRED);
        newState ^= globalPVS.getState();
        return TraversalResult::COMPLETE;
      })
      .Default([&](Operation *op) { return TraversalResult::COMPLETE; });
}

//===----------------------------------------------------------------------===//
// ValueProducerAffinityPVS
//===----------------------------------------------------------------------===//

void ValueProducerAffinityPVS::initializeValue(Value value,
                                               DFX::Solver &solver) {}

ChangeStatus ValueProducerAffinityPVS::updateValue(Value value,
                                                   DFX::Solver &solver) {
  // We special case some ops that act as barriers in the program. This prevents
  // us from walking past boundaries that are not profitable to do so with; for
  // example, globals are usually stored in independent contexts from where they
  // are consumed.
  if (auto barrierOp = dyn_cast_if_present<IREE::Util::OptimizationBarrierOp>(
          value.getDefiningOp())) {
    auto operand =
        barrierOp.getOperand(cast<OpResult>(value).getResultNumber());
    auto operandPVS = solver.getElementFor<ValueProducerAffinityPVS>(
        *this, Position::forValue(operand), DFX::Resolution::REQUIRED);
    LLVM_DEBUG({
      llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
      value.printAsOperand(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << " affinity using barrier op operand as ";
      operandPVS.print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    return DFX::clampStateAndIndicateChange(getState(), operandPVS.getState());
  } else if (auto loadOp =
                 dyn_cast_if_present<IREE::Util::GlobalLoadOpInterface>(
                     value.getDefiningOp())) {
    auto *globalInfo = solver.getExplorer().queryGlobalInfoFrom(
        loadOp.getGlobalName(), loadOp);
    auto &globalPVS = solver.getElementFor<GlobalAffinityPVS>(
        *this, Position::forOperation(globalInfo->op),
        DFX::Resolution::REQUIRED);
    LLVM_DEBUG({
      llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
      value.printAsOperand(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << " affinity using global op affinity from "
                   << loadOp.getGlobalName() << " as ";
      globalPVS.print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    return DFX::clampStateAndIndicateChange(getState(), globalPVS.getState());
  }

  // Walk the program up into any possible producers of the value.
  StateType newState;
  auto traversalResult = TraversalResult::COMPLETE;
  traversalResult |= solver.getExplorer().walkDefiningOps(
      value,
      [&](OpResult result) {
        if (isa<CallOpInterface>(result.getOwner())) {
          return WalkResult::advance();
        }

        // If coming from an affinity-aware op that pins the value storage to a
        // particular affinity that defines the required producer, even if this
        // op isn't producing it. We continue walking through tied ops below in
        // case we need to _also_ find the producer affinity. This is how alias
        // ops work: some op produces the value and then an alias op pins it -
        // the producer need not have the same affinity as the alias op as
        // that's just indicating where it is stored.
        //
        // Pinning has an important property: we know exactly where the resource
        // *must* be (because the user told us that). What we may not be able to
        // tell during analysis is where it _also_ is. That's fine (I think)
        // because pinning is how users provide us information we otherwise
        // can't get and we have to go on trust that they are telling us the
        // proper placement. Below all state combinations we perform are guarded
        // on whether it's pinned: if it is, we ignore invalid analysis state
        // and only use it to augment the pinned affinity.
        bool isPinned = false;
        if (auto affinityOp =
                dyn_cast_if_present<IREE::Stream::AffinityOpInterface>(
                    result.getDefiningOp())) {
          if (affinityOp.pinsValueAffinity()) {
            isPinned = true;
            auto &opPVS = solver.getElementFor<OpAffinityPVS>(
                *this, Position::forOperation(affinityOp),
                DFX::Resolution::REQUIRED);
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
              value.printAsOperand(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << " affinity using pinned affinity from ";
              result.printAsOperand(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << " as ";
              opPVS.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            newState ^= opPVS;
          }
        }

        // If the result value is tied to an operand of the defining op then
        // inherit the operand affinity.
        if (auto tiedOp = dyn_cast_if_present<IREE::Util::TiedOpInterface>(
                result.getDefiningOp())) {
          auto operand = tiedOp.getTiedResultOperand(result);
          if (operand) {
            auto &valuePVS = solver.getElementFor<ValueProducerAffinityPVS>(
                *this, Position::forValue(operand), DFX::Resolution::OPTIONAL);
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
              value.printAsOperand(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << " affinity referencing tied operand ";
              operand.printAsOperand(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << " as ";
              valuePVS.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            if (!isPinned || valuePVS.isValidState()) {
              newState ^= valuePVS;
            }
            return WalkResult::advance();
          }
        }

        // If the value is produced by the defining op then assume that the
        // execution affinity dictates the result affinity.
        if (auto affinityOp =
                dyn_cast_if_present<IREE::Stream::AffinityOpInterface>(
                    result.getDefiningOp())) {
          auto &opPVS = solver.getOrCreateElementFor<OpAffinityPVS>(
              Position::forOperation(result.getOwner()), *this,
              DFX::Resolution::OPTIONAL, /*forceUpdate=*/false,
              /*updateAfterInit=*/false);
          LLVM_DEBUG({
            llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
            value.printAsOperand(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << " affinity using op affinity from result ";
            result.printAsOperand(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << " as ";
            opPVS.print(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << "\n";
          });
          if (!isPinned || opPVS.isValidState()) {
            newState ^= opPVS;
          }
          return WalkResult::advance();
        }

        // Special handling for specific ops.
        TypeSwitch<Operation *>(result.getOwner())
            .Case([&](IREE::Stream::AsyncTransferOp op) {
              if (auto sourceAffinityAttr = op.getSourceAffinityAttr()) {
                LLVM_DEBUG({
                  llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
                  value.printAsOperand(llvm::dbgs(), solver.getAsmState());
                  llvm::dbgs()
                      << " affinity unioning with transfer source affinity as "
                      << sourceAffinityAttr << "\n";
                });
                newState.unionAssumed(sourceAffinityAttr);
              }
            })
            .Case<IREE::Util::GlobalLoadOpInterface>([&](auto loadOp) {
              auto *globalInfo = solver.getExplorer().queryGlobalInfoFrom(
                  loadOp.getGlobalName(), loadOp);
              auto &globalPVS = solver.getElementFor<GlobalAffinityPVS>(
                  *this, Position::forOperation(globalInfo->op),
                  DFX::Resolution::REQUIRED);
              LLVM_DEBUG({
                llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
                value.printAsOperand(llvm::dbgs(), solver.getAsmState());
                llvm::dbgs()
                    << " affinity using global op affinity from result ";
                result.printAsOperand(llvm::dbgs(), solver.getAsmState());
                llvm::dbgs() << " as ";
                globalPVS.print(llvm::dbgs(), solver.getAsmState());
                llvm::dbgs() << "\n";
              });
              if (!isPinned || globalPVS.isValidState()) {
                newState ^= globalPVS.getState();
              }
            })
            .Case<mlir::arith::SelectOp>([&](auto op) {
              auto &truePVS = solver.getElementFor<ValueProducerAffinityPVS>(
                  *this, Position::forValue(op.getTrueValue()),
                  DFX::Resolution::REQUIRED);
              if (!isPinned || truePVS.isValidState()) {
                newState ^= truePVS.getState();
              }
              auto &falsePVS = solver.getElementFor<ValueProducerAffinityPVS>(
                  *this, Position::forValue(op.getFalseValue()),
                  DFX::Resolution::REQUIRED);
              if (!isPinned || falsePVS.isValidState()) {
                newState ^= falsePVS.getState();
              }
            })
            .Default([&](auto op) {
              auto valuePVS = solver.getElementFor<ValueProducerAffinityPVS>(
                  *this, Position::forValue(result), DFX::Resolution::OPTIONAL);
              LLVM_DEBUG({
                llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
                value.printAsOperand(llvm::dbgs(), solver.getAsmState());
                llvm::dbgs() << " affinity using generic producer affinity of ";
                result.printAsOperand(llvm::dbgs(), solver.getAsmState());
                llvm::dbgs() << " as ";
                valuePVS.print(llvm::dbgs(), solver.getAsmState());
                llvm::dbgs() << "\n";
              });
              if (!isPinned || valuePVS.isValidState()) {
                newState ^= valuePVS;
              }
            });
        return WalkResult::advance();
      },
      (TraversalBehavior::DEFAULT | TraversalBehavior::DONT_WALK_TIED_VALUES));

  if (traversalResult == TraversalResult::INCOMPLETE) {
    // Incomplete traversal because of external call graph edges or pointers.
    newState.unionAssumedWithUndef();
    newState.indicatePessimisticFixpoint();
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

//===----------------------------------------------------------------------===//
// GlobalAffinityPVS
//===----------------------------------------------------------------------===//

void GlobalAffinityPVS::initializeOperation(
    IREE::Util::GlobalOpInterface globalOp, DFX::Solver &solver) {
  // If an affinity is explicitly specified we take that over all analysis.
  if (auto affinityAttr = IREE::Stream::AffinityAttr::lookup(globalOp)) {
    LLVM_DEBUG({
      llvm::dbgs() << "[GlobalAffinityPVS] global @"
                   << globalOp.getGlobalName().getValue()
                   << " affinity explicitly specified as ";
      affinityAttr.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    unionAssumed(affinityAttr);
    indicateOptimisticFixpoint();
    return;
  }
}

ChangeStatus
GlobalAffinityPVS::updateOperation(IREE::Util::GlobalOpInterface globalOp,
                                   DFX::Solver &solver) {
  StateType newState;
  auto traversalResult = TraversalResult::COMPLETE;

  const auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
  if (globalInfo->isIndirect) {
    traversalResult = TraversalResult::INCOMPLETE;
  }

  // Traverse all transitive uses of the global.
  // We try to place globals where they are used as the common case is weights
  // or parameters that are read more frequently than they are written.
  // The reasoning is that if there are more writes than reads there's unneeded
  // work being done and otherwise there's always at least one read per write
  // or more reads than writes.
  bool anyLoads = false;
  for (auto loadOp : globalInfo->getLoads()) {
    anyLoads = true;
    auto &valuePVS = solver.getElementFor<ValueConsumerAffinityPVS>(
        *this, Position::forValue(loadOp.getLoadedGlobalValue()),
        DFX::Resolution::OPTIONAL);
    if (valuePVS.isValidState()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[GlobalAffinityPVS] global @"
                     << globalOp.getGlobalName().getValue()
                     << " affinity using consumer affinity from ";
        valuePVS.print(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      newState ^= valuePVS;
    }
  }

  // If there were no loads then take the affinity from stores.
  // This is not common but can arise in tests or where the globals may be used
  // to model side-effecting behavior.
  if (!anyLoads) {
    for (auto storeOp : globalInfo->getStores()) {
      auto &valuePVS = solver.getElementFor<ValueProducerAffinityPVS>(
          *this, Position::forValue(storeOp.getStoredGlobalValue()),
          DFX::Resolution::OPTIONAL);
      if (valuePVS.isValidState()) {
        LLVM_DEBUG({
          llvm::dbgs() << "[GlobalAffinityPVS] global @"
                       << globalOp.getGlobalName().getValue()
                       << " affinity using producer affinity from ";
          valuePVS.print(llvm::dbgs(), solver.getAsmState());
          llvm::dbgs() << "\n";
        });
        newState ^= valuePVS;
      }
    }
  }

  if (traversalResult == TraversalResult::INCOMPLETE) {
    // Incomplete traversal because of external call graph edges or pointers.
    newState.unionAssumedWithUndef();
    newState.indicatePessimisticFixpoint();
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

//===----------------------------------------------------------------------===//
// OpAffinityPVS
//===----------------------------------------------------------------------===//

void OpAffinityPVS::initializeOperation(Operation *op, DFX::Solver &solver) {
  // If an affinity is explicitly specified we take that over all analysis.
  if (auto affinityAttr = IREE::Stream::AffinityAttr::lookup(op)) {
    LLVM_DEBUG({
      llvm::dbgs() << "[OpAffinityPVS] op ";
      op->getName().print(llvm::dbgs());
      llvm::dbgs() << " affinity explicitly specified as ";
      affinityAttr.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    unionAssumed(affinityAttr);
    indicateOptimisticFixpoint();
    return;
  }
}

ChangeStatus OpAffinityPVS::updateOperation(Operation *op,
                                            DFX::Solver &solver) {
  StateType newState;

  const bool consumesAny = llvm::any_of(
      op->getOperandTypes(), +[](Type type) {
        return isa<IREE::Stream::AffinityTypeInterface>(type);
      });
  if (consumesAny) {
    for (auto operand : op->getOperands()) {
      if (isa<IREE::Stream::AffinityTypeInterface>(operand.getType())) {
        auto valuePVS = solver.getElementFor<ValueProducerAffinityPVS>(
            *this, Position::forValue(operand), DFX::Resolution::REQUIRED);
        LLVM_DEBUG({
          llvm::dbgs() << "[OpAffinityPVS] op ";
          op->getName().print(llvm::dbgs());
          llvm::dbgs() << " consumes ";
          valuePVS.print(llvm::dbgs(), solver.getAsmState());
          llvm::dbgs() << "\n";
        });
        newState ^= valuePVS;
      }
    }
  } else {
    for (auto result : op->getResults()) {
      if (isa<IREE::Stream::AffinityTypeInterface>(result.getType())) {
        auto valuePVS = solver.getElementFor<ValueConsumerAffinityPVS>(
            *this, Position::forValue(result), DFX::Resolution::REQUIRED);
        LLVM_DEBUG({
          llvm::dbgs() << "[OpAffinityPVS] op ";
          op->getName().print(llvm::dbgs());
          llvm::dbgs() << " produces ";
          valuePVS.print(llvm::dbgs(), solver.getAsmState());
          llvm::dbgs() << "\n";
        });
        newState ^= valuePVS;
      }
    }
  }

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

//===----------------------------------------------------------------------===//
// AffinityAnalysis
//===----------------------------------------------------------------------===//

// Tries to find a default affinity specified on an ancestor of |fromOp| and
// adds it to |affinities|. Returns true if an affinity was found.
static bool tryLookupDefaultAffinity(
    Operation *fromOp,
    SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities) {
  while (fromOp) {
    auto affinityAttr = fromOp->getAttrOfType<IREE::Stream::AffinityAttr>(
        "stream.affinity.default");
    if (affinityAttr) {
      affinities.push_back(affinityAttr);
      return true;
    }
    fromOp = fromOp->getParentOp();
  }
  return false;
}

// Returns the first affinity if all affinities are compatible and otherwise
// returns nullptr.
static IREE::Stream::AffinityAttr
trySelectLeadAffinity(ArrayRef<IREE::Stream::AffinityAttr> affinities) {
  if (affinities.empty()) {
    return {};
  }
  auto leadAffinityAttr = affinities.front();
  for (size_t i = 1; i < affinities.size(); ++i) {
    if (!IREE::Stream::AffinityAttr::areCompatible(affinities[i],
                                                   leadAffinityAttr)) {
      return {};
    }
  }
  return leadAffinityAttr;
}

// Sorts |affinities| in the natural affinity sort order.
// We unfortunately have to do this as the PVS elements we source from are
// unsorted.
static void
sortAffinities(SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities) {
  // HACK: this should probably do a type id ordering followed by a
  // type-specific ordering (interface compare method?). We just need this to be
  // stable as the affinities come from multiple DenseSets that have run-to-run
  // ordering variance. This is very inefficient but is only used when there are
  // multiple possible affinities and we try to avoid that anyway.
  if (affinities.size() <= 1) {
    return;
  }
  llvm::stable_sort(affinities, [](IREE::Stream::AffinityAttr lhs,
                                   IREE::Stream::AffinityAttr rhs) {
    std::string lhsStr;
    llvm::raw_string_ostream lhsStream(lhsStr);
    lhs.print(lhsStream);
    std::string rhsStr;
    llvm::raw_string_ostream rhsStream(rhsStr);
    rhs.print(rhsStream);
    return lhsStr < rhsStr;
  });
}

AffinityAnalysis::AffinityAnalysis(Operation *rootOp)
    : explorer(rootOp, TraversalAction::RECURSE), solver(explorer, allocator) {
  explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
      TraversalAction::RECURSE);

  explorer.setDialectAction<mlir::scf::SCFDialect>(TraversalAction::RECURSE);

  explorer.setDialectAction<IREE::Stream::StreamDialect>(
      TraversalAction::RECURSE);
  explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::IGNORE);

  explorer.initialize();
}

AffinityAnalysis::~AffinityAnalysis() = default;

IREE::Stream::AffinityAttr
AffinityAnalysis::lookupGlobalAffinity(Operation *op) {
  SmallVector<IREE::Stream::AffinityAttr> affinities;
  if (!tryLookupGlobalAffinity(op, affinities) || affinities.empty()) {
    return {};
  }
  if (affinities.size() == 1) {
    return affinities.front();
  }
  return trySelectLeadAffinity(affinities);
}

bool AffinityAnalysis::tryLookupGlobalAffinity(
    Operation *op, SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities) {
  auto globalPVS =
      solver.lookupElementFor<GlobalAffinityPVS>(Position::forOperation(op));
  if (!globalPVS) {
    // Global was never analyzed (probably not an executable op); try to find a
    // default.
    return tryLookupDefaultAffinity(op, affinities);
  } else if (!globalPVS->isValidState() || globalPVS->isUndefContained()) {
    // Analysis failed.
    return false;
  } else if (globalPVS->getAssumedSet().empty()) {
    // Analysis completed but no affinity was specified; try to find a default.
    return tryLookupDefaultAffinity(op, affinities);
  }
  for (auto affinityAttr : globalPVS->getAssumedSet()) {
    affinities.push_back(affinityAttr);
  }
  sortAffinities(affinities);
  return true;
}

IREE::Stream::AffinityAttr
AffinityAnalysis::lookupExecutionAffinity(Operation *op) {
  SmallVector<IREE::Stream::AffinityAttr> affinities;
  if (!tryLookupExecutionAffinity(op, affinities) || affinities.empty()) {
    return {};
  }
  if (affinities.size() == 1) {
    return affinities.front();
  }
  return trySelectLeadAffinity(affinities);
}

bool AffinityAnalysis::tryLookupExecutionAffinity(
    Operation *op, SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities) {
  auto opPVS =
      solver.lookupElementFor<OpAffinityPVS>(Position::forOperation(op));
  if (!opPVS) {
    // Op was never analyzed (probably not an executable op); try to find a
    // default.
    return tryLookupDefaultAffinity(op, affinities);
  } else if (!opPVS->isValidState() || opPVS->isUndefContained()) {
    // Analysis failed.
    return false;
  } else if (opPVS->getAssumedSet().empty()) {
    // Analysis completed but no affinity was specified; try to find a default.
    return tryLookupDefaultAffinity(op, affinities);
  }
  for (auto affinityAttr : opPVS->getAssumedSet()) {
    affinities.push_back(affinityAttr);
  }
  sortAffinities(affinities);
  return true;
}

IREE::Stream::AffinityAttr
AffinityAnalysis::inferExecutionAffinity(Operation *op) {
  SmallVector<IREE::Stream::AffinityAttr> affinities;
  if (!tryInferExecutionAffinity(op, affinities) || affinities.empty()) {
    return {};
  }
  if (affinities.size() == 1) {
    return affinities.front();
  }
  return trySelectLeadAffinity(affinities);
}

bool AffinityAnalysis::tryInferExecutionAffinity(
    Operation *op, SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities) {
  if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
    return tryLookupExecutionAffinity(op, affinities);
  }
  DFX::PotentialValuesState<IREE::Stream::AffinityAttr> opPVS;
  const bool consumesAny = llvm::any_of(
      op->getOperandTypes(), +[](Type type) {
        return isa<IREE::Stream::AffinityTypeInterface>(type);
      });
  if (consumesAny) {
    for (auto operand : op->getOperands()) {
      if (isa<IREE::Stream::AffinityTypeInterface>(operand.getType())) {
        auto valuePVS = solver.lookupElementFor<ValueProducerAffinityPVS>(
            Position::forValue(operand), nullptr, DFX::Resolution::REQUIRED);
        if (valuePVS && valuePVS->isValidState()) {
          opPVS.unionAssumed(valuePVS->getState());
        } else {
          return false;
        }
      }
    }
  } else {
    for (auto result : op->getResults()) {
      if (isa<IREE::Stream::AffinityTypeInterface>(result.getType())) {
        auto valuePVS = solver.lookupElementFor<ValueConsumerAffinityPVS>(
            Position::forValue(result), nullptr, DFX::Resolution::REQUIRED);
        if (valuePVS && valuePVS->isValidState()) {
          opPVS.unionAssumed(valuePVS->getState());
        } else {
          return false;
        }
      }
    }
  }
  if (!opPVS.isValidState() || opPVS.isUndefContained()) {
    // Analysis failed.
    return false;
  }
  if (opPVS.getAssumedSet().empty()) {
    // Analysis completed but no affinity was specified; try to find a default.
    return tryLookupDefaultAffinity(op, affinities);
  }
  for (auto affinityAttr : opPVS.getAssumedSet()) {
    affinities.push_back(affinityAttr);
  }
  sortAffinities(affinities);
  return true;
}

IREE::Stream::AffinityAttr
AffinityAnalysis::lookupResourceAffinity(Value value) {
  SmallVector<IREE::Stream::AffinityAttr> affinities;
  if (!tryLookupResourceAffinity(value, affinities) || affinities.empty()) {
    return {};
  }
  if (affinities.size() == 1) {
    return affinities.front();
  }
  return trySelectLeadAffinity(affinities);
}

bool AffinityAnalysis::tryLookupResourceAffinity(
    Value value, SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities) {
  // If the value is pinned then we always use that.
  if (tryLookupPinnedAffinities(value, affinities)) {
    return true;
  }

  auto valuePVS = solver.lookupElementFor<ValueProducerAffinityPVS>(
      Position::forValue(value));
  if (!valuePVS || !valuePVS->isValidState() || valuePVS->isUndefContained()) {
    // Analysis failed.
    return false;
  }
  if (valuePVS->getAssumedSet().empty()) {
    // Analysis completed but no affinity was specified; try to find a default.
    return tryLookupDefaultAffinity(value.getParentBlock()->getParentOp(),
                                    affinities);
  }
  for (auto affinityAttr : valuePVS->getAssumedSet()) {
    affinities.push_back(affinityAttr);
  }
  sortAffinities(affinities);
  return true;
}

bool AffinityAnalysis::tryLookupResourceUsageAffinity(
    Value value, SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities) {
  SetVector<IREE::Stream::AffinityAttr> affinitySet;
  auto producerPVS = solver.lookupElementFor<ValueProducerAffinityPVS>(
      Position::forValue(value));
  if (producerPVS && producerPVS->isValidState() &&
      !producerPVS->isUndefContained()) {
    affinitySet.insert_range(producerPVS->getAssumedSet());
  }
  auto consumerPVS = solver.lookupElementFor<ValueConsumerAffinityPVS>(
      Position::forValue(value));
  if (consumerPVS && consumerPVS->isValidState() &&
      !consumerPVS->isUndefContained()) {
    affinitySet.insert_range(consumerPVS->getAssumedSet());
  }
  // Merge pinned affinities - they are indicating some assumption that we may
  // not be able to analyze (like external modules).
  SmallVector<IREE::Stream::AffinityAttr> pinnedAffinities;
  tryLookupPinnedAffinities(value, pinnedAffinities);
  affinitySet.insert_range(pinnedAffinities);
  if (affinitySet.empty()) {
    // Analysis completed but no affinity was specified; try to find a default.
    return tryLookupDefaultAffinity(value.getParentBlock()->getParentOp(),
                                    affinities);
  }
  llvm::append_range(affinities, affinitySet);
  sortAffinities(affinities);
  return true;
}

bool AffinityAnalysis::tryLookupPinnedAffinities(
    Value value, SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities) {
  auto it = precomputedQueries.pinnedAffinities.find(value);
  if (it == precomputedQueries.pinnedAffinities.end() || it->second.empty()) {
    return false;
  }
  DenseSet<IREE::Stream::AffinityAttr> pinnedAffinities;
  for (auto pinningOp : it->second) {
    SmallVector<IREE::Stream::AffinityAttr> pinnerAffinities;
    if (!tryLookupExecutionAffinity(pinningOp, pinnerAffinities)) {
      return false;
    }
    pinnedAffinities.insert_range(pinnerAffinities);
  }
  llvm::append_range(affinities, pinnedAffinities);
  sortAffinities(affinities);
  return true;
}

LogicalResult AffinityAnalysis::run() {
  // An unfortunate full-module walk.
  // We should try to do everything that we can before touching the solver in
  // this walk and cache it on precomputedQueries.
  precomputedQueries.compute(explorer);
  LLVM_DEBUG(precomputedQueries.print(llvm::dbgs(), solver.getAsmState()));

  // Inject sideband data into the solver. This allows our elements to lookup
  // the data by querying on the root op.
  precomputedQueries.inject(solver);

  // Initialize globals so that we can assign them affinity.
  explorer.forEachGlobal([&](const auto *globalInfo) {
    if (isa<IREE::Stream::AffinityTypeInterface>(
            globalInfo->op.getGlobalType())) {
      solver.getOrCreateElementFor<GlobalAffinityPVS>(
          Position::forOperation(globalInfo->op));
    }
  });

  // Initialize op execution affinities for any ops that use tracked types.
  //
  // TODO(benvanik): avoid doing this initialization for the entire module and
  // instead rely on DFX to automatically populate the required abstract values.
  // There's some missing logic in the element initialization, though, and by
  // initializing all values we side-step that and work with test programs that
  // may not have I/O edges that we could easily latch on to here.
  explorer.forEachFunctionLikeOp([&](FunctionOpInterface funcOp) {
    for (auto &block : funcOp.getBlocks()) {
      for (auto arg : block.getArguments()) {
        if (isa<IREE::Stream::AffinityTypeInterface>(arg.getType())) {
          solver.getOrCreateElementFor<ValueProducerAffinityPVS>(
              Position::forValue(arg));
        }
      }
    }
    funcOp.walk([&](Operation *op) {
      if (isa<FunctionOpInterface>(op)) {
        return; // ignore func/initializers/etc.
      }
      if (auto regionOp = dyn_cast<RegionBranchOpInterface>(op)) {
        for (auto &region : regionOp->getRegions()) {
          for (auto arg : region.getArguments()) {
            if (isa<IREE::Stream::AffinityTypeInterface>(arg.getType())) {
              solver.getOrCreateElementFor<ValueProducerAffinityPVS>(
                  Position::forValue(arg));
            }
          }
        }
      }
      if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
        if (affinityOp.requiresAffinity()) {
          solver.getOrCreateElementFor<OpAffinityPVS>(
              Position::forOperation(op));
        }
      }
      for (auto result : op->getResults()) {
        if (isa<IREE::Stream::AffinityTypeInterface>(result.getType())) {
          solver.getOrCreateElementFor<ValueProducerAffinityPVS>(
              Position::forValue(result));
          solver.getOrCreateElementFor<ValueConsumerAffinityPVS>(
              Position::forValue(result));
        }
      }
    });
  });

  if (failed(solver.run(clSolverMaxIterations))) {
    return failure(); // did not converge
  }

  LLVM_DEBUG({
    llvm::dbgs()
        << "\n\n[Analysis] affinity analysis results for the whole module:\n";
    solver.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  return success();
}

} // namespace mlir::iree_compiler::IREE::Stream
