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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

#define DEBUG_TYPE "iree-util-dfx"

namespace mlir::iree_compiler::IREE::Stream {

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

  // Operations that the value is pinned to.
  SetVector<Operation *> pinnedOps;
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
                                               DFX::Solver &solver) {
  // DO NOT SUBMIT
}

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
      (TraversalBehavior::DEFAULT & ~TraversalBehavior::WALK_TIED_VALUES));

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
  if (auto affinityOp =
          dyn_cast<IREE::Stream::AffinityOpInterface>(operand.getOwner())) {
    // DO NOT SUBMIT
    // if (affinityOp.pinsValueAffinity()) {
    auto opPVS = solver.getElementFor<OpAffinityPVS>(
        *this, Position::forOperation(operand.getOwner()),
        DFX::Resolution::REQUIRED);
    LLVM_DEBUG({
      llvm::dbgs() << "[ValueConsumerAffinityPVS] value ";
      value.printAsOperand(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << " affinity using consumer affinity from ";
      operand.get().printAsOperand(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << " as ";
      opPVS.print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    newState ^= opPVS;
  }

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

  return TypeSwitch<Operation *, TraversalResult>(operand.getOwner())
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
                                               DFX::Solver &solver) {
  solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
    if (!isa<IREE::Stream::AffinityTypeInterface>(result.getType())) {
      return WalkResult::skip();
    }
    if (auto affinityOp =
            dyn_cast_if_present<IREE::Stream::AffinityOpInterface>(
                result.getOwner())) {
      if (affinityOp.pinsValueAffinity()) {
        pinnedOps.insert(result.getOwner());
      }
    }
    return WalkResult::advance();
  });
  solver.getExplorer().walkTransitiveUses(value, [&](OpOperand &operand) {
    if (!isa<IREE::Stream::AffinityTypeInterface>(operand.get().getType())) {
      return WalkResult::skip();
    }
    if (auto affinityOp =
            dyn_cast_if_present<IREE::Stream::AffinityOpInterface>(
                operand.getOwner())) {
      if (affinityOp.pinsValueAffinity()) {
        pinnedOps.insert(operand.getOwner());
      }
    }
    return WalkResult::advance();
  });
// DO NOT SUBMIT
#if 0
  if (auto affinityOp = dyn_cast_if_present<IREE::Stream::AffinityOpInterface>(
          value.getDefiningOp())) {
    if (auto affinityAttr = affinityOp.getAffinityAttr()) {
      if (affinityOp.pinsValueAffinity()) {
        LLVM_DEBUG({
          llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
          value.printAsOperand(llvm::dbgs(), solver.getAsmState());
          llvm::dbgs() << " affinity pinned to defining op affinity: ";
          affinityAttr.print(llvm::dbgs());
          llvm::dbgs() << "\n";
        });
        unionAssumed(affinityAttr);
        indicateOptimisticFixpoint();
        return;
      }
    }
  }
#endif
}

ChangeStatus ValueProducerAffinityPVS::updateValue(Value value,
                                                   DFX::Solver &solver) {
  StateType newState;

  if (!pinnedOps.empty()) {
    for (auto pinnedOp : pinnedOps) {
      auto &opPVS = solver.getElementFor<OpAffinityPVS>(
          *this, Position::forOperation(pinnedOp), DFX::Resolution::REQUIRED);
      newState ^= opPVS;
    }
    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

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
    newState ^= operandPVS;
    return DFX::clampStateAndIndicateChange(getState(), newState);
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
    newState ^= globalPVS.getState();
    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

  // DO NOT SUBMIT
  auto traversalResult = TraversalResult::COMPLETE;
  traversalResult |= solver.getExplorer().walkDefiningOps(
      value,
      [&](OpResult result) {
        // DO NOT SUBMIT skip branch/call/etc
        // need an attr for device touch?
        if (isa<CallOpInterface>(result.getOwner()))
          return WalkResult::advance();

        // DO NOT SUBMIT
        // interface that pins storage?
        if (auto affinityOp =
                dyn_cast_if_present<IREE::Stream::AffinityOpInterface>(
                    result.getDefiningOp())) {
          if (affinityOp.pinsValueAffinity()) {
            auto &opPVS = solver.getElementFor<OpAffinityPVS>(
                *this, Position::forOperation(affinityOp),
                DFX::Resolution::REQUIRED);
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
              value.printAsOperand(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << " affinity using assuming pinned affinity from ";
              result.printAsOperand(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << " as ";
              opPVS.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            newState ^= opPVS;
            newState.indicateOptimisticFixpoint();
            return WalkResult::advance();
          }
        }

        // if tied then use tied value
        // else if affinity then use op value
        // else use result value
        if (auto tiedOp =
                dyn_cast<IREE::Util::TiedOpInterface>(result.getDefiningOp())) {
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
            newState ^= valuePVS;
            return WalkResult::advance();
          }
        }

        if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(
                result.getDefiningOp())) {
          auto &opPVS = solver.getElementFor<OpAffinityPVS>(
              *this, Position::forOperation(result.getOwner()),
              DFX::Resolution::OPTIONAL);
          LLVM_DEBUG({
            llvm::dbgs() << "[ValueProducerAffinityPVS] value ";
            value.printAsOperand(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << " affinity using op affinity from result ";
            result.printAsOperand(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << " as ";
            opPVS.print(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << "\n";
          });
          newState ^= opPVS;
          return WalkResult::advance();
        }

        TypeSwitch<Operation *>(result.getOwner())
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
              newState ^= globalPVS.getState();
            })
            .Case<mlir::arith::SelectOp>([&](auto op) {
              auto &truePVS = solver.getElementFor<ValueProducerAffinityPVS>(
                  *this, Position::forValue(op.getTrueValue()),
                  DFX::Resolution::REQUIRED);
              newState ^= truePVS.getState();
              auto &falsePVS = solver.getElementFor<ValueProducerAffinityPVS>(
                  *this, Position::forValue(op.getFalseValue()),
                  DFX::Resolution::REQUIRED);
              newState ^= falsePVS.getState();
            })
            // DO NOT SUBMIT branch
            // DO NOT SUBMIT scf
            .Default([&](auto op) {
              auto valuePVS = solver.getElementFor<ValueProducerAffinityPVS>(
                  *this, Position::forValue(result), DFX::Resolution::OPTIONAL);
              newState ^= valuePVS;
            });
        return WalkResult::advance();
      },
      (TraversalBehavior::DEFAULT & ~TraversalBehavior::WALK_TIED_VALUES));

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
        newState ^= valuePVS;
      }
    }
  } else {
    for (auto result : op->getResults()) {
      if (isa<IREE::Stream::AffinityTypeInterface>(result.getType())) {
        auto valuePVS = solver.getElementFor<ValueConsumerAffinityPVS>(
            *this, Position::forValue(result), DFX::Resolution::REQUIRED);
        newState ^= valuePVS;
      }
    }
  }

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

//===----------------------------------------------------------------------===//
// AffinityAnalysis
//===----------------------------------------------------------------------===//

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
  if (!globalPVS || !globalPVS->isValidState() ||
      globalPVS->isUndefContained()) {
    return false;
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
  if (!opPVS || !opPVS->isValidState() || opPVS->isUndefContained()) {
    return false;
  }
  for (auto affinityAttr : opPVS->getAssumedSet()) {
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
  auto valuePVS = solver.lookupElementFor<ValueProducerAffinityPVS>(
      Position::forValue(value));
  if (!valuePVS || !valuePVS->isValidState() || valuePVS->isUndefContained()) {
    return false;
  }
  for (auto affinityAttr : valuePVS->getAssumedSet()) {
    affinities.push_back(affinityAttr);
  }
  sortAffinities(affinities);
  return true;
}

LogicalResult AffinityAnalysis::run() {
  // Initialize globals so that we can assign them affinity.
  explorer.forEachGlobal([&](const auto *globalInfo) {
    if (isa<IREE::Stream::AffinityTypeInterface>(
            globalInfo->op.getGlobalType())) {
      solver.getOrCreateElementFor<GlobalAffinityPVS>(
          Position::forOperation(globalInfo->op));
    }
  });

  // Initialize op execution affinities for any ops that use tracked types.
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
        solver.getOrCreateElementFor<OpAffinityPVS>(Position::forOperation(op));
      }
      for (auto result : op->getResults()) {
        if (isa<IREE::Stream::AffinityTypeInterface>(result.getType())) {
          solver.getOrCreateElementFor<ValueProducerAffinityPVS>(
              Position::forValue(result));
        }
      }
    });
  });

  if (failed(solver.run())) {
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
