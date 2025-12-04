// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-elide-timepoints"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ELIDETIMEPOINTSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Resource usage query/application patterns
//===----------------------------------------------------------------------===//

// Returns true if |value| is defined as a #stream.timepoint.immediate.
static bool isDefinedImmediate(Value value) {
  return isa_and_nonnull<IREE::Stream::TimepointImmediateOp>(
      value.getDefiningOp());
}

// Tracks whether a util.global of !stream.timepoint is immediately resolved.
// Boolean state will be set to false if any stores are non-immediate.
class IsGlobalImmediate
    : public DFX::StateWrapper<
          DFX::BooleanState, DFX::TypedOperationElement<IREE::Util::GlobalOp>> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::BooleanState,
                        DFX::TypedOperationElement<IREE::Util::GlobalOp>>;

  static IsGlobalImmediate &createForPosition(const Position &pos,
                                              DFX::Solver &solver) {
    return *(new (solver.getAllocator()) IsGlobalImmediate(pos));
  }

  bool isImmediate() const { return isAssumed(); }

  const std::string getName() const override { return "IsGlobalImmediate"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return std::string("is_immediate: ") + std::to_string(isAssumed());
  }

private:
  explicit IsGlobalImmediate(const Position &pos) : BaseType(pos) {}

  void initializeOperation(IREE::Util::GlobalOp globalOp,
                           DFX::Solver &solver) override {
    // Immutable constant globals are all immediate. Initialized globals may
    // end up not being immediate and we'll need to analyze.
    if (!globalOp.getIsMutable() && globalOp.getInitialValue().has_value()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[ElideTimepoints] immutable immediate global: ";
        globalOp.print(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      setKnown(true);
      indicateOptimisticFixpoint();
      return;
    }

    // Globals must have been analyzed in order to be tracked.
    // Indirectly-accessed globals are not currently supported.
    auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
    if (!globalInfo || globalInfo->isIndirect) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "[ElideTimepoints] unanalyzed/indirect global ignored: ";
        globalOp.print(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      setKnown(false);
      indicatePessimisticFixpoint();
      return;
    }

    // Assume true until proven otherwise.
    setAssumed(true);
  }

  ChangeStatus updateOperation(IREE::Util::GlobalOp globalOp,
                               DFX::Solver &solver) override;

  friend class DFX::Solver;
};
const char IsGlobalImmediate::ID = 0;

// Tracks whether a !stream.timepoint is immediately resolved.
// Boolean state will be set to false if any sources are non-immediate.
class IsImmediate
    : public DFX::StateWrapper<DFX::BooleanState, DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<DFX::BooleanState, DFX::ValueElement>;

  static IsImmediate &createForPosition(const Position &pos,
                                        DFX::Solver &solver) {
    return *(new (solver.getAllocator()) IsImmediate(pos));
  }

  bool isImmediate() const { return isAssumed(); }

  const std::string getName() const override { return "IsImmediate"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return std::string("is_immediate: ") + std::to_string(isAssumed());
  }

private:
  explicit IsImmediate(const Position &pos) : BaseType(pos) {}

  void initializeValue(Value value, DFX::Solver &solver) override {
    // Immediate timepoints (constant resolved) are always available and cover
    // everything. We check for this as a special case to short-circuit the
    // solver.
    if (isDefinedImmediate(value)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[ElideTimepoints] defined immediate: ";
        value.printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      setKnown(true);
      indicateOptimisticFixpoint();
      return;
    }

    // Assume true until proven otherwise.
    setAssumed(true);
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    StateType newState = getState();

    auto traversalResult = TraversalResult::COMPLETE;

    // Scan IR to see if we can identify whether this definitely comes from an
    // immediate op. This will reach across block and call edges and may fan out
    // into many incoming ops - all of them must be immediate for this op to be
    // considered immediate.
    traversalResult |=
        solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
          updateFromDefiningOp(newState, value, result, solver);
          return WalkResult::advance();
        });

    if (traversalResult == TraversalResult::INCOMPLETE) {
      newState.indicatePessimisticFixpoint();
    }

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

  // Updates the usage based on the op defining the value.
  void updateFromDefiningOp(StateType &newState, Value value, OpResult result,
                            DFX::Solver &solver) {
    TypeSwitch<Operation *, void>(result.getOwner())
        .Case([&](IREE::Util::GlobalLoadOpInterface op) {
          auto *globalInfo =
              solver.getExplorer().queryGlobalInfoFrom(op.getGlobalName(), op);
          if (!globalInfo || globalInfo->isIndirect) {
            LLVM_DEBUG(
                {
                  llvm::dbgs()
                      << "[ElideTimepoints] indirect usage global backing ";
                  value.printAsOperand(llvm::dbgs(), solver.getAsmState());
                  llvm::dbgs() << "; marking undef\n";
                });
            newState.indicatePessimisticFixpoint();
            return;
          }
          auto isImmediate = solver.getElementFor<IsGlobalImmediate>(
              *this, Position::forOperation(globalInfo->op),
              DFX::Resolution::REQUIRED);
          LLVM_DEBUG({
            llvm::dbgs() << "[ElideTimepoints] global load ";
            isImmediate.print(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << "\n";
          });
          newState ^= isImmediate.getState();
        })
        .Case([&](IREE::Stream::TimepointImmediateOp op) {
          // Defined by an immediate op; definitely immediate.
          newState.setAssumed(true);
        })
        .Case([&](IREE::Stream::TimepointJoinOp op) {
          // Only immediate if all inputs to the join are immediate.
          for (auto operand : op.getAwaitTimepoints()) {
            auto isImmediate = solver.getElementFor<IsImmediate>(
                *this, Position::forValue(operand), DFX::Resolution::REQUIRED);
            LLVM_DEBUG({
              llvm::dbgs() << "[ElideTimepoints] join operand ";
              isImmediate.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            newState ^= isImmediate.getState();
          }
        })
        .Case([&](IREE::Stream::TimelineOpInterface op) {
          // Defined by a timeline operation that ensures it's never immediate.
          LLVM_DEBUG({
            llvm::dbgs() << "[ElideTimepoints] non-immediate timeline op: ";
            value.printAsOperand(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << "\n";
          });
          newState.indicatePessimisticFixpoint();
        })
        .Case([&](arith::SelectOp op) {
          auto isTrueImmediate = solver.getElementFor<IsImmediate>(
              *this, Position::forValue(op.getTrueValue()),
              DFX::Resolution::REQUIRED);
          auto isFalseImmediate = solver.getElementFor<IsImmediate>(
              *this, Position::forValue(op.getFalseValue()),
              DFX::Resolution::REQUIRED);
          LLVM_DEBUG({
            llvm::dbgs() << "[ElideTimepoints] select join ";
            isTrueImmediate.print(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << " OR ";
            isFalseImmediate.print(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << "\n";
          });
          newState ^= isTrueImmediate.getState();
          newState ^= isFalseImmediate.getState();
        })
        // Allowed because traversal will take care of things:
        .Case([&](mlir::CallOpInterface) {})
        .Case([&](mlir::BranchOpInterface) {})
        .Case([&](scf::IfOp) {})
        .Case([&](scf::ForOp) {})
        .Default([&](Operation *op) {
          // Unknown op defines the value - we can't make any assumptions.
          LLVM_DEBUG({
            llvm::dbgs() << "[ElideTimepoints] unknown usage of ";
            value.printAsOperand(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << " by " << op->getName() << "\n";
          });
          newState.indicatePessimisticFixpoint();
        });
  }

  friend class DFX::Solver;
};
const char IsImmediate::ID = 0;

ChangeStatus IsGlobalImmediate::updateOperation(IREE::Util::GlobalOp globalOp,
                                                DFX::Solver &solver) {
  IsGlobalImmediate::StateType newState = getState();

  auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
  assert(globalInfo && "analysis required");

  // Walk all stores and clamp to their status.
  for (auto storeOp : globalInfo->getStores()) {
    auto isImmediate = solver.getElementFor<IsImmediate>(
        *this, Position::forValue(storeOp.getStoredGlobalValue()),
        DFX::Resolution::REQUIRED);
    LLVM_DEBUG({
      llvm::dbgs() << "[ElideTimepoints] global store: ";
      storeOp.getStoredGlobalValue().printAsOperand(llvm::dbgs(),
                                                    solver.getAsmState());
      llvm::dbgs() << "; ";
      isImmediate.print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    newState ^= isImmediate;
  }

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

// Tracks which timepoints are covered by a fence from timeline-aware ops.
// This is used to bridge HAL fence values to stream timepoints for operations
// like util.call with async fence arguments.
class FenceCoverage : public DFX::StateWrapper<DFX::PotentialValuesState<Value>,
                                               DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<Value>, DFX::ValueElement>;

  static FenceCoverage &createForPosition(const Position &pos,
                                          DFX::Solver &solver) {
    return *(new (solver.getAllocator()) FenceCoverage(pos));
  }

  const std::string getName() const override { return "FenceCoverage"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    std::string str;
    llvm::raw_string_ostream sstream(str);
    sstream << "fence_covered: ";
    if (isValidState()) {
      sstream << "[";
      llvm::interleaveComma(getAssumedSet(), sstream, [&](Value value) {
        value.printAsOperand(sstream, asmState);
      });
      sstream << "]";
    } else {
      sstream << "(invalid)";
    }
    sstream.flush();
    return str;
  }

private:
  explicit FenceCoverage(const Position &pos) : BaseType(pos) {}

  void initializeValue(Value fence, DFX::Solver &solver) override {
    // Find timeline-aware ops that use this fence directly.
    // Transitive coverage is handled in updateValue() through FenceCoverage
    // dependencies to avoid the O(F × U × A) nested walk complexity.
    auto processTimelineAwareOp =
        [&](IREE::Stream::TimelineAwareOpInterface awareOp) {
          if (!awareOp.participatesInTimeline() ||
              awareOp.getSignalFence() != fence) {
            return;
          }
          // Found a timeline-aware op signaling this fence.
          // Record its await fences for processing in updateValue().
          for (Value awaitFence : awareOp.getAwaitFences()) {
            // We'll process this await fence's coverage in updateValue().
            // For now, just record it by walking to find direct exports.
            solver.getExplorer().walkDefiningOps(
                awaitFence, [&](OpResult awaitResult) {
                  if (auto exportOp = dyn_cast<IREE::Stream::TimepointExportOp>(
                          awaitResult.getOwner())) {
                    // Check if this export result is the await fence.
                    for (auto result : exportOp.getResults()) {
                      if (result == awaitResult) {
                        unionAssumed(exportOp.getAwaitTimepoint());
                        LLVM_DEBUG({
                          llvm::dbgs() << "[ElideTimepoints] fence ";
                          fence.printAsOperand(llvm::dbgs(),
                                               solver.getAsmState());
                          llvm::dbgs() << " directly covers timepoint ";
                          exportOp.getAwaitTimepoint().printAsOperand(
                              llvm::dbgs(), solver.getAsmState());
                          llvm::dbgs() << "\n";
                        });
                        break;
                      }
                    }
                  }
                  return WalkResult::advance();
                });
          }
        };

    // Check if any users are timeline-aware ops (for fence operands).
    for (auto *user : fence.getUsers()) {
      if (auto awareOp =
              dyn_cast<IREE::Stream::TimelineAwareOpInterface>(user)) {
        processTimelineAwareOp(awareOp);
      }
    }

    // Walk defining ops to find timeline-aware producers (for fence results).
    solver.getExplorer().walkDefiningOps(fence, [&](OpResult valueResult) {
      auto *definingOp = valueResult.getOwner();
      if (auto awareOp =
              dyn_cast<IREE::Stream::TimelineAwareOpInterface>(definingOp)) {
        if (awareOp.getSignalFence() == valueResult) {
          processTimelineAwareOp(awareOp);
        }
      }
      return WalkResult::advance();
    });
  }

  // Defined after TimepointCoverage since it uses TimepointCoverage methods.
  ChangeStatus updateValue(Value fence, DFX::Solver &solver) override;

  friend class DFX::Solver;
};
const char FenceCoverage::ID = 0;

class TimepointCoverage
    : public DFX::StateWrapper<DFX::PotentialValuesState<Value>,
                               DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<Value>, DFX::ValueElement>;

  static TimepointCoverage &createForPosition(const Position &pos,
                                              DFX::Solver &solver) {
    return *(new (solver.getAllocator()) TimepointCoverage(pos));
  }

  const std::string getName() const override { return "TimepointCoverage"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  // Returns true if the given |value| is known to be covered by this value
  // indicating that any time this value is reached |value| must also have been.
  bool covers(Value value) const { return getAssumedSet().contains(value); }

  const std::string getAsStr(AsmState &asmState) const override {
    std::string str;
    llvm::raw_string_ostream sstream(str);
    sstream << "covered: ";
    if (isValidState()) {
      sstream << "[";
      if (isUndefContained()) {
        sstream << "undef, ";
      }
      llvm::interleaveComma(getAssumedSet(), sstream, [&](Value value) {
        value.printAsOperand(sstream, asmState);
        sstream << "(" << (void *)value.getImpl() << ")";
      });
      sstream << "]";
    } else {
      sstream << "(invalid)";
    }
    sstream.flush();
    return str;
  }

private:
  explicit TimepointCoverage(const Position &pos) : BaseType(pos) {}

  void initializeValue(Value value, DFX::Solver &solver) override {
    // Immediate timepoints (constant resolved) are always available and cover
    // everything. We check for this as a special case to short-circuit the
    // solver.
    if (isDefinedImmediate(value)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[ElideTimepoints] defined immediate: ";
        value.printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      unionAssumed(value);
      indicateOptimisticFixpoint();
      return;
    }
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    StateType newState;

    // Intersect coverage of all incoming block edge operands.
    // This will also step outside the entry block and into callee functions.
    // The intersection prevents back-edges from polluting block arguments.
    auto gatherBlockOperands = [&](BlockArgument blockArg) {
      StateType uniformState;
      bool firstEdge = true;
      if (solver.getExplorer().walkIncomingBlockArgument(
              blockArg, [&](Block *sourceBlock, Value operand) {
                auto operandCoverage = solver.getElementFor<TimepointCoverage>(
                    *this, Position::forValue(operand),
                    DFX::Resolution::REQUIRED);
                LLVM_DEBUG({
                  llvm::dbgs()
                      << "[ElideTimepoints] intersect incoming branch operand ";
                  operandCoverage.print(llvm::dbgs(), solver.getAsmState());
                  llvm::dbgs() << "\n";
                });
                if (firstEdge) {
                  uniformState = operandCoverage.getState();
                  firstEdge = false;
                } else {
                  uniformState.intersectAssumed(operandCoverage.getState());
                }
                return WalkResult::advance();
              }) == TraversalResult::INCOMPLETE) {
        LLVM_DEBUG(llvm::dbgs() << "[ElideTimepoints] incomplete branch arg "
                                   "traversal; assuming unknown\n");
        uniformState.unionAssumedWithUndef();
      }
      newState ^= uniformState;
      newState.unionAssumed(blockArg);
    };

    // Intersect coverage of all callee/child region return operands.
    // The intersection prevents multiple return sites from interfering.
    auto gatherRegionReturns = [&](Operation *regionOp, unsigned resultIndex) {
      StateType uniformState;
      bool firstEdge = true;
      if (solver.getExplorer().walkReturnOperands(
              regionOp, [&](OperandRange operands) {
                auto operand = operands[resultIndex];
                auto operandCoverage = solver.getElementFor<TimepointCoverage>(
                    *this, Position::forValue(operand),
                    DFX::Resolution::REQUIRED);
                LLVM_DEBUG({
                  llvm::dbgs()
                      << "[ElideTimepoints] intersect incoming return operand ";
                  operandCoverage.print(llvm::dbgs(), solver.getAsmState());
                  llvm::dbgs() << "\n";
                });
                if (firstEdge) {
                  uniformState = operandCoverage.getState();
                  firstEdge = false;
                } else {
                  uniformState.intersectAssumed(operandCoverage.getState());
                }
                return WalkResult::advance();
              }) == TraversalResult::INCOMPLETE) {
        LLVM_DEBUG(llvm::dbgs() << "[ElideTimepoints] incomplete region "
                                   "traversal; assuming unknown\n");
        uniformState.unionAssumedWithUndef();
      }
      newState ^= uniformState;
    };

    auto *definingOp = value.getDefiningOp();
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      // Block arguments need an intersection of all incoming branch/call edges.
      gatherBlockOperands(blockArg);
      return DFX::clampStateAndIndicateChange(getState(), newState);
    }

    TypeSwitch<Operation *>(definingOp)
        .Case([&](IREE::Stream::TimelineOpInterface timelineOp) {
          // Value defined from a timeline op and we can mark all awaits of
          // the op as covered by the result.
          for (auto operand : timelineOp.getAwaitTimepoints()) {
            auto operandCoverage = solver.getElementFor<TimepointCoverage>(
                *this, Position::forValue(operand), DFX::Resolution::REQUIRED);
            LLVM_DEBUG({
              llvm::dbgs() << "[ElideTimepoints] dependent timeline operand ";
              operandCoverage.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            newState.unionAssumed(operand);
            newState &= operandCoverage;
          }
          // Timepoints cover themselves; this is redundant but simplifies the
          // set logic later on.
          if (auto resultTimepoint = timelineOp.getResultTimepoint()) {
            newState.unionAssumed(resultTimepoint);
            LLVM_DEBUG({
              llvm::dbgs() << "[ElideTimepoints] produced timeline result ";
              resultTimepoint.printAsOperand(llvm::dbgs(),
                                             solver.getAsmState());
              llvm::dbgs() << "\n";
            });
          }
        })
        .Case([&](IREE::Stream::TimepointImportOp importOp) {
          // Special handling for stream.timepoint.import: check if the imported
          // value comes from a timeline-aware op and query FenceCoverage for
          // each imported fence to find covered timepoints.
          for (Value importedValue : importOp.getOperands()) {
            auto &fenceCoverage = solver.getElementFor<FenceCoverage>(
                *this, Position::forValue(importedValue),
                DFX::Resolution::REQUIRED);
            if (fenceCoverage.isValidState()) {
              // Add all timepoints covered by this fence.
              for (Value coveredTimepoint : fenceCoverage.getAssumedSet()) {
                newState.unionAssumed(coveredTimepoint);
                auto coverage = solver.getElementFor<TimepointCoverage>(
                    *this, Position::forValue(coveredTimepoint),
                    DFX::Resolution::REQUIRED);
                newState &= coverage;
                LLVM_DEBUG({
                  llvm::dbgs() << "[ElideTimepoints] imported fence ";
                  importedValue.printAsOperand(llvm::dbgs(),
                                               solver.getAsmState());
                  llvm::dbgs() << " covers exported timepoint ";
                  coveredTimepoint.printAsOperand(llvm::dbgs(),
                                                  solver.getAsmState());
                  llvm::dbgs() << "\n";
                });
              }
            }
          }
        })
        .Case([&](mlir::CallOpInterface callOp) {
          // Step into called functions and get a coverage intersection of all
          // return sites.
          auto callableOp = callOp.resolveCallableInTable(
              &solver.getExplorer().getSymbolTables());
          unsigned resultIndex = cast<OpResult>(value).getResultNumber();
          gatherRegionReturns(callableOp, resultIndex);
        })
        .Case([&](RegionBranchOpInterface regionOp) {
          // Step into regions and get a coverage intersection of all return
          // sites.
          unsigned resultIndex = cast<OpResult>(value).getResultNumber();
          gatherRegionReturns(regionOp, resultIndex);
        })
        .Case([&](arith::SelectOp op) {
          auto trueCoverage = solver.getElementFor<TimepointCoverage>(
              *this, Position::forValue(op.getTrueValue()),
              DFX::Resolution::REQUIRED);
          auto falseCoverage = solver.getElementFor<TimepointCoverage>(
              *this, Position::forValue(op.getFalseValue()),
              DFX::Resolution::REQUIRED);
          LLVM_DEBUG({
            llvm::dbgs() << "[ElideTimepoints] select join ";
            trueCoverage.print(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << " AND ";
            falseCoverage.print(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << "\n";
          });
          newState &= trueCoverage;
          newState &= falseCoverage;
        });

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

  friend class DFX::Solver;
};
const char TimepointCoverage::ID = 0;

// Define FenceCoverage::updateValue() after TimepointCoverage is fully defined.
ChangeStatus FenceCoverage::updateValue(Value fence, DFX::Solver &solver) {
  StateType newState = getState();

  // Gather transitive coverage from await fences.
  // We need to query FenceCoverage for await fences to get their transitive
  // coverage, and TimepointCoverage for the timepoints we directly cover.
  auto processTimelineAwareOp =
      [&](IREE::Stream::TimelineAwareOpInterface awareOp) {
        if (!awareOp.participatesInTimeline() ||
            awareOp.getSignalFence() != fence) {
          return;
        }
        // For each await fence, get its transitive coverage.
        for (Value awaitFence : awareOp.getAwaitFences()) {
          // Query FenceCoverage for this await fence to get transitive
          // coverage. This lets the DFX solver handle fixpoint iteration.
          auto &awaitFenceCoverage = solver.getElementFor<FenceCoverage>(
              *this, Position::forValue(awaitFence), DFX::Resolution::REQUIRED);
          if (awaitFenceCoverage.isValidState()) {
            LLVM_DEBUG({
              llvm::dbgs() << "[ElideTimepoints] fence ";
              fence.printAsOperand(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << " transitively covers via await fence ";
              awaitFence.printAsOperand(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << ": ";
              awaitFenceCoverage.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            newState &= awaitFenceCoverage;
          }
        }
      };

  // Check users for timeline-aware ops.
  for (auto *user : fence.getUsers()) {
    if (auto awareOp = dyn_cast<IREE::Stream::TimelineAwareOpInterface>(user)) {
      processTimelineAwareOp(awareOp);
    }
  }

  // Walk defining ops to find timeline-aware producers.
  solver.getExplorer().walkDefiningOps(fence, [&](OpResult valueResult) {
    auto *definingOp = valueResult.getOwner();
    if (auto awareOp =
            dyn_cast<IREE::Stream::TimelineAwareOpInterface>(definingOp)) {
      if (awareOp.getSignalFence() == valueResult) {
        processTimelineAwareOp(awareOp);
      }
    }
    return WalkResult::advance();
  });

  // Also propagate coverage from the timepoints we directly cover.
  for (Value timepoint : getAssumedSet()) {
    auto &timepointCoverage = solver.getElementFor<TimepointCoverage>(
        *this, Position::forValue(timepoint), DFX::Resolution::REQUIRED);
    if (timepointCoverage.isValidState()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[ElideTimepoints] fence ";
        fence.printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << " covers timepoint ";
        timepoint.printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << " which covers: ";
        timepointCoverage.print(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      newState &= timepointCoverage;
    }
  }

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

class TimepointCoverageAnalysis {
public:
  explicit TimepointCoverageAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::RECURSE),
        solver(explorer, allocator) {
    // Ignore the contents of executables (linalg goo, etc) and execution
    // regions (they don't impact timepoints).
    explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::IGNORE);
    explorer.setOpAction<IREE::Stream::AsyncExecuteOp>(
        TraversalAction::SHALLOW);
    explorer.setOpAction<IREE::Stream::CmdExecuteOp>(TraversalAction::SHALLOW);
    explorer.initialize();

    assert(rootOp->getNumRegions() == 1 && "expected module-like root op");
    topLevelOps = llvm::to_vector(
        rootOp->getRegions().front().getOps<mlir::CallableOpInterface>());
  }

  AsmState &getAsmState() { return solver.getAsmState(); }
  Explorer &getExplorer() { return explorer; }

  // Runs analysis and populates the state cache.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run() {
    explorer.forEachGlobal([&](const auto *globalInfo) {
      solver.getOrCreateElementFor<IsGlobalImmediate>(
          Position::forOperation(globalInfo->op));
      for (auto loadOp : globalInfo->getLoads()) {
        solver.getOrCreateElementFor<IsImmediate>(
            Position::forValue(loadOp.getLoadedGlobalValue()));
      }
    });
    std::function<void(Region &)> seedRegion;
    seedRegion = [&](Region &region) {
      for (auto &block : region) {
        // Seed all block arguments.
        for (auto arg : block.getArguments()) {
          if (isa<IREE::Stream::TimepointType>(arg.getType())) {
            solver.getOrCreateElementFor<IsImmediate>(Position::forValue(arg));
          }
        }

        // Seed the timepoints created from any timeline ops.
        for (auto op : block.getOps<IREE::Stream::TimelineOpInterface>()) {
          for (auto operand : op.getAwaitTimepoints()) {
            solver.getOrCreateElementFor<TimepointCoverage>(
                Position::forValue(operand));
            solver.getOrCreateElementFor<IsImmediate>(
                Position::forValue(operand));
          }
          if (auto resultTimepoint = op.getResultTimepoint()) {
            solver.getOrCreateElementFor<TimepointCoverage>(
                Position::forValue(resultTimepoint));
            solver.getOrCreateElementFor<IsImmediate>(
                Position::forValue(resultTimepoint));
          }
        }

        // Seed FenceCoverage for fence values that might be relevant.
        // This includes fences from timeline-aware ops and imported fences.
        for (auto op : block.getOps<IREE::Stream::TimelineAwareOpInterface>()) {
          if (!op.participatesInTimeline())
            continue;
          if (Value signalFence = op.getSignalFence()) {
            solver.getOrCreateElementFor<FenceCoverage>(
                Position::forValue(signalFence));
          }
          for (Value awaitFence : op.getAwaitFences()) {
            solver.getOrCreateElementFor<FenceCoverage>(
                Position::forValue(awaitFence));
          }
        }

        // Also seed for fences being imported by TimepointImportOp.
        for (auto importOp : block.getOps<IREE::Stream::TimepointImportOp>()) {
          for (Value importedValue : importOp.getOperands()) {
            solver.getOrCreateElementFor<FenceCoverage>(
                Position::forValue(importedValue));
          }
        }

        // Seed all terminator operands.
        if (auto *terminatorOp = block.getTerminator()) {
          for (auto operand : terminatorOp->getOperands()) {
            if (isa<IREE::Stream::TimepointType>(operand.getType())) {
              solver.getOrCreateElementFor<TimepointCoverage>(
                  Position::forValue(operand));
              solver.getOrCreateElementFor<IsImmediate>(
                  Position::forValue(operand));
            }
          }
        }
      }

      // Walk into nested ops.
      region.walk([&](RegionBranchOpInterface nestedOp) {
        for (auto &nestedRegion : nestedOp->getRegions()) {
          seedRegion(nestedRegion);
        }
      });
    };
    for (auto callableOp : getTopLevelOps()) {
      auto *region = callableOp.getCallableRegion();
      if (!region || region->empty())
        continue;
      seedRegion(*region);
    }

    // Run solver to completion.
    auto result = solver.run();
    LLVM_DEBUG(solver.print(llvm::dbgs()));
    return result;
  }

  // Returns a list of all top-level callable ops in the root op.
  ArrayRef<mlir::CallableOpInterface> getTopLevelOps() const {
    return topLevelOps;
  }

  // Returns true if |value| is known to be immediately resolved.
  bool isImmediate(Value value) {
    if (isDefinedImmediate(value))
      return true;
    auto &isImmediate =
        solver.getOrCreateElementFor<IsImmediate>(Position::forValue(value));
    return isImmediate.isValidState() && isImmediate.isKnown();
  }

  // Returns true if the given |consumerTimepoints| are covered by the
  // |producerTimepoint|.
  bool covers(Value producerTimepoint, ValueRange consumerTimepoints) {
    // Empty await list covers nothing - need proactive folding.
    if (consumerTimepoints.empty()) {
      return false;
    }
    for (Value consumerTimepoint : consumerTimepoints) {
      auto coverage = solver.getOrCreateElementFor<TimepointCoverage>(
          Position::forValue(producerTimepoint));
      if (!coverage.isValidState() || !coverage.covers(consumerTimepoint)) {
        return false;
      }
      LLVM_DEBUG({
        llvm::dbgs() << "[ElideTimepoints] producer timepoint ";
        producerTimepoint.printAsOperand(llvm::dbgs(), getAsmState());
        llvm::dbgs() << " already covered by ";
        consumerTimepoint.printAsOperand(llvm::dbgs(), getAsmState());
        llvm::dbgs() << "\n";
      });
    }
    return true;
  }

  // Unions all transitively reached timepoints by the time |value| is reached.
  bool unionTransitivelyReachedTimepoints(Value value, SetVector<Value> &set) {
    auto coverage = solver.getOrCreateElementFor<TimepointCoverage>(
        Position::forValue(value));
    if (!coverage.isValidState() || coverage.isUndefContained())
      return false;
    for (auto reached : coverage.getAssumedSet()) {
      set.insert(reached);
    }
    return true;
  }

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
  SmallVector<mlir::CallableOpInterface> topLevelOps;
};

// Prunes |possibleTimepoints| into a set of required timepoints.
// Any timepoints not in the resulting set are required.
static SetVector<Value>
buildRequiredCoverageSet(SmallVector<Value> possibleTimepoints,
                         TimepointCoverageAnalysis &analysis) {
  // Build a map that effectively tracks an incoming edge counter for each
  // timepoint. Values with no incoming edges are required.
  DenseMap<Value, int> coverageMap;
  for (auto possibleTimepoint : possibleTimepoints) {
    // Query all transitively reached timepoints from this potentially required
    // timepoint. If analysis failed we skip it and ensure the timepoint is
    // pulled in unless something else covers it.
    SetVector<Value> reachedTimepoints;
    bool isValid = analysis.unionTransitivelyReachedTimepoints(
        possibleTimepoint, reachedTimepoints);
    if (isValid) {
      for (auto reachedTimepoint : reachedTimepoints) {
        // TODO(benvanik): avoid self-references so we don't need this check.
        if (reachedTimepoint == possibleTimepoint)
          continue;
        ++coverageMap[reachedTimepoint];
      }
    }
  }
  // Any possibly required timepoint that has no coverage is a root (no refs)
  // and is required.
  SetVector<Value> requiredTimepoints;
  for (auto possibleTimepoint : possibleTimepoints) {
    auto it = coverageMap.find(possibleTimepoint);
    if (it == coverageMap.end() || it->second <= 0) {
      LLVM_DEBUG({
        llvm::dbgs() << "   ++ requiring uncovered ";
        possibleTimepoint.printAsOperand(llvm::dbgs(), analysis.getAsmState());
        llvm::dbgs() << " (root)\n";
      });
      requiredTimepoints.insert(possibleTimepoint);
    } else {
      LLVM_DEBUG({
        llvm::dbgs() << "   -- omitting covered ";
        possibleTimepoint.printAsOperand(llvm::dbgs(), analysis.getAsmState());
        llvm::dbgs() << "\n";
      });
    }
  }
  return requiredTimepoints;
}

//===----------------------------------------------------------------------===//
// Await hoisting/sinking for control flow
//===----------------------------------------------------------------------===//

// Returns true if |value| is used directly (not captured by stream ops) in
// |region|. Stream ops with await() clauses don't count as direct usage since
// the value is captured for async execution.
static bool isValueUsedDirectlyInRegion(Value value, Region &region) {
  for (Block &block : region) {
    for (Operation &op : block) {
      // Check if op uses value as operand (but not in stream op await clause).
      for (OpOperand &operand : op.getOpOperands()) {
        if (operand.get() == value) {
          // Check if value is in await list.
          if (auto timelineOp =
                  dyn_cast<IREE::Stream::TimelineOpInterface>(&op)) {
            bool isInAwaitList = false;
            for (Value awaitTimepoint : timelineOp.getAwaitTimepoints()) {
              if (awaitTimepoint == value) {
                isInAwaitList = true;
                break;
              }
            }
            if (isInAwaitList) {
              // Not a direct use, captured by await().
              continue;
            }
          }
          // Direct use found.
          return true;
        }
      }

      // Recursively check nested regions.
      for (Region &nestedRegion : op.getRegions()) {
        if (isValueUsedDirectlyInRegion(value, nestedRegion)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Identifies which regions of |controlFlowOp| use |value| directly.
static void getRegionsThatUseValue(Value value, Operation *controlFlowOp,
                                   SmallVectorImpl<Region *> &regions) {
  for (Region &region : controlFlowOp->getRegions()) {
    if (isValueUsedDirectlyInRegion(value, region)) {
      regions.push_back(&region);
    }
  }
}

// Sinks awaits into specific branches of control flow.
// Conservative: only optimizes ops with mutually exclusive execution paths.
static bool trySinkAwaitIntoBranch(IREE::Stream::TimepointAwaitOp awaitOp,
                                   Operation *controlFlowOp) {
  // Generic analysis: which regions use the awaited values?
  // We need at least one region to use it for sinking to make sense.
  SmallVector<Region *> regionsWithDirectUse;
  for (Value awaitedValue : awaitOp.getResults()) {
    getRegionsThatUseValue(awaitedValue, controlFlowOp, regionsWithDirectUse);
  }
  if (regionsWithDirectUse.empty()) {
    return false;
  }

  // Conservative: only handle ops with mutually exclusive branches.
  // TODO(benvanik): maybe peel an scf.if from an scf.for/while and make the
  //     wait conditional? That may mess up other analysis so we'd only want
  //     to do that if all other analysis failed. Better would be to try to
  //     rotate it into the loop.
  auto sinkIntoRegion = [&](Region *targetRegion) {
    // Clone await into target region.
    OpBuilder builder(&targetRegion->front(), targetRegion->front().begin());
    auto *newAwaitOp = builder.clone(*awaitOp);

    // Replace uses in target region with new await.
    for (auto [oldResult, newResult] :
         llvm::zip(awaitOp.getResults(), newAwaitOp->getResults())) {
      oldResult.replaceUsesWithIf(newResult, [&](OpOperand &use) {
        return targetRegion->isAncestor(use.getOwner()->getParentRegion());
      });
    }
  };
  return TypeSwitch<Operation *, bool>(controlFlowOp)
      .Case<scf::IfOp>([&](auto ifOp) {
        // scf.if has mutually exclusive branches - sink into all that need it.
        LLVM_DEBUG({
          llvm::dbgs() << "[ElideTimepoints] sinking await into scf.if ";
          bool first = true;
          for (Region *region : regionsWithDirectUse) {
            if (!first)
              llvm::dbgs() << " and ";
            if (region == &ifOp.getThenRegion()) {
              llvm::dbgs() << "then";
            } else {
              llvm::dbgs() << "else";
            }
            first = false;
          }
          llvm::dbgs() << " branch(es)\n";
        });
        for (Region *region : regionsWithDirectUse) {
          sinkIntoRegion(region);
        }
        // If no uses remain outside then erase the original op we're sinking.
        if (llvm::all_of(awaitOp.getResults(),
                         [](Value v) { return v.use_empty(); })) {
          awaitOp.erase();
        }
        return true;
      })
      .Case<scf::IndexSwitchOp>([&](auto switchOp) {
        // scf.index_switch has mutually exclusive case regions - sink into all
        // that need it.
        LLVM_DEBUG({
          llvm::dbgs()
              << "[ElideTimepoints] sinking await into scf.index_switch ";
          bool first = true;
          auto caseRegions = switchOp.getCaseRegions();
          for (Region *region : regionsWithDirectUse) {
            if (!first)
              llvm::dbgs() << ", ";
            // Find which case this is.
            bool foundCase = false;
            for (auto [idx, caseRegion] : llvm::enumerate(caseRegions)) {
              if (&caseRegion == region) {
                llvm::dbgs() << "case " << idx;
                foundCase = true;
                break;
              }
            }
            if (!foundCase && region == &switchOp.getDefaultRegion()) {
              llvm::dbgs() << "default";
            }
            first = false;
          }
          llvm::dbgs() << " region(s)\n";
        });
        for (Region *region : regionsWithDirectUse) {
          sinkIntoRegion(region);
        }
        // If no uses remain outside then erase the original op we're sinking.
        if (llvm::all_of(awaitOp.getResults(),
                         [](Value v) { return v.use_empty(); })) {
          awaitOp.erase();
        }
        return true;
      })
      .Default([](auto) { return false; });
}

// Hoists await past control flow when the value only used after.
// Safe for any control flow if no regions use the value as our timepoints are
// immutable and cannot be impacted by side-effects. I hope.
static bool tryHoistAwaitPastControlFlow(IREE::Stream::TimepointAwaitOp awaitOp,
                                         Operation *controlFlowOp) {
  // Check if awaited values used in any region of control flow.
  for (Value awaitedValue : awaitOp.getResults()) {
    for (Region &region : controlFlowOp->getRegions()) {
      if (isValueUsedDirectlyInRegion(awaitedValue, region)) {
        // Used in region, can't hoist.
        return false;
      }
    }

    // Also check if used by control flow op itself (condition, bounds, etc).
    for (OpOperand &operand : controlFlowOp->getOpOperands()) {
      if (operand.get() == awaitedValue) {
        return false;
      }
    }
  }

  // All uses are after control flow - safe to move.
  awaitOp->moveAfter(controlFlowOp);

  LLVM_DEBUG({
    llvm::dbgs() << "[ElideTimepoints] hoisted await past ";
    llvm::dbgs() << controlFlowOp->getName() << "\n";
  });

  return true;
}

// Folds await with timeline ops that consume the awaited values.
// Handles both proactive absorption and reactive cleanup:
// - Proactive: if timeline op doesn't cover the await timepoint absorb it by
//   adding to the timeline op's await list.
// - Reactive: if timeline op already covers the await timepoint just eliminate
//   the redundant await.
// Returns true if any folding occurred.
static bool tryFoldAwaitWithTimelineOps(IREE::Stream::TimepointAwaitOp awaitOp,
                                        TimepointCoverageAnalysis &analysis,
                                        DominanceInfo &domInfo) {
  // Walk all uses (which may be across CFG or call graph edges).
  // Note that we don't care if we can't find all of them - this is an
  // optimization and there will always be cases we miss.
  bool didChange = false;
  Value awaitTimepoint = awaitOp.getAwaitTimepoint();

  // Validate the await timepoint before using it.
  if (!awaitTimepoint) {
    LLVM_DEBUG({
      llvm::dbgs() << "[ElideTimepoints] skipping fold: null await timepoint\n";
    });
    return false;
  }

  // Collect modifications to apply after walking (don't modify IR during walk).
  // We store enough information to reconstruct what to do without holding
  // OpOperand* pointers which can become invalid.
  struct Modification {
    IREE::Stream::TimelineOpInterface timelineOp;
    Value oldValue;                // value to replace
    Value newValue;                // replacement value
    bool needsTimepointAbsorption; // true if we need to add awaitTimepoint
  };
  SmallVector<Modification> modifications;

  // Validate resource operands before accessing.
  auto resourceOperands = awaitOp.getResourceOperands();
  if (resourceOperands.size() != awaitOp->getNumResults()) {
    LLVM_DEBUG({
      llvm::dbgs() << "[ElideTimepoints] skipping fold: resource operands "
                      "mismatch with results\n";
    });
    return false;
  }

  for (auto result : awaitOp->getResults()) {
    Value asyncValue = resourceOperands[result.getResultNumber()];

    // Validate the async value before using it.
    if (!asyncValue) {
      LLVM_DEBUG({
        llvm::dbgs() << "[ElideTimepoints] skipping fold: null async value\n";
      });
      continue;
    }

    analysis.getExplorer().walkTransitiveUses(
        result,
        [&](OpOperand &operand) {
          if (auto timelineOp = dyn_cast<IREE::Stream::TimelineOpInterface>(
                  operand.getOwner())) {
            // Await result value is being consumed by a timeline op.

            // Only fold if the async value dominates the timeline op.
            // This prevents dominance violations when the async value is
            // defined inside a nested region (e.g., scf.if branch).
            if (!domInfo.properlyDominates(asyncValue, timelineOp)) {
              LLVM_DEBUG({
                llvm::dbgs() << "[ElideTimepoints] skipping fold: async value ";
                asyncValue.printAsOperand(llvm::dbgs(), analysis.getAsmState());
                llvm::dbgs() << " does not dominate timeline op\n";
              });
              return WalkResult::advance();
            }

            // Check if timeline op's awaits cover this await timepoint.
            bool isCovered = analysis.covers(awaitTimepoint,
                                             timelineOp.getAwaitTimepoints());

            if (!isCovered) {
              // PROACTIVE: Timeline op DOESN'T cover the await timepoint.
              // We'll absorb the await into the timeline op after the walk
              // completes.
              LLVM_DEBUG({
                llvm::dbgs()
                    << "[ElideTimepoints] will absorb await into timeline op "
                    << "adding timepoint ";
                awaitTimepoint.printAsOperand(llvm::dbgs(),
                                              analysis.getAsmState());
                llvm::dbgs() << "\n";
              });
              modifications.push_back(
                  {timelineOp, operand.get(), asyncValue, true});
              didChange = true;
            } else {
              // REACTIVE: Timeline op DOES cover the await timepoint.
              // The timeline op already awaits this timepoint (or something
              // that covers it), so we can just eliminate the redundant await.
              LLVM_DEBUG({
                llvm::dbgs() << "[ElideTimepoints] timeline op consumed sync "
                                "operand already covered by timepoint ";
                awaitTimepoint.printAsOperand(llvm::dbgs(),
                                              analysis.getAsmState());
                llvm::dbgs() << "\n";
              });
              modifications.push_back(
                  {timelineOp, operand.get(), asyncValue, false});
              didChange = true;
            }
          }
          return WalkResult::advance();
        },
        TraversalBehavior::DONT_WALK_TIED_VALUES);
  }

  // Phase 1: Update all timeline ops that need timepoint absorption.
  // Group by timeline op to avoid adding the same timepoint multiple times.
  llvm::DenseMap<Operation *, SmallVector<Value>> timelineAbsorptions;
  for (auto &mod : modifications) {
    if (mod.needsTimepointAbsorption) {
      timelineAbsorptions[mod.timelineOp].push_back(awaitTimepoint);
    }
  }

  // Apply timepoint absorptions.
  for (auto &[timelineOpPtr, timepoints] : timelineAbsorptions) {
    auto timelineOp = cast<IREE::Stream::TimelineOpInterface>(timelineOpPtr);
    OpBuilder builder(timelineOp);
    SmallVector<Value> newAwaitTimepoints(timelineOp.getAwaitTimepoints());

    // Add unique timepoints (avoid duplicates).
    for (Value timepoint : timepoints) {
      if (!llvm::is_contained(newAwaitTimepoints, timepoint)) {
        newAwaitTimepoints.push_back(timepoint);
      }
    }

    timelineOp.setAwaitTimepoints(newAwaitTimepoints, builder);
    LLVM_DEBUG({
      llvm::dbgs() << "[ElideTimepoints] absorbed await into timeline op ";
      timelineOp.print(llvm::dbgs(), analysis.getAsmState());
      llvm::dbgs() << "\n";
    });
  }

  // Phase 2: Replace all operand values with async values.
  // Use replaceUsesWithIf to safely update only the uses within the timeline
  // ops we're modifying.
  for (auto &mod : modifications) {
    mod.oldValue.replaceUsesWithIf(mod.newValue, [&](OpOperand &use) {
      if (use.getOwner() != mod.timelineOp.getOperation()) {
        return false;
      }
      // Verify that mod.newValue dominates mod.oldValue.
      // For block arguments, check if the block containing mod.newValue
      // dominates the block argument's owner block.
      // For regular values, check if mod.newValue dominates the defining op.
      if (auto blockArg = dyn_cast<BlockArgument>(mod.oldValue)) {
        // mod.oldValue is a block argument.
        Block *argBlock = blockArg.getOwner();

        // Check if this is a CF loop header (block has a back-edge to itself).
        // A back-edge exists if any predecessor is dominated by this block.
        // This indicates the block is part of a loop where the same block
        // argument receives different values on different iterations. Skip this
        // check for blocks inside SCF regions (they use
        // RegionBranchOpInterface).
        bool isInSCFRegion = false;
        Operation *parentOp = argBlock->getParentOp();
        while (parentOp) {
          if (isa<mlir::RegionBranchOpInterface>(parentOp)) {
            isInSCFRegion = true;
            break;
          }
          if (parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
            break; // Stop at function/module boundary.
          }
          parentOp = parentOp->getParentOp();
        }

        if (!isInSCFRegion) {
          for (Block *pred : argBlock->getPredecessors()) {
            if (pred == argBlock || domInfo.dominates(argBlock, pred)) {
              // Back-edge detected: either self-loop or pred is dominated by
              // argBlock. Block argument could receive different values on
              // different paths.
              return false;
            }
          }
        }

        // Get the block containing mod.newValue and check if it dominates.
        Operation *defOp = mod.newValue.getDefiningOp();
        if (!defOp) {
          // mod.newValue is itself a block argument, check block dominance.
          auto newBlockArg = cast<BlockArgument>(mod.newValue);
          return domInfo.properlyDominates(newBlockArg.getOwner(), argBlock);
        }
        // Check if the block containing the defining op dominates the block
        // argument's owner block.
        return domInfo.properlyDominates(defOp->getBlock(), argBlock);
      } else {
        // mod.oldValue is a regular SSA value. Check if mod.newValue dominates
        // its defining operation.
        return domInfo.properlyDominates(mod.newValue,
                                         mod.oldValue.getDefiningOp());
      }
    });
  }

  // Phase 3: Erase the await op if all its results are now unused.
  // This is safe now that all modifications are complete and we're not in a
  // walk anymore.
  if (didChange && llvm::all_of(awaitOp->getResults(), [](Value result) {
        return result.use_empty();
      })) {
    LLVM_DEBUG({
      llvm::dbgs() << "[ElideTimepoints] erasing completely folded await op\n";
    });
    awaitOp.erase();
  }

  return didChange;
}

// Tries to eliminate redundant awaits before region branch terminators when
// both the awaited resource and a covering timepoint are passed to the
// successor region. Uses RegionBranchOpInterface to handle all region branch
// ops generically (scf.if, scf.for, scf.while, etc.).
//
// Pattern: %r = await %tp => %res; <region terminator> %r, %tp
// Transform: <region terminator> %res, %tp (eliminates await)
//
// This is safe because:
// 1. The timepoint is passed to the successor, so consumers will wait on it
// 2. Coverage analysis confirms the timepoint covers the await
// 3. Dominance analysis ensures the original resource is available
//
// Returns true if any awaits were eliminated.
static bool
tryEliminateAwaitBeforeRegionBranchYield(IREE::Stream::TimepointAwaitOp awaitOp,
                                         TimepointCoverageAnalysis &analysis,
                                         DominanceInfo &domInfo) {
  bool didChange = false;

  // Collect await's timepoint and resources for checking.
  Value awaitTimepoint = awaitOp.getAwaitTimepoint();
  if (!awaitTimepoint) {
    return false;
  }
  auto awaitResources = awaitOp.getResourceOperands();
  if (awaitResources.empty()) {
    return false;
  }

  // Walk all transitive uses of the await results to find region branch
  // terminators. This handles indirect flows through arith.select, phi nodes,
  // etc., which getDefiningOp would miss.
  for (auto awaitResult : awaitOp->getResults()) {
    unsigned resultIdx = cast<OpResult>(awaitResult).getResultNumber();
    Value originalResource = awaitResources[resultIdx];

    // Track modifications to apply after walking (don't modify during walk).
    struct Modification {
      Operation *terminator;
      unsigned operandIdx;
      Value newValue;
    };
    SmallVector<Modification> modifications;
    analysis.getExplorer().walkTransitiveUses(
        awaitResult,
        [&](OpOperand &use) {
          Operation *user = use.getOwner();

          // Check if user is a terminator in a region branch op.
          if (!user->hasTrait<OpTrait::IsTerminator>()) {
            return WalkResult::advance();
          }
          auto parentOp =
              dyn_cast<RegionBranchOpInterface>(user->getParentOp());
          if (!parentOp) {
            return WalkResult::advance();
          }

          // Query where this terminator's operands flow.
          // We need to cast the terminator to RegionBranchTerminatorOpInterface
          // to construct a proper RegionBranchPoint.
          auto terminatorInterface =
              dyn_cast<RegionBranchTerminatorOpInterface>(user);
          if (!terminatorInterface) {
            // Terminator doesn't implement the interface, skip it.
            return WalkResult::advance();
          }

          SmallVector<RegionSuccessor> successors;
          parentOp.getSuccessorRegions(RegionBranchPoint(terminatorInterface),
                                       successors);
          for (auto successor : successors) {
            // Only handle yields to parent (region -> parent op results).
            // Yields to other regions would require different analysis.
            if (!successor.isParent()) {
              continue;
            }

            // Get operands passed to this successor.
            // For scf.condition, exclude the condition boolean from operands.
            OperandRange successorOperands = user->getOperands();
            if (auto condOp = dyn_cast<scf::ConditionOp>(user)) {
              successorOperands = condOp.getArgs();
            }

            // Collect all timepoint operands passed to the successor.
            SmallVector<Value> yieldedTimepoints;
            for (Value operand : successorOperands) {
              if (isa<IREE::Stream::TimepointType>(operand.getType())) {
                yieldedTimepoints.push_back(operand);
              }
            }
            if (yieldedTimepoints.empty()) {
              // No timepoints yielded - can't eliminate await.
              continue;
            }

            // Check if the yielded timepoints cover the awaited timepoint.
            // This handles both exact matches and transitive coverage
            // (e.g., yielding a joined timepoint that covers this one).
            if (!analysis.covers(awaitTimepoint, yieldedTimepoints)) {
              LLVM_DEBUG({
                llvm::dbgs()
                    << "[ElideTimepoints] cannot eliminate await "
                       "before region branch yield: yielded timepoints "
                       "don't cover awaited timepoint\n";
              });
              continue;
            }

            // Verify the original resource dominates the terminator.
            // This ensures it's safe to use the original instead of awaited.
            if (!domInfo.properlyDominates(originalResource, user)) {
              LLVM_DEBUG({
                llvm::dbgs() << "[ElideTimepoints] cannot eliminate await "
                                "before region branch yield: original resource "
                                "doesn't dominate terminator\n";
              });
              continue;
            }

            // Safe to eliminate - record the modification.
            LLVM_DEBUG({
              llvm::dbgs() << "[ElideTimepoints] eliminating redundant await "
                              "before region branch yield - timepoint ";
              awaitTimepoint.printAsOperand(llvm::dbgs(),
                                            analysis.getAsmState());
              llvm::dbgs() << " is covered by yielded timepoints\n";
            });

            modifications.push_back({
                user,
                use.getOperandNumber(),
                originalResource,
            });
            didChange = true;
          }

          return WalkResult::advance();
        },
        TraversalBehavior::DONT_WALK_TIED_VALUES);

    // Apply modifications.
    for (auto &mod : modifications) {
      mod.terminator->setOperand(mod.operandIdx, mod.newValue);
    }
  }

  return didChange;
}

// Tries to elide timepoints nested within |region| when safe.
// Returns true if any ops were elided.
static bool tryElideTimepointsInRegion(Region &region,
                                       TimepointCoverageAnalysis &analysis,
                                       DominanceInfo &domInfo) {
  bool didChange = false;

  // We batch up all results we're going to change to prevent SSA value
  // breakages in the debug print out. This maps old->new values.
  DenseMap<Value, Value> pendingReplacements;

  // Inserts an immediate timepoint or reuses an existing replacement (if
  // any).
  auto makeImmediate = [&](Value elidedTimepoint, OpBuilder builder) -> Value {
    auto existingReplacement = pendingReplacements.find(elidedTimepoint);
    if (existingReplacement != pendingReplacements.end()) {
      return existingReplacement->second;
    }
    return IREE::Stream::TimepointImmediateOp::create(builder,
                                                      elidedTimepoint.getLoc());
  };

  // Elides |elidedTimepoint| by replacing all its uses by |op| with an
  // immediate timepoint value.
  auto elideTimepointOperand = [&](Operation *op, Value elidedTimepoint) {
    if (isDefinedImmediate(elidedTimepoint))
      return; // already immediate
    auto immediateTimepoint = makeImmediate(elidedTimepoint, OpBuilder(op));
    elidedTimepoint.replaceUsesWithIf(
        immediateTimepoint,
        [&](OpOperand &operand) { return operand.getOwner() == op; });
    didChange = true;
  };

  // Elides all timepoint operands of |op| that are immediately resolved.
  auto elideTimepointOperands = [&](Operation *op) {
    for (auto operand : llvm::make_early_inc_range(op->getOperands())) {
      if (!isa<IREE::Stream::TimepointType>(operand.getType()))
        continue;
      if (isDefinedImmediate(operand))
        continue;
      if (analysis.isImmediate(operand)) {
        LLVM_DEBUG({
          llvm::dbgs() << "  >>> eliding known-immediate operand ";
          operand.printAsOperand(llvm::dbgs(), analysis.getAsmState());
          llvm::dbgs() << " consumed by " << op->getName() << "\n";
        });
        elideTimepointOperand(op, operand);
      }
    }
  };

  // Elides |elidedTimepoint| by replacing all its uses with an immediate
  // timepoint value. The original value will end up with zero uses.
  auto elideTimepointResult = [&](Operation *op, Value elidedTimepoint) {
    if (elidedTimepoint.use_empty())
      return; // no-op
    if (isDefinedImmediate(elidedTimepoint))
      return; // already immediate
    OpBuilder afterBuilder(op);
    afterBuilder.setInsertionPointAfterValue(elidedTimepoint);
    Value immediateTimepoint = IREE::Stream::TimepointImmediateOp::create(
        afterBuilder, elidedTimepoint.getLoc());
    // Defer actually swapping until later.
    pendingReplacements.insert(
        std::make_pair(elidedTimepoint, immediateTimepoint));
    didChange = true;
  };

  // Elides all timepoint results of |op| that are immediately resolved.
  auto elideTimepointResults = [&](Operation *op) {
    // Reverse so that we insert in return order:
    //  %0, %1 = ...
    //  %imm0 = immediate
    //  %imm1 = immediate
    for (auto result : llvm::reverse(op->getResults())) {
      if (!isa<IREE::Stream::TimepointType>(result.getType()))
        continue;
      if (isDefinedImmediate(result))
        continue;
      if (analysis.isImmediate(result)) {
        LLVM_DEBUG({
          llvm::dbgs() << "  >>> eliding known-immediate result ";
          result.printAsOperand(llvm::dbgs(), analysis.getAsmState());
          llvm::dbgs() << " produced by " << op->getName() << " (result "
                       << result.getResultNumber() << ")\n";
        });
        elideTimepointResult(op, result);
      }
    }
  };

  // Processes timeline |op| by eliding its await and result timepoints if
  // possible.
  auto processTimelineOp = [&](IREE::Stream::TimelineOpInterface op) {
    auto resultTimepoint = op.getResultTimepoint();
    auto awaitTimepoints = op.getAwaitTimepoints();
    if (awaitTimepoints.empty())
      return;

    LLVM_DEBUG({
      llvm::dbgs() << "[ElideTimepoints] pruning " << op->getName()
                   << " await(";
      llvm::interleaveComma(awaitTimepoints, llvm::dbgs(), [&](Value value) {
        value.printAsOperand(llvm::dbgs(), analysis.getAsmState());
      });
      llvm::dbgs() << ")";
      if (resultTimepoint) {
        llvm::dbgs() << " producing ";
        resultTimepoint.printAsOperand(llvm::dbgs(), analysis.getAsmState());
      }
      llvm::dbgs() << "\n";
    });

    // If the result of the op is immediate then we can elide the resulting
    // timepoint.
    if (resultTimepoint && analysis.isImmediate(resultTimepoint)) {
      LLVM_DEBUG({
        llvm::dbgs() << "  >>> eliding entire known-immediate result ";
        resultTimepoint.printAsOperand(llvm::dbgs(), analysis.getAsmState());
        llvm::dbgs() << " produced by " << op->getName() << "\n";
      });
      elideTimepointResult(op, resultTimepoint);
    }

    // Prune all immediately reached timepoints.
    // This may let us avoid doing the full pruning pass by getting us down to
    // 0 or 1 timepoints.
    SmallVector<Value> possibleTimepoints;
    for (auto awaitTimepoint : awaitTimepoints) {
      if (analysis.isImmediate(awaitTimepoint)) {
        // Timepoint is definitely immediate and can be pruned.
        LLVM_DEBUG({
          llvm::dbgs() << "  >>> eliding use of known-immediate ";
          awaitTimepoint.printAsOperand(llvm::dbgs(), analysis.getAsmState());
          llvm::dbgs() << " in " << op->getName() << "\n";
        });
        elideTimepointOperand(op, awaitTimepoint);
      } else {
        // May be immediate but not certain; preserve.
        possibleTimepoints.push_back(awaitTimepoint);
      }
    }

    // If there's only one timepoint we don't have to worry with coverage.
    if (possibleTimepoints.size() <= 1)
      return;

    // Perform the analysis on the possible timepoints to find which are covered
    // by others and elide all of those known-covered.
    auto requiredTimepoints =
        buildRequiredCoverageSet(possibleTimepoints, analysis);
    for (auto possibleTimepoint : possibleTimepoints) {
      if (!requiredTimepoints.contains(possibleTimepoint)) {
        // Timepoint is covered (or immediate) and can be pruned.
        LLVM_DEBUG({
          llvm::dbgs() << "  >>> eliding use of covered ";
          possibleTimepoint.printAsOperand(llvm::dbgs(),
                                           analysis.getAsmState());
          llvm::dbgs() << "(" << (void *)possibleTimepoint.getImpl() << ")\n";
        });
        elideTimepointOperand(op, possibleTimepoint);
      }
    }
  };

  // Walk all blocks and elide timepoints.
  // We walk pre-order to make the debug output easier to read.
  region.walk<WalkOrder::PreOrder>([&](Operation *op) {
    // TODO(benvanik): handle more ops from scf or other dialects.
    TypeSwitch<Operation *>(op)
        .Case([&](IREE::Stream::TimelineOpInterface op) {
          // Most of the interesting stream.* stuff happens here.
          processTimelineOp(op);
        })
        .Case<scf::IfOp, scf::ForOp, IREE::Util::GlobalLoadOp>(
            [&](Operation *op) { elideTimepointResults(op); })
        .Case<CallOpInterface, arith::SelectOp>([&](Operation *op) {
          elideTimepointOperands(op);
          elideTimepointResults(op);
        })
        .Case<cf::BranchOp, cf::CondBranchOp, IREE::Util::ReturnOp>(
            [&](Operation *op) { elideTimepointOperands(op); });
  });

  // Apply await movement optimizations around control flow.
  // Iterate until no more structural changes occur (sinking/hoisting).
  // Folding is not iterated because it doesn't change await placement.
  bool awaitMovementChanged = false;
  do {
    awaitMovementChanged = false;
    // Walk awaits and try to optimize their placement.
    region.walk([&](IREE::Stream::TimepointAwaitOp awaitOp) {
      // Try moving await around control flow.
      Operation *nextOp = awaitOp->getNextNode();
      if (!nextOp || nextOp->getNumRegions() == 0) {
        // Try folding if no control flow after.
        if (tryFoldAwaitWithTimelineOps(awaitOp, analysis, domInfo)) {
          didChange = true;
        }
        return WalkResult::advance();
      }

      // Try sinking into specific branch.
      if (trySinkAwaitIntoBranch(awaitOp, nextOp)) {
        didChange = true;
        awaitMovementChanged = true;
        return WalkResult::skip();
      }

      // Try hoisting past control flow.
      if (tryHoistAwaitPastControlFlow(awaitOp, nextOp)) {
        didChange = true;
        awaitMovementChanged = true;
      }

      return WalkResult::advance();
    });
  } while (awaitMovementChanged);

  // After all structural movement is complete, do a final folding pass.
  // This catches awaits that may now be adjacent to timeline ops after sinking.
  region.walk([&](IREE::Stream::TimepointAwaitOp awaitOp) {
    if (tryFoldAwaitWithTimelineOps(awaitOp, analysis, domInfo)) {
      didChange = true;
    }
  });

  // Eliminate redundant awaits before region branch yields.
  // This uses coverage analysis to determine if yielded timepoints cover the
  // awaits, handling all RegionBranchOpInterface ops generically.
  region.walk([&](IREE::Stream::TimepointAwaitOp awaitOp) {
    if (tryEliminateAwaitBeforeRegionBranchYield(awaitOp, analysis, domInfo)) {
      didChange = true;
    }
  });

  // Process elided results; we do this afterward to keep the debug output
  // cleaner by not adding <<UNKNOWN VALUES>>.
  for (auto replacement : pendingReplacements) {
    replacement.first.replaceAllUsesWith(replacement.second);
  }

  return didChange;
}

//===----------------------------------------------------------------------===//
// --iree-stream-elide-timepoints
//===----------------------------------------------------------------------===//

struct ElideTimepointsPass
    : public IREE::Stream::impl::ElideTimepointsPassBase<ElideTimepointsPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    // Perform whole-program analysis to find for each timepoint what other
    // timepoints are known to be reached.
    TimepointCoverageAnalysis analysis(moduleOp);
    if (failed(analysis.run())) {
      moduleOp.emitError() << "failed to solve for timepoint coverage";
      return signalPassFailure();
    }

    bool didChange = false;

    // Apply analysis by replacing known-covered timepoint usage with immediate
    // values. If we change something we'll indicate that so that the parent
    // fixed-point iteration continues.
    for (auto callableOp : analysis.getTopLevelOps()) {
      auto *region = callableOp.getCallableRegion();
      if (!region || region->empty()) {
        continue;
      }

      // Compute dominance info once for the entire function and reuse across
      // all await folding within. This is safe because we never modify the CFG
      // structure (blocks/branches), only operation operands, uses, and
      // non-CFG operations.
      DominanceInfo domInfo(callableOp);

      didChange =
          tryElideTimepointsInRegion(*region, analysis, domInfo) || didChange;
    }

    if (didChange)
      signalFixedPointModified(moduleOp);
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
