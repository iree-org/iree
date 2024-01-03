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
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(value)) {
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
            LLVM_DEBUG({
              llvm::dbgs() << "[ElideTimepoints] produced timeline result ";
              resultTimepoint.printAsOperand(llvm::dbgs(),
                                             solver.getAsmState());
              llvm::dbgs() << "\n";
            });
          }
        })
        .Case([&](mlir::CallOpInterface callOp) {
          // Step into callees and get a coverage intersection of all return
          // sites.
          auto callableOp =
              callOp.resolveCallable(&solver.getExplorer().getSymbolTables());
          unsigned resultIndex = llvm::cast<OpResult>(value).getResultNumber();
          gatherRegionReturns(callableOp, resultIndex);
        })
        .Case([&](RegionBranchOpInterface regionOp) {
          // Step into regions and get a coverage intersection of all return
          // sites.
          unsigned resultIndex = llvm::cast<OpResult>(value).getResultNumber();
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

class TimepointCoverageAnalysis {
public:
  explicit TimepointCoverageAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    explorer.setOpAction<IREE::Util::InitializerOp>(TraversalAction::RECURSE);
    explorer.setOpAction<mlir::func::FuncOp>(TraversalAction::RECURSE);
    explorer.setOpAction<mlir::scf::IfOp>(TraversalAction::RECURSE);
    explorer.setDialectAction<IREE::Stream::StreamDialect>(
        TraversalAction::RECURSE);
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
          if (llvm::isa<IREE::Stream::TimepointType>(arg.getType())) {
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

        // Seed all terminator operands.
        if (auto *terminatorOp = block.getTerminator()) {
          for (auto operand : terminatorOp->getOperands()) {
            if (llvm::isa<IREE::Stream::TimepointType>(operand.getType())) {
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

  // Union all transitively reached timepoints by the time |value| is reached.
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

// Tries to elide timepoints nested within |region| when safe.
// Returns true if any ops were elided.
static bool tryElideTimepointsInRegion(Region &region,
                                       TimepointCoverageAnalysis &analysis) {
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
    return builder.create<IREE::Stream::TimepointImmediateOp>(
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
      if (!llvm::isa<IREE::Stream::TimepointType>(operand.getType()))
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
    Value immediateTimepoint =
        afterBuilder.create<IREE::Stream::TimepointImmediateOp>(
            elidedTimepoint.getLoc());
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
      if (!llvm::isa<IREE::Stream::TimepointType>(result.getType()))
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
        .Case<cf::BranchOp, cf::CondBranchOp>(
            [&](Operation *op) { elideTimepointOperands(op); })
        .Case<func::ReturnOp, scf::YieldOp>(
            [&](Operation *op) { elideTimepointOperands(op); });
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
    auto moduleOp = getOperation();
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
      if (!region || region->empty())
        continue;
      didChange = tryElideTimepointsInRegion(*region, analysis) || didChange;
    }

    if (didChange)
      signalFixedPointModified(moduleOp);
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
