// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-elide-async-copies"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// Resource usage query/application patterns
//===----------------------------------------------------------------------===//

// TODO(benvanik): change this to just be an AbstractState - there's no real
// need for PVS as we don't track dynamically and are just using this as a
// cache.
class LastUsers
    : public DFX::StateWrapper<DFX::PotentialValuesState<Operation *>,
                               DFX::ValueElement> {
 public:
  using BaseType = DFX::StateWrapper<DFX::PotentialValuesState<Operation *>,
                                     DFX::ValueElement>;

  static LastUsers &createForPosition(const Position &pos,
                                      DFX::Solver &solver) {
    return *(new (solver.getAllocator()) LastUsers(pos));
  }

  const std::string getName() const override { return "LastUsers"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  // Returns true if the given |op| is known to be a last user of the value.
  // Note that a single op may use a value multiple times.
  bool isAssumedLastUser(Operation *op) const {
    return getAssumedSet().contains(op);
  }

  const std::string getAsStr(AsmState &asmState) const override {
    return std::string("last users: ") + std::to_string(getAssumedSet().size());
  }

 private:
  explicit LastUsers(const Position &pos) : BaseType(pos) {}

  void initializeValue(Value value, DFX::Solver &solver) override {
    // NOTE: this is only for the local region; we don't touch transitive users.
    // TODO(benvanik): touch transitive users? We could evaluate with
    //     solver.getExplorer().walkTransitiveUsers() and ensure all tied uses
    //     go out of scope at the right time. For now we assume that the SSA
    //     value last users are all we care about.
    auto parentOp =
        value.getParentRegion()->getParentOfType<mlir::CallableOpInterface>();
    auto liveness = solver.getExplorer()
                        .getAnalysisManager()
                        .nest(parentOp)
                        .getAnalysis<Liveness>();
    for (auto user : value.getUsers()) {
      if (liveness.isDeadAfter(value, user)) {
        unionAssumed(user);
      }
    }
    indicateOptimisticFixpoint();

    LLVM_DEBUG({
      AsmState asmState(value.getParentBlock()->getParentOp());
      llvm::dbgs() << "[elide-copies] initialized value last users for ";
      value.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << ": " << getAssumedSet().size() << "\n";
      for (auto user : getAssumedSet()) {
        llvm::dbgs() << "  ";
        user->print(llvm::dbgs(), OpPrintingFlags().elideLargeElementsAttrs());
        llvm::dbgs() << "\n";
      }
    });
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    // NOTE: this is purely a cache and is based only on the initial value;
    // this should never be called.
    return ChangeStatus::UNCHANGED;
  }

  friend class DFX::Solver;
};
const char LastUsers::ID = 0;

class ArgumentSemantics
    : public DFX::StateWrapper<DFX::BitIntegerState<uint8_t, 3, 0>,
                               DFX::ValueElement> {
 public:
  using BaseType =
      DFX::StateWrapper<DFX::BitIntegerState<uint8_t, 3, 0>, DFX::ValueElement>;

  // Inverted bits so that we can go from best (all bits set) to worst (no bits
  // set).
  enum {
    // Argument is _not_ mutated within the region it is used.
    NOT_MUTATED = 1u << 0,
    // Argument is _not_ by reference (so: by value). Indicates that the
    // argument is not retained at any predecessor/caller and is owned by the
    // receiver.
    NOT_BY_REFERENCE = 1u << 1,

    BEST_STATE = NOT_MUTATED | NOT_BY_REFERENCE,
  };
  static_assert(BEST_STATE == BaseType::getBestState(),
                "unexpected BEST_STATE value");

  static ArgumentSemantics &createForPosition(const Position &pos,
                                              DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ArgumentSemantics(pos));
  }

  const std::string getName() const override { return "ArgumentSemantics"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  // Returns true if the argument is known to be passed by-value from all
  // predecessors/callers.
  bool getKnownByValue() const {
    return (this->getKnown() & NOT_BY_REFERENCE) == NOT_BY_REFERENCE;
  }

  // Returns true if the argument is assumed to be passed by-value from all
  // predecessors/callers.
  bool getAssumedByValue() const {
    return (this->getAssumed() & NOT_BY_REFERENCE) == NOT_BY_REFERENCE;
  }

  const std::string getAsStr(AsmState &asmState) const override {
    std::string str;
    auto append = [&](const char *part) {
      if (!str.empty()) str += '|';
      str += part;
    };
    append(this->isAssumed(NOT_MUTATED) ? "immutable" : "mutable");
    append(this->isAssumed(NOT_BY_REFERENCE) ? "by-value" : "by-reference");
    return str.empty() ? "*" : str;
  }

 private:
  explicit ArgumentSemantics(const Position &pos) : BaseType(pos) {}

  // Returns true if |operand| is tied to a result on its owner indicating an
  // in-place operation.
  static bool isTiedUse(OpOperand &operand) {
    if (auto tiedOp =
            dyn_cast<IREE::Util::TiedOpInterface>(operand.getOwner())) {
      if (tiedOp.isOperandTied(operand.getOperandNumber())) return true;
    }
    return false;
  }

  // Starts analysis of the |value| with known bits based on IR structure.
  void initializeValue(Value value, DFX::Solver &solver) override {
    // Start as NOT_MUTATED and NOT_BY_REFERENCE (by-value).
    intersectAssumedBits(BEST_STATE);

    // If any use is tied then we know we are mutated in-place.
    // Note that this walks into call targets and across branches.
    auto traversalResult = solver.getExplorer().walkTransitiveUses(
        value, [&](OpOperand &operand) -> WalkResult {
          if (isTiedUse(operand)) {
            // Mutated in-place; nothing more we need to do.
            removeKnownBits(NOT_MUTATED);
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (traversalResult == TraversalResult::INCOMPLETE) {
      // Analysis incomplete - mark as conservatively by reference/mutated.
      removeKnownBits(NOT_MUTATED | NOT_BY_REFERENCE);
    }
  }

  // Updates the element state based on _a_ predecessor operand that is the
  // source of the argument value. Will be called once per predecessors/caller.
  void updateFromPredecessorUse(OpOperand &operand, DFX::Solver &solver) {
    // If the operand is a block argument then we need to ask for the argument
    // semantics first - if it's by reference then it's definitely not the last
    // use and we can short-circuit this.
    if (auto arg = operand.get().dyn_cast<BlockArgument>()) {
      auto &argumentSemantics = solver.getElementFor<ArgumentSemantics>(
          *this, Position::forValue(operand.get()), DFX::Resolution::REQUIRED);
      LLVM_DEBUG(llvm::dbgs()
                 << "  pred is arg; combining state: "
                 << argumentSemantics.getAsStr(solver.getAsmState()) << "\n");
      getState() ^= argumentSemantics.getState();
    }

    auto &lastUsers = solver.getElementFor<LastUsers>(
        *this, Position::forValue(operand.get()), DFX::Resolution::REQUIRED);
    bool isLastUser = lastUsers.isAssumedLastUser(operand.getOwner());
    if (!isLastUser) {
      // Not the last user - value is passed in by reference.
      LLVM_DEBUG(llvm::dbgs() << "  not the last user\n");
      removeAssumedBits(NOT_BY_REFERENCE | NOT_MUTATED);
    }
  }

  // Updates the semantics of |value| by walking all predecessors/callers (up
  // through function arguments, branch arguments, and tied results) and all
  // transitive uses (down through function calls, branches, and tied operands)
  // by way of usage analysis.
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    auto assumedBits = getAssumed();
    auto traversalResult = TraversalResult::COMPLETE;

    auto arg = value.cast<BlockArgument>();
    bool isEntryArg = arg.getParentBlock()->isEntryBlock();
    if (isEntryArg) {
      // Call argument.
      auto callableOp =
          cast<mlir::CallableOpInterface>(arg.getParentBlock()->getParentOp());
      traversalResult |= solver.getExplorer().walkIncomingCalls(
          callableOp, [&](mlir::CallOpInterface callOp) -> WalkResult {
            unsigned baseIdx = callOp.getArgOperands().getBeginOperandIndex();
            auto &sourceOperand =
                callOp->getOpOperand(baseIdx + arg.getArgNumber());
            updateFromPredecessorUse(sourceOperand, solver);
            return WalkResult::advance();
          });
    } else {
      // Branch argument.
      traversalResult |= solver.getExplorer().walkIncomingBranchOperands(
          arg.getParentBlock(),
          [&](Block *sourceBlock, OperandRange operands) -> WalkResult {
            unsigned baseIdx = operands.getBeginOperandIndex();
            auto &sourceOperand = sourceBlock->getTerminator()->getOpOperand(
                baseIdx + arg.getArgNumber());
            updateFromPredecessorUse(sourceOperand, solver);
            return WalkResult::advance();
          });
    }

    if (traversalResult == TraversalResult::INCOMPLETE) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "  !! traversal result incomplete; assuming by reference\n");
      removeAssumedBits(NOT_BY_REFERENCE | NOT_MUTATED);
    }
    return assumedBits == getAssumed() ? ChangeStatus::UNCHANGED
                                       : ChangeStatus::CHANGED;
  }

  friend class DFX::Solver;
};
const char ArgumentSemantics::ID = 0;

// TODO(benvanik): change into something we can use for ref counting. We need
// that to insert stream-ordered deallocs and know when timepoints have been
// discard as they go out of scope. For now this strictly checks last use.
class LastUseAnalysis {
 public:
  explicit LastUseAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    explorer.setOpAction<IREE::Util::InitializerOp>(TraversalAction::RECURSE);
    explorer.setOpAction<mlir::func::FuncOp>(TraversalAction::RECURSE);
    explorer.setDialectAction<IREE::Stream::StreamDialect>(
        TraversalAction::RECURSE);
    // Ignore the contents of executables (linalg goo, etc).
    explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::IGNORE);
    explorer.initialize();

    assert(rootOp->getNumRegions() == 1 && "expected module-like root op");
    topLevelOps = llvm::to_vector<4>(
        rootOp->getRegions().front().getOps<mlir::CallableOpInterface>());
  }

  // Runs analysis and populates the state cache.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run() {
    // Seed all block arguments throughout the program.
    for (auto callableOp : getTopLevelOps()) {
      auto *region = callableOp.getCallableRegion();
      if (!region) continue;
      for (auto &block : *region) {
        for (auto arg : block.getArguments()) {
          if (arg.getType().isa<IREE::Stream::ResourceType>()) {
            solver.getOrCreateElementFor<ArgumentSemantics>(
                Position::forValue(arg));
          }
        }
      }
    }

    // Run solver to completion.
    return solver.run();
  }

  // Returns a list of all top-level callable ops in the root op.
  ArrayRef<mlir::CallableOpInterface> getTopLevelOps() const {
    return topLevelOps;
  }

  // Returns true if block argument |arg| is passed in by-value/move (it's the
  // last use from all callers/predecessor branches). When false the value
  // represented by the argument may have other uses outside of its block.
  bool isArgMoved(BlockArgument arg) {
    auto argumentSemantics =
        solver.lookupElementFor<ArgumentSemantics>(Position::forValue(arg));
    if (!argumentSemantics) return false;
    return argumentSemantics->getAssumedByValue();
  }

  // Returns true if |userOp| is the last user of |operand|.
  bool isLastUser(Value operand, Operation *userOp) {
    auto lastUsers =
        solver.getOrCreateElementFor<LastUsers>(Position::forValue(operand));
    return lastUsers.isAssumedLastUser(userOp);
  }

 private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
  SmallVector<mlir::CallableOpInterface> topLevelOps;
};

// Returns true if the given |operand| value does not need a copy on write.
// This is a conservative check and will return false ("not safe to elide") in
// many cases that otherwise don't need a copy. The
// --iree-stream-elide-async-copies pass will do a whole-program analysis and
// remove the copies we insert here when possible.
//
// No-op clone is elidable:
//   %0 ---> %1 = clone(%0) ---> use(%1)  // last use of %0
//
// Clone required for correctness:
//   %0 ---> %1 = clone(%0) ---> use(%1)
//      \--> use(%0)
//
// Second clone elidable, first required:
//   %0 ---> %1 = clone(%0) ---> use(%1)
//      \--> %2 = clone(%0) ---> use(%2)  // last use of %0
static bool isSafeToElideCloneOp(IREE::Stream::AsyncCloneOp cloneOp,
                                 LastUseAnalysis &analysis) {
  LLVM_DEBUG({
    llvm::dbgs() << "isSafeToElideCloneOp:\n";
    llvm::dbgs() << "  ";
    cloneOp.print(llvm::dbgs(),
                  OpPrintingFlags().elideLargeElementsAttrs().assumeVerified());
    llvm::dbgs() << "\n";
  });

  // If this clone is performing a type change we need to preserve it.
  // TODO(benvanik): remove this carveout - could make clone not change type
  // and transfer be needed instead.
  auto sourceType =
      cloneOp.getSource().getType().cast<IREE::Stream::ResourceType>();
  auto targetType =
      cloneOp.getResult().getType().cast<IREE::Stream::ResourceType>();
  if (sourceType != targetType &&
      sourceType.getLifetime() == IREE::Stream::Lifetime::Constant) {
    LLVM_DEBUG(llvm::dbgs()
               << "  + clone source is a constant; cannot elide\n");
    return false;
  }

  // If the source is a block argument we have to look into the analysis cache
  // to see if it's been classified as a last use/by-value move. If it isn't
  // then we cannot mutate it in-place as it could be used by the caller/another
  // branch and we need to respect the forking of the value.
  if (auto arg = cloneOp.getSource().dyn_cast<BlockArgument>()) {
    if (!analysis.isArgMoved(arg)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  - clone source is a by-ref arg; cannot elide\n");
      return false;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  ? clone source is a by-value arg; may elide\n");
  }

  // If there's only one user of the source we know it's this clone and can
  // bypass all the more expensive liveness analysis.
  if (cloneOp.getSource().hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs()
               << "  + clone source SSA value has one use; can elide\n");
    return true;
  }

  // If this is the last user of the source SSA value then we can elide the
  // clone knowing that any mutations won't impact the source.
  if (analysis.isLastUser(cloneOp.getSource(), cloneOp)) {
    LLVM_DEBUG(llvm::dbgs() << "  + clone source use is the last; can elide\n");
    return true;
  }

  // Not safe.
  LLVM_DEBUG(llvm::dbgs() << "  - clone source cannot be elided\n");
  return false;
}

// Tries to elide |cloneOp| by replacing all uses with its source if safe.
// Returns true if the op was elided.
static bool tryElideCloneOp(IREE::Stream::AsyncCloneOp cloneOp,
                            LastUseAnalysis &analysis) {
  if (!isSafeToElideCloneOp(cloneOp, analysis)) return false;
  cloneOp.replaceAllUsesWith(cloneOp.getSource());
  cloneOp.erase();
  return true;
}

// Tries to elide copies nested within |region| when safe.
// Returns true if any ops were elided.
static bool tryElideAsyncCopiesInRegion(Region &region,
                                        LastUseAnalysis &analysis) {
  bool didChange = false;
  for (auto &block : region) {
    for (auto cloneOp : llvm::make_early_inc_range(
             block.getOps<IREE::Stream::AsyncCloneOp>())) {
      if (!isSafeToElideCloneOp(cloneOp, analysis)) continue;
      cloneOp.replaceAllUsesWith(cloneOp.getSource());
      cloneOp.erase();
      didChange = true;
    }
  }
  return didChange;
}

//===----------------------------------------------------------------------===//
// -iree-stream-elide-async-copies
//===----------------------------------------------------------------------===//

// Elides async copies that perform no meaningful work - such as clones of the
// last use of a value. This is designed to be run after
// --iree-stream-materialize-copy-on-write to clean up the copies it introduces
// but will also pick up any copies that came from the frontend.
//
// This should never remove copies that are required for correctness: we err on
// the side of leaving copies when we cannot perform full analysis.
//
// This operates using a whole-program data flow analysis to first determine
// which block arguments have move semantics (they are passed the last use of
// a resource) and the last users of all cloned values. Once analyzed all copies
// in the program are checked to see if they can be safely removed and if so are
// rerouted to the cloned source value. This process repeats until no more
// copies are elided: we are guaranteed to reach a fixed point as we are only
// removing copies in this pass and not introducing any new ops.
class ElideAsyncCopiesPass : public ElideAsyncCopiesBase<ElideAsyncCopiesPass> {
 public:
  ElideAsyncCopiesPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) return;

    // Try analyzing the program and eliding the unneeded copies until we reach
    // a fixed point (no more copies can be elided).
    unsigned maxIterationCount = 30;
    unsigned iterationCount = 0;
    for (; iterationCount < maxIterationCount; ++iterationCount) {
      // Perform whole-program analysis.
      // TODO(benvanik): reuse allocator across iterations.
      LastUseAnalysis analysis(moduleOp);
      if (failed(analysis.run())) {
        moduleOp.emitError() << "failed to solve for last users";
        return signalPassFailure();
      }

      // Apply analysis by eliding all copies that are safe to elide.
      // If we can't elide any we'll consider the iteration complete and exit.
      bool didChange = false;
      for (auto callableOp : analysis.getTopLevelOps()) {
        auto *region = callableOp.getCallableRegion();
        if (!region) continue;
        didChange = tryElideAsyncCopiesInRegion(*region, analysis) || didChange;
      }
      if (!didChange) break;
    }
    if (iterationCount == maxIterationCount) {
      // If you find yourself hitting this we can evaluate increasing the
      // iteration count (if it would eventually converge) or whether we allow
      // this to happen without remarking. For now all our programs coverge in
      // just one or two iterations and this needs to be tuned with more complex
      // control flow.
      moduleOp.emitRemark()
          << "copy elision pass failed to reach a fixed point after "
          << maxIterationCount << " iterations; unneeded copies may be present";
      return;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createElideAsyncCopiesPass() {
  return std::make_unique<ElideAsyncCopiesPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
