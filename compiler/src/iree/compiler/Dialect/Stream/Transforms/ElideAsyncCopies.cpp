// Copyright 2021 The IREE Authors
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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-elide-async-copies"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ELIDEASYNCCOPIESPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

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
      llvm::dbgs() << "[elide-copies] initialized value last users for ";
      value.printAsOperand(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << ": " << getAssumedSet().size() << "\n";
      for (auto user : getAssumedSet()) {
        llvm::dbgs() << "  ";
        user->print(llvm::dbgs(), solver.getAsmState());
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
      if (!str.empty())
        str += '|';
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
      if (tiedOp.isOperandTied(operand.getOperandNumber()))
        return true;
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
    if (auto arg = llvm::dyn_cast<BlockArgument>(operand.get())) {
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

    auto arg = llvm::cast<BlockArgument>(value);
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
          [&](Block *sourceBlock, OperandRange operands,
              size_t offset) -> WalkResult {
            unsigned baseIdx = operands.getBeginOperandIndex();
            auto &sourceOperand = sourceBlock->getTerminator()->getOpOperand(
                baseIdx + arg.getArgNumber() + offset);
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
class ElisionAnalysis {
public:
  explicit ElisionAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
        TraversalAction::RECURSE);
    explorer.setDialectAction<IREE::Stream::StreamDialect>(
        TraversalAction::RECURSE);
    // Ignore the contents of executables (linalg goo, etc).
    explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::IGNORE);
    explorer.initialize();

    assert(rootOp->getNumRegions() == 1 && "expected module-like root op");
    topLevelOps = llvm::to_vector(
        rootOp->getRegions().front().getOps<mlir::CallableOpInterface>());
  }

  AsmState &getAsmState() { return solver.getAsmState(); }

  // Runs analysis and populates the state cache.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run() {
    // Seed all block arguments throughout the program.
    for (auto callableOp : getTopLevelOps()) {
      auto *region = callableOp.getCallableRegion();
      if (!region)
        continue;
      for (auto &block : *region) {
        for (auto arg : block.getArguments()) {
          if (llvm::isa<IREE::Stream::ResourceType>(arg.getType())) {
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
    if (!argumentSemantics)
      return false;
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

//===----------------------------------------------------------------------===//
// IREE::Stream::AsyncCloneOp elision
//===----------------------------------------------------------------------===//

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
                                 ElisionAnalysis &analysis) {
  LLVM_DEBUG({
    llvm::dbgs() << "isSafeToElideCloneOp:\n";
    llvm::dbgs() << "  ";
    cloneOp.print(llvm::dbgs(), analysis.getAsmState());
    llvm::dbgs() << "\n";
  });

  // If this clone is performing a type change we need to preserve it.
  // TODO(benvanik): remove this carveout - could make clone not change type
  // and transfer be needed instead.
  auto sourceType =
      llvm::cast<IREE::Stream::ResourceType>(cloneOp.getSource().getType());
  auto targetType =
      llvm::cast<IREE::Stream::ResourceType>(cloneOp.getResult().getType());
  if (sourceType != targetType &&
      sourceType.getLifetime() == IREE::Stream::Lifetime::Constant) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - clone source is a constant; cannot elide\n");
    return false;
  }

  // If the source is a block argument we have to look into the analysis cache
  // to see if it's been classified as a last use/by-value move. If it isn't
  // then we cannot mutate it in-place as it could be used by the caller/another
  // branch and we need to respect the forking of the value.
  if (auto arg = llvm::dyn_cast<BlockArgument>(cloneOp.getSource())) {
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

// Elides a stream.async.clone op by replacing all uses with the cloned source.
static void elideCloneOp(IREE::Stream::AsyncCloneOp cloneOp) {
  cloneOp.replaceAllUsesWith(cloneOp.getSource());
  cloneOp.erase();
}

//===----------------------------------------------------------------------===//
// IREE::Stream::AsyncSliceOp elision
//===----------------------------------------------------------------------===//

// Filter to slices that are supported by the folding code.
static bool areSliceUsesSupported(IREE::Stream::AsyncSliceOp sliceOp) {
  for (auto &use : sliceOp.getResult().getUses()) {
    if (!TypeSwitch<Operation *, bool>(use.getOwner())
             .Case<IREE::Stream::AsyncCopyOp>([&](auto copyOp) {
               // Only support folding into source today.
               return !copyOp.isOperandTied(use.getOperandNumber());
             })
             .Case<IREE::Stream::AsyncDispatchOp>([&](auto dispatchOp) {
               // Only support folding into reads today.
               return !dispatchOp.isOperandTied(use.getOperandNumber());
             })
             .Default([](auto *op) { return false; })) {
      return false;
    }
  }
  return true;
}

// Returns true if |sliceOp| is safe to elide.
// This is only the case if the users are all supported ops.
static bool isSafeToElideSliceOp(IREE::Stream::AsyncSliceOp sliceOp,
                                 ElisionAnalysis &analysis) {
  LLVM_DEBUG({
    llvm::dbgs() << "isSafeToElideSliceOp:\n";
    llvm::dbgs() << "  ";
    sliceOp.print(llvm::dbgs(), analysis.getAsmState());
    llvm::dbgs() << "\n";
  });

  // Ensure all uses are ones we can support.
  if (!areSliceUsesSupported(sliceOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - slice consumers not supported; cannot elide\n");
    return false;
  }

  // Currently we don't analyze up a tied op chain and require the defining op
  // to be the producer.
  Value source = sliceOp.getSource();
  Value sourceBase = IREE::Util::TiedOpInterface::findTiedBaseValue(source);
  if (source != sourceBase) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - source is tied; cannot be elided (today)\n");
    return false;
  }

  AsyncAccessRange sliceRange;
  sliceRange.access = ResourceAccessBitfield::Read;
  sliceRange.resource = source;
  sliceRange.start = sliceOp.getSourceOffset();
  sliceRange.end = sliceOp.getSourceEnd();
  sliceRange.length = sliceOp.getResultSize();

  // Gather all accesses of the source by all other ops (not the slice being
  // inspected).
  SmallVector<AsyncAccessRange> consumerRanges;
  SmallVector<AsyncAccessRange> queryRanges;
  for (auto user : source.getUsers()) {
    if (user == sliceOp)
      continue;
    if (auto accessOp = dyn_cast<IREE::Stream::AsyncAccessOpInterface>(user)) {
      // Async op consuming part of the resource. We can query it to see what
      // it's doing to its operands/results and filter to just the accesses of
      // the source value.
      accessOp.getAsyncAccessRanges(queryRanges);
      for (auto range : queryRanges) {
        if (range.resource == source)
          consumerRanges.push_back(range);
      }
      queryRanges.clear();
    } else {
      // Unknown user - for now we skip analysis. If we made the access range
      // things elements in the solver we could traverse further.
      LLVM_DEBUG({
        llvm::dbgs()
            << "  - analysis failure on unhandled user of slice source:\n";
        user->print(llvm::dbgs(), analysis.getAsmState());
      });
      return false;
    }
  }

  // If all other users don't overlap with the slice we can directly use the
  // source resource.
  for (auto &otherRange : consumerRanges) {
    if (IREE::Stream::AsyncAccessRange::mayOverlap(sliceRange, otherRange)) {
      // Potential overlap detected (or analysis failed) - if both are reads
      // then we allow the elision (today) as there should be no hazard.
      if (!otherRange.isReadOnly()) {
        LLVM_DEBUG({
          llvm::dbgs() << "  - consumer overlap, skipping elision today\n";
          llvm::dbgs() << "    v slice ";
          sliceRange.print(llvm::dbgs(), analysis.getAsmState());
          llvm::dbgs() << "\n";
          llvm::dbgs() << "    ^ conflict ";
          otherRange.print(llvm::dbgs(), analysis.getAsmState());
          llvm::dbgs() << "\n";
        });
        return false;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  + slice can (probably) be elided\n");
  return true;
}

// arith.addi folders are terrible and don't handle adds of 0 so we handle that
// here and then avoid doing the folding.
static Value addOffset(Value lhs, Value rhs, OpBuilder &builder) {
  if (matchPattern(lhs, m_Zero()))
    return rhs;
  if (matchPattern(rhs, m_Zero()))
    return lhs;
  return builder.createOrFold<arith::AddIOp>(
      builder.getFusedLoc(lhs.getLoc(), rhs.getLoc()), lhs, rhs);
}

// TODO(benvanik): move these into patterns and use subview ops, maybe.
// That would allow us to support a lot more op types but if we can't guarantee
// a fold then we'd be left with unanalyzable subview ops. For now we handle the
// cases we care about here.

// Folds a stream.async.slice into a stream.async.copy source.
static void foldSliceIntoCopy(IREE::Stream::AsyncSliceOp sliceOp,
                              IREE::Stream::AsyncCopyOp copyOp,
                              unsigned operandNumber) {
  copyOp.getSourceMutable().set(sliceOp.getSource());
  OpBuilder builder(copyOp);
  copyOp.getSourceOffsetMutable().set(
      addOffset(sliceOp.getSourceOffset(), copyOp.getSourceOffset(), builder));
  copyOp.getSourceEndMutable().set(
      addOffset(sliceOp.getSourceOffset(), copyOp.getSourceEnd(), builder));
  copyOp.getSourceSizeMutable().set(sliceOp.getSourceSize());
}

// Folds a stream.async.slice into a stream.async.dispatch operand.
static void foldSliceIntoDispatch(IREE::Stream::AsyncSliceOp sliceOp,
                                  IREE::Stream::AsyncDispatchOp dispatchOp,
                                  unsigned operandNumber) {
  unsigned operandIndex =
      operandNumber - dispatchOp.getTiedOperandsIndexAndLength().first;
  dispatchOp.getResourceOperandsMutable()[operandIndex].set(
      sliceOp.getSource());
  unsigned resourceIndex = llvm::count_if(
      dispatchOp.getResourceOperands().slice(0, operandIndex),
      [](Value operand) {
        return llvm::isa<IREE::Stream::ResourceType>(operand.getType());
      });
  OpBuilder builder(dispatchOp);
  dispatchOp.getResourceOperandOffsetsMutable()[resourceIndex].set(addOffset(
      sliceOp.getSourceOffset(),
      dispatchOp.getResourceOperandOffsets()[resourceIndex], builder));
  dispatchOp.getResourceOperandEndsMutable()[resourceIndex].set(
      addOffset(sliceOp.getSourceOffset(),
                dispatchOp.getResourceOperandEnds()[resourceIndex], builder));
  dispatchOp.getResourceOperandSizesMutable()[resourceIndex].set(
      sliceOp.getSourceSize());
}

// Elides a stream.async.slice op (assuming able) by folding it into consumers.
static void elideSliceOp(IREE::Stream::AsyncSliceOp sliceOp) {
  SmallVector<std::pair<Operation *, unsigned>> consumers;
  for (auto &use : sliceOp.getResult().getUses())
    consumers.push_back(std::make_pair(use.getOwner(), use.getOperandNumber()));
  for (auto [owner, operandNumberIt] : consumers) {
    unsigned operandNumber = operandNumberIt; // need C++20 to avoid this :|
    TypeSwitch<Operation *>(owner)
        .Case<IREE::Stream::AsyncCopyOp>([=](auto copyOp) {
          foldSliceIntoCopy(sliceOp, copyOp, operandNumber);
        })
        .Case<IREE::Stream::AsyncDispatchOp>([=](auto dispatchOp) {
          foldSliceIntoDispatch(sliceOp, dispatchOp, operandNumber);
        })
        .Default([](auto *op) {});
  }
  sliceOp.erase();
}

//===----------------------------------------------------------------------===//
// --iree-stream-elide-async-copies
//===----------------------------------------------------------------------===//

// Tries to elide copies nested within |region| when safe.
// Returns true if any ops were elided.
static bool tryElideAsyncCopiesInRegion(Region &region,
                                        ElisionAnalysis &analysis) {
  bool didChange = false;
  for (auto &block : region) {
    block.walk([&](Operation *op) {
      return TypeSwitch<Operation *, WalkResult>(op)
          .Case<IREE::Stream::AsyncCloneOp>([&](auto cloneOp) {
            if (isSafeToElideCloneOp(cloneOp, analysis)) {
              elideCloneOp(cloneOp);
              didChange = true;
            }
            return WalkResult::advance();
          })
          .Case<IREE::Stream::AsyncSliceOp>([&](auto sliceOp) {
            if (isSafeToElideSliceOp(sliceOp, analysis)) {
              elideSliceOp(sliceOp);
              didChange = true;
            }
            return WalkResult::advance();
          })
          .Default([&](auto *op) { return WalkResult::advance(); });
    });
  }
  return didChange;
}

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
struct ElideAsyncCopiesPass
    : public IREE::Stream::impl::ElideAsyncCopiesPassBase<
          ElideAsyncCopiesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    // Try analyzing the program and eliding the unneeded copies until we reach
    // a fixed point (no more copies can be elided).
    unsigned maxIterationCount = 30;
    unsigned iterationCount = 0;
    for (; iterationCount < maxIterationCount; ++iterationCount) {
      // Perform whole-program analysis.
      // TODO(benvanik): reuse allocator across iterations.
      ElisionAnalysis analysis(moduleOp);
      if (failed(analysis.run())) {
        moduleOp.emitError() << "failed to solve for last users";
        return signalPassFailure();
      }

      // Apply analysis by eliding all copies that are safe to elide.
      // If we can't elide any we'll consider the iteration complete and exit.
      bool didChange = false;
      for (auto callableOp : analysis.getTopLevelOps()) {
        auto *region = callableOp.getCallableRegion();
        if (!region)
          continue;
        didChange = tryElideAsyncCopiesInRegion(*region, analysis) || didChange;
      }
      if (!didChange)
        break;
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

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
