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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-resource-refcounting"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_RESOURCEREFCOUNTINGPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

// TODO: Cleanup - reference from ElideAsyncCopiesPass
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
      llvm::dbgs() << "[refcounting] initialized value last users for ";
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
class LivenessAnalysis {
public:
  explicit LivenessAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
        TraversalAction::RECURSE);
    explorer.setDialectAction<IREE::Stream::StreamDialect>(
        TraversalAction::RECURSE);
    explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::SHALLOW);
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
            if (llvm::isa<IREE::Stream::ResourceType>(arg.getType())) {
              solver.getOrCreateElementFor<ArgumentSemantics>(
                  Position::forValue(arg));
            }
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

llvm::DenseMap<Value, int> refCountMap;
llvm::DenseMap<Value, Operation *> lastUseMap;
void insertDealloca(Value, Operation *);

// on alloca, find all users and add reference counts
// at each execute region block  resource is used
// that execution region is last user, decrement count,
//      if not last user, refCount for that resource stays incremented
//
// walk to another execute region, increment

static void performRefCountingInRegion(Region &region,
                                       LivenessAnalysis &analysis) {

  region.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<IREE::Stream::ResourceAllocaOp>([&](auto allocaOp) {
          auto resource =
              cast<IREE::Stream::ResourceAllocaOp>(op)->getResult(0);
          // Adds refCounting
          if (!resource.getUses().empty()) {
            refCountMap.insert(std::make_pair(resource, 0));
            for (Operation *user : resource.getUsers()) {
              refCountMap[resource]++;
              if (analysis.isLastUser(resource, user))
                lastUseMap[resource] = user;
            }
          }
        })
        .Case<IREE::Stream::ResourceDeallocaOp>([&](auto deallocaOp) {
          auto resource =
              cast<IREE::Stream::ResourceDeallocaOp>(op)->getResult(0);

          refCountMap[resource]--;
          LLVM_DEBUG({
            llvm::dbgs() << "\n last user is dealloca, just decrement refcount "
                            "and do nothing: "
                         << resource;
          });
        })
        .Case<IREE::Stream::CmdExecuteOp>([&](auto executeOp) {
          for (auto operand : executeOp->getOperands()) {
            if (refCountMap.count(operand)) {

              // Decrement reference after last use
              if (analysis.isLastUser(operand, executeOp)) {
                lastUseMap.insert(std::make_pair(operand, executeOp));

                if (refCountMap[operand] > 0) {
                  refCountMap[operand]--;
                }

                if (refCountMap[operand] == 0) {
                  insertDealloca(operand, executeOp);
                }
              } else {
                refCountMap[operand]--;
              }
            }
          }
        });
  });
}

void insertDealloca(Value resource, Operation *lastuseOp) {
  auto refCount = refCountMap[resource];
  auto definingOp = resource.getDefiningOp();

  auto allocaOp = cast<IREE::Stream::ResourceAllocaOp>(definingOp);

  OpBuilder builder(lastuseOp);
  builder.setInsertionPointAfter(lastuseOp);
  auto loc = lastuseOp->getLoc();
  auto timepoint = allocaOp.getResultTimepoint();
  if (auto op = dyn_cast<IREE::Stream::CmdExecuteOp>(lastuseOp)) {
    timepoint = op.getResultTimepoint();
  }

  Type i32Type = builder.getIntegerType(32);
  Value countVal = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(i32Type, refCount));
  Value zero = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(i32Type, 0));

  auto cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            countVal, zero);
  builder.create<mlir::scf::IfOp>(
      loc, cond, [&](OpBuilder &builder, Location loc) {
        builder.create<IREE::Stream::ResourceDeallocaOp>(
            loc, resource, allocaOp.getResultSize(0), timepoint,
            allocaOp.getAffinityAttr());

        builder.create<mlir::scf::YieldOp>(loc);
      });
}

// This operates using a whole-program analysis to track reference counts of a
// resource refCount map of SSA + refCount (they are passed the last use of a
// resource) and the last users of resource values. Once analyzed resource with
// ref count being 0 need to have deallocas inserted in place. This process
// repeats until no more Last use opes are present
struct ResourceRefCountingPass
    : public IREE::Stream::impl::ResourceRefCountingPassBase<
          ResourceRefCountingPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    // Perform whole-program analysis to find last user each resource
    LivenessAnalysis analysis(moduleOp);
    if (failed(analysis.run())) {
      moduleOp.emitError() << "failed to solve for liveness analysis";
      return signalPassFailure();
    }

    for (auto callableOp : analysis.getTopLevelOps()) {

      auto *region = callableOp.getCallableRegion();
      if (!region)
        continue;

      // Use analysis to find last-witness op for each resource,
      // adding reference count for each user, and
      // insert dealloca when refcount reaches zero
      performRefCountingInRegion(*region, analysis);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
