// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
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
#include "mlir/IR/Matchers.h"
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

// Returns the default affinity of |fromOp| or its ancestors, if available.
// Note that default affinities are optional (and generally discouraged) but
// useful for quick hacking/testing - by this point in the full pipeline we
// assume all ops have explicit affinities assigned.
//
// TODO(benvanik): hide this implementation detail; it's currently hoisted from
// AffinityAnalysis (and I hate that).
static IREE::Stream::AffinityAttr tryLookupDefaultAffinity(Operation *fromOp) {
  while (fromOp) {
    if (auto affinityAttr = fromOp->getAttrOfType<IREE::Stream::AffinityAttr>(
            "stream.affinity.default")) {
      return affinityAttr;
    }
    fromOp = fromOp->getParentOp();
  }
  return nullptr;
}

// Returns true if |value| is known to be constant zero.
static bool isConstantZero(Value value) {
  if (!value)
    return true; // No offset means 0.
  return matchPattern(value, m_Zero());
}

// Returns true if |result| is produced by an operation that fully overwrites
// the tied resource with new data. In this case, the result's data is "owned"
// (not borrowed) even if the underlying buffer came from an external source.
//
// This is a tactical check using constant matching. A proper fix would use
// DFX-based range analysis and integer range analysis.
static bool doesOpFullyOverwriteTiedResult(OpResult result) {
  Operation *op = result.getOwner();

  // Must be a tied operation.
  auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
  if (!tiedOp)
    return false;

  auto resultIdx = result.getResultNumber();
  if (!tiedOp.getTiedResultOperandIndex(resultIdx).has_value()) {
    return false;
  }

  // Must implement AsyncAccessOpInterface to query access ranges.
  auto accessOp = dyn_cast<IREE::Stream::AsyncAccessOpInterface>(op);
  if (!accessOp)
    return false;

  // Must implement SizeAwareOpInterface to get result size.
  auto sizeAwareOp = dyn_cast<IREE::Util::SizeAwareOpInterface>(op);
  if (!sizeAwareOp)
    return false;

  Value resultSize = sizeAwareOp.getResultSizeFromValue(result);
  if (!resultSize)
    return false;

  // Check access ranges for a full overwrite of the result.
  SmallVector<AsyncAccessRange> accessRanges;
  accessOp.getAsyncAccessRanges(accessRanges);
  for (auto &range : accessRanges) {
    // Look for write-ONLY access to this result. Read+Write means the op
    // depends on the existing data (e.g., in/out dispatch operand), so we
    // can't treat it as a full overwrite of borrowed data.
    if (range.resource == result &&
        range.access == ResourceAccessBitfield::Write) {
      // Check if write covers full resource: offset=0, length=size.
      if (isConstantZero(range.start) && range.length == resultSize) {
        return true;
      }
    }
  }

  return false;
}

// Returns true if we can safely propagate |newLifetime| through all users of
// |value|. This is used for eliding lifetime-changing clones (e.g., external
// -> *) by propagating the source lifetime through the use chain.
//
// Collects operations that need type updates in |opsToUpdate|. Each entry is
// (op, operandIdx) where operandIdx identifies which operand receives |value|.
//
// We can propagate through:
// - TiedOpInterface ops: result is tied to operand, so result type must match
// - Ops whose result is already the target lifetime (will become same-type)
static bool canPropagateLifetimeToUsers(
    Value value, IREE::Stream::Lifetime newLifetime,
    SmallVectorImpl<std::pair<Operation *, unsigned>> &opsToUpdate,
    SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return true; // Already visited, avoid cycles.

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    unsigned operandIdx = use.getOperandNumber();

    // Check if this op ties the operand to a result via TiedOpInterface.
    if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(user)) {
      // Find any results tied to this operand.
      auto tiedResults = tiedOp.getOperandTiedResults(operandIdx);
      if (!tiedResults.empty()) {
        // This op passes the resource through - we need to update result types.
        opsToUpdate.push_back({user, operandIdx});

        // Recursively check all tied results.
        for (auto result : tiedResults) {
          if (isa<IREE::Stream::ResourceType>(result.getType())) {
            if (!canPropagateLifetimeToUsers(result, newLifetime, opsToUpdate,
                                             visited))
              return false;
          }
        }
        continue;
      }
    }

    // Non-tied use: check if result converts BACK to our target lifetime.
    // (e.g., clone/transfer from * -> external when we're propagating external)
    // These will become same-type ops after we update the operand.
    if (user->getNumResults() == 1) {
      if (auto resultType = dyn_cast<IREE::Stream::ResourceType>(
              user->getResult(0).getType())) {
        if (resultType.getLifetime() == newLifetime) {
          // This op's result is already the target lifetime - it will become
          // a same-type operation and can be elided in a subsequent pass.
          opsToUpdate.push_back({user, operandIdx});
          continue;
        }
      }
    }

    // Unknown pattern - conservatively fail.
    return false;
  }
  return true;
}

// Updates the types of operations collected during canPropagateLifetimeToUsers.
// For tied ops, we update the tied result types to match the new lifetime.
static void updateTypesForLifetimePropagation(
    ArrayRef<std::pair<Operation *, unsigned>> opsToUpdate,
    IREE::Stream::Lifetime newLifetime) {
  for (auto [op, operandIdx] : opsToUpdate) {
    // For tied ops: update the tied result types to match new operand lifetime.
    if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op)) {
      auto tiedResults = tiedOp.getOperandTiedResults(operandIdx);
      for (auto result : tiedResults) {
        if (auto oldType =
                dyn_cast<IREE::Stream::ResourceType>(result.getType())) {
          auto newType = IREE::Stream::ResourceType::get(oldType.getContext(),
                                                         newLifetime);
          result.setType(newType);
        }
      }
    }
    // For non-tied ops with single result converting back to target lifetime:
    // No type update needed - they already have the correct result type.
    // After RAUW, they will have same source/target type and become elidable.
  }
}

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
      if (!str.empty()) {
        str += '|';
      }
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
      if (tiedOp.isOperandTied(operand.getOperandNumber())) {
        return true;
      }
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
    if (auto arg = dyn_cast<BlockArgument>(operand.get())) {
      auto &argumentSemantics = solver.getElementFor<ArgumentSemantics>(
          *this, Position::forValue(operand.get()), DFX::Resolution::REQUIRED);
      LLVM_DEBUG(llvm::dbgs()
                 << "  pred is arg; combining state: "
                 << argumentSemantics.getAsStr(solver.getAsmState()) << "\n");
      getState() ^= argumentSemantics.getState();
    }

    // Check if the source is a stream.tensor.import with the consume attribute.
    // This indicates transfer of ownership (by-value semantics).
    //
    // We should probably make this part of an interface, but today the tensor
    // import op is the only thing we support this on.
    if (auto importOp =
            operand.get().getDefiningOp<IREE::Stream::TensorImportOp>()) {
      if (importOp.getConsume()) {
        // Consume attribute present - value is owned by importer (by-value).
        // Already in BEST_STATE (NOT_MUTATED | NOT_BY_REFERENCE).
        LLVM_DEBUG(llvm::dbgs()
                   << "  stream.tensor.import with consume attribute\n");
        return;
      }
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

    auto arg = cast<BlockArgument>(value);
    bool isEntryArg = arg.getParentBlock()->isEntryBlock();
    if (isEntryArg) {
      // Entry block argument - could be a callable function argument or
      // an SCF/region operation argument (scf.for iter_args, scf.if, etc).
      auto *parentOp = arg.getParentBlock()->getParentOp();
      if (auto callableOp = dyn_cast<mlir::CallableOpInterface>(parentOp)) {
        // Call argument.
        traversalResult |= solver.getExplorer().walkIncomingCalls(
            callableOp, [&](mlir::CallOpInterface callOp) -> WalkResult {
              unsigned baseIdx = callOp.getArgOperands().getBeginOperandIndex();
              auto &sourceOperand =
                  callOp->getOpOperand(baseIdx + arg.getArgNumber());
              updateFromPredecessorUse(sourceOperand, solver);
              return WalkResult::advance();
            });
      } else {
        // SCF or other region operation argument (scf.for iter_args, CFG
        // blocks, etc). We use the explorer to get the incoming values from
        // parent/incoming CFG edges. Note that we can't tell in a multi-region
        // op which region will execute so behavior is the same regardless (e.g.
        // consuming in the then/else region of an scf.if will both walk to
        // producers).
        traversalResult |= solver.getExplorer().walkIncomingBlockArgument(
            arg, [&](Block *sourceBlock, Value operand) -> WalkResult {
              // |operand| is the produced value feeding this argument and we
              // need to find the OpOperand that corresponds to this use.
              // For scf.for this could be from the ForOp (init args) or
              // from the yield terminator (loop-back).
              Operation *sourceOp = arg.getParentBlock()->getParentOp();
              if (sourceBlock == sourceOp->getBlock()) {
                // Init args case: operand is from the region op itself.
                for (unsigned i = 0; i < sourceOp->getNumOperands(); ++i) {
                  if (sourceOp->getOperand(i) == operand) {
                    updateFromPredecessorUse(sourceOp->getOpOperand(i), solver);
                    return WalkResult::advance();
                  }
                }
              } else {
                // Loop-back case: operand is from a terminator in the region.
                auto *terminator = sourceBlock->getTerminator();
                for (unsigned i = 0; i < terminator->getNumOperands(); ++i) {
                  if (terminator->getOperand(i) == operand) {
                    updateFromPredecessorUse(terminator->getOpOperand(i),
                                             solver);
                    return WalkResult::advance();
                  }
                }
              }
              // Fallback: couldn't find the operand.
              return WalkResult::advance();
            });
      }
    } else {
      // Non-entry block branch argument.
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

// Tracks whether a resource value is ever mutated anywhere in the program.
// This performs whole-program transitive use analysis to detect any writes.
class ResourceMutationSemantics
    : public DFX::StateWrapper<DFX::BitIntegerState<uint8_t, 1, 0>,
                               DFX::ValueElement> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::BitIntegerState<uint8_t, 1, 0>, DFX::ValueElement>;

  enum {
    // Inverted bit so we start optimistic (assume not mutated).
    NOT_MUTATED = 1u << 0,
    BEST_STATE = NOT_MUTATED,
  };
  static_assert(BEST_STATE == BaseType::getBestState(),
                "unexpected BEST_STATE value");

  static ResourceMutationSemantics &createForPosition(const Position &pos,
                                                      DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ResourceMutationSemantics(pos));
  }

  const std::string getName() const override {
    return "ResourceMutationSemantics";
  }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  // Returns true if the resource is assumed to never be mutated.
  bool isAssumedNotMutated() const {
    return (this->getAssumed() & NOT_MUTATED) == NOT_MUTATED;
  }

  const std::string getAsStr(AsmState &asmState) const override {
    return this->isAssumed(NOT_MUTATED) ? "not-mutated" : "mutated";
  }

private:
  explicit ResourceMutationSemantics(const Position &pos) : BaseType(pos) {}

  // Analyzes LOCAL mutations of a value by checking AsyncAccessOpInterface.
  // This does NOT follow tied operations - that's done in updateValue to
  // participate in fixed-point iteration.
  void analyzeLocalMutation(Value value, DFX::Solver &solver) {
    SmallVector<AsyncAccessRange> accessRanges;

    // Walk ALL transitive uses (crosses function boundaries).
    auto traversalResult = solver.getExplorer().walkTransitiveUses(
        value, [&](OpOperand &operand) -> WalkResult {
          Operation *op = operand.getOwner();

          // Check AsyncAccessOpInterface for write access.
          // This is the primary way to detect mutations.
          if (auto accessOp =
                  dyn_cast<IREE::Stream::AsyncAccessOpInterface>(op)) {
            accessRanges.clear();
            accessOp.getAsyncAccessRanges(accessRanges);
            for (auto &range : accessRanges) {
              if (range.resource == value && !range.isReadOnly()) {
                LLVM_DEBUG({
                  llvm::dbgs() << "    found write access at op: ";
                  op->print(llvm::dbgs(), solver.getAsmState());
                  llvm::dbgs() << "\n";
                });
                removeAssumedBits(NOT_MUTATED);
                return WalkResult::interrupt();
              }
            }
          }

          return WalkResult::advance();
        });

    handleIncompleteTraversal(value, traversalResult);
  }

  // Handle incomplete traversal results (value escapes).
  void handleIncompleteTraversal(Value value, TraversalResult traversalResult) {
    if (traversalResult != TraversalResult::INCOMPLETE)
      return;

    // Analysis is incomplete when a value escapes (e.g., returned from
    // public functions). Whether we assume mutation depends on the lifetime:
    auto resourceType = llvm::cast<IREE::Stream::ResourceType>(value.getType());
    auto lifetime = resourceType.getLifetime();
    if (lifetime == IREE::Stream::Lifetime::Constant) {
      // Constants maintain immutability guarantees.
      LLVM_DEBUG(
          llvm::dbgs()
          << "    incomplete analysis for constant; assuming immutable\n");
    } else if (lifetime == IREE::Stream::Lifetime::External) {
      // External resources are expected to cross function boundaries (they
      // are imported/exported). Escaping via return/export doesn't mean
      // mutation - it just means the resource is being returned to the
      // caller who owns it. The caller may mutate it later, but that's
      // THEIR concern after we return.
      LLVM_DEBUG(
          llvm::dbgs()
          << "    incomplete analysis for external; assuming immutable\n");
    } else {
      // Conservative: assume mutated if analysis fails for other lifetimes.
      LLVM_DEBUG(
          llvm::dbgs()
          << "    incomplete analysis; conservatively assuming mutated\n");
      removeAssumedBits(NOT_MUTATED);
    }
  }

  // Analyzes tied operations to check if any tied result is mutated.
  // This is called during updateValue to participate in fixed-point iteration.
  // If a value is used by a tied op and the tied result is mutated, the
  // original value is also considered mutated (they share storage).
  void analyzeTiedMutation(Value value, DFX::Solver &solver) {
    // Walk uses to find tied operations.
    solver.getExplorer().walkTransitiveUses(
        value, [&](OpOperand &operand) -> WalkResult {
          Operation *op = operand.getOwner();

          // For tied ops (barrier, await), check if the tied result is mutated.
          // Since they share storage, mutation of the result means mutation of
          // the operand.
          if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op)) {
            // Find which result is tied to our operand.
            unsigned operandIdx = operand.getOperandNumber();
            for (auto result : op->getResults()) {
              if (!llvm::isa<IREE::Stream::ResourceType>(result.getType()))
                continue;
              auto tiedOperandIdx =
                  tiedOp.getTiedResultOperandIndex(result.getResultNumber());
              if (tiedOperandIdx.has_value() &&
                  tiedOperandIdx.value() == operandIdx) {
                // This result is tied to our value - check if IT is mutated.
                // Use the solver to get the mutation status, which will trigger
                // analysis if needed and participate in fixed-point iteration.
                auto &resultMutation =
                    solver.getOrCreateElementFor<ResourceMutationSemantics>(
                        Position::forValue(result));
                if (!resultMutation.isAssumed(NOT_MUTATED)) {
                  LLVM_DEBUG({
                    llvm::dbgs() << "    tied result is mutated: ";
                    result.printAsOperand(llvm::dbgs(), solver.getAsmState());
                    llvm::dbgs() << "\n";
                  });
                  removeAssumedBits(NOT_MUTATED);
                  return WalkResult::interrupt();
                }
              }
            }
          }

          return WalkResult::advance();
        });
  }

  void initializeValue(Value value, DFX::Solver &solver) override {
    // Start optimistic - assume not mutated.
    intersectAssumedBits(BEST_STATE);

    LLVM_DEBUG({
      llvm::dbgs() << "  analyzing mutations for value: ";
      value.printAsOperand(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });

    // Only do local analysis during initialization (direct write access).
    // Tied-chain analysis is deferred to updateValue for fixed-point iteration.
    analyzeLocalMutation(value, solver);

    LLVM_DEBUG({
      if ((this->getAssumed() & NOT_MUTATED) == NOT_MUTATED) {
        llvm::dbgs() << "    result: not mutated\n";
      } else {
        llvm::dbgs() << "    result: mutated\n";
      }
    });

    // Don't call indicateOptimisticFixpoint() here - let the solver iterate.
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    // Check if already known to be mutated - no need to re-analyze.
    if (!isAssumed(NOT_MUTATED)) {
      return ChangeStatus::UNCHANGED;
    }

    auto assumedBits = getAssumed();

    // Re-run local analysis (may find mutations through control flow joins).
    analyzeLocalMutation(value, solver);

    // Check tied operations to propagate mutation through the alias chain.
    // This is the key for fixed-point iteration: if a value flows through
    // barrier/await and the tied result is mutated, this value is also
    // mutated since they share the same underlying storage.
    analyzeTiedMutation(value, solver);

    return assumedBits == getAssumed() ? ChangeStatus::UNCHANGED
                                       : ChangeStatus::CHANGED;
  }

  friend class DFX::Solver;
};
const char ResourceMutationSemantics::ID = 0;

// TODO(benvanik): change into something we can use for ref counting. We need
// that to insert stream-ordered deallocs and know when timepoints have been
// discard as they go out of scope. For now this strictly checks last use.
class ElisionAnalysis {
public:
  explicit ElisionAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::RECURSE),
        solver(explorer, allocator) {
    // Default is RECURSE to support arbitrary other dialects that may use
    // stream resources in their bodies (SCF, etc). We explicitly ignore
    // stream.executable here for clarity but the explorer will ignore all
    // ObjectLike ops (hal.executable, etc) that may exist in the IR
    // automatically.
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
    // Seed all block arguments throughout the program (including nested
    // regions). This ensures SCF operations (scf.for, scf.if, scf.while) and
    // other region-bearing ops have their block arguments analyzed.
    for (auto callableOp : getTopLevelOps()) {
      auto *region = callableOp.getCallableRegion();
      if (!region) {
        continue;
      }
      region->walk([&](Block *block) {
        for (auto arg : block->getArguments()) {
          if (isa<IREE::Stream::ResourceType>(arg.getType())) {
            solver.getOrCreateElementFor<ArgumentSemantics>(
                Position::forValue(arg));
          }
        }
      });
    }

    // Seed ResourceMutationSemantics for all Stream resource values.
    // This ensures they participate in the fixed-point iteration.
    int seedCount = 0;
    for (auto callableOp : getTopLevelOps()) {
      auto *region = callableOp.getCallableRegion();
      if (!region) {
        continue;
      }
      region->walk([&](Operation *op) {
        for (auto result : op->getResults()) {
          if (llvm::isa<IREE::Stream::ResourceType>(result.getType())) {
            solver.getOrCreateElementFor<ResourceMutationSemantics>(
                Position::forValue(result));
            ++seedCount;
          }
        }
      });
    }
    LLVM_DEBUG(llvm::dbgs() << "seeded " << seedCount
                            << " ResourceMutationSemantics elements\n");

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
    if (!argumentSemantics) {
      return false;
    }
    return argumentSemantics->getAssumedByValue();
  }

  // Returns true if |userOp| is the last user of |operand|.
  bool isLastUser(Value operand, Operation *userOp) {
    auto lastUsers =
        solver.getOrCreateElementFor<LastUsers>(Position::forValue(operand));
    return lastUsers.isAssumedLastUser(userOp);
  }

  // Returns true if |value| is never mutated anywhere in the program.
  // This uses whole-program transitive use analysis.
  bool isNeverMutated(Value value) {
    // Get or create the element. It will be initialized with the current
    // solver state if created after the solver has run.
    auto &mutationSemantics =
        solver.getOrCreateElementFor<ResourceMutationSemantics>(
            Position::forValue(value));
    return mutationSemantics.isAssumedNotMutated();
  }

  // Returns true if |value| is "borrowed" from an external source, meaning
  // the caller retains ownership and we cannot safely mutate it in-place.
  // This is true when:
  // - The value is a block argument that was passed by-reference (not moved)
  // - The value traces back to a stream.tensor.import WITHOUT consume
  //
  // If borrowed, cloning the value creates a copy that we can mutate safely.
  bool isBorrowedValue(Value value) {
    // For block arguments, use ArgumentSemantics which already handles
    // cross-function analysis of whether the arg was passed by-value or
    // by-reference.
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      return !isArgMoved(arg);
    }

    // For operation results, walk up the def chain (through tied ops only)
    // to find the ultimate source. If it's a non-consuming import the value
    // is borrowed.
    bool foundBorrowed = false;
    explorer.walkDefiningOps(
        value,
        [&](OpResult result) {
          Operation *op = result.getOwner();

          // Check for stream.tensor.import - if present without consume,
          // this is the borrowing point.
          if (auto importOp = dyn_cast<IREE::Stream::TensorImportOp>(op)) {
            if (!importOp.getConsume()) {
              foundBorrowed = true;
              return WalkResult::interrupt();
            }
            // Import with consume = owned, stop searching this path
            return WalkResult::interrupt();
          }

          // For tied operations, the result may inherit ownership from the
          // tied operand - unless the operation fully overwrites with new data.
          if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op)) {
            auto resultIdx = result.getResultNumber();
            if (tiedOp.getTiedResultOperandIndex(resultIdx).has_value()) {
              // If this operation fully overwrites the result with new data,
              // the data is owned (not borrowed) - stop walking.
              if (doesOpFullyOverwriteTiedResult(result)) {
                return WalkResult::interrupt();
              }
              // Partial/no overwrite - continue walking to find ultimate
              // source.
              return WalkResult::advance();
            }
          }

          // Non-tied result = owned, stop searching this path.
          return WalkResult::interrupt();
        },
        TraversalBehavior::DEFAULT);

    return foundBorrowed;
  }

  // Attempts to infer the affinity of |value| by walking up its defining ops
  // and checking for explicit affinity annotations or default affinities.
  // Returns nullptr if no affinity can be determined.
  IREE::Stream::AffinityAttr tryInferValueAffinity(Value value) {
    // Fast path: check if defining op has explicit affinity.
    if (auto definingOp = value.getDefiningOp()) {
      if (auto affinityOp =
              dyn_cast<IREE::Stream::AffinityOpInterface>(definingOp)) {
        if (auto affinity = affinityOp.getResultAffinityAttr()) {
          return affinity;
        }
      }
    }

    // Walk defining ops using Explorer to find an affinity-annotated producer.
    IREE::Stream::AffinityAttr foundAffinity;
    explorer.walkDefiningOps(
        value,
        [&](OpResult result) {
          Operation *op = result.getOwner();

          // Check if this defining op has an explicit affinity.
          if (auto affinityOp =
                  dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
            if (auto affinity = affinityOp.getResultAffinityAttr()) {
              // Found explicit affinity, stop.
              foundAffinity = affinity;
              return WalkResult::interrupt();
            }
          }

          // Check op for explicit affinity attribute.
          if (auto affinityAttr = IREE::Stream::AffinityAttr::lookup(op)) {
            foundAffinity = affinityAttr;
            return WalkResult::interrupt();
          }

          // Keep searching for explicit affinity.
          return WalkResult::advance();
        },
        TraversalBehavior::DEFAULT);
    if (foundAffinity) {
      return foundAffinity;
    }

    // No explicit affinity found anywhere in the producer chain.
    // Fall back to default affinity at the value's location.
    if (auto definingOp = value.getDefiningOp()) {
      return tryLookupDefaultAffinity(definingOp);
    } else if (auto arg = dyn_cast<BlockArgument>(value)) {
      return tryLookupDefaultAffinity(arg.getParentBlock()->getParentOp());
    }
    return nullptr;
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

  // If this clone is performing a lifetime conversion we need to preserve it.
  // Clones can change resource lifetime (e.g., * -> variable, external -> *)
  // and these conversions are semantically meaningful and must be preserved.
  auto sourceType =
      cast<IREE::Stream::ResourceType>(cloneOp.getSource().getType());
  auto targetType =
      cast<IREE::Stream::ResourceType>(cloneOp.getResult().getType());
  if (sourceType.getLifetime() != targetType.getLifetime()) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - clone is a resource type cast (" << sourceType << " to "
               << targetType << "); cannot elide\n");
    return false;
  }

  // Constant-to-constant clones can be elided only if both source and result
  // are never mutated anywhere in the program (interprocedural analysis).
  // Constants are immutable after initialization, but during initialization
  // they may be mutated.
  if (sourceType.getLifetime() == IREE::Stream::Lifetime::Constant &&
      targetType.getLifetime() == IREE::Stream::Lifetime::Constant) {

    // Check if source constant is ever mutated anywhere (whole program).
    bool sourceSafe = analysis.isNeverMutated(cloneOp.getSource());
    // Check if clone result is ever mutated anywhere (whole program).
    bool resultSafe = analysis.isNeverMutated(cloneOp.getResult());

    if (!sourceSafe) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  - source constant is mutated somewhere; cannot elide\n");
      return false;
    }

    if (!resultSafe) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  - clone result is mutated somewhere; cannot elide\n");
      return false;
    }

    LLVM_DEBUG(llvm::dbgs() << "  + constant->constant clone with no mutations "
                               "anywhere; can elide\n");
    return true;
  }

  // If the source is a block argument we have to look into the analysis cache
  // to see if it's been classified as a last use/by-value move. If it isn't
  // then we cannot mutate it in-place as it could be used by the caller/another
  // branch and we need to respect the forking of the value.
  if (auto arg = dyn_cast<BlockArgument>(cloneOp.getSource())) {
    if (!analysis.isArgMoved(arg)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  - clone source is a by-ref arg; cannot elide\n");
      return false;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  ? clone source is a by-value arg; may elide\n");
  }

  // If the source is "borrowed" (comes from a non-consuming import or is a
  // by-reference block argument), the caller retains ownership. If the clone
  // result is ever mutated (including transitively through tied ops), we need
  // to preserve the clone to protect the caller's buffer from in-place
  // mutation. This handles cases like:
  // - An imported buffer used as an in-place accumulator (GEMM accumulate)
  // - A value passed by-reference through function calls
  // - A value flowing through tied ops from a borrowed source
  if (analysis.isBorrowedValue(cloneOp.getSource())) {
    // Check if the clone result is ever mutated anywhere (whole program).
    if (!analysis.isNeverMutated(cloneOp.getResult())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  - source is borrowed and result is mutated; "
                    "cannot elide\n");
      return false;
    }
    LLVM_DEBUG(llvm::dbgs() << "  ? source is borrowed but result is never "
                               "mutated; may elide\n");
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

// Attempts to elide a lifetime-changing clone (e.g., external -> *) by
// propagating the source lifetime through all downstream users.
//
// Returns true if the clone was elided, false if elision is not possible.
//
// This handles the case where an external buffer flows through internal ops
// (barriers, awaits) and eventually returns to external. By propagating the
// external lifetime through the chain, we avoid allocating an intermediate
// buffer.
static bool tryElideLifetimeChangingClone(IREE::Stream::AsyncCloneOp cloneOp,
                                          ElisionAnalysis &analysis) {
  auto sourceType =
      cast<IREE::Stream::ResourceType>(cloneOp.getSource().getType());
  auto targetType =
      cast<IREE::Stream::ResourceType>(cloneOp.getResult().getType());

  // Only handle lifetime changes.
  if (sourceType.getLifetime() == targetType.getLifetime())
    return false;

  LLVM_DEBUG({
    llvm::dbgs() << "tryElideLifetimeChangingClone:\n";
    llvm::dbgs() << "  ";
    cloneOp.print(llvm::dbgs(), analysis.getAsmState());
    llvm::dbgs() << "\n";
  });

  // Only handle external -> * for now (most common case for output buffers).
  if (sourceType.getLifetime() != IREE::Stream::Lifetime::External ||
      targetType.getLifetime() != IREE::Stream::Lifetime::Unknown) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - unsupported lifetime change (not external -> *)\n");
    return false;
  }

  // Check if source is borrowed - if so, we need to be careful about mutation.
  if (analysis.isBorrowedValue(cloneOp.getSource())) {
    // Source is borrowed from caller. We can only elide if the result is
    // never mutated (otherwise we'd mutate the caller's buffer).
    if (!analysis.isNeverMutated(cloneOp.getResult())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  - borrowed source and result is mutated; keeping\n");
      return false;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  ? borrowed source but result never mutated; may elide\n");
  }

  // Check if we can propagate the external lifetime through all users.
  SmallVector<std::pair<Operation *, unsigned>> opsToUpdate;
  SmallPtrSet<Value, 8> visited;
  if (!canPropagateLifetimeToUsers(cloneOp.getResult(),
                                   sourceType.getLifetime(), opsToUpdate,
                                   visited)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - cannot propagate lifetime to all users; keeping\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "  + can propagate lifetime through "
                          << opsToUpdate.size() << " ops; eliding\n");

  // Update types of all downstream operations.
  updateTypesForLifetimePropagation(opsToUpdate, sourceType.getLifetime());

  // Replace uses and erase the clone.
  cloneOp.replaceAllUsesWith(cloneOp.getSource());
  cloneOp.erase();

  return true;
}

//===----------------------------------------------------------------------===//
// IREE::Stream::AsyncTransferOp elision
//===----------------------------------------------------------------------===//

// Returns true if |transferOp| performs no work or is safe to elide as a copy.
//
// This handles several common cases:
// 1. No-op transfers (same affinity + same type)
// 2. Lifetime-changing copies (same affinity, different type)
// 3. Transfer chain collapsing (A->B->C becomes A->C)
// 4. Topology-based elision (unified memory between devices)
//
// The analysis must ensure we don't elide protective copies needed for:
// - Tied operations (in-place mutations)
// - Values with multiple users (fork semantics)
// - Block arguments passed by-reference
static bool isSafeToElideTransferOp(
    IREE::Stream::AsyncTransferOp transferOp, ElisionAnalysis &analysis,
    IREE::Stream::AffinityTopologyAttrInterface topologyAttr) {
  LLVM_DEBUG({
    llvm::dbgs() << "isSafeToElideTransferOp:\n";
    llvm::dbgs() << "  ";
    transferOp.print(llvm::dbgs(), analysis.getAsmState());
    llvm::dbgs() << "\n";
  });

  auto source = transferOp.getSource();
  auto result = transferOp.getResult();
  auto sourceType = cast<IREE::Stream::ResourceType>(source.getType());
  auto resultType = cast<IREE::Stream::ResourceType>(result.getType());

  // Don't elide transfers that change lifetime (usage casts).
  // These encode important semantic information about how the resource is used.
  if (sourceType.getLifetime() != resultType.getLifetime()) {
    LLVM_DEBUG(llvm::dbgs() << "  - transfer changes lifetime; cannot elide\n");
    return false;
  }

  // Infer source affinity if not explicitly specified.
  // Source affinity inference walks defining ops and checks for result affinity
  // annotations. At this point in the pipeline most ops have explicit
  // affinities or use defaults so this covers the remaining edge cases without
  // requiring full affinity analysis integration (which is expensive).
  auto sourceAffinityAttr = transferOp.getSourceAffinityAttr();
  if (!sourceAffinityAttr) {
    sourceAffinityAttr = analysis.tryInferValueAffinity(source);
  }
  auto resultAffinityAttr = transferOp.getResultAffinityAttr();

  // Track whether topology analysis allows elision.
  bool topologyAllowsElision = false;

  // Check if topology allows elision (unified memory, etc).
  // This check happens first as it's the primary gate for whether to remove
  // transfers that the hardware topology doesn't require.
  if (sourceAffinityAttr && resultAffinityAttr && topologyAttr) {
    // Check if we can elide based on topology (unified memory, transparent
    // access, etc).
    // Note that even with unified memory we may need the transfer for other
    // reasons (protective copy, etc) and continue checking below.
    if (topologyAttr.hasUnifiedMemory(sourceAffinityAttr, resultAffinityAttr) ||
        topologyAttr.hasTransparentAccess(sourceAffinityAttr,
                                          resultAffinityAttr)) {
      LLVM_DEBUG(llvm::dbgs() << "  + topology allows elision (unified memory "
                                 "or transparent access)\n");
      topologyAllowsElision = true;
    } else {
      // Topology requires an actual transfer - cannot elide.
      LLVM_DEBUG(llvm::dbgs()
                 << "  - topology requires transfer; cannot elide\n");
      return false;
    }
  }

  // Different affinity without topology info = real transfer operation.
  // These must be preserved as they represent data movement and we don't have
  // topology information to determine if they're needed.
  if (!topologyAllowsElision && transferOp.getSourceAffinityAttr() !=
                                    transferOp.getResultAffinityAttr()) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - transfer changes affinity and no topology info; cannot "
                  "elide\n");
    return false;
  }

  // Same affinity = either no-op (same type) or lifetime change.
  // Only elide if this is truly a no-op or if we can prove it's safe.
  const bool isSameType =
      (sourceType.getLifetime() == resultType.getLifetime());

  // If the immediate source is a block argument we have to look into the
  // analysis cache to see if it's been classified as a last use/by-value move.
  // However, same-type transfers are always safe to elide regardless of move
  // semantics, since they're true no-ops.
  if (auto arg = llvm::dyn_cast<BlockArgument>(transferOp.getSource())) {
    if (!analysis.isArgMoved(arg) && !isSameType) {
      LLVM_DEBUG(llvm::dbgs() << "  - transfer source is a by-ref arg and "
                                 "changes type; cannot elide\n");
      return false;
    }
    LLVM_DEBUG(
        llvm::dbgs()
        << "  ? transfer source is a by-value arg or same-type; may elide\n");
  }

  // If there's only one user of the source we know it's this transfer and can
  // bypass all the more expensive liveness analysis. Single-use transfers can
  // usually be elided (they compose with downstream transfers) but we need to
  // check if the result type is constrained by the user (e.g., return type).
  if (transferOp.getSource().hasOneUse()) {
    // Check if this transfer is the only thing preventing a type mismatch.
    // This happens when the source and result have different types and the
    // result type is required by the context (return, call, etc).
    if (!isSameType) {
      // For type-changing transfers we need to be conservative.
      // Many operations have specific type requirements (e.g.,
      // stream.async.load requires staging resources). Rather than try to
      // enumerate all such cases we conservatively preserve type-changing
      // single-use transfers unless:
      // - The user is another transfer (chain collapsing is safe)
      // - The user explicitly accepts * (unknown) resources
      for (auto &use : transferOp.getResult().getUses()) {
        Operation *user = use.getOwner();
        // Allow elision if the user is another transfer (chain collapsing).
        if (isa<IREE::Stream::AsyncTransferOp>(user)) {
          continue;
        }
        // Conservative: preserve type-changing transfers for other users.
        LLVM_DEBUG(llvm::dbgs()
                   << "  - type-changing transfer with non-transfer "
                      "user; cannot elide\n");
        return false;
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  + transfer source SSA value has one use; can elide\n");
    return true;
  }

  // If this is the last user of the source SSA value then we can elide the
  // transfer knowing that any mutations won't impact the source but only
  // for same-type transfers since type changes may encode required semantics.
  if (isSameType && analysis.isLastUser(transferOp.getSource(), transferOp)) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "  + transfer source use is the last and same type; can elide\n");
    return true;
  }

  // Special case: same-type transfers are always safe to elide regardless of
  // multiple users since they're true no-ops. The transfer result can be
  // replaced with the source without any semantic changes.
  if (isSameType) {
    LLVM_DEBUG(llvm::dbgs()
               << "  + same-type transfer (true no-op); can always elide\n");
    return true;
  }

  // Now we have multiple users, are not the last user, or have a type-changing
  // transfer. At this point all elision opportunities have been exhausted.
  // Type-changing transfers with multiple non-last users cannot be safely
  // elided as the type change encodes semantic information needed by those
  // users.
  LLVM_DEBUG(llvm::dbgs() << "  - transfer source cannot be elided\n");
  return false;
}

// Elides a stream.async.transfer op by replacing all uses with the source.
static void elideTransferOp(IREE::Stream::AsyncTransferOp transferOp) {
  transferOp.replaceAllUsesWith(transferOp.getSource());
  transferOp.erase();
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
    if (user == sliceOp) {
      continue;
    }
    if (auto accessOp = dyn_cast<IREE::Stream::AsyncAccessOpInterface>(user)) {
      // Async op consuming part of the resource. We can query it to see what
      // it's doing to its operands/results and filter to just the accesses of
      // the source value.
      accessOp.getAsyncAccessRanges(queryRanges);
      for (auto range : queryRanges) {
        if (range.resource == source) {
          consumerRanges.push_back(range);
        }
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
  if (matchPattern(lhs, m_Zero())) {
    return rhs;
  }
  if (matchPattern(rhs, m_Zero())) {
    return lhs;
  }
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
        return isa<IREE::Stream::ResourceType>(operand.getType());
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
  for (auto &use : sliceOp.getResult().getUses()) {
    consumers.push_back(std::make_pair(use.getOwner(), use.getOperandNumber()));
  }
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
// IREE::Stream::AsyncUpdateOp elision
//===----------------------------------------------------------------------===//

// Returns true if |user| is a tied operation that fully overwrites the region
// written by |updateOp|. When a tied operation's result aliases the update's
// target buffer, downstream reads might access the update region. This is only
// safe if the tied operation writes to the exact same region (fully
// overwriting the update's data).
static bool doesTiedOpFullyOverwriteUpdate(Operation *user,
                                           IREE::Stream::AsyncUpdateOp updateOp,
                                           Value target, Value result) {
  auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(user);
  if (!tiedOp) {
    return true; // Not a tied op, no aliasing concern.
  }

  // Check if any operand using our result is tied.
  for (auto &operand : user->getOpOperands()) {
    if (operand.get() != result) {
      continue;
    }
    if (!tiedOp.isOperandTied(operand.getOperandNumber())) {
      continue;
    }

    // Result is tied - the operation's result aliases our buffer.
    // Check if the operation fully overwrites our update region.
    auto accessOp = dyn_cast<IREE::Stream::AsyncAccessOpInterface>(user);
    if (!accessOp) {
      // Tied but no access info - conservatively assume it doesn't overwrite.
      return false;
    }

    SmallVector<AsyncAccessRange> ranges;
    accessOp.getAsyncAccessRanges(ranges);
    for (auto &range : ranges) {
      // Only check writes to our result/target.
      if (range.resource != result && range.resource != target) {
        continue;
      }
      if (range.isReadOnly()) {
        continue;
      }
      // Check if this write fully covers our update region.
      bool sameStart = (range.start == updateOp.getTargetOffset());
      bool sameEnd = (range.end == updateOp.getTargetEnd());
      if (sameStart && sameEnd) {
        return true; // Found a write that fully overwrites.
      }
    }

    // Tied operand but no write fully covers the update region.
    return false;
  }

  return true; // No tied operands using our result.
}

// Returns true if |updateOp| is safe to elide by proving the mutation has no
// observable effect. An update is not observable if no subsequent operation
// reads from the mutated region.
static bool isSafeToElideUpdateOp(IREE::Stream::AsyncUpdateOp updateOp,
                                  ElisionAnalysis &analysis) {
  LLVM_DEBUG({
    llvm::dbgs() << "isSafeToElideUpdateOp:\n";
    llvm::dbgs() << "  ";
    updateOp.print(llvm::dbgs(), analysis.getAsmState());
    llvm::dbgs() << "\n";
  });

  // Check if target (or its base through tied chains) is a block argument
  // with reference semantics. If passed by-reference, mutation might be
  // observable to caller.
  Value target = updateOp.getTarget();
  Value targetBase = IREE::Util::TiedOpInterface::findTiedBaseValue(target);
  if (auto arg = llvm::dyn_cast<BlockArgument>(targetBase)) {
    if (!analysis.isArgMoved(arg)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  - update target traces to by-ref arg; cannot elide\n");
      return false;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  ? update target traces to by-value arg; may elide\n");
  }

  // Verify target has no other users that might observe mutation through
  // the shared underlying buffer. In well-formed CoW IR, target should
  // only be used by this update, but we check defensively.
  if (!target.hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs() << "  - target has multiple users; cannot elide\n");
    return false;
  }

  // Build access range for the update operation.
  // The update writes to target[offset, end), and result aliases target.
  // We use result as the resource since we're checking against users of result.
  Value result = updateOp.getResult();
  AsyncAccessRange updateRange;
  updateRange.access = ResourceAccessBitfield::Write;
  updateRange.resource = result; // Use result, not target, for alias matching.
  updateRange.start = updateOp.getTargetOffset();
  updateRange.end = updateOp.getTargetEnd();
  updateRange.length = updateOp.getUpdateSize();

  // Check all uses of the result to see if they read the mutated region.
  SmallVector<AsyncAccessRange> queryRanges;
  LLVM_DEBUG(llvm::dbgs() << "  Checking users of update result\n");
  for (auto user : result.getUsers()) {
    LLVM_DEBUG({
      llvm::dbgs() << "  Examining user: ";
      user->print(llvm::dbgs(), analysis.getAsmState());
      llvm::dbgs() << "\n";
    });

    // Check if result is used as target of another update.
    // If so, that update's result will alias our written region, and
    // downstream reads of that result will read our region indirectly.
    if (auto nextUpdate = dyn_cast<IREE::Stream::AsyncUpdateOp>(user)) {
      if (nextUpdate.getTarget() == result) {
        // Check if the next update fully overwrites our write region.
        // This handles patterns like: alloca -> update[0:4] -> update[0:4]
        // where the second update completely replaces the first.
        bool sameStart =
            (nextUpdate.getTargetOffset() == updateOp.getTargetOffset());
        bool sameEnd = (nextUpdate.getTargetEnd() == updateOp.getTargetEnd());
        if (sameStart && sameEnd) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  ? chained update fully overwrites same region; "
                        "continuing analysis\n");
          // This chained update overwrites our write - safe to continue
          // checking other users.
          continue;
        }
        // Different regions or can't prove full coverage - be conservative.
        LLVM_DEBUG(
            llvm::dbgs()
            << "  - result used as target of chained update; cannot elide\n");
        return false;
      }
    }

    // Check if result is used by a tied operation. When a tied operation's
    // result aliases our buffer, downstream reads of that result might access
    // our update region even if the operation itself writes to a different
    // region. Block elision unless the operation fully overwrites our update
    // region.
    if (!doesTiedOpFullyOverwriteUpdate(user, updateOp, target, result)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  - result used by tied op; cannot elide (downstream "
                    "reads may access update region through tied result)\n");
      return false;
    }

    if (auto accessOp = dyn_cast<IREE::Stream::AsyncAccessOpInterface>(user)) {
      // Query the user operation for its access ranges.
      accessOp.getAsyncAccessRanges(queryRanges);
      LLVM_DEBUG(llvm::dbgs() << "    Got " << queryRanges.size()
                              << " access ranges from user\n");
      for (auto &userRange : queryRanges) {
        LLVM_DEBUG({
          llvm::dbgs() << "    Range: ";
          userRange.print(llvm::dbgs(), analysis.getAsmState());
          llvm::dbgs() << "\n";
          llvm::dbgs() << "      resource == target? "
                       << (userRange.resource == target) << "\n";
          llvm::dbgs() << "      resource == result? "
                       << (userRange.resource == result) << "\n";
        });
        // Only check accesses to the same resource.
        // Note: result aliases target (tied operand), so check both.
        if (userRange.resource != target && userRange.resource != result) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    Skipping range - different resource\n");
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "    Checking overlap...\n");
        // TODO(benvanik): use integer range analysis (Presburger) to prove
        // disjoint constant ranges don't overlap. Currently we conservatively
        // assume overlap unless start == end (adjacent) or different resources.
        // Check for overlap with the updated region.
        if (IREE::Stream::AsyncAccessRange::mayOverlap(updateRange,
                                                       userRange)) {
          LLVM_DEBUG(llvm::dbgs() << "    Ranges overlap!\n");
          // If the overlapping access has a read component, mutation is
          // observable. Only pure writes (overwrites) are safe.
          if (!userRange.isWriteOnly()) {
            LLVM_DEBUG(
                {
                  llvm::dbgs()
                      << "  - user reads from mutated region; cannot elide\n";
                  llvm::dbgs() << "    v update ";
                  updateRange.print(llvm::dbgs(), analysis.getAsmState());
                  llvm::dbgs() << "\n";
                  llvm::dbgs() << "    ^ conflict ";
                  userRange.print(llvm::dbgs(), analysis.getAsmState());
                  llvm::dbgs() << "\n";
                });
            return false;
          }
          // If the overlapping access is write-only, mutation is overwritten
          // (safe to elide).
        }
      }
      queryRanges.clear();
    } else {
      // Unknown user - conservatively assume it reads the mutation.
      LLVM_DEBUG({
        llvm::dbgs()
            << "  - analysis failure on unhandled user of update result:\n";
        user->print(llvm::dbgs(), analysis.getAsmState());
      });
      return false;
    }
  }

  // No overlapping reads found - mutation is not observable.
  LLVM_DEBUG(llvm::dbgs() << "  + update can be elided\n");
  return true;
}

// Elides an update operation by proving the mutation has no observable effect.
static void elideUpdateOp(IREE::Stream::AsyncUpdateOp updateOp) {
  Value source = updateOp.getUpdate();
  Value target = updateOp.getTarget();
  Value result = updateOp.getResult();

  // If this is a full-buffer update with matching types, replace with source.
  // This handles patterns like: alloca -> full-buffer update -> use
  // where we can bypass the staging buffer entirely.
  if (updateOp.getUpdateSize() == updateOp.getTargetSize() &&
      source.getType() == result.getType()) {
    // Replace result with source (bypass staging buffer).
    result.replaceAllUsesWith(source);
  } else {
    // Replace result with target (mutation didn't matter).
    result.replaceAllUsesWith(target);
  }
  updateOp.erase();
}

//===----------------------------------------------------------------------===//
// --iree-stream-elide-async-copies
//===----------------------------------------------------------------------===//

// Results from trying to elide copies in a region.
struct ElisionResults {
  unsigned clonesElided = 0;
  unsigned transfersElided = 0;
  unsigned slicesElided = 0;
  unsigned updatesElided = 0;
  bool didChange() const {
    return clonesElided > 0 || transfersElided > 0 || slicesElided > 0 ||
           updatesElided > 0;
  }
  void add(ElisionResults &other) {
    clonesElided += other.clonesElided;
    transfersElided += other.transfersElided;
    slicesElided += other.slicesElided;
    updatesElided += other.updatesElided;
  }
};

// Tries to elide copies nested within |region| when safe.
// Returns true if any ops were elided.
static ElisionResults tryElideAsyncCopiesInRegion(
    Region &region, ElisionAnalysis &analysis,
    IREE::Stream::AffinityTopologyAttrInterface topologyAttr) {
  ElisionResults results;
  for (auto &block : region) {
    block.walk([&](Operation *op) {
      return TypeSwitch<Operation *, WalkResult>(op)
          .Case<IREE::Stream::AsyncCloneOp>([&](auto cloneOp) {
            // First, try to elide lifetime-changing clones by propagating
            // the source lifetime through downstream ops.
            if (tryElideLifetimeChangingClone(cloneOp, analysis)) {
              ++results.clonesElided;
              return WalkResult::advance();
            }
            // Fall back to same-lifetime clone elision.
            if (isSafeToElideCloneOp(cloneOp, analysis)) {
              elideCloneOp(cloneOp);
              ++results.clonesElided;
            }
            return WalkResult::advance();
          })
          .Case<IREE::Stream::AsyncTransferOp>([&](auto transferOp) {
            if (isSafeToElideTransferOp(transferOp, analysis, topologyAttr)) {
              elideTransferOp(transferOp);
              ++results.transfersElided;
            }
            return WalkResult::advance();
          })
          .Case<IREE::Stream::AsyncSliceOp>([&](auto sliceOp) {
            if (isSafeToElideSliceOp(sliceOp, analysis)) {
              elideSliceOp(sliceOp);
              ++results.slicesElided;
            }
            return WalkResult::advance();
          })
          .Case<IREE::Stream::AsyncUpdateOp>([&](auto updateOp) {
            if (isSafeToElideUpdateOp(updateOp, analysis)) {
              elideUpdateOp(updateOp);
              ++results.updatesElided;
            }
            return WalkResult::advance();
          })
          .Default([&](auto *op) { return WalkResult::advance(); });
    });
  }
  return results;
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
    mlir::ModuleOp moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) {
      return;
    }

    // Get the topology attribute from the module (if present).
    // This enables topology-based transfer elision (unified memory, etc).
    //
    // TODO(benvanik): add a helper for this (we do this lookup in a few
    // places).
    auto topologyAttr =
        moduleOp->getAttrOfType<IREE::Stream::AffinityTopologyAttrInterface>(
            "stream.topology");

    // Track total elisions across all iterations.
    ElisionResults totalResults;

    // Try analyzing the program and eliding the unneeded copies until we reach
    // a fixed point (no more copies can be elided).
    unsigned maxIterationCount = 30;
    unsigned iterationCount = 0;
    for (; iterationCount < maxIterationCount; ++iterationCount) {
      // Perform whole-program analysis.
      // TODO(benvanik): reuse allocator across iterations.
      ElisionAnalysis analysis(moduleOp);
      if (failed(analysis.run())) {
        moduleOp.emitError() << "failed to solve for elision analysis";
        return signalPassFailure();
      }

      // Apply analysis by eliding all copies that are safe to elide.
      // If we can't elide any we'll consider the iteration complete and exit.
      ElisionResults iterationResults;
      for (auto callableOp : analysis.getTopLevelOps()) {
        if (auto *region = callableOp.getCallableRegion()) {
          ElisionResults regionResults =
              tryElideAsyncCopiesInRegion(*region, analysis, topologyAttr);
          iterationResults.add(regionResults);
        }
      }
      totalResults.add(iterationResults);
      if (!iterationResults.didChange()) {
        break; // quiesced
      }
    }

    // Update pass statistics.
    numClonesElided += totalResults.clonesElided;
    numTransfersElided += totalResults.transfersElided;
    numSlicesElided += totalResults.slicesElided;
    numIterations += iterationCount;

    if (iterationCount == maxIterationCount) {
      // If you find yourself hitting this we can evaluate increasing the
      // iteration count (if it would eventually converge) or whether we allow
      // this to happen without remarking. For now all our programs converge in
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
