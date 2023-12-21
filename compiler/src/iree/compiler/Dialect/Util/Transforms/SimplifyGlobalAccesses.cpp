// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-util-simplify-global-accesses"

namespace mlir::iree_compiler::IREE::Util {

namespace {

// Builds symbol ref set for all immutable globals in |moduleOp|.
static DenseSet<StringRef> gatherImmutableGlobals(mlir::ModuleOp moduleOp) {
  DenseSet<StringRef> set;
  for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
    if (!globalOp.isGlobalMutable()) {
      set.insert(globalOp.getGlobalName());
    }
  }
  return set;
}

// Hoists all loads of immutable globals in |funcOp| to the entry block.
// |immutableGlobals| is used for lookups of which globals are immutable.
static void hoistImmutableLoads(Region &region,
                                DenseSet<StringRef> &immutableGlobals) {
  // Since CSE of loads isn't a thing yet we perform a basic deduping here by
  // folding all subsequent loads into the first one found. This works only for
  // immutable globals as otherwise we'd have to ensure stores and
  // side-effects were properly observed.
  DenseMap<Attribute, Operation *> loadOps;
  auto *entryBlock = &region.getBlocks().front();
  Operation *lastEntryOp = nullptr;
  SmallVector<std::pair<Operation *, Operation *>> opReplacements;
  for (auto &block : region) {
    auto ops =
        llvm::to_vector<8>(block.getOps<IREE::Util::GlobalLoadOpInterface>());
    for (auto &op : ops) {
      if (!immutableGlobals.contains(op.getGlobalName()))
        continue;
      auto globalRef = llvm::cast<Attribute>(op.getGlobalAttr());
      auto it = loadOps.find(globalRef);
      if (it == loadOps.end()) {
        // Move to entry block; even if it's already there (so loads are
        // hoisted at the same time).
        LLVM_DEBUG(llvm::dbgs()
                   << "moving immutable global " << op.getGlobalName()
                   << " load to the entry block\n");
        if (lastEntryOp) {
          op->moveAfter(lastEntryOp);
        } else {
          op->moveBefore(entryBlock, entryBlock->begin());
        }
        loadOps[globalRef] = op;
        lastEntryOp = op;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "CSE'ing immutable global "
                                << op.getGlobalName() << "\n");
        opReplacements.push_back({op, it->getSecond()});
      }
    }
  }
  for (auto &replacement : opReplacements) {
    replacement.first->replaceAllUsesWith(replacement.second);
    replacement.first->erase();
  }
}

// Class to track the number of global accessors a particular operation has
// nested inside, per global. The map constructed by the analysis is used to
// track the number of globals accesses as they are hoisted out of nested
// regions.
class GlobalAccessor
    : public DFX::StateWrapper<DFX::PotentialCountsState<StringAttr>,
                               DFX::OperationElement> {
public:
  using BaseType = DFX::StateWrapper<DFX::PotentialCountsState<StringAttr>,
                                     DFX::OperationElement>;

  static GlobalAccessor &createForPosition(const Position &pos,
                                           DFX::Solver &solver) {
    return *(new (solver.getAllocator()) GlobalAccessor(pos));
  }

  const std::string getName() const override { return "GlobalAccessor"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  // Returns true if the global with the given name is known to be accessed by
  // this operation.
  bool isAssumedGlobalAccessor(StringAttr globalName) const {
    if (isValidState()) {
      return getAssumedMultiSet().contains(globalName);
    }
    return true;
  }

  const std::string getAsStr(AsmState &asmState) const override {
    std::string sizeStr =
        isValidState() ? std::to_string(getAssumedMultiSet().size()) : "ALL";
    return std::string("Global accessor: ") + sizeStr;
  }

private:
  explicit GlobalAccessor(const Position &pos) : BaseType(pos) {}
  // Initialize global (indirect) loads/stores and calls to external functions.
  void initializeOperation(Operation *op, DFX::Solver &solver) override {
    // Direct globals loads/stores only perform a single access of the
    // associated global.
    if (auto load = dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
      unionAssumed(load.getGlobalAttr().getAttr(), 1);
      indicateOptimisticFixpoint();
    } else if (auto store = dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
      unionAssumed(store.getGlobalAttr().getAttr(), 1);
      indicateOptimisticFixpoint();
    } else if (isa<IREE::Util::GlobalLoadIndirectOpInterface,
                   IREE::Util::GlobalStoreIndirectOpInterface>(op)) {
      // Indirect global accessors can access any global and need to be treated
      // as pessimistically as possible.
      indicatePessimisticFixpoint();
    } else if (auto callOp = dyn_cast<mlir::CallOpInterface>(op)) {
      auto callableOp = llvm::dyn_cast_if_present<mlir::CallableOpInterface>(
          callOp.resolveCallable(&solver.getExplorer().getSymbolTables()));
      // If we cannot resolve the callee of a call op, we block all global
      // movement around it.
      if (!callableOp) {
        indicatePessimisticFixpoint();
      } else if (callableOp
                     ->getParentWithTrait<OpTrait::IREE::Util::ObjectLike>()) {
        // Calls into executables are non-blocking.
        indicateOptimisticFixpoint();
      } else if (!callableOp.getCallableRegion() ||
                 callableOp.getCallableRegion()->empty()) {
        // No assumptions can be made about an external calls.
        indicatePessimisticFixpoint();
      } else {
        return;
      }
    } else if (op->hasTrait<OpTrait::IREE::Util::YieldPoint>()) {
      // Asynchronous yield points need to maintain global accessor order before
      // and after the yield.
      indicatePessimisticFixpoint();
    } else if (op->hasTrait<OpTrait::IREE::Util::ObjectLike>()) {
      // Object-like (i.e. *.executable) operations cannot access globals within
      // the module so we can resolve it optimistically.
      indicateOptimisticFixpoint();
    } else if (auto callableOp = dyn_cast<mlir::CallableOpInterface>(op)) {
      // External callable ops can potentially call back in to the current
      // module and thus we pessimistically assume it can access any global.
      if (!callableOp.getCallableRegion() ||
          callableOp.getCallableRegion()->empty()) {
        indicatePessimisticFixpoint();
      }
    } else if (!op->getNumRegions()) {
      // All other regionless operations are non-blocking.
      indicateOptimisticFixpoint();
    } else {
      return;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[simplify-globals] initialized global accessor ";
      op->print(llvm::dbgs(), solver.getAsmState());
      if (isValidState()) {
        llvm::dbgs() << ": " << getAssumedMultiSet().size() << "\n";
        for (auto &[globalName, c] : getAssumedMultiSet()) {
          llvm::dbgs() << "  " << globalName << ":  " << c << "\n";
        }
      } else {
        llvm::dbgs() << ": ALL GLOBALS\n";
      }
    });
  }

  // The only operation kinds we need to update are those with regions, for
  // example SCF and callable ops.
  ChangeStatus updateOperation(Operation *op, DFX::Solver &solver) override {
    // Callers inherit the accesses from its callee.
    if (auto callOp = dyn_cast<mlir::CallOpInterface>(op)) {
      auto callableOp =
          callOp.resolveCallable(&solver.getExplorer().getSymbolTables());
      assert(callableOp &&
             "Failed to find callee for potential global accessor");
      auto accessorFunction = solver.getElementFor<GlobalAccessor>(
          *this, Position::forOperation(callableOp), DFX::Resolution::REQUIRED);
      LLVM_DEBUG({
        llvm::dbgs() << "[simplify-globals] call: ";
        callOp.print(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "; ";
        accessorFunction.print(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      StateType newState = accessorFunction;
      return DFX::inheritStateAndIndicateChange(getState(), newState);
    }

    // At this point the only unhandled operations are those with regions. To
    // determine the current known number of global accessors, sum the accesses
    // of the ops within this immediate region.
    assert(op->getNumRegions() > 0 && "updating regionless global accessor");
    StateType bestState = getBestState();
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &containedOp : block) {
          auto accessor = solver.getElementFor<GlobalAccessor>(
              *this, Position::forOperation(&containedOp),
              DFX::Resolution::REQUIRED);
          bestState ^= accessor;
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[simplify-globals] region op: ";
      op->print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "; " << bestState;
      llvm::dbgs() << "\n";
    });
    return DFX::inheritStateAndIndicateChange(getState(), bestState);
  }

  friend class DFX::Solver;
};
const char GlobalAccessor::ID = 0;

class GlobalAccessorAnalysis {
public:
  explicit GlobalAccessorAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::RECURSE,
                 OpPrintingFlags().elideLargeElementsAttrs().skipRegions()),
        solver(explorer, allocator) {
    explorer.initialize();

    assert(rootOp->getNumRegions() == 1 && "expected module-like root op");

    // Populate the list of globals to optimize. This is primarily to force the
    // order in which globals are processed to be deterministic, reducing minor
    // ordering variations in the resulting IR.
    rootOp->walk([&](IREE::Util::GlobalOpInterface global) {
      globalNames.push_back(global.getGlobalName());
    });
  }

  AsmState &getAsmState() { return solver.getAsmState(); }
  Explorer &getExplorer() { return explorer; }

  // Runs analysis and populates the state cache.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run() {
    // Initialize all potential accessors.
    explorer.walk([&](Operation *op) {
      if (isa<IREE::Util::GlobalLoadOpInterface>(op) ||
          isa<IREE::Util::GlobalStoreOpInterface>(op) ||
          isa<IREE::Util::GlobalLoadIndirectOpInterface,
              IREE::Util::GlobalStoreIndirectOpInterface>(op) ||
          isa<mlir::CallOpInterface>(op) ||
          isa<mlir::CallableOpInterface>(op) ||
          op->hasTrait<OpTrait::IREE::Util::YieldPoint>() ||
          op->hasTrait<OpTrait::IREE::Util::ObjectLike>() ||
          op->getNumRegions() > 0) {
        solver.getOrCreateElementFor<GlobalAccessor>(
            Position::forOperation(op));
      }
      return WalkResult::advance();
    });

    // Run solver to completion.
    auto result = solver.run();
    LLVM_DEBUG(solver.print(llvm::dbgs()));

    // Populate the accessor map and other cached state for use when
    // propagating direct accessors.
    if (succeeded(result)) {
      solver.forEachElement([&](const DFX::AbstractElement *element) {
        // We only use a single element type in this analysis so this cast is
        // always valid.
        auto accessor = static_cast<const GlobalAccessor *>(element);
        Operation *op = &accessor->getOperation();

        if (accessor->isValidState()) {
          auto accessorMap = accessor->getAssumedMultiSet();
          // If the map is empty, this operation has no hazards for accessor
          // motion. Because this is true for most operations (i.e. anything
          // regionless and not specifically marked with a relevant side effect)
          // we elide the map in such cases.
          if (!accessorMap.empty()) {
            accessorCounts[op] = accessorMap;
            globalAccessors.insert(op);
          }
        } else {
          // An invalid state means the solver either couldn't resolve the
          // accessors for this operation, or it reached a pessimistic fixed
          // point. In such cases treat the operation as always blocking.
          globalAccessors.insert(op);
        }
        if (auto store = dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
          bucketedAccessors[store.getGlobalAttr().getAttr()].insert(store);
        } else if (auto load =
                       dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
          bucketedAccessors[load.getGlobalAttr().getAttr()].insert(load);
        } else if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
          compositeAccessors.push_back(loop);
        }
        return;
      });
    }
    return result;
  }

  // Process the given load by moving up to the nearest blocking operation and
  // performing RAR or RAW forwarding if the blocking operation is a load/store.
  bool processGlobalLoad(IREE::Util::GlobalLoadOpInterface load);
  // Process the given store by moving down to the nearest blocking operation
  // and performing WAW forwarding if the blocking operation is a store.
  bool processGlobalStore(IREE::Util::GlobalStoreOpInterface store);
  // Perform LICM style hoisting of single globals accessors.
  bool processLoopLikeOp(LoopLikeOpInterface loop);
  // Process the given loop, hoisting redundant accessors. On success this
  // returns the newly created loop and replaces the accessor map entry for
  // the original loop with the new one.
  Operation *hoistRedundantAccessors(LoopLikeOpInterface loop);
  // Process all composite accessors (e.g. scf.for).
  bool processCompositeAccessors();
  // Process all of loads/stores for the given global.
  bool processDirectAccessors(StringAttr globalName);
  // Process all accessors by bucket in |bucketedAccessors|.
  bool processGlobalAccessors();

private:
  // Move an operation right after the nearest blocking operation above |op|
  // within its block. An operation is blocking if it *may* access the global
  // referenced by |globalName|.
  void moveOpUpInBlock(Operation *op, StringAttr globalName);
  // Move an operation right before the nearest blocking operation below |op|
  // within its block. An operation is blocking if it *may* access the global
  // referenced by |globalName|.
  void moveOpDownInBlock(Operation *op, StringAttr globalName);

  StringAttr getGlobalName(Operation *op) {
    StringAttr globalName;
    if (auto store = dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
      globalName = store.getGlobalAttr().getAttr();
    } else if (auto load = dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
      globalName = load.getGlobalAttr().getAttr();
    } else {
      llvm_unreachable("Unhandled direct accessor type");
    }
    return globalName;
  }

  void decrementAccessorCount(StringAttr globalName, Operation *op,
                              int64_t i = 1) {
    if (accessorCounts.contains(op)) {
      auto &map = accessorCounts[op];
      DFX::SaturatedInteger newVal = map[globalName] - i;
      LLVM_DEBUG({
        llvm::dbgs() << "Decrementing accessor count of\n";
        op->print(llvm::dbgs(), getAsmState());
        llvm::dbgs() << "\n";
        llvm::dbgs() << "for global " << globalName << " by " << i << " to "
                     << newVal << "\n";
      });
      if (newVal == 0) {
        map.erase(globalName);
      } else {
        map[globalName] = newVal;
      }
    }
  }

  void eraseAccessor(StringAttr globalName, Operation *op) {
    bucketedAccessors[globalName].erase(op);
    Operation *parent = op->getParentOp();
    // Decrement the accessor count for this global and propagate to all
    // parents.
    // TODO: This does not properly track accessor changes across function
    // bounaries, however without hoisting across function boundaries, this
    // has no effect on the current reached fixedpoint given that accessors
    // are only ever removed one at a time from a pair within the same region.
    while (parent) {
      decrementAccessorCount(globalName, parent);
      parent = parent->getParentOp();
    }
    op->erase();
  }

  DFX::SaturatedInteger getAccessorCount(Operation *op, StringAttr globalName) {
    // If not an accessor, the number of accesses is necessarily zero.
    if (!globalAccessors.contains(op)) {
      return DFX::SaturatedInteger{false, 0};
    }
    // The counts are unspecified for invalid nodes, meaning an infinite number
    // of accesses for any global.
    if (!accessorCounts.contains(op)) {
      return DFX::SaturatedInteger{true, 0};
    }
    // If there is no entry for this global by this accessor, then this accessor
    // does not access this global.
    if (!accessorCounts[op].contains(globalName)) {
      return DFX::SaturatedInteger{false, 0};
    }
    return accessorCounts[op][globalName];
  }

  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;

  // Set of all potentially blocking operations.
  DenseSet<Operation *> globalAccessors;
  // Cached map from operations to the globals they *may* access. For example:
  //
  // |globalAccessors| = {opA, opB, opC}
  // |accessorCounts| = {
  //    opA: {
  //      varA:  2 accessors
  //    }
  //    opB: {
  //      varB: "inf" accessors
  //    }
  // }
  //
  // This implies |opA| has 2 accessors to global |varA|, |opB| is always
  // blocking for global |varB|, and |opC| has no accessor information about
  // it and thus blocks all motion. This map is automatically updated as ops
  // are removed during propagation.
  DenseMap<Operation *, DenseMap<StringAttr, DFX::SaturatedInteger>>
      accessorCounts;

  SmallVector<StringAttr> globalNames;
  DenseMap<StringAttr, DenseSet<Operation *>> bucketedAccessors;
  SmallVector<Operation *> compositeAccessors;
};

void GlobalAccessorAnalysis::moveOpUpInBlock(Operation *op,
                                             StringAttr globalName) {
  // Find the earliest node that does not block op motion then move before it.
  mlir::Operation *earliestValidNode = op;
  while (earliestValidNode->getPrevNode()) {
    auto prev = earliestValidNode->getPrevNode();
    // If the accessor is invalid or is known to access the given global,
    // block propagation through that operation.
    if (getAccessorCount(prev, globalName) != 0) {
      break;
    }
    earliestValidNode = prev;
  }
  if (earliestValidNode != op) {
    op->moveBefore(earliestValidNode);
    LLVM_DEBUG({
      llvm::dbgs() << "Moving op: ";
      op->print(llvm::dbgs(), getAsmState());
      llvm::dbgs() << "\n";
      llvm::dbgs() << "before: ";
      earliestValidNode->print(llvm::dbgs(), getAsmState());
      llvm::dbgs() << "\n";
    });
  }
}

void GlobalAccessorAnalysis::moveOpDownInBlock(Operation *op,
                                               StringAttr globalName) {
  // Find the latest node that does not block op motion then move after it.
  mlir::Operation *latestValidNode = op;
  while (!latestValidNode->getNextNode()->hasTrait<OpTrait::IsTerminator>()) {
    auto next = latestValidNode->getNextNode();
    // Check if the accessor might access the given global.
    if (getAccessorCount(next, globalName) != 0) {
      break;
    }
    latestValidNode = next;
  }
  if (latestValidNode != op) {
    op->moveAfter(latestValidNode);
    LLVM_DEBUG({
      llvm::dbgs() << "Moving op: ";
      op->print(llvm::dbgs(), getAsmState());
      llvm::dbgs() << "\n";
      llvm::dbgs() << "after: ";
      latestValidNode->print(llvm::dbgs(), getAsmState());
      llvm::dbgs() << "\n";
    });
  }
}

bool GlobalAccessorAnalysis::processGlobalLoad(
    IREE::Util::GlobalLoadOpInterface load) {
  moveOpUpInBlock(load, load.getGlobalAttr().getAttr());
  Operation *prev = load->getPrevNode();
  if (!prev) {
    return false;
  }

  bool changedLoad = false;
  if (auto store = dyn_cast<IREE::Util::GlobalStoreOpInterface>(prev)) {
    // RAW - forward the stored global to the following use.
    assert(store.getGlobalName() == load.getGlobalName() &&
           "Global RAW forwarding different globals");
    auto storedValue = store.getStoredGlobalValue();
    LLVM_DEBUG({
      llvm::dbgs() << "RAW: replacing load with previous store value:\n";
      load->print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "->\n" << storedValue;
    });
    load->replaceAllUsesWith(ValueRange{storedValue});
    eraseAccessor(load.getGlobalAttr().getAttr(), load);
    changedLoad = true;
  } else if (isa<IREE::Util::GlobalLoadOpInterface>(prev)) {
    // RAR - forward the loaded global to the following use.
    LLVM_DEBUG({
      llvm::dbgs() << "RAR: replacing subsequent load with op:\n";
      load->print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "->\n";
      prev->print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    load->replaceAllUsesWith(prev);
    eraseAccessor(load.getGlobalAttr().getAttr(), load);
    changedLoad = true;
  }
  return changedLoad;
}

bool GlobalAccessorAnalysis::processGlobalStore(
    IREE::Util::GlobalStoreOpInterface store) {
  moveOpDownInBlock(store, store.getGlobalAttr().getAttr());
  Operation *next = store->getNextNode();
  if (next->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }

  bool changedStore = false;
  if (auto nextStore = dyn_cast<IREE::Util::GlobalStoreOpInterface>(next)) {
    // WAW - remove the first store.
    LLVM_DEBUG({
      llvm::dbgs() << "WAW: erasing source op:\n";
      store->print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\nand keeping subsequent op:\n";
      nextStore->print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    eraseAccessor(store.getGlobalAttr().getAttr(), store);
    changedStore = true;
  }
  return changedStore;
}

bool GlobalAccessorAnalysis::processLoopLikeOp(LoopLikeOpInterface loop) {
  return moveLoopInvariantCode(
      loop.getLoopRegions(),
      [&](Value value, Region *) { return loop.isDefinedOutsideOfLoop(value); },
      [&](Operation *op, Region *) {
        if (auto store = dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
          return getAccessorCount(loop, store.getGlobalAttr().getAttr()) == 1;
        } else if (auto load =
                       dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
          return getAccessorCount(loop, load.getGlobalAttr().getAttr()) == 1;
        }
        return false;
      },
      [&](Operation *op, Region *) {
        StringAttr globalName = getGlobalName(op);
        decrementAccessorCount(globalName, loop);
        LLVM_DEBUG({
          llvm::dbgs() << "Hoisting invariant accessor\n";
          op->print(llvm::dbgs(), solver.getAsmState());
          llvm::dbgs() << "of loop:\n";
          loop->print(llvm::dbgs(), solver.getAsmState());
          llvm::dbgs() << "\n";
        });
        loop.moveOutOfLoop(op);
      });
}

Operation *
GlobalAccessorAnalysis::hoistRedundantAccessors(LoopLikeOpInterface loop) {
  auto regions = loop.getLoopRegions();
  // To avoid scanning the loop regions each iteration for all hoistable
  // accessors, restrict to single block region loops and hoist accessors only
  // when at the boundaries of the block.
  if (regions.size() != 1 || !regions[0]->hasOneBlock()) {
    return loop;
  }
  auto &body = regions[0]->front();
  Operation *terminator = body.getTerminator();
  Operation *lastOp = terminator->getPrevNode();
  if (!lastOp) {
    return loop;
  }
  auto maybeStore = dyn_cast<IREE::Util::GlobalStoreOpInterface>(lastOp);
  auto maybeLoad = dyn_cast<IREE::Util::GlobalLoadOpInterface>(body.front());
  if (!maybeStore || !maybeLoad) {
    return loop;
  }
  if (maybeStore.getGlobalAttr() != maybeLoad.getGlobalAttr()) {
    return loop;
  }
  StringAttr globalName = getGlobalName(maybeStore);
  if (getAccessorCount(loop, globalName) != 2) {
    return loop;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Hoisting redundant accessors\n";
    maybeLoad->print(llvm::dbgs(), solver.getAsmState());
    llvm::dbgs() << "\n";
    maybeStore->print(llvm::dbgs(), solver.getAsmState());
    llvm::dbgs() << "\n";
    llvm::dbgs() << "of loop:\n";
    loop->print(llvm::dbgs(), solver.getAsmState());
    llvm::dbgs() << "\n";
  });

  IRRewriter rewriter(loop.getContext());
  NewYieldValuesFn newYieldValuesFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> innerNewBBArgs) -> SmallVector<Value> {
    return {maybeStore.getStoredGlobalValue()};
  };
  FailureOr<LoopLikeOpInterface> maybeLoop = loop.replaceWithAdditionalYields(
      rewriter, maybeLoad.getLoadedGlobalValue(),
      /*replaceInitOperandUsesInLoop=*/true, newYieldValuesFn);
  if (failed(maybeLoop))
    return loop;
  auto newLoop = *maybeLoop;

  // Hoist the load/store ops and replace the store operand with the new loop
  // result.
  maybeLoad->moveBefore(newLoop);
  maybeStore->moveAfter(newLoop);
  OpResult loopResult = newLoop->getResults().back();
  maybeStore->setOperand(0, loopResult);

  globalAccessors.erase(loop);
  globalAccessors.insert(newLoop);
  accessorCounts[newLoop] = accessorCounts[loop];
  accessorCounts.erase(loop);
  decrementAccessorCount(globalName, newLoop, 2);
  return newLoop;
}

bool GlobalAccessorAnalysis::processDirectAccessors(StringAttr globalName) {
  if (!bucketedAccessors.contains(globalName)) {
    return false;
  }
  bool directAccessorsDidChange = false;
  auto accessors = bucketedAccessors[globalName];
  // Record the current set of loads/stores for this global.
  SmallVector<Operation *> directAccessors = llvm::to_vector(accessors);
  for (auto accessor : directAccessors) {
    // Process loads/stores one at a time, tracking whether any were removed.
    if (auto store = dyn_cast<IREE::Util::GlobalStoreOpInterface>(accessor)) {
      directAccessorsDidChange |= processGlobalStore(store);
    } else if (auto load =
                   dyn_cast<IREE::Util::GlobalLoadOpInterface>(accessor)) {
      directAccessorsDidChange |= processGlobalLoad(load);
    } else {
      llvm_unreachable("unexpected direct accessor type");
    }
  }
  return directAccessorsDidChange;
}

bool GlobalAccessorAnalysis::processCompositeAccessors() {
  bool compositeAccessorsDidChange = false;
  for (int64_t i = 0, e = compositeAccessors.size(); i < e; ++i) {
    Operation *accessor = compositeAccessors[i];
    if (auto loop = dyn_cast<LoopLikeOpInterface>(accessor)) {
      compositeAccessorsDidChange |= processLoopLikeOp(loop);
      auto newLoop = hoistRedundantAccessors(loop);
      compositeAccessorsDidChange |= newLoop != accessor;
      // Replace with the new loop.
      compositeAccessors[i] = newLoop;
    }
  }
  return compositeAccessorsDidChange;
}

bool GlobalAccessorAnalysis::processGlobalAccessors() {
  bool globalAccessorsDidChange = false;
  for (auto globalName : globalNames) {
    globalAccessorsDidChange |= processDirectAccessors(globalName);
    globalAccessorsDidChange |= processCompositeAccessors();
  }
  return globalAccessorsDidChange;
}

}; // namespace

namespace {

class SimplifyGlobalAccessesPass
    : public SimplifyGlobalAccessesBase<SimplifyGlobalAccessesPass> {
public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    // Build a set of all immutable globals for fast lookup.
    DenseSet<StringRef> immutableGlobals = gatherImmutableGlobals(moduleOp);

    // Hoist immutable globals first. These have no hazards and don't care
    // about control flow - like `constant` - so getting them handled first
    // avoids the need for us to do the full analysis.
    moduleOp.walk([&](CallableOpInterface callableOp) {
      if (!callableOp.getCallableRegion() ||
          callableOp.getCallableRegion()->empty()) {
        return;
      }
      auto &region = *callableOp.getCallableRegion();
      // Skip initializers as we might be initializing the immutable global. In
      // such cases we fall back to treating it like normal global accessors.
      if (region.getParentOfType<IREE::Util::InitializerOp>()) {
        return;
      }
      hoistImmutableLoads(region, immutableGlobals);
    });

    // This runs an analysis to count the number of unique accessors per region
    // operation.  Execution order of direct accesses relative to external
    // calls, indirect accessors, and asynchronous yields must remain invariant.
    // Thus those operations block all nearby motion.
    // The result of the analysis caches a map describing the number of unique
    // direct accessors each operation contains. This is used to inform the
    // propagation logic of whether an operation blocks motion.
    GlobalAccessorAnalysis analysis(moduleOp);
    if (failed(analysis.run())) {
      moduleOp.emitError() << "failed to solve for global accessors";
      return signalPassFailure();
    }

    while (analysis.processGlobalAccessors()) {
      // Propagate global accessors to a fixed point.
      // 1. Collect the current set of direct loads/stores for a single global.
      // 2. One at a time, move loads up just above the nearest non-blocking
      //    operation, and stores just after the latest non-blocking op. Note
      //    that loads/stores to the same global always block each other.
      //   a. Attempt to resolve various RAW/WAW/WAR folding rules immediately
      //      after motion. When an operation is removed, the access counts of
      //      all parent operations are decremented.
      // 3. Process all composite accessors (e.g. loop like/region ops).
      // 4. Repeat for each global.
      // 5. Repeat until no operations are removed.
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createSimplifyGlobalAccessesPass() {
  return std::make_unique<SimplifyGlobalAccessesPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
