// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-util-explorer"

namespace mlir {
namespace iree_compiler {

static StringRef getOpName(Operation *op) {
  auto symbol = dyn_cast<mlir::SymbolOpInterface>(op);
  return symbol ? symbol.getName() : op->getName().getStringRef();
}

static StringRef getRegionName(Region &region) {
  return getOpName(region.getParentOp());
}

// Returns the remapped successor operand index if the branch operand is
// passed to a successor (vs being used by the op itself, such as the cond_br
// condition).
static std::optional<unsigned> mapSuccessorOperand(BranchOpInterface branchOp,
                                                   unsigned successorIdx,
                                                   unsigned operandIdx) {
  // I don't know if there's a better way to do this - the interface doesn't
  // help.
  auto operandRange = branchOp.getSuccessorOperands(successorIdx);
  if (operandRange.empty())
    return std::nullopt;
  unsigned beginIdx =
      operandRange.getForwardedOperands().getBeginOperandIndex();
  if (operandIdx >= beginIdx && operandIdx < beginIdx + operandRange.size()) {
    // Covered.
    return {operandIdx - beginIdx};
  }
  return std::nullopt;
}

Explorer::Explorer(Operation *rootOp, TraversalAction defaultAction)
    : rootOp(rootOp),
      asmState(rootOp, OpPrintingFlags().elideLargeElementsAttrs()),
      callGraph(rootOp), defaultAction(defaultAction),
      analysisManager(rootOp, /*passInstrumentor=*/nullptr) {}

Explorer::~Explorer() = default;

TraversalAction Explorer::getTraversalAction(Operation *op) {
  auto opIt = opActions.find(op->getName());
  if (opIt != opActions.end())
    return opIt->second;
  auto *dialect = op->getDialect();
  if (!dialect) {
    // Unregistered dialect/op - ignore.
    // TODO(benvanik): fail traversal with INCOMPLETE? this is only used in
    // tests today and we don't generally allow unknown ops inside of IREE
    // dialects where we use this.
    LLVM_DEBUG(llvm::dbgs() << "  -- ignoring unregistered dialect op "
                            << op->getName() << "\n");
    return TraversalAction::IGNORE;
  }
  auto dialectIt = dialectActions.find(dialect->getNamespace());
  if (dialectIt != dialectActions.end())
    return dialectIt->second;
  return defaultAction;
}

void Explorer::setDialectAction(StringRef dialectNamespace,
                                TraversalAction action) {
  dialectActions[dialectNamespace] = action;
}

void Explorer::setOpAction(OperationName op, TraversalAction action) {
  opActions[op] = action;
}

void Explorer::initialize() {
  initializeGlobalInfos();
  initializeInverseCallGraph();
}

// SymbolTable::getSymbolUses always walks the IR even when specifying the
// symbol; to prevent us needing to walk the module O(globals) times we eagerly
// initialize the cache here. This can hurt if globals are never accessed, but
// that's rare (globals store model parameters and such even in simple cases).
void Explorer::initializeGlobalInfos() {
  auto *symbolTableOp = SymbolTable::getNearestSymbolTable(rootOp);
  auto &symbolTable = symbolTables.getSymbolTable(symbolTableOp);

  // TODO(benvanik): filter the use list by traversal actions; where this runs
  // today we don't yet have the actions specified so we can't.

  auto allUses = symbolTable.getSymbolUses(&symbolTableOp->getRegion(0));
  if (!allUses.has_value())
    return;
  for (auto use : allUses.value()) {
    auto *symbolOp =
        symbolTable.lookupNearestSymbolFrom(use.getUser(), use.getSymbolRef());
    if (!isa_and_nonnull<IREE::Util::GlobalOpInterface>(symbolOp))
      continue;
    auto &globalInfo = globalInfos[symbolOp];
    globalInfo.op = cast<IREE::Util::GlobalOpInterface>(symbolOp);
    if (isa<IREE::Util::GlobalAddressOpInterface>(use.getUser())) {
      globalInfo.isIndirect = true;
    } else {
      globalInfo.uses.push_back(use.getUser());
    }
  }
}

// CallGraph is outgoing edges only but when performing backward traversal we
// want incoming edges as well. It's unfortunate we have to rerun effectively
// the same calculation over again. Maybe there's a way to use all that
// GraphTraits goo to do this, but I don't know it.
void Explorer::initializeInverseCallGraph() {
  rootOp->walk([&](CallOpInterface callOp) {
    if (callOp.getCallableForCallee().is<Value>()) {
      // Indirect calls can't be tracked in the call graph, so ensure we mark
      // the incomplete flag so that any call graph queries return
      // TraversalResult::INCOMPLETE.
      isCallGraphIncomplete = true;
    } else {
      auto *node = callGraph.resolveCallable(callOp, symbolTables);
      if (!node->isExternal()) {
        callGraphInv[node->getCallableRegion()].push_back(callOp);
      }
    }
  });
}

const Explorer::GlobalInfo *
Explorer::getGlobalInfo(IREE::Util::GlobalOpInterface globalOp) {
  auto it = globalInfos.find(globalOp);
  if (it == globalInfos.end())
    return nullptr;
  return &it->second;
}

const Explorer::GlobalInfo *Explorer::queryGlobalInfoFrom(StringRef globalName,
                                                          Operation *from) {
  auto *symbolTableOp = SymbolTable::getNearestSymbolTable(from);
  auto &symbolTable = symbolTables.getSymbolTable(symbolTableOp);
  auto op = symbolTable.lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(
      from, StringAttr::get(from->getContext(), globalName));
  if (!op)
    return nullptr;
  auto it = globalInfos.find(op);
  if (it == globalInfos.end())
    return nullptr;
  return &it->second;
}

void Explorer::forEachGlobal(std::function<void(const GlobalInfo *)> fn) {
  for (auto it : globalInfos) {
    fn(&it.second);
  }
}

bool Explorer::mayValuesAlias(Value a, Value b) {
  if (a == b)
    return true;
  bool mayAlias = false;
  auto traversalResult = walkTransitiveUses(a, [&](OpOperand &value) {
    mayAlias = value.get() == b;
    return mayAlias ? WalkResult::interrupt() : WalkResult::advance();
  });
  return mayAlias || traversalResult == TraversalResult::INCOMPLETE;
}

bool Explorer::mayBeUsedBy(Value value, Operation *user) {
  bool mayBeUsed = false;
  auto traversalResult = walkTransitiveUses(value, [&](OpOperand &use) {
    mayBeUsed = use.getOwner() == user;
    return mayBeUsed ? WalkResult::interrupt() : WalkResult::advance();
  });
  return mayBeUsed || traversalResult == TraversalResult::INCOMPLETE;
}

TraversalResult Explorer::walk(OperationWalkFn fn) {
  LLVM_DEBUG(llvm::dbgs() << "[[ Explorer::walk ]]\n");
  TraversalResult result = TraversalResult::COMPLETE;

  for (auto &scc : llvm::make_range(llvm::scc_begin(&callGraph),
                                    llvm::scc_end(&callGraph))) {
    LLVM_DEBUG(llvm::dbgs()
               << "? entering scc slice with " << scc.size() << " callables\n");
    for (auto *node : scc) {
      if (node->isExternal())
        continue;

      // Ensure we want to step into this region.
      // Note that SCC returns every function like in the whole program,
      // where we only care about ones rooted in our rootOp.
      auto &callableRegion = *node->getCallableRegion();
      auto *callableOp = callableRegion.getParentOp();
      auto action = getTraversalAction(callableOp);
      if (action == TraversalAction::IGNORE)
        continue;
      bool validInPlace = true;
      for (auto *parentOp = callableOp->getParentOp(); parentOp != rootOp;
           parentOp = parentOp->getParentOp()) {
        if (getTraversalAction(parentOp) != TraversalAction::RECURSE) {
          validInPlace = false;
          break;
        }
      }
      if (!validInPlace) {
        LLVM_DEBUG(llvm::dbgs() << "  -- ignoring callable region @"
                                << getRegionName(callableRegion) << "\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "   + entering callable region @"
                              << getRegionName(callableRegion) << "\n");
      auto emitResult = recursiveWalk(callableOp, fn);
      if (emitResult.wasInterrupted())
        break;
      if (emitResult.wasSkipped())
        continue;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "<< Explorer::walk >> is " << result << "\n");
  return result;
}

WalkResult Explorer::recursiveWalk(Operation *parentOp,
                                   const OperationWalkFn &fn) {
  auto parentAction = getTraversalAction(parentOp);
  if (parentAction == TraversalAction::IGNORE) {
    LLVM_DEBUG(llvm::dbgs()
               << "  -- ignoring op " << getOpName(parentOp) << "\n");
    return WalkResult::advance();
  }

  LLVM_DEBUG(llvm::dbgs() << "  == emitting op " << getOpName(parentOp)
                          << "\n");
  auto emitResult = fn(parentOp);
  if (emitResult.wasInterrupted())
    return WalkResult::interrupt();
  if (emitResult.wasSkipped())
    return WalkResult::advance();

  if (parentOp->getNumRegions() == 0 ||
      parentAction != TraversalAction::RECURSE) {
    return WalkResult::advance();
  }

  LLVM_DEBUG(llvm::dbgs() << "   + recursing into op " << getOpName(parentOp)
                          << "\n");
  for (auto &region : parentOp->getRegions()) {
    // TODO(benvanik): walk in topological order? or scc?
    for (auto &block : region.getBlocks()) {
      for (auto &op : block) {
        auto opResult = recursiveWalk(&op, fn);
        if (opResult.wasInterrupted())
          return WalkResult::interrupt();
      }
    }
  }
  return WalkResult::advance();
}

TraversalResult Explorer::walkValues(ValueWalkFn fn) {
  LLVM_DEBUG(llvm::dbgs() << "[[ Explorer::walkValues ]]\n");
  TraversalResult result = TraversalResult::COMPLETE;

  DenseSet<Value> visitedValues;
  for (auto &scc : llvm::make_range(llvm::scc_begin(&callGraph),
                                    llvm::scc_end(&callGraph))) {
    LLVM_DEBUG(llvm::dbgs()
               << "? entering scc slice with " << scc.size() << " callables\n");
    for (auto *node : scc) {
      if (node->isExternal())
        continue;

      // Ensure we want to step into this region.
      // Note that SCC returns every function like in the whole program,
      // where we only care about ones rooted in our rootOp.
      auto &callableRegion = *node->getCallableRegion();
      auto *callableOp = callableRegion.getParentOp();
      auto action = getTraversalAction(callableOp);
      if (action == TraversalAction::IGNORE)
        continue;
      bool validInPlace = true;
      for (auto *parentOp = callableOp->getParentOp(); parentOp != rootOp;
           parentOp = parentOp->getParentOp()) {
        if (getTraversalAction(parentOp) != TraversalAction::RECURSE) {
          validInPlace = false;
          break;
        }
      }
      if (!validInPlace) {
        LLVM_DEBUG(llvm::dbgs() << "  -- ignoring callable region @"
                                << getRegionName(callableRegion) << "\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "   + entering callable region @"
                              << getRegionName(callableRegion) << "\n");
      auto emitResult = recursiveWalkValues(callableOp, visitedValues, fn);
      if (emitResult.wasInterrupted())
        break;
      if (emitResult.wasSkipped())
        continue;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "<< Explorer::walkValues >> is " << result
                          << "\n");
  return result;
}

TraversalResult Explorer::walkValues(Operation *op, ValueWalkFn fn) {
  LLVM_DEBUG(llvm::dbgs() << "[[ Explorer::walkValues ]]\n");
  TraversalResult result = TraversalResult::COMPLETE;

  DenseSet<Value> visitedValues;
  recursiveWalkValues(op, visitedValues, fn);

  LLVM_DEBUG(llvm::dbgs() << "<< Explorer::walkValues >> is " << result
                          << "\n");
  return result;
}

WalkResult Explorer::recursiveWalkValues(Operation *parentOp,
                                         DenseSet<Value> &visitedValues,
                                         const ValueWalkFn &fn) {
  auto parentAction = getTraversalAction(parentOp);
  if (parentAction == TraversalAction::IGNORE) {
    LLVM_DEBUG(llvm::dbgs()
               << "  -- ignoring op " << getOpName(parentOp) << "\n");
    return WalkResult::advance();
  }

  if (parentOp->getNumResults() > 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "   + processing op results " << getOpName(parentOp) << "\n");
    for (auto result : parentOp->getResults()) {
      if (visitedValues.insert(result).second) {
        LLVM_DEBUG({
          llvm::dbgs() << "  == emitting value ";
          result.printAsOperand(llvm::dbgs(), asmState);
          llvm::dbgs() << "\n";
        });
        if (fn(result).wasInterrupted())
          return WalkResult::interrupt();
      }
    }
  }

  if (parentOp->getNumRegions() == 0 ||
      parentAction != TraversalAction::RECURSE) {
    return WalkResult::advance();
  }

  LLVM_DEBUG(llvm::dbgs() << "   + recursing into op " << getOpName(parentOp)
                          << "\n");
  for (auto &region : parentOp->getRegions()) {
    // TODO(benvanik): walk in topological order? or scc?
    for (auto &block : region.getBlocks()) {
      if (block.getNumArguments() > 0) {
        LLVM_DEBUG({
          llvm::dbgs() << "   + processing block ";
          block.printAsOperand(llvm::dbgs(), asmState);
          llvm::dbgs() << " arguments\n";
        });
        for (auto arg : block.getArguments()) {
          if (visitedValues.insert(arg).second) {
            LLVM_DEBUG({
              llvm::dbgs() << "  == emitting block arg ";
              arg.printAsOperand(llvm::dbgs(), asmState);
              llvm::dbgs() << "\n";
            });
            if (fn(arg).wasInterrupted())
              return WalkResult::interrupt();
          }
        }
      }
      for (auto &op : block) {
        auto opResult = recursiveWalkValues(&op, visitedValues, fn);
        if (opResult.wasInterrupted())
          return WalkResult::interrupt();
      }
    }
  }
  return WalkResult::advance();
}

TraversalResult
Explorer::walkIncomingCalls(CallableOpInterface callableOp,
                            std::function<WalkResult(CallOpInterface)> fn) {
  auto it = callGraphInv.find(callableOp.getCallableRegion());
  if (it != callGraphInv.end()) {
    for (auto &callOp : it->second) {
      if (fn(callOp).wasInterrupted())
        break;
    }
  }
  bool isPublic = false;
  if (auto symbolOp = dyn_cast<SymbolOpInterface>(callableOp.getOperation())) {
    isPublic = symbolOp.isPublic();
    LLVM_DEBUG({
      if (isPublic) {
        llvm::dbgs()
            << "  !! traversal incomplete due to public function-like op @"
            << symbolOp.getName() << "\n";
      }
    });
  }
  return isPublic || isCallGraphIncomplete ? TraversalResult::INCOMPLETE
                                           : TraversalResult::COMPLETE;
}

TraversalResult Explorer::walkReturnOps(Operation *parentOp,
                                        OperationWalkFn fn) {
  LLVM_DEBUG(llvm::dbgs() << "[[ Explorer::walkReturnOps ]]\n");
  if (getTraversalAction(parentOp) != TraversalAction::RECURSE) {
    LLVM_DEBUG(llvm::dbgs() << "  -- ignoring region op "
                            << parentOp->getName().getStringRef() << "\n");
    return TraversalResult::COMPLETE;
  }
  TraversalResult result = TraversalResult::COMPLETE;
  if (auto regionOp = dyn_cast<RegionBranchOpInterface>(parentOp)) {
    auto enumerateTerminatorOps = [&](Region &region) {
      for (auto &block : region) {
        if (auto *terminatorOp = block.getTerminator()) {
          // TODO(benvanik): ensure this terminator can return to parent? this
          // region op interface confuses me.
          LLVM_DEBUG({
            llvm::dbgs() << "  == emitting region branch terminator op ";
            terminatorOp->print(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          return fn(terminatorOp);
        }
      }
      return WalkResult::advance();
    };
    for (auto &region : regionOp->getRegions()) {
      if (enumerateTerminatorOps(region).wasInterrupted())
        break;
    }
  } else if (auto parentFuncOp =
                 llvm::dyn_cast<FunctionOpInterface>(parentOp)) {
    if (parentFuncOp->getNumRegions() == 0 ||
        parentFuncOp->getRegion(0).empty()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "  !! traversal incomplete due to external function-like op @"
          << cast<SymbolOpInterface>(parentOp).getName() << "\n");
      result |= TraversalResult::INCOMPLETE;
    } else {
      for (auto &region : parentFuncOp->getRegions()) {
        for (auto &block : region) {
          auto *terminatorOp = block.getTerminator();
          if (terminatorOp->hasTrait<OpTrait::ReturnLike>()) {
            LLVM_DEBUG({
              llvm::dbgs() << "  == emitting return-like op ";
              terminatorOp->print(llvm::dbgs(), asmState);
              llvm::dbgs() << "\n";
            });
            if (fn(terminatorOp).wasInterrupted())
              break;
          }
        }
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "<< Explorer::walkReturnOps >> is " << result
                          << "\n");
  return result;
}

TraversalResult Explorer::walkReturnOperands(Operation *parentOp,
                                             OperandRangeWalkFn fn) {
  return walkReturnOps(parentOp, [&](Operation *returnOp) {
    if (auto terminatorOp =
            dyn_cast<RegionBranchTerminatorOpInterface>(returnOp)) {
      return fn(terminatorOp.getSuccessorOperands(RegionBranchPoint::parent()));
    } else {
      return fn(returnOp->getOperands());
    }
  });
}

TraversalResult Explorer::walkIncomingBranchOperands(
    Block *targetBlock,
    std::function<WalkResult(Block *sourceBlock, OperandRange operands)> fn) {
  TraversalResult result = TraversalResult::COMPLETE;

  // If the block is an entry (or only) block then we need to walk up to the
  // containing region.
  if (targetBlock->isEntryBlock()) {
    auto *parentOp = targetBlock->getParentOp();

    // If the block is owned by a WhileOp we need to walk to the other region.
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      if (whileOp.getBeforeBody() == targetBlock) {
        fn(parentOp->getBlock(), whileOp.getYieldOp().getOperands());
      }
      if (whileOp.getAfterBody() == targetBlock) {
        fn(parentOp->getBlock(), whileOp.getConditionOp().getArgs());
      }
    } else if (auto regionOp = dyn_cast<RegionBranchOpInterface>(parentOp)) {
      SmallVector<RegionSuccessor, 2> entrySuccessors;
      regionOp.getSuccessorRegions(RegionBranchPoint::parent(),
                                   entrySuccessors);
      for (auto &entrySuccessor : entrySuccessors) {
        if (fn(regionOp->getBlock(), regionOp.getEntrySuccessorOperands(
                                         entrySuccessor.getSuccessor()))
                .wasInterrupted()) {
          break;
        }
      }
    } else if (auto callableOp = dyn_cast<CallableOpInterface>(parentOp)) {
      result |= walkIncomingCalls(callableOp, [&](CallOpInterface callOp) {
        return fn(callOp->getBlock(), callOp.getArgOperands());
      });
    } else {
      LLVM_DEBUG({
        llvm::dbgs() << "  !! unhandled entry point block parent\n";
        parentOp->print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      result |= TraversalResult::INCOMPLETE;
    }
  }

  // Walk any internal branches to this block.
  for (auto *sourceBlock : targetBlock->getPredecessors()) {
    auto branchOp = cast<BranchOpInterface>(sourceBlock->getTerminator());
    // I couldn't find a way to get the successor index from the predecessor,
    // so here we have to scan the source to see which one we are. This is
    // required to make things like cond_br work where there are multiple
    // successors and one or more may end up in our target block.
    for (unsigned i = 0; i < sourceBlock->getNumSuccessors(); ++i) {
      if (sourceBlock->getSuccessor(i) == targetBlock) {
        auto operandRange =
            branchOp.getSuccessorOperands(i).getForwardedOperands();
        if (fn(sourceBlock, operandRange).wasInterrupted()) {
          return result;
        }
      }
    }
  }

  return result;
}

TraversalResult Explorer::walkIncomingBlockArgument(
    BlockArgument blockArg,
    std::function<WalkResult(Block *sourceBlock, Value operand)> fn) {
  return walkIncomingBranchOperands(
      blockArg.getParentBlock(),
      [&](Block *sourceBlock, OperandRange operands) {
        return fn(sourceBlock, operands[blockArg.getArgNumber()]);
      });
}

TraversalResult Explorer::walkOutgoingBranchArguments(
    Block *sourceBlock,
    std::function<WalkResult(Block *targetBlock, Block::BlockArgListType args)>
        fn) {
  for (auto *targetBlock : sourceBlock->getSuccessors()) {
    if (fn(targetBlock, targetBlock->getArguments()).wasInterrupted()) {
      break;
    }
  }
  return TraversalResult::COMPLETE;
}

TraversalResult Explorer::walkOutgoingBranchOperandArguments(
    mlir::BranchOpInterface branchOp, unsigned operandIdx,
    std::function<WalkResult(Block *targetBlock, BlockArgument arg)> fn) {
  for (unsigned successorIdx = 0; successorIdx < branchOp->getNumSuccessors();
       ++successorIdx) {
    auto successorOperandIdx =
        mapSuccessorOperand(branchOp, successorIdx, operandIdx);
    if (!successorOperandIdx.has_value())
      continue;
    auto *targetBlock = branchOp->getSuccessor(successorIdx);
    auto blockArg = targetBlock->getArgument(*successorOperandIdx);
    if (fn(targetBlock, blockArg).wasInterrupted()) {
      break;
    }
  }
  return TraversalResult::COMPLETE;
}

// Depth-first search through the program.
// Breadth-first may be better here depending on scenario and we may want to
// allow the caller to select. For example, anything that is looking locally
// may find its solution much faster if it doesn't immediately pop out to
// some other place in the program.
//
// A hybrid approach would really be nice: process all work in the current
// region before moving on to work in other regions (knowing that traversal
// may cycle back around).
//
// TODO(benvanik): rework this using a visitor so that we have the backward
// traversal algorithm separated from the policy here. This would let us
// reuse the traversal for other kinds of walks that are more specific (like
// only getting the ops or values instead of both, etc).
TraversalResult Explorer::walkDefiningOps(Value value, ResultWalkFn fn) {
  // Fast-path short-circuit for constants, which are like 25% of all IR.
  if (value.getDefiningOp() &&
      value.getDefiningOp()->hasTrait<OpTrait::ConstantLike>()) {
    fn(llvm::cast<OpResult>(value));
    return TraversalResult::COMPLETE;
  }

  LLVM_DEBUG(llvm::dbgs() << "[[ Explorer::walkDefiningOps ]]\n");
  TraversalResult result = TraversalResult::COMPLETE;

  // We only want to issue the callback once per definition.
  // We may not need to use a set for the worklist as it's likely cheaper to
  // just add dupes and let the required processedValues check handle skipping.
  // Right now we do ~2-3 map lookups per value traversed.
  DenseSet<void *> processedValues;
  SetVector<Value> worklist;

  // Move from a block argument to all predecessors.
  auto traverseBlockArg = [&](BlockArgument arg) {
    auto *targetBlock = arg.getParentBlock();
    return walkIncomingBranchOperands(
        targetBlock, [&](Block *sourceBlock, OperandRange operands) {
          auto branchOperand = operands[arg.getArgNumber()];
          LLVM_DEBUG({
            llvm::dbgs() << "   + queuing ";
            sourceBlock->printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << " branch operand ";
            branchOperand.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          worklist.insert(branchOperand);
          return WalkResult::advance();
        });
  };

  // Move from a call into all return points from the target.
  auto traverseCallOp = [&](CallOpInterface callOp, unsigned idx) {
    // Indirect calls would require us to perform an analysis to first see if we
    // can make them direct or annotate the call sites with the possible
    // targets.
    if (callOp.getCallableForCallee().is<Value>()) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "  !! traversal incomplete due to unanalyzable indirect call: ";
        callOp.print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      return TraversalResult::INCOMPLETE;
    }
    auto targetSymbol = callOp.getCallableForCallee().get<SymbolRefAttr>();
    auto targetOp = symbolTables.lookupNearestSymbolFrom<CallableOpInterface>(
        callOp, targetSymbol);
    assert(targetOp && "call target not found");
    if (getTraversalAction(targetOp) != TraversalAction::RECURSE) {
      LLVM_DEBUG(llvm::dbgs() << "  -- ignoring call target op "
                              << targetOp->getName().getStringRef() << "\n");
      return TraversalResult::COMPLETE;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  -> traversing into call target " << targetSymbol << "\n");
    return walkReturnOperands(targetOp, [&](OperandRange returnOperands) {
      auto returnOperand = returnOperands[idx];
      LLVM_DEBUG({
        llvm::dbgs() << "   + queuing ";
        returnOperand.getParentBlock()->printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " return value ";
        returnOperand.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      worklist.insert(returnOperand);
      return WalkResult::advance();
    });
  };

  // Move from a util.global.load into all stores into that global.
  auto traverseGlobalLoadOp = [&](IREE::Util::GlobalLoadOpInterface loadOp) {
    // Indirect globals would require us to perform an analysis to first see if
    // we can make them direct or annotate the load/store sites with the
    // possible targets.
    auto *globalInfo = queryGlobalInfoFrom(loadOp.getGlobalName(), loadOp);
    if (!globalInfo || globalInfo->isIndirect) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "  !! traversal incomplete due to unanalyzable indirect global @"
            << loadOp.getGlobalName() << ": ";
        loadOp.print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      return TraversalResult::INCOMPLETE;
    }
    LLVM_DEBUG(llvm::dbgs() << "  -> traversing into global stores to @"
                            << loadOp.getGlobalName() << ":\n");
    for (auto *user : globalInfo->uses) {
      auto storeOp = dyn_cast<IREE::Util::GlobalStoreOpInterface>(user);
      if (!storeOp)
        continue;
      LLVM_DEBUG({
        llvm::dbgs() << "   + queuing stored value from ";
        storeOp.print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      worklist.insert(storeOp.getStoredGlobalValue());
    }
    return TraversalResult::COMPLETE;
  };

  // Move from a region op into all nested return points.
  auto traverseRegionOp = [&](RegionBranchOpInterface regionOp, unsigned idx) {
    if (getTraversalAction(regionOp.getOperation()) !=
        TraversalAction::RECURSE) {
      LLVM_DEBUG(llvm::dbgs() << "  -- ignoring region op "
                              << regionOp->getName().getStringRef() << "\n");
      return TraversalResult::COMPLETE;
    }
    LLVM_DEBUG(llvm::dbgs() << "  -> traversing into region op "
                            << regionOp->getName().getStringRef() << "\n");
    return walkReturnOperands(
        regionOp.getOperation(), [&](OperandRange returnOperands) {
          auto returnOperand = returnOperands[idx];
          LLVM_DEBUG({
            llvm::dbgs() << "   + queuing ";
            returnOperand.getParentBlock()->printAsOperand(llvm::dbgs(),
                                                           asmState);
            llvm::dbgs() << " return value ";
            returnOperand.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          worklist.insert(returnOperand);
          return WalkResult::advance();
        });
  };

  // Seed the queried value.
  worklist.insert(value);
  do {
    // Pop the next work item; avoiding processing values more than once.
    auto work = worklist.pop_back_val();
    if (!processedValues.insert(work.getAsOpaquePointer()).second)
      continue;

    LLVM_DEBUG({
      llvm::dbgs() << "   ? working on ";
      work.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });

    auto *definingOp = work.getDefiningOp();
    if (!definingOp) {
      // Op comes from a block argument; we need to continue walking through all
      // predecessors.
      result |= traverseBlockArg(llvm::cast<BlockArgument>(work));
      continue;
    }

    // If the op is excluded we skip it entirely.
    auto action = getTraversalAction(definingOp);
    if (action == TraversalAction::IGNORE) {
      // TODO(benvanik): determine if we should still follow ties?
      LLVM_DEBUG(llvm::dbgs() << "  -- ignoring op "
                              << definingOp->getName().getStringRef() << "\n");
      continue;
    }

    // Op is visible in the CFG as a leaf.
    auto resultValue = llvm::cast<OpResult>(work);
    LLVM_DEBUG(llvm::dbgs() << "  == emitting op "
                            << definingOp->getName().getStringRef() << "\n");
    auto fnResult = fn(resultValue);
    if (fnResult.wasInterrupted())
      break;
    if (fnResult.wasSkipped())
      continue;

    // If the op is tied we may need to walk up to the operand the result is
    // tied to.
    if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(definingOp)) {
      auto tiedOperand = tiedOp.getTiedResultOperand(resultValue);
      if (tiedOperand) {
        LLVM_DEBUG({
          llvm::dbgs() << "   + queuing tied operand ";
          tiedOperand.printAsOperand(llvm::dbgs(), asmState);
          llvm::dbgs() << "\n";
        });
        worklist.insert(tiedOperand);
      }
    }

    // If the op is a call then we need to walk up into the call sites.
    if (auto callOp = dyn_cast<CallOpInterface>(definingOp)) {
      result |= traverseCallOp(callOp, resultValue.getResultNumber());
    }

    // Step across global loads and into all of the stores across the program.
    if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(definingOp)) {
      result |= traverseGlobalLoadOp(loadOp);
    }

    // If the op is a region then we need to walk up into all of its exits.
    if (action == TraversalAction::RECURSE) {
      if (auto regionOp = dyn_cast<RegionBranchOpInterface>(definingOp)) {
        result |= traverseRegionOp(regionOp, resultValue.getResultNumber());
      }
    }
  } while (!worklist.empty());

  LLVM_DEBUG(llvm::dbgs() << "<< Explorer::walkDefiningOps >> is " << result
                          << "\n");
  return result;
}

TraversalResult Explorer::walkTransitiveUses(Value value, UseWalkFn fn) {
  LLVM_DEBUG(llvm::dbgs() << "[[ Explorer::walkTransitiveUses ]]\n");
  TraversalResult result = TraversalResult::COMPLETE;

  // We only want to issue the callback once per use.
  // We may not need to use a set for the worklist as it's likely cheaper to
  // just add dupes and let the required processedValues check handle skipping.
  // Right now we do ~2-3 map lookups per value traversed.
  DenseSet<void *> processedValues;
  SetVector<Value> worklist;

  // Move into all region entry successors. We may then cycle around inside of
  // the region for a bit.
  auto traverseRegionOp = [&](RegionBranchOpInterface regionOp,
                              unsigned operandIdx) {
    SmallVector<RegionSuccessor, 2> entrySuccessors;
    regionOp.getSuccessorRegions(RegionBranchPoint::parent(), entrySuccessors);
    for (auto &entrySuccessor : entrySuccessors) {
      auto successorInputs = entrySuccessor.getSuccessorInputs();
      if (operandIdx >= successorInputs.size()) {
        // Implicit capture; argument has the same SSA value on the inside of
        // the region. Uses show up as normal so we ignore here.
        LLVM_DEBUG(llvm::dbgs() << "  -- ignoring implicit region capture\n");
      } else {
        // Normal captured entry argument.
        auto entryArg = successorInputs[operandIdx];
        LLVM_DEBUG({
          llvm::dbgs() << "   + queuing region argument ";
          entryArg.printAsOperand(llvm::dbgs(), asmState);
          llvm::dbgs() << "\n";
        });
        worklist.insert(entryArg);
      }
    }
    return TraversalResult::COMPLETE;
  };

  // Move within/out-of a region.
  auto traverseRegionBranchOp = [&](RegionBranchTerminatorOpInterface branchOp,
                                    unsigned operandIdx) {
    auto successorOperands =
        branchOp.getSuccessorOperands(RegionBranchPoint::parent());
    unsigned beginIdx = successorOperands.getBeginOperandIndex();
    if (operandIdx < beginIdx ||
        operandIdx >= beginIdx + successorOperands.size()) {
      // Used by the op itself (or something else); ignore.
      LLVM_DEBUG(llvm::dbgs()
                 << "  -- ignoring non-succesor region branch operand "
                 << operandIdx << "\n");
      return TraversalResult::COMPLETE;
    }
    auto result = branchOp.getSuccessorOperands(
        RegionBranchPoint::parent())[operandIdx - beginIdx];
    LLVM_DEBUG({
      llvm::dbgs() << "   + queuing region result ";
      result.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });
    worklist.insert(result);
    return TraversalResult::COMPLETE;
  };

  // Move across a branch to all successors.
  auto traverseBranchOp = [&](BranchOpInterface branchOp, unsigned operandIdx) {
    return walkOutgoingBranchOperandArguments(
        branchOp, operandIdx, [&](Block *targetBlock, BlockArgument arg) {
          LLVM_DEBUG({
            llvm::dbgs() << "   + queuing ";
            targetBlock->printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << " branch argument ";
            arg.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          worklist.insert(arg);
          return WalkResult::advance();
        });
  };

  // Move across a call to the callee entry block.
  auto traverseCallOp = [&](CallOpInterface callOp, unsigned operandIdx) {
    auto callable = callOp.getCallableForCallee();
    if (callable.is<Value>()) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "  !! traversal incomplete due to unanalyzable indirect call: ";
        callOp.print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      return TraversalResult::INCOMPLETE;
    }
    auto targetSymbol = callable.get<SymbolRefAttr>();
    auto targetOp = symbolTables.lookupNearestSymbolFrom<CallableOpInterface>(
        callOp, targetSymbol);
    assert(targetOp && "call target not found");
    if (!targetOp.getCallableRegion() ||
        targetOp.getCallableRegion()->empty()) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "  !! traversal incomplete due to unanalyzable external call: ";
        callOp.print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      return TraversalResult::INCOMPLETE;
    }
    auto entryArg = targetOp.getCallableRegion()->getArgument(operandIdx);
    LLVM_DEBUG({
      llvm::dbgs() << "   + queuing call to @" << targetSymbol
                   << " entry argument ";
      entryArg.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });
    worklist.insert(entryArg);
    return TraversalResult::COMPLETE;
  };

  // Move across a return to all caller return points.
  auto traverseReturnOp = [&](Operation *returnOp, unsigned operandIdx) {
    auto callableOp = cast<CallableOpInterface>(returnOp->getParentOp());
    return walkIncomingCalls(callableOp, [&](CallOpInterface callOp) {
      auto callResult = callOp->getResult(operandIdx);
      LLVM_DEBUG({
        auto calleeOp = dyn_cast<SymbolOpInterface>(callableOp.getOperation());
        llvm::dbgs() << "   + queuing call to @" << calleeOp.getName()
                     << " result ";
        callResult.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      worklist.insert(callResult);
      return WalkResult::advance();
    });
  };

  // Move from a util.global.store into all loads from that global.
  auto traverseGlobalStoreOp = [&](IREE::Util::GlobalStoreOpInterface storeOp) {
    // Indirect globals would require us to perform an analysis to first see if
    // we can make them direct or annotate the load/store sites with the
    // possible targets.
    auto *globalInfo = queryGlobalInfoFrom(storeOp.getGlobalName(), storeOp);
    if (!globalInfo || globalInfo->isIndirect) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "  !! traversal incomplete due to unanalyzable indirect global @"
            << storeOp.getGlobalName() << ": ";
        storeOp.print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      return TraversalResult::INCOMPLETE;
    }
    LLVM_DEBUG(llvm::dbgs() << "  -> traversing into global loads from @"
                            << storeOp.getGlobalName() << ":\n");
    for (auto *user : globalInfo->uses) {
      auto loadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(user);
      if (!loadOp)
        continue;
      LLVM_DEBUG({
        llvm::dbgs() << "   + queuing loaded value from ";
        loadOp.print(llvm::dbgs(), asmState);
        llvm::dbgs() << "\n";
      });
      worklist.insert(loadOp.getLoadedGlobalValue());
    }
    return TraversalResult::COMPLETE;
  };

  // Seed the queried value.
  worklist.insert(value);
  do {
    // Pop the next work item; avoiding processing values more than once.
    auto work = worklist.pop_back_val();

    LLVM_DEBUG({
      llvm::dbgs() << "   ? working on ";
      work.printAsOperand(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    });

    // Walk each use of the value (of which a single op may use it multiple
    // times!).
    for (auto &use : work.getUses()) {
      auto *ownerOp = use.getOwner();
      if (!processedValues.insert(&use).second)
        continue;

      auto action = getTraversalAction(ownerOp);
      if (action == TraversalAction::IGNORE) {
        // TODO(benvanik): determine if we should still follow ties?
        LLVM_DEBUG(llvm::dbgs() << "  -- ignoring op "
                                << ownerOp->getName().getStringRef() << "\n");
        continue;
      }

      // Emit for the op itself.
      LLVM_DEBUG(llvm::dbgs() << "  == emitting op "
                              << ownerOp->getName().getStringRef() << "\n");
      if (fn(use).wasInterrupted())
        break;

      // If the op is tied we may need to walk down to the results the operand
      // is tied to (multiple results can tie the same operand).
      if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(ownerOp)) {
        for (auto tiedResult :
             tiedOp.getOperandTiedResults(use.getOperandNumber())) {
          LLVM_DEBUG({
            llvm::dbgs() << "   + queuing tied result ";
            tiedResult.printAsOperand(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          worklist.insert(tiedResult);
        }
      }

      // If the op is a region then we need to walk down into all of its entry
      // points.
      if (action == TraversalAction::RECURSE) {
        if (auto regionOp = dyn_cast<RegionBranchOpInterface>(ownerOp)) {
          result |= traverseRegionOp(regionOp, use.getOperandNumber());
        }
      }

      // If op is a branch then we need to walk down into all successors.
      if (auto branchOp = dyn_cast<BranchOpInterface>(ownerOp)) {
        result |= traverseBranchOp(branchOp, use.getOperandNumber());
      } else if (auto branchOp =
                     dyn_cast<RegionBranchTerminatorOpInterface>(ownerOp)) {
        result |= traverseRegionBranchOp(branchOp, use.getOperandNumber());
      }

      // If op is a call then we need to walk into the callees.
      if (auto callOp = dyn_cast<CallOpInterface>(ownerOp)) {
        result |= traverseCallOp(callOp, use.getOperandNumber());
      }

      // If op is a return then we need to walk into the caller results.
      if (ownerOp->hasTrait<OpTrait::ReturnLike>() &&
          llvm::isa<CallableOpInterface>(ownerOp->getParentOp())) {
        result |= traverseReturnOp(ownerOp, use.getOperandNumber());
      }

      if (ownerOp->hasTrait<OpTrait::ReturnLike>() &&
          !llvm::isa<CallableOpInterface>(ownerOp->getParentOp())) {
        auto parent = ownerOp->getParentOp();
        auto result = parent->getResult(use.getOperandNumber());
        worklist.insert(result);
      }

      // Step across global stores and into all of the loads across the program.
      if (auto storeOp =
              dyn_cast<IREE::Util::GlobalStoreOpInterface>(ownerOp)) {
        result |= traverseGlobalStoreOp(storeOp);
      }
    }
  } while (!worklist.empty());

  LLVM_DEBUG(llvm::dbgs() << "<< Explorer::walkTransitiveUses >> is " << result
                          << "\n");
  return result;
}

TraversalResult Explorer::walkTransitiveUsers(Value value, OperationWalkFn fn) {
  DenseSet<Operation *> visitedOwners;
  return walkTransitiveUses(value, [&](OpOperand &use) {
    if (visitedOwners.insert(use.getOwner()).second) {
      return fn(use.getOwner());
    }
    return WalkResult::advance();
  });
}

} // namespace iree_compiler
} // namespace mlir
