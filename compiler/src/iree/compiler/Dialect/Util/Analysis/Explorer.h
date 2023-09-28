// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_ANALYSIS_EXPLORER_H_
#define IREE_COMPILER_DIALECT_UTIL_ANALYSIS_EXPLORER_H_

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PointerIntPair.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Traversal control
//===----------------------------------------------------------------------===//
// TODO(benvanik): separate out into an OpActionSet (or something) for reuse.

// Controls how traversal is performed.
// By default ops are processed with the default traversal action.
enum class TraversalAction {
  // Traversal walks just the target while ignoring nested regions.
  SHALLOW = 0,
  // Traversal walks both the target and all nested regions.
  RECURSE,
  // Target is entirely ignored during traversal.
  // No results or nested regions will be walked.
  IGNORE,
};

// Boolean operations on TraversalResult behave as though `INCOMPLETE` is
// truthy to allow for |='ing results.
enum class TraversalResult {
  // All values that could be walked were. The set walked is considered
  // complete. The walk results can be used for _must_ expressions.
  COMPLETE = 0,
  // Only some values were walked out of all potential values. This indicates
  // that analysis could not see certain operations (such as external or
  // indirect ones). The walk results can only be used for _may_ expressions.
  INCOMPLETE,
};
inline TraversalResult operator|(TraversalResult lhs, TraversalResult rhs) {
  return lhs == TraversalResult::INCOMPLETE ? lhs : rhs;
}
inline TraversalResult &operator|=(TraversalResult &lhs, TraversalResult rhs) {
  lhs = lhs | rhs;
  return lhs;
}
inline TraversalResult operator&(TraversalResult lhs, TraversalResult rhs) {
  return lhs == TraversalResult::COMPLETE ? lhs : rhs;
}
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     TraversalResult result) {
  return os << (result == TraversalResult::COMPLETE ? "complete"
                                                    : "incomplete");
}

//===----------------------------------------------------------------------===//
// IR Explorer Utility
//===----------------------------------------------------------------------===//

// IR exploration acceleration structure and queries.
// Enables efficient traversal of IR as if it were a graph using cached state.
// Expensive to create and behavior is undefined if IR is modified out from
// under it. Prefer to create and cache this once per pass and perform all
// queries prior to manipulating the IR.
//
// TODO(#7389): make this an abstract interface and hide the IREE details.
class Explorer {
public:
  Explorer(Operation *rootOp, TraversalAction defaultAction);
  ~Explorer();

  Operation *getRootOp() const { return rootOp; }
  AsmState &getAsmState() { return asmState; }
  SymbolTableCollection &getSymbolTables() { return symbolTables; }
  const CallGraph &getCallGraph() const { return callGraph; }

  // Returns the traversal action to perform for the given op.
  TraversalAction getTraversalAction(Operation *op);

  // Registers a default action for all ops in the given dialect namespace.
  // Individual op actions can override this.
  void setDialectAction(StringRef dialectNamespace, TraversalAction action);
  template <typename DialectT>
  void setDialectAction(TraversalAction action) {
    setDialectAction(DialectT::getDialectNamespace(), action);
  }
  template <typename DialectT, typename DialectT2, typename... DialectTs>
  void setDialectAction(TraversalAction action) {
    setDialectAction<DialectT>(action);
    setDialectAction<DialectT2, DialectTs...>(action);
  }

  // Registers a traversal action for the given op, overriding the explorer
  // default and any dialect action specified.
  void setOpAction(OperationName op, TraversalAction action);
  template <typename OpT>
  void setOpAction(TraversalAction action) {
    setOpAction(OperationName(OpT::getOperationName(), rootOp->getContext()),
                action);
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void setOpAction(TraversalAction action) {
    setOpAction<OpT>(action);
    setOpAction<OpT2, OpTs...>(action);
  }

  // Initializes the explorer. Must be called after all dialect/op actions have
  // been specified.
  void initialize();

  // Returns a cached analysis manager for the root op.
  AnalysisManager getAnalysisManager() { return analysisManager; }

  // Cached information about a global variable.
  struct GlobalInfo {
    // Global variable definition.
    mutable IREE::Util::GlobalOpInterface op;
    // True if the global is ever used indirectly anywhere in the program.
    // The explorer cannot (currently) see through these and the global should
    // be considered volatile.
    bool isIndirect = false;
    // All loads and stores of the global across the program.
    SmallVector<Operation *> uses;

    // Returns a range of all direct loads of the global.
    auto getLoads() const {
      assert(!isIndirect && "indirect loads not yet tracked");
      return llvm::map_range(
          llvm::make_filter_range(
              uses,
              [](Operation *op) {
                return isa<IREE::Util::GlobalLoadOpInterface>(op);
              }),
          [](Operation *op) {
            return cast<IREE::Util::GlobalLoadOpInterface>(op);
          });
    }

    // Returns a range of all direct stores to the global.
    auto getStores() const {
      assert(!isIndirect && "indirect stores not yet tracked");
      return llvm::map_range(
          llvm::make_filter_range(
              uses,
              [](Operation *op) {
                return isa<IREE::Util::GlobalStoreOpInterface>(op);
              }),
          [](Operation *op) {
            return cast<IREE::Util::GlobalStoreOpInterface>(op);
          });
    }
  };

  // Gets analyzed global information for the given global operation.
  const GlobalInfo *getGlobalInfo(IREE::Util::GlobalOpInterface globalOp);

  // Queries memoized information about a global variable, returning nullptr if
  // not found.
  const GlobalInfo *queryGlobalInfoFrom(StringRef globalName, Operation *from);

  // Calls |fn| once for each global in the root.
  void forEachGlobal(std::function<void(const GlobalInfo *)> fn);

  // Returns true if the two values _may_ alias each other via a tie or a join.
  // Conservative: returns true if value usage cannot be tracked.
  //
  // Example:
  //  func.func @root(%arg0: index) -> index {
  //    %0 = some.region(%arg0 as %innerArg : index) -> index {
  //      %1 = some.tied_op(%innerArg) : (index) -> %innerArg
  //      yield %1 : index
  //    }
  //    return %0 : index
  //  }
  //  mayValuesAlias(%arg0, %innerArg) = true
  //  mayValuesAlias(%arg0, %0) = true
  //  mayValuesAlias(%arg0, %1) = true
  //  mayValuesAlias(%innerArg, %1) = true
  bool mayValuesAlias(Value a, Value b);

  // Returns true if |value| _may_ be used by |op|.
  // Conservative: returns true if value usage cannot be tracked.
  bool mayBeUsedBy(Value value, Operation *user);

  using ValueWalkFn = std::function<WalkResult(Value)>;
  using OperationWalkFn = std::function<WalkResult(Operation *)>;
  using UseWalkFn = std::function<WalkResult(OpOperand &)>;
  using ResultWalkFn = std::function<WalkResult(OpResult)>;
  using OperandRangeWalkFn = std::function<WalkResult(OperandRange)>;

  // Walks the entire root op scope by traversing strongly connected components.
  // Operations are filtered based on the explorer configuration.
  TraversalResult walk(OperationWalkFn fn);

  // Walks all unique SSA values nested within the root op.
  TraversalResult walkValues(ValueWalkFn fn);

  // Walks all unique SSA values used/defined by |op| and all nested regions.
  TraversalResult walkValues(Operation *op, ValueWalkFn fn);

  // TODO(benvanik): walk reachable ops from op, optional limitOp scope.

  // Walks all of the call ops calling into the given |callableOp|.
  // May be incomplete if there are indirect calls in the program.
  TraversalResult
  walkIncomingCalls(CallableOpInterface callableOp,
                    std::function<WalkResult(CallOpInterface)> fn);

  // Walks all return-like (or region terminators to parent) ops in |parentOp|.
  // The operations enumerated will be either ReturnLike or implement
  // RegionBranchTerminatorOpInterface.
  TraversalResult walkReturnOps(Operation *parentOp, OperationWalkFn fn);

  // Walks all return-like (or region terminators to parent) ops in |parentOp|
  // and provides the operands passed to them. These are the values that will
  // be returned from a function call or as the parent op result values.
  TraversalResult walkReturnOperands(Operation *parentOp,
                                     OperandRangeWalkFn fn);

  // Walks all predecessor blocks of |targetBlock| and provides the operands
  // passed to them along the incoming edge. Note that |targetBlock| may be
  // enumerated if there is recursion.
  TraversalResult walkIncomingBranchOperands(
      Block *targetBlock,
      std::function<WalkResult(Block *sourceBlock, OperandRange operands,
                               size_t offset)>
          fn);

  // Walks all predecessor blocks providing values for |blockArg|.
  TraversalResult walkIncomingBlockArgument(
      BlockArgument blockArg,
      std::function<WalkResult(Block *sourceBlock, Value operand)> fn);

  // Walks all successor blocks of |sourceBlock| and provides their arguments.
  // Note that |sourceBlock| may be enumerated if there is recursion.
  TraversalResult walkOutgoingBranchArguments(
      Block *sourceBlock,
      std::function<WalkResult(Block *targetBlock,
                               Block::BlockArgListType args)>
          fn);

  // Walks all successors of |branchOp| and provides the successor block
  // argument corresponding to the given branch |operandIdx|.
  TraversalResult walkOutgoingBranchOperandArguments(
      mlir::BranchOpInterface branchOp, unsigned operandIdx,
      std::function<WalkResult(Block *targetBlock, BlockArgument arg)> fn);

  // Walks all potential defining ops of |value|.
  // The defining ops may come from any part of the program. There may be
  // multiple defining ops in cases of arguments that may come from multiple
  // source blocks/calls. Does not include parent regions of block arguments.
  //
  // Includes the producer of |value|. Returns TraversalResult::INCOMPLETE if
  // only a partial walk could be performed due to incomplete information.
  //
  // Example:
  //  func.func @root(%arg0: index) -> index
  //    %0 = producer.a %arg0 : index
  //    %1 = call @some_user(%0) : (index) -> index
  //    return %1 : index
  //  }
  //  func.func @some_user(%arg0: index) -> index {
  //    %2 = producer.b %arg0 : index
  //    return %2 : index
  //  }
  //  Walk %0: [%0 of producer.a]
  //  Walk %1: [%2 of producer.b]
  //  Walk %2: [%2 of producer.b]
  //  Walk @some_user::%arg0: [%0 of producer.a]
  //  Walk @some_user::ret0: [%2 of producer.b]
  TraversalResult walkDefiningOps(Value value, ResultWalkFn fn);

  // Randomly walks uses of |value| and any transitive alias of |value|.
  // The uses may come from any part of the program.
  //
  // Does not include the producer |value|. Returns TraversalResult::INCOMPLETE
  // if only a partial walk could be performed due to incomplete information.
  //
  // Example:
  //  func.func @root(%arg0: index) -> index
  //    %0 = producer.a %arg0 : index
  //    %1 = call @some_user(%0) : (index) -> index
  //    return %1 : index
  //  }
  //  func.func @some_user(%arg0: index) -> index {
  //    %2 = producer.b %arg0 : index
  //    return %2 : index
  //  }
  //  Walk %arg0: [%arg0 of producer.a]
  //  Walk %0: [%0 of call @some_user, %arg0 of producer.b]
  //  Walk %2: [%2 of return, %1 of return]
  TraversalResult walkTransitiveUses(Value value, UseWalkFn fn);

  // Randomly walks uses of |value| and any transitive alias of |value| and
  // returns each owner operation once. As a value may be used multiple times
  // by a single operation this is equivalent to a walkTransitiveUses with
  // deduplication on the owner of the use.
  TraversalResult walkTransitiveUsers(Value value, OperationWalkFn fn);

private:
  // Maps callee callable region -> call sites.
  using InverseCallGraph = DenseMap<Region *, SmallVector<CallOpInterface>>;

  void initializeGlobalInfos();
  void initializeInverseCallGraph();

  WalkResult recursiveWalk(Operation *parentOp, const OperationWalkFn &fn);
  WalkResult recursiveWalkValues(Operation *parentOp,
                                 DenseSet<Value> &visitedValues,
                                 const ValueWalkFn &fn);

  Operation *rootOp = nullptr;
  AsmState asmState;
  SymbolTableCollection symbolTables;

  // TODO(benvanik): build on-demand to reduced fixed cost for local queries.
  // NOTE: may be incomplete if there are indirect calls; isCallGraphIncomplete.
  const CallGraph callGraph;
  InverseCallGraph callGraphInv;
  bool isCallGraphIncomplete = false;

  TraversalAction defaultAction;
  DenseMap<StringRef, TraversalAction> dialectActions;
  DenseMap<OperationName, TraversalAction> opActions;

  DenseMap<Operation *, GlobalInfo> globalInfos;
  ModuleAnalysisManager analysisManager;
};

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_UTIL_ANALYSIS_EXPLORER_H_
