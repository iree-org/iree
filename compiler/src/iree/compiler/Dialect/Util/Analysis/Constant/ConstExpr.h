// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_CONSTANT_CONST_EXPR_H_
#define IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_CONSTANT_CONST_EXPR_H_

#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

// Analyzes an entire module to determine all operations/values that are
// purely derived from constants or immutable data and builds a
// dependency tree.
//
// Modifying any of the analyzed operations invalidates this analysis.
class ConstExprAnalysis {
public:
  struct ConstValueInfo;
  explicit ConstExprAnalysis(Operation *rootOp);

  void print(raw_ostream &os) const;
  void dump() const;

  // Returns const-expr info for an operation (or nullptr if unknown).
  const ConstValueInfo *lookup(Value queryValue) const {
    return constInfoMap.lookup(queryValue);
  }

  // Return const-expr info for an operation (or nullptr if unknown). Presently,
  // an operation's results will either all be const-expr or not, so we just
  // check the first. 0-result ops cannot be const-expr.
  const ConstValueInfo *lookup(Operation *queryOp) const {
    if (queryOp->getNumResults() == 0)
      return nullptr;
    return lookup(queryOp->getResult(0));
  }

  // Returns true if the given value is only derived from immutable inputs.
  // Note that this only returns true for derived values. Direct use of
  // existing constants returns false.
  bool isConstExprValue(Value queryValue) const {
    ConstValueInfo *found = constInfoMap.lookup(queryValue);
    if (!found)
      return false;
    return found->state == ConstValueInfo::CONSTANT && !found->isRoot;
  }

  // Returns whether the given operation is considered const-expr. Presently,
  // an operation's results will either all be const-expr or not, so we just
  // check the first. 0-result ops cannot be const-expr.
  bool isConstExprOperation(Operation *queryOp) const {
    if (queryOp->getNumResults() == 0)
      return false;
    return isConstExprValue(queryOp->getResult(0));
  }

  // Populates a set, in arbitrary order, of all const-expr ops in the
  // program. This includes root ops.
  void populateConstExprOperations(llvm::DenseSet<Operation *> &ops) const {
    for (auto it : constInfoMap) {
      ConstValueInfo *info = it.second;
      if (info->state == ConstValueInfo::CONSTANT) {
        Operation *definingOp = info->constValue.getDefiningOp();
        assert(definingOp && "const-expr values must have a defining op");
        ops.insert(definingOp);
      }
    }
  }

  // Map of a root value in the program that should be considered constant
  // to the operation that defines the constant. Two cases:
  //   LoadGlobalOp.result -> GlobalOp
  //   ConstantOp.result -> ConstantOp
  // Entries can come from the whole program.
  using ConstRootMap = llvm::DenseMap<Value, Operation *>;

  // Information about a Value that is has been analyzed.
  struct ConstValueInfo {
    ConstValueInfo(Value constValue) : constValue(constValue) {}

    // UNKNOWN: Not all producers have been validated.
    // CONSTANT: Producers have all been validated as constants.
    // NON_CONSTANT: The op is not eligible to be treated as a constant or
    //   one or more producers is non constant.
    enum State { UNKNOWN, CONSTANT, NON_CONSTANT };
    State state = UNKNOWN;

    // The presumed constant value.
    Value constValue;

    // Root values (in ConstRootMap) that this value (indirectly) derives from.
    SmallPtrSet<Value, 4> roots;

    // Direct producers that feed into this constant value.
    SmallPtrSet<ConstValueInfo *, 8> producers;

    // Direct consumers (constant and non-constant) of this value.
    SmallPtrSet<ConstValueInfo *, 8> consumers;

    // Whether this is a root.
    bool isRoot = false;

    // Whether this is a const-expr value.
    bool isConstExpr() const { return state == CONSTANT; }

    // If the value is consumed by an operation that was not analyzed, returns
    // true. This can be considered a non-constexpr escape.
    bool hasNonAnalyzedConsumer() const;

    // Gets the defining operation.
    Operation *getOperation() const {
      Operation *ret = constValue.getDefiningOp();
      assert(ret && "const-expr must have a defining op");
      return ret;
    }
  };

  // Define an iterator over the second value of constInfoMap.
  using ConstValueMapT = llvm::DenseMap<Value, ConstValueInfo *>;
  class ConstValueIterator final
      : public llvm::mapped_iterator<
            ConstValueMapT::const_iterator,
            ConstValueInfo *(*)(const ConstValueMapT::value_type &)> {

    static ConstValueInfo *unwrap(const ConstValueMapT::value_type &value) {
      return value.second;
    }

  public:
    ConstValueIterator(ConstValueMapT::const_iterator it)
        : llvm::mapped_iterator<
              ConstValueMapT::const_iterator,
              ConstValueInfo *(*)(const ConstValueMapT::value_type &)>(
              it, &unwrap) {}
  };

  ConstValueIterator begin() const { return constInfoMap.begin(); }
  ConstValueIterator end() const { return constInfoMap.end(); }

private:
  // Expands the frontier to include all results of a given op in an UNKNOWN
  // state. This also checks that all of its operands are known, adding
  // them recusrively if not.
  void expandToOp(Operation *op);

  // Add a new info record for a value to analyze for const-ness.
  ConstValueInfo *addInfo(Value constValue);

  // Map of a root value in the program that should be considered constant
  // to the operation that defines the constant. Two cases:
  //   LoadGlobalOp.result -> GlobalOp
  //   ConstantOp.result -> ConstantOp
  // Entries can come from the whole program.
  llvm::DenseMap<Value, Operation *> constantRoots;

  // Map of analyzed value to corresponding info struct.
  llvm::DenseMap<Value, ConstValueInfo *> constInfoMap;

  // Define an iterator over std::unique_ptr<T> as a pointer range on T*.

  // Allocated ConstValueInfo structs (to preserve pointer stability).
  llvm::SmallVector<std::unique_ptr<ConstValueInfo>> allocedConstInfos;

  // Worklist of const value info structs which need more resolution.
  using ConstValueWorklist = llvm::SmallVector<ConstValueInfo *>;
  ConstValueWorklist worklist;
  friend class ConstExprHoistingPolicy;
};

// Mutable base class for implementing policies that make decisions on
// which expressions to hoist. This wraps a read-only ConstExprAnalysis,
// overlaying it with cost and decisions about which specific expressions to
// hoist.
//
// The default base class will hoist everything that is eligible.
class ConstExprHoistingPolicy {
public:
  using Worklist = llvm::SmallVector<const ConstExprAnalysis::ConstValueInfo *>;
  enum Outcome {
    UNDECIDED = 0,
    ENABLE_HOIST = 1,
    DISABLE_HOIST = 2,
  };
  class Decision {
  public:
    void disableHoist() {
      assert(outcome == UNDECIDED &&
             "can only disable hoisting of an undecided decision");
      outcome = DISABLE_HOIST;
    }
    void enableHoist() {
      assert(outcome == UNDECIDED &&
             "can only disable hoisting of an undecided decision");
      outcome = ENABLE_HOIST;
    }

    Outcome getOutcome() const { return outcome; }

  private:
    Outcome outcome = UNDECIDED;
  };

  void printDotGraph(raw_ostream &os) const;
  void dumpDotGraph() const;

  const ConstExprAnalysis &getAnalysis() const { return analysis; }

  ConstExprHoistingPolicy(const ConstExprAnalysis &analysis, int64_t threshold);
  void initialize();
  Decision *getDecision(const ConstExprAnalysis::ConstValueInfo *info) {
    return &decisions[info];
  }

  Outcome getOutcome(const ConstExprAnalysis::ConstValueInfo *info) const {
    return decisions.lookup(info).getOutcome();
  }

private:
  // At initialization time, makes any fixed decisions. This hook can only
  // make decisions that do not depend on any const-exprs outside of what is
  // passed.
  void makeInvariantDecision(const ConstExprAnalysis::ConstValueInfo *info,
                             Decision *decision);
  // Makes a decision that depends on producers and consumers of a value. This
  // may be called repeatedly until convergence. The implementation should
  // call decision.disableHoist() or decision.enableHoist() if it can reach a
  // decision.
  void makeDecision(const ConstExprAnalysis::ConstValueInfo *info,
                    Decision *decision);

  const ConstExprAnalysis &analysis;

  int64_t constExprMaxSizeIncreaseThreshold;

  // Map of ConstValueInfo * to decision structs. All are allocated at
  // initialization and then the structure is not changed.
  llvm::DenseMap<const ConstExprAnalysis::ConstValueInfo *, Decision> decisions;
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const ConstExprAnalysis &analysis) {
  analysis.print(os);
  return os;
}

} // namespace Util
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

namespace llvm {
template <>
struct GraphTraits<
    mlir::iree_compiler::IREE::Util::ConstExprAnalysis::ConstValueInfo *> {
  using NodeRef =
      mlir::iree_compiler::IREE::Util::ConstExprAnalysis::ConstValueInfo *;
  using ChildIteratorType = SmallPtrSetImpl<NodeRef>::iterator;

  static NodeRef getEntryNode(NodeRef info) { return info; }

  static ChildIteratorType child_begin(NodeRef info) {
    return info->consumers.begin();
  }

  static ChildIteratorType child_end(NodeRef info) {
    return info->consumers.end();
  }
};

template <>
struct GraphTraits<
    const mlir::iree_compiler::IREE::Util::ConstExprHoistingPolicy *>
    : public GraphTraits<mlir::iree_compiler::IREE::Util::ConstExprAnalysis::
                             ConstValueInfo *> {

  using nodes_iterator =
      mlir::iree_compiler::IREE::Util::ConstExprAnalysis::ConstValueIterator;

  static NodeRef getEntryNode(
      const mlir::iree_compiler::IREE::Util::ConstExprHoistingPolicy *graph) {
    return *graph->getAnalysis().begin();
  }

  static nodes_iterator nodes_begin(
      const mlir::iree_compiler::IREE::Util::ConstExprHoistingPolicy *graph) {
    return graph->getAnalysis().begin();
  }

  static nodes_iterator nodes_end(
      const mlir::iree_compiler::IREE::Util::ConstExprHoistingPolicy *graph) {
    return graph->getAnalysis().end();
  }
};

} // namespace llvm

#endif // IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_CONSTANT_CONST_EXPR_H_
