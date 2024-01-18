// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_DEPGRAPH_H_
#define IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_DEPGRAPH_H_

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::DFX {

class AbstractElement;
class Solver;

// Indicates the type of dependency between two elements in the graph.
enum class Resolution {
  // The target cannot be valid if the source is not.
  REQUIRED,
  // The target may be valid if the source is not.
  OPTIONAL,
  // Do not track a dependence between source and target.
  NONE,
};

// The data structure for the nodes of a dependency graph
class DepGraphNode {
public:
  using DepTy = llvm::PointerIntPair<DepGraphNode *, 1>;

  virtual ~DepGraphNode() = default;

protected:
  // Set of dependency graph nodes which should be updated if this one
  // is updated. The bit encodes if it is optional.
  TinyPtrVector<DepTy> deps;

  static DepGraphNode *DepGetVal(DepTy &dt) { return dt.getPointer(); }
  static AbstractElement *DepGetValAA(DepTy &dt) {
    return cast<AbstractElement>(dt.getPointer());
  }

  operator AbstractElement *() { return cast<AbstractElement>(this); }

public:
  using iterator = llvm::mapped_iterator<TinyPtrVector<DepTy>::iterator,
                                         decltype(&DepGetVal)>;
  using aaiterator = llvm::mapped_iterator<TinyPtrVector<DepTy>::iterator,
                                           decltype(&DepGetValAA)>;

  aaiterator begin() { return aaiterator(deps.begin(), &DepGetValAA); }
  aaiterator end() { return aaiterator(deps.end(), &DepGetValAA); }
  iterator child_begin() { return iterator(deps.begin(), &DepGetVal); }
  iterator child_end() { return iterator(deps.end(), &DepGetVal); }

  virtual void print(raw_ostream &os, AsmState &asmState) const {
    os << "DepGraphNode unimpl\n";
  }
  TinyPtrVector<DepTy> &getDeps() { return deps; }

  friend class Solver;
  friend struct DepGraph;
};

// The data structure for the dependency graph
//
// Note that in this graph if there is an edge from A to B (A -> B),
// then it means that B depends on A, and when the state of A is
// updated, node B should also be updated
struct DepGraph {
  using DepTy = DepGraphNode::DepTy;
  static DepGraphNode *DepGetVal(DepTy &DT) { return DT.getPointer(); }
  using iterator = llvm::mapped_iterator<llvm::TinyPtrVector<DepTy>::iterator,
                                         decltype(&DepGetVal)>;

  explicit DepGraph(AsmState &asmState) : asmState(asmState) {}
  ~DepGraph() = default;

  // There is no root node for the dependency graph. But the SCCIterator
  // requires a single entry point, so we maintain a fake("synthetic") root
  // node that depends on every node.
  DepGraphNode syntheticRoot;
  DepGraphNode *getEntryNode() { return &syntheticRoot; }

  iterator begin() { return syntheticRoot.child_begin(); }
  iterator end() { return syntheticRoot.child_end(); }

  void print(llvm::raw_ostream &os);
  void dumpGraph();

  AsmState &asmState;
};

} // namespace mlir::iree_compiler::DFX

namespace llvm {

using DFXDepGraph = mlir::iree_compiler::DFX::DepGraph;
using DFXDepGraphNode = mlir::iree_compiler::DFX::DepGraphNode;

template <>
struct GraphTraits<DFXDepGraphNode *> {
  using NodeRef = DFXDepGraphNode *;
  using DepTy = llvm::PointerIntPair<DFXDepGraphNode *, 1>;
  using EdgeRef = llvm::PointerIntPair<DFXDepGraphNode *, 1>;

  static NodeRef getEntryNode(DFXDepGraphNode *node) { return node; }
  static NodeRef DepGetVal(DepTy &dt) { return dt.getPointer(); }

  using ChildIteratorType =
      llvm::mapped_iterator<TinyPtrVector<DepTy>::iterator,
                            decltype(&DepGetVal)>;
  using ChildEdgeIteratorType = TinyPtrVector<DepTy>::iterator;

  static ChildIteratorType child_begin(NodeRef n) { return n->child_begin(); }
  static ChildIteratorType child_end(NodeRef n) { return n->child_end(); }
};

template <>
struct GraphTraits<DFXDepGraph *> : public GraphTraits<DFXDepGraphNode *> {
  static NodeRef getEntryNode(DFXDepGraph *graph) {
    return graph->getEntryNode();
  }

  using nodes_iterator = llvm::mapped_iterator<TinyPtrVector<DepTy>::iterator,
                                               decltype(&DepGetVal)>;

  static nodes_iterator nodes_begin(DFXDepGraph *graph) {
    return graph->begin();
  }
  static nodes_iterator nodes_end(DFXDepGraph *graph) { return graph->end(); }
};

} // end namespace llvm

#endif // IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_DEPGRAPH_H_
