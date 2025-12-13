// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTraits.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/SCCIterator.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::VM {

#define GEN_PASS_DEF_ANNOTATEFUNCTIONSPASS
#include "iree/compiler/Dialect/VM/Transforms/Passes.h.inc"

namespace {

// Information collected about a function in a single walk.
struct FuncInfo {
  bool needsYield = false;
  bool needsUnwind = false;
  SmallVector<Operation *> callees;
};

// Analyzes a function in a single walk to collect yield/unwind requirements
// and callees.
static FuncInfo analyzeFunction(IREE::VM::FuncOp funcOp,
                                SymbolTable &symbolTable) {
  FuncInfo info;

  // Check existing attributes.
  info.needsYield = funcOp->hasAttr("vm.yield");
  info.needsUnwind = funcOp->hasAttr("vm.unwind");

  // Check signature for refs.
  bool hasRefs = llvm::any_of(funcOp.getArgumentTypes(),
                              [](Type t) { return isa<IREE::VM::RefType>(t); });
  hasRefs |= llvm::any_of(funcOp.getResultTypes(),
                          [](Type t) { return isa<IREE::VM::RefType>(t); });

  bool hasFailableOps = false;

  // Single walk: collect callees, check for refs/failable ops/yield ops.
  funcOp.walk([&](Operation *op) {
    // Collect callees.
    if (auto callOp = dyn_cast<IREE::VM::CallOp>(op)) {
      if (auto callee = symbolTable.lookup(callOp.getCallee()))
        info.callees.push_back(callee);
    } else if (auto callOp = dyn_cast<IREE::VM::CallVariadicOp>(op)) {
      if (auto callee = symbolTable.lookup(callOp.getCallee()))
        info.callees.push_back(callee);
    }
    // Check for yield ops.
    if (isa<IREE::VM::YieldOp>(op)) {
      info.needsYield = true;
    }
    // Check for MayFail trait.
    if (op->hasTrait<OpTrait::IREE::VM::MayFail>()) {
      hasFailableOps = true;
    }
    // Check results for refs.
    for (Value result : op->getResults()) {
      if (isa<IREE::VM::RefType>(result.getType())) {
        hasRefs = true;
        break;
      }
    }
  });

  // Compute initial unwind requirement.
  if (hasRefs && hasFailableOps) {
    info.needsUnwind = true;
  }

  return info;
}

// Analyzes an import to collect yield/unwind requirements.
static FuncInfo analyzeImport(IREE::VM::ImportOp importOp) {
  FuncInfo info;
  info.needsYield = importOp->hasAttr("vm.yield");
  info.needsUnwind = importOp->hasAttr("vm.unwind");
  // Imports have no callees.
  return info;
}

class AnnotateFunctionsPass
    : public IREE::VM::impl::AnnotateFunctionsPassBase<AnnotateFunctionsPass> {
public:
  void runOnOperation() override {
    IREE::VM::ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Map from operation to its info.
    DenseMap<Operation *, FuncInfo> funcInfos;

    // Phase 1: Analyze all functions and imports (single walk each).
    for (auto funcOp : moduleOp.getOps<IREE::VM::FuncOp>()) {
      funcInfos[funcOp] = analyzeFunction(funcOp, symbolTable);
    }
    for (auto importOp : moduleOp.getOps<IREE::VM::ImportOp>()) {
      funcInfos[importOp] = analyzeImport(importOp);
    }

    // Phase 2: Build adjacency list for reverse graph (callee -> callers).
    // We need to propagate from callees to callers, so we process in
    // reverse topological order. Using the call graph directly gives us
    // the right order with scc_iterator.

    // Build the MLIR CallGraph.
    const CallGraph callGraph(moduleOp);

    // Phase 3: Process SCCs in reverse topological order (callees before
    // callers). The scc_iterator provides SCCs in reverse post-order, which
    // means callees are visited before callers (what we want).
    for (auto sccIt = llvm::scc_begin(&callGraph); !sccIt.isAtEnd(); ++sccIt) {
      const std::vector<CallGraphNode *> &scc = *sccIt;

      // Collect initial bits from all nodes in this SCC.
      bool sccYield = false;
      bool sccUnwind = false;

      for (CallGraphNode *node : scc) {
        if (node->isExternal())
          continue;
        Operation *op = node->getCallableRegion()->getParentOp();
        auto it = funcInfos.find(op);
        if (it != funcInfos.end()) {
          sccYield |= it->second.needsYield;
          sccUnwind |= it->second.needsUnwind;
        }
      }

      // Propagate from callees (already processed, outside this SCC).
      for (CallGraphNode *node : scc) {
        if (node->isExternal())
          continue;
        Operation *op = node->getCallableRegion()->getParentOp();
        auto it = funcInfos.find(op);
        if (it == funcInfos.end())
          continue;

        for (Operation *calleeOp : it->second.callees) {
          auto calleeIt = funcInfos.find(calleeOp);
          if (calleeIt == funcInfos.end())
            continue;

          // Only propagate from callees outside this SCC (they have final
          // bits).
          bool inScc = false;
          for (CallGraphNode *sccNode : scc) {
            if (!sccNode->isExternal() &&
                sccNode->getCallableRegion()->getParentOp() == calleeOp) {
              inScc = true;
              break;
            }
          }
          if (!inScc) {
            sccYield |= calleeIt->second.needsYield;
            sccUnwind |= calleeIt->second.needsUnwind;
          }
        }
      }

      // Apply to all nodes in this SCC.
      for (CallGraphNode *node : scc) {
        if (node->isExternal())
          continue;
        Operation *op = node->getCallableRegion()->getParentOp();
        auto it = funcInfos.find(op);
        if (it != funcInfos.end()) {
          it->second.needsYield = sccYield;
          it->second.needsUnwind = sccUnwind;
        }
      }
    }

    // Phase 4: Apply attributes to functions.
    for (auto funcOp : moduleOp.getOps<IREE::VM::FuncOp>()) {
      auto it = funcInfos.find(funcOp);
      if (it == funcInfos.end())
        continue;

      if (it->second.needsYield && !funcOp->hasAttr("vm.yield")) {
        funcOp->setAttr("vm.yield", UnitAttr::get(funcOp.getContext()));
      }
      if (it->second.needsUnwind && !funcOp->hasAttr("vm.unwind")) {
        funcOp->setAttr("vm.unwind", UnitAttr::get(funcOp.getContext()));
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::VM
