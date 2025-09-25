// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/GlobalTable.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_VERIFYINITIALIZATIONORDERPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

// Tracks which functions are only reachable from initializers.
struct InitializerReachability {
  // Functions that are only callable from initializers (not from external).
  DenseSet<Operation *> initializerOnlyFunctions;
  // Functions reachable from external entry points.
  DenseSet<Operation *> externallyReachableFunctions;
};

class VerifyInitializationOrderPass
    : public impl::VerifyInitializationOrderPassBase<
          VerifyInitializationOrderPass> {
public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    GlobalTable globalTable(moduleOp);
    globalTable.rebuild();

    // Collect initialization points in module order.
    SmallVector<IREE::Util::InitializerOpInterface> initializerOps;
    DenseMap<Operation *, size_t> opOrdinalMap;
    collectInitializationOrder(moduleOp, initializerOps, opOrdinalMap);

    // Phase 1: simple checks (no CFG analysis needed).
    if (failed(verifyModuleOrderDependencies(globalTable, initializerOps,
                                             opOrdinalMap))) {
      return;
    }

    if (failed(verifyDoubleInitialization(globalTable, initializerOps))) {
      return;
    }

    if (failed(verifyInitialValueConflicts(globalTable, initializerOps,
                                           opOrdinalMap))) {
      return;
    }

    // Phase 2: conservative reachability analysis with warnings.
    auto reachability = computeInitializerReachability(moduleOp);
    if (failed(verifyImmutableStoresWithWarnings(globalTable, reachability))) {
      return;
    }
  }

private:
  enum DiagnosticLevel { ERROR, WARNING, INFO };

  void emitDiagnostic(DiagnosticLevel level, Location loc, const Twine &msg) {
    switch (level) {
    case ERROR:
      emitError(loc) << msg;
      signalPassFailure();
      break;
    case WARNING:
      emitWarning(loc) << msg << " (verification may be overly conservative)";
      break;
    case INFO:
      emitRemark(loc) << msg;
      break;
    }
  }

  // Collects all initialization points (initializers and globals) in order.
  void collectInitializationOrder(
      mlir::ModuleOp moduleOp,
      SmallVector<IREE::Util::InitializerOpInterface> &initializerOps,
      DenseMap<Operation *, size_t> &opOrdinalMap) {
    size_t ordinal = 0;
    for (auto &op : moduleOp.getOps()) {
      opOrdinalMap[&op] = ordinal++;
      if (auto initializerOp =
              dyn_cast<IREE::Util::InitializerOpInterface>(&op)) {
        initializerOps.push_back(initializerOp);
      }
    }
  }

  // Verifies that initializers only access globals defined before them.
  LogicalResult verifyModuleOrderDependencies(
      GlobalTable &globalTable,
      ArrayRef<IREE::Util::InitializerOpInterface> initializerOps,
      DenseMap<Operation *, size_t> &opOrdinalMap) {
    for (auto initializerOp : initializerOps) {
      size_t initOrdinal = opOrdinalMap[initializerOp];

      // Check all global accesses in the initializer.
      auto result = initializerOp.walk([&](Operation *op) -> WalkResult {
        if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
          auto globalName = loadOp.getGlobalName();
          auto &global = globalTable.lookup(globalName);
          if (global.op) {
            size_t globalOrdinal = opOrdinalMap[global.op];
            if (globalOrdinal > initOrdinal) {
              emitDiagnostic(ERROR, op->getLoc(),
                             "initializer at position " + Twine(initOrdinal) +
                                 " accesses global '@" + globalName +
                                 "' defined at position " +
                                 Twine(globalOrdinal));
              return WalkResult::interrupt();
            }
          }
        }

        // Also check stores for forward references.
        if (auto storeOp = dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
          auto globalName = storeOp.getGlobalName();
          auto &global = globalTable.lookup(globalName);
          if (global.op) {
            size_t globalOrdinal = opOrdinalMap[global.op];
            if (globalOrdinal > initOrdinal) {
              emitDiagnostic(ERROR, op->getLoc(),
                             "initializer at position " + Twine(initOrdinal) +
                                 " stores to global '@" + globalName +
                                 "' defined at position " +
                                 Twine(globalOrdinal));
              return WalkResult::interrupt();
            }
          }
        }

        // Check function calls for transitive dependencies.
        if (auto callOp = dyn_cast<IREE::Util::CallOp>(op)) {
          // We'll verify the called functions separately in a more
          // sophisticated pass. For now, just ensure the function exists. DO
          // NOT SUBMIT
        }

        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        return failure();
      }
    }
    return success();
  }

  // Verifies that immutable globals are only initialized once.
  LogicalResult verifyDoubleInitialization(
      GlobalTable &globalTable,
      ArrayRef<IREE::Util::InitializerOpInterface> initializerOps) {
    // Track which immutable globals have been initialized.
    llvm::StringMap<Location> immutableGlobalInitializations;

    // First pass: check globals with initial values.
    for (size_t i = 0; i < globalTable.size(); ++i) {
      auto globalName = globalTable.lookupByOrdinal(i);
      auto &global = globalTable.lookup(globalName);
      if (global.op && !global.op.isGlobalMutable() &&
          global.op.getGlobalInitialValue()) {
        // Immutable global with initial value.
        immutableGlobalInitializations.try_emplace(globalName,
                                                   global.op->getLoc());
      }
    }

    // Second pass: check for stores in initializers.
    for (auto initializerOp : initializerOps) {
      auto result = initializerOp.walk(
          [&](IREE::Util::GlobalStoreOpInterface storeOp) -> WalkResult {
            auto globalName = storeOp.getGlobalName();
            auto &global = globalTable.lookup(globalName);
            if (global.op && !global.op.isGlobalMutable()) {
              // Check if already initialized.
              auto it = immutableGlobalInitializations.find(globalName);
              if (it != immutableGlobalInitializations.end()) {
                emitDiagnostic(ERROR, storeOp->getLoc(),
                               "immutable global '@" + globalName +
                                   "' is initialized both by initial value and "
                                   "by store in initializer");
                return WalkResult::interrupt();
              }
              immutableGlobalInitializations.try_emplace(globalName,
                                                         storeOp->getLoc());
            }
            return WalkResult::advance();
          });
      if (result.wasInterrupted()) {
        return failure();
      }
    }
    return success();
  }

  // Verifies that globals with initial values aren't modified by earlier
  // initializers.
  LogicalResult verifyInitialValueConflicts(
      GlobalTable &globalTable,
      ArrayRef<IREE::Util::InitializerOpInterface> initializerOps,
      DenseMap<Operation *, size_t> &opOrdinalMap) {
    // Check each initializer for stores to globals that come after it.
    for (auto initializerOp : initializerOps) {
      size_t initOrdinal = opOrdinalMap[initializerOp];
      auto result = initializerOp.walk(
          [&](IREE::Util::GlobalStoreOpInterface storeOp) -> WalkResult {
            auto globalName = storeOp.getGlobalName();
            auto &global = globalTable.lookup(globalName);
            if (global.op && global.op.getGlobalInitialValue()) {
              // Global has an initial value. Check if it comes after this
              // initializer.
              size_t globalOrdinal = opOrdinalMap[global.op];
              if (globalOrdinal > initOrdinal) {
                emitDiagnostic(
                    ERROR, storeOp->getLoc(),
                    "global '@" + globalName +
                        "' with initial value at position " +
                        Twine(globalOrdinal) +
                        " is modified by earlier initializer at position " +
                        Twine(initOrdinal));
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          });
      if (result.wasInterrupted()) {
        return failure();
      }
    }
    return success();
  }

  // Computes which functions are only reachable from initializers.
  InitializerReachability
  computeInitializerReachability(mlir::ModuleOp moduleOp) {
    InitializerReachability result;

    // Build the call graph.
    CallGraph callGraph(moduleOp);

    // Find all externally reachable functions.
    SetVector<CallGraphNode *> externalWorklist;
    for (CallGraphNode *node : callGraph) {
      if (node->isExternal()) {
        continue;
      }
      auto *callableOp = node->getCallableRegion()->getParentOp();
      if (isa<IREE::Util::InitializerOpInterface>(callableOp)) {
        // Initializers are not externally reachable.
        continue;
      }
      if (auto funcOp = dyn_cast<FunctionOpInterface>(callableOp)) {
        if (funcOp.isPublic()) {
          // Public function - externally reachable.
          result.externallyReachableFunctions.insert(callableOp);
          externalWorklist.insert(node);
        }
      }
    }

    // Transitively mark all functions called from external entry points.
    while (!externalWorklist.empty()) {
      auto *node = externalWorklist.pop_back_val();
      for (auto &edge : *node) {
        auto *targetNode = edge.getTarget();
        if (!targetNode->isExternal()) {
          auto *targetOp = targetNode->getCallableRegion()->getParentOp();
          if (result.externallyReachableFunctions.insert(targetOp).second) {
            externalWorklist.insert(targetNode);
          }
        }
      }
    }

    // Find functions only called from initializers.
    SetVector<CallGraphNode *> initializerWorklist;
    for (CallGraphNode *node : callGraph) {
      if (node->isExternal()) {
        continue;
      }
      auto *callableOp = node->getCallableRegion()->getParentOp();
      if (isa<IREE::Util::InitializerOpInterface>(callableOp)) {
        initializerWorklist.insert(node);
      }
    }

    // Mark all functions transitively called from initializers.
    DenseSet<Operation *> initializerReachable;
    while (!initializerWorklist.empty()) {
      auto *node = initializerWorklist.pop_back_val();
      for (auto &edge : *node) {
        auto *targetNode = edge.getTarget();
        if (!targetNode->isExternal()) {
          auto *targetOp = targetNode->getCallableRegion()->getParentOp();
          if (initializerReachable.insert(targetOp).second) {
            initializerWorklist.insert(targetNode);
          }
        }
      }
    }

    // Functions that are initializer-reachable but NOT externally reachable
    // are initializer-only.
    for (auto *op : initializerReachable) {
      if (!result.externallyReachableFunctions.contains(op)) {
        result.initializerOnlyFunctions.insert(op);
      }
    }

    return result;
  }

  // Verifies stores to immutable globals with appropriate warnings.
  LogicalResult verifyImmutableStoresWithWarnings(
      GlobalTable &globalTable, const InitializerReachability &reachability) {
    bool hadError = false;
    globalTable.forEach([&](Global &global) {
      if (global.op.isGlobalMutable()) {
        // Mutable global - stores are allowed from anywhere.
        return GlobalAction::PRESERVE;
      }

      // Immutable global - check all stores.
      for (auto storeOp : global.storeOps) {
        Operation *parent = storeOp->getParentOp();

        // Walk up to find the containing function or initializer.
        while (parent && !isa<FunctionOpInterface>(parent) &&
               !isa<IREE::Util::InitializerOpInterface>(parent)) {
          parent = parent->getParentOp();
        }

        if (!parent) {
          emitDiagnostic(ERROR, storeOp->getLoc(),
                         "store to immutable global '@" + global.getName() +
                             "' outside of function or initializer context");
          hadError = true;
          continue;
        }

        if (isa<IREE::Util::InitializerOpInterface>(parent)) {
          // Store directly in initializer - always OK.
          continue;
        }

        // Check if the function is initializer-only.
        if (reachability.externallyReachableFunctions.contains(parent)) {
          // Function is externally reachable - this is an error.
          auto funcOp = cast<FunctionOpInterface>(parent);
          emitDiagnostic(
              ERROR, storeOp->getLoc(),
              "store to immutable global '@" + global.getName() +
                  "' in function '" + funcOp.getName() +
                  "' which is reachable from non-initializer contexts");
          hadError = true;
        } else if (!reachability.initializerOnlyFunctions.contains(parent)) {
          // Function reachability is unknown - shouldn't happen but be
          // defensive.
          auto funcOp = cast<FunctionOpInterface>(parent);
          emitDiagnostic(WARNING, storeOp->getLoc(),
                         "store to immutable global '@" + global.getName() +
                             "' in function '" + funcOp.getName() +
                             "' with unknown reachability");
        } else {
          // Function is initializer-only. Check if it's in a conditional
          // context by looking for ops that implement RegionBranchOpInterface.
          bool inConditional = false;
          Operation *checkOp = storeOp;
          while (checkOp != parent) {
            checkOp = checkOp->getParentOp();
            if (isa<RegionBranchOpInterface>(checkOp)) {
              inConditional = true;
              break;
            }
          }

          // Else: unconditional store in initializer-only function - OK.
          if (inConditional) {
            auto funcOp = cast<FunctionOpInterface>(parent);
            emitDiagnostic(
                WARNING, storeOp->getLoc(),
                "conditional store to immutable global '@" + global.getName() +
                    "' in initializer-only function '" + funcOp.getName() +
                    "' may indicate complex initialization pattern");
          }
        }
      }

      return GlobalAction::PRESERVE;
    });

    return hadError ? failure() : success();
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
