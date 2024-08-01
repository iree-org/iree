// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/GlobalTable.h"

#include "mlir/Analysis/CallGraph.h"

namespace mlir::iree_compiler::IREE::Util {

// Returns a set of all top-level callable ops that are externally reachable.
// Callables only reachable from initializers are excluded.
static DenseSet<Operation *>
calculateExternallyReachableOps(ModuleOp moduleOp) {
  DenseSet<Operation *> externallyReachableOps;

  // Expensive; we want to avoid this unless the call graph changes.
  CallGraph callGraph(moduleOp);

  SetVector<CallGraphNode *> worklist;
  worklist.insert(callGraph.begin(), callGraph.end());
  while (!worklist.empty()) {
    auto *node = worklist.pop_back_val();
    if (node->isExternal()) {
      // Skip declarations.
      continue;
    }
    auto *callableOp = node->getCallableRegion()->getParentOp();
    if (isa<IREE::Util::InitializerOpInterface>(callableOp)) {
      // Initializers are never externally reachable.
      continue;
    }
    bool isExternallyReachable = externallyReachableOps.contains(callableOp);
    if (auto funcOp = dyn_cast<FunctionOpInterface>(callableOp)) {
      // Public functions exported on the module are externally reachable.
      isExternallyReachable |= funcOp.isPublic();
    }
    if (isExternallyReachable) {
      // Insert into the set of reachable ops and also any outgoing calls.
      // Queue up the edges in the worklist for further processing.
      externallyReachableOps.insert(callableOp);
      for (auto outgoingEdge : *node) {
        auto *calleeNode = outgoingEdge.getTarget();
        if (!calleeNode->isExternal()) {
          externallyReachableOps.insert(
              calleeNode->getCallableRegion()->getParentOp());
          worklist.insert(outgoingEdge.getTarget());
        }
      }
    }
  }

  return externallyReachableOps;
}

GlobalTable::GlobalTable(mlir::ModuleOp moduleOp) : moduleOp(moduleOp) {
  externallyReachableOps = calculateExternallyReachableOps(moduleOp);
}

void GlobalTable::rebuild() {
  globalOrder.clear();
  globalMap.clear();

  for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
    auto globalName = globalOp.getGlobalName();
    globalMap[globalName] = Global{globalOrder.size(), globalOp};
    globalOrder.push_back(globalName);
  }

  for (auto callableOp : moduleOp.getOps<CallableOpInterface>()) {
    if (auto uses = SymbolTable::getSymbolUses(callableOp)) {
      for (auto use : *uses) {
        auto leafRef = use.getSymbolRef().getLeafReference().getValue();
        auto it = globalMap.find(leafRef);
        if (it != globalMap.end()) {
          auto &global = it->second;
          auto *op = use.getUser();
          if (auto addressOp =
                  dyn_cast<IREE::Util::GlobalAddressOpInterface>(op)) {
            global.isIndirect = true;
          } else if (auto loadOp =
                         dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
            global.loadOps.push_back(loadOp);
          } else if (auto storeOp =
                         dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
            global.storeOps.push_back(storeOp);
          } else {
            global.referencingOps.push_back(op);
          }
        }
      }
    }
  }

  for (auto &[globalName, global] : globalMap) {
    bool anyNonInitializerStores = false;
    for (auto storeOp : global.storeOps) {
      auto callableOp = storeOp->getParentOfType<CallableOpInterface>();
      if (externallyReachableOps.contains(callableOp)) {
        anyNonInitializerStores = true;
        break;
      }
    }
    global.onlyInitialized = !anyNonInitializerStores;
  }
}

Global &GlobalTable::lookup(StringRef globalName) {
  return globalMap[globalName];
}

StringRef GlobalTable::lookupByOrdinal(size_t ordinal) const {
  return globalOrder[ordinal];
}

bool GlobalTable::forEach(std::function<GlobalAction(Global &global)> fn) {
  bool didChange = false;
  for (size_t i = 0; i < size();) {
    auto globalName = globalOrder[i];
    auto action = fn(globalMap[globalName]);
    switch (action) {
    case GlobalAction::PRESERVE: {
      ++i;
      break;
    }
    case GlobalAction::UPDATE: {
      didChange |= true;
      ++i;
      break;
    }
    case GlobalAction::DELETE: {
      didChange |= true;
      eraseGlobal(globalName);
      break;
    }
    }
  }
  return didChange;
}

void GlobalTable::renameGlobalUses(Global &sourceGlobal, Global &targetGlobal) {
  auto sourceAttr = FlatSymbolRefAttr::get(sourceGlobal.op.getGlobalName());
  auto targetAttr = FlatSymbolRefAttr::get(targetGlobal.op.getGlobalName());

  // Rename all global load ops.
  for (auto loadOp : sourceGlobal.loadOps) {
    loadOp.setGlobalAttr(targetAttr);
    targetGlobal.loadOps.push_back(loadOp);
  }
  sourceGlobal.loadOps.clear();

  // Rename all references via op attributes.
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](Attribute originalAttr)
                              -> AttrTypeReplacer::ReplaceFnResult<Attribute> {
    if (originalAttr == sourceAttr) {
      return std::make_pair(cast<Attribute>(targetAttr), WalkResult::advance());
    }
    return std::nullopt;
  });
  for (auto refOp : sourceGlobal.referencingOps) {
    replacer.recursivelyReplaceElementsIn(refOp);
    targetGlobal.referencingOps.push_back(refOp);
  }
  sourceGlobal.referencingOps.clear();
}

void GlobalTable::eraseGlobal(StringRef globalName) {
  auto &global = globalMap[globalName];
  assert(global.op.isGlobalPrivate() && "can't delete public globals");
  assert(global.loadOps.empty() && "must not be used");
  assert(global.referencingOps.empty() && "must not be referenced");
  global.eraseStores();
  globalMap.erase(globalName);
  llvm::erase(globalOrder, globalName);
  global.op.erase();
}

} // namespace mlir::iree_compiler::IREE::Util
