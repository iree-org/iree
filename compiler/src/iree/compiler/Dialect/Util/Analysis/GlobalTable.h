// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_ANALYSIS_GLOBALTABLE_H_
#define IREE_COMPILER_DIALECT_IREE_ANALYSIS_GLOBALTABLE_H_

#include <functional>

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::Util {

// An entry in a GlobalTable representing a util.global (or related) op.
struct Global {
  // Ordinal of the global in the parent module.
  size_t ordinal = 0;

  // Global this information relates to.
  IREE::Util::GlobalOpInterface op;

  // True if the address of the global is ever taken.
  // This disables most optimizations here; we could use some data flow analysis
  // to track potential operations on globals via the addresses but we don't
  // currently have any input programs that require doing so.
  bool isIndirect = false;

  // All util.global.load ops referencing the global.
  SmallVector<IREE::Util::GlobalLoadOpInterface> loadOps;
  // All util.global.store ops referencing the global.
  SmallVector<IREE::Util::GlobalStoreOpInterface> storeOps;
  // All other operations that reference the global via an attribute.
  SmallVector<Operation *> referencingOps;

  // Returns the symbol name of the global op.
  StringRef getName() { return op.getGlobalName().getValue(); }

  // Returns true if the global is a candidate for folding.
  bool isCandidate() { return !isIndirect && op.isGlobalPrivate(); }

  // TODO(benvanik): refine how we determine whether we can DCE things. Today we
  // can be too aggressive with certain types that may be side-effecting though
  // that shouldn't be the case: the IREE execution model should not require
  // globals to be stored to be correct as anything using a reference type
  // should be capturing it. Unfortunately today our DCE is not comprehensive
  // enough to be safe.
  //
  // Returns true if the global can be DCEd if there are no loads.
  // This is generally only the case for value types as reference types may be
  // aliased or have side effects on creation.
  bool canDCE() {
    return isCandidate() &&
           !isa<IREE::Util::ReferenceTypeInterface>(op.getGlobalType());
  }

  // Erases all stores to the global.
  void eraseStores() {
    for (auto storeOp : storeOps) {
      storeOp.erase();
    }
    storeOps.clear();
  }
};

// Action to perform on a global under enumeration.
enum class GlobalAction {
  // Preserve the global in the program as-is.
  PRESERVE,
  // Global has been updated and another iteration of the pass may be required.
  UPDATE,
  // Delete the global as it is unused.
  DELETE,
};

// A constructed table of analyzed globals in a module with some utilities for
// manipulating them. This is designed for simple uses and more advanced
// analysis should be performed with an Explorer or DFX.
struct GlobalTable {
  GlobalTable() = delete;
  explicit GlobalTable(mlir::ModuleOp moduleOp);

  MLIRContext *getContext() { return moduleOp.getContext(); }

  // Total number of globals in the module.
  size_t size() const { return globalOrder.size(); }

  // Returns the information for the given global.
  Global &lookup(StringRef globalName);
  Global &lookup(StringAttr globalName) {
    return lookup(globalName.getValue());
  }

  // Returns the global with the given ordinal.
  StringRef lookupByOrdinal(size_t ordinal) const;

  // Enumerates all globals in the program and calls the given |fn|.
  // The function should return an action as to what should be done with the
  // global.
  // Returns true if any changes were made.
  bool forEach(std::function<GlobalAction(Global &global)> fn);

  // Renames all uses of |sourceGlobal| into |targetGlobal|.
  // The source global and stores to it are preserved.
  void renameGlobalUses(Global &sourceGlobal, Global &targetGlobal);

  // Erases the global with the given name.
  // Must have no loads or references remaining.
  void eraseGlobal(StringRef globalName);

private:
  void rebuild();

  // Module under analysis.
  mlir::ModuleOp moduleOp;
  // All globals in the order they are declared by symbol name.
  SmallVector<StringRef> globalOrder;
  // A map of global symbol names to analysis results.
  DenseMap<StringRef, Global> globalMap;
};

} // namespace mlir::iree_compiler::IREE::Util

#endif // IREE_COMPILER_DIALECT_IREE_ANALYSIS_GLOBALTABLE_H_
