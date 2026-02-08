// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_BUILTINREGISTRY_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_BUILTINREGISTRY_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

// Registry of VM builtin helper functions used during dialect conversion.
//
// During conversion to the VM dialect, some source ops expand into calls to
// shared helper functions (e.g., integer-to-string conversion). Naively
// creating these helpers on-demand via lookupSymbol + create during pattern
// application causes O(n) module scans and mutates the module mid-conversion.
//
// BuiltinRegistry solves this with a four-phase lifecycle:
//   1. Registration: populate*Patterns functions call add() to register
//      builtins with their signatures and deferred body builders.
//   2. Declaration: declareAll() inserts empty vm.func declarations into the
//      module before conversion begins.
//   3. Conversion: patterns call use() to obtain the FuncOp and record usage.
//   4. Finalization: finalize() erases unused declarations and populates used
//      ones by invoking their body builders.
class BuiltinRegistry {
public:
  // Callback that populates a declared vm.func with its body.
  // The FuncOp has the correct signature but no entry block; the callback
  // must add blocks and populate the function body.
  using BodyBuilder =
      std::function<void(IREE::VM::FuncOp funcOp, OpBuilder &builder)>;

  // Registers a builtin helper with the given name, signature, and deferred
  // body builder. Must be called before declareAll().
  void add(StringRef name, FunctionType type, BodyBuilder bodyBuilder);

  // Inserts empty vm.func declarations for all registered builtins into the
  // module. Must be called once, before applyPartialConversion.
  void declareAll(Operation *moduleOp);

  // Returns the FuncOp for the named builtin and increments its use count.
  // Called by conversion patterns during matchAndRewrite.
  IREE::VM::FuncOp use(StringRef name);

  // Erases unused declarations (use count == 0) and populates used ones
  // (use count >= 1) by invoking their body builders. Must be called once,
  // after applyPartialConversion completes.
  void finalize();

private:
  struct Builtin {
    std::string name;
    FunctionType type;
    BodyBuilder bodyBuilder;
    IREE::VM::FuncOp funcOp; // Set by declareAll().
    unsigned useCount = 0;
  };
  // Ordered list for deterministic declaration order.
  SmallVector<Builtin> builtins;
  // Index for O(1) lookup by name.
  llvm::StringMap<unsigned> nameToIndex;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_BUILTINREGISTRY_H_
