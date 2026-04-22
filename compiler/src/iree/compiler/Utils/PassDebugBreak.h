// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_PASSDEBUGBREAK_H_
#define IREE_COMPILER_UTILS_PASSDEBUGBREAK_H_

#include <memory>
#include <string>

#include "llvm/ADT/StringSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassInstrumentation.h"

namespace mlir::iree_compiler {

// How the interactive break waits for a resume decision.
enum class DebugBreakMode {
  // Reads a line from stdin ("continue"/"c" -> resume, "abort"/"a" -> abort).
  // Default. Best for humans at a terminal.
  Stdin,
  // Polls for sentinel files next to the break file:
  //   <breakFile>.continue -> resume
  //   <breakFile>.abort    -> abort
  // The sentinel is deleted once observed. Best for agent loops and other
  // out-of-process drivers that can't own the compiler's stdin.
  File,
};

// Pauses the pass pipeline around selected passes so the user can hand-edit
// the IR the pass operates on, then resumes with the edited IR. Intended for
// interactive prototyping — pair with agent loops that patch IR between
// passes without rebuilding the compiler.
//
// The instrumentation fires:
//   * Before a pass whose `getArgument()` is in |breakBeforePasses| — lets
//     the user edit the IR entering that pass.
//   * After a pass whose `getArgument()` is in |breakAfterPasses| — lets
//     the user edit the IR that pass produced.
//
// Works at any op scope. Both `ModuleOp` (most pipeline-level passes) and
// nested ops like `func.func`/`util.func` (function-scope passes inside a
// `FunctionLikeNest`) and other symbol-bearing ops are supported. The
// banner reports the op name and (when applicable) symbol name so the user
// can tell which iteration of a per-function pass fired the break.
//
// At each break, the instrumentation:
//   1. Writes the running op (or its enclosing ModuleOp, when the op is
//      not itself a ModuleOp) to |breakFile|. Dumping the enclosing
//      module ensures cross-references like `util.call` to sibling
//      functions resolve when the file is re-parsed and verified.
//   2. Prints a machine-parseable banner to stderr (phase=before|after,
//      pass=<arg>, op=<op-name> [@<symbol>], file=<path>).
//   3. Either (a) if |patchFile| is non-empty, immediately re-parses
//      |patchFile| and splices the matching op back into the live op,
//      skipping the wait (non-interactive "auto-patch" mode — useful for
//      scripted workflows and lit tests); or (b) blocks waiting for a
//      resume decision using |mode|, then re-parses |breakFile| after
//      the user edits it.
//
// Splice semantics: for `ModuleOp` we transfer the parsed module's body
// region wholesale. For other ops we walk the parsed module to find an op
// matching the live op by op-name (and symbol-name when present), then
// transfer that op's regions and discardable attributes into the live op.
// Inherent attributes (sym_name, function_type, etc.) are preserved on
// the live op so type/signature info stays consistent.
//
// Caveat for function-scope (and other non-module) breaks: the dump shows
// the enclosing module for context, but only edits to the op the break
// fired on take effect. Edits to sibling ops in the dump are silently
// discarded — the splice rewires only the live op's regions and attrs.
// If you need to mutate multiple ops, break at module scope instead.
//
// Side effect on construction/destruction: this instrumentation disables
// MLIRContext multithreading for its own lifetime, then restores it.
//
// Why disable: a `PassManager` may run nested `OpPassManager`s in parallel
// across all matching ops in a pipeline (controlled by
// `MLIRContext::isMultithreadingEnabled()`; each nested adaptor consults
// the context's thread pool). When that's on, several worker threads can
// hit `runBeforePass` / `runAfterPass` concurrently for sibling ops at the
// same nest level. Our protocol has a single stdin and a single
// |breakFile| on disk, so multiple concurrent breaks would race on both
// ends. Disabling multithreading on the context serializes the pass
// invocations the manager dispatches, ensuring at most one break is
// active at a time.
//
// We capture `priorMultithreadingEnabled` at construction so the
// destructor can put it back: a long-lived session (e.g. Python bindings)
// that reuses the same context for subsequent non-debug compiles
// shouldn't be silently left single-threaded after we go out of scope.
class PassDebugBreakInstrumentation : public PassInstrumentation {
public:
  PassDebugBreakInstrumentation(MLIRContext *context,
                                llvm::StringSet<> breakBeforePasses,
                                llvm::StringSet<> breakAfterPasses,
                                std::string breakFile, std::string patchFile,
                                DebugBreakMode mode, unsigned pollIntervalMs,
                                unsigned timeoutSec);
  ~PassDebugBreakInstrumentation() override;

  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;

private:
  // Shared break logic. |phase| is "before" or "after" and only controls the
  // banner text.
  void breakHere(Pass *pass, Operation *op, llvm::StringRef phase);

  MLIRContext *context;
  // The context's `isMultithreadingEnabled()` state captured at
  // construction. The destructor uses it to restore the original setting
  // (instead of unconditionally re-enabling) so a context that was
  // already single-threaded before us stays single-threaded after us.
  // See the class comment above for why we toggle this in the first
  // place.
  bool priorMultithreadingEnabled;

  llvm::StringSet<> breakBeforePasses;
  llvm::StringSet<> breakAfterPasses;
  std::string breakFile;
  // When non-empty, the break auto-splices from this path and resumes without
  // waiting. Empty means "interactive mode via |mode|".
  std::string patchFile;
  DebugBreakMode mode;
  // File-mode poll interval and timeout. Ignored in stdin mode.
  // |timeoutSec| == 0 means wait forever (mirrors stdin semantics).
  unsigned pollIntervalMs;
  unsigned timeoutSec;
};

// Returns a debug-break instrumentation configured from command-line flags,
// or nullptr if both `--iree-debug-break-before` and `--iree-debug-break-after`
// are empty. |context| is the MLIRContext the pass manager will run on —
// its multithreading state is captured at construction and restored when
// the instrumentation is destroyed.
std::unique_ptr<PassDebugBreakInstrumentation>
createPassDebugBreakInstrumentationFromFlags(MLIRContext *context);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_PASSDEBUGBREAK_H_
