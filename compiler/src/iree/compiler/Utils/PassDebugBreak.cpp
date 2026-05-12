// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/PassDebugBreak.h"

#include <chrono>
#include <iostream>
#include <string>
#include <system_error>
#include <thread>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler {

namespace {

llvm::cl::list<std::string> clBreakBefore(
    "iree-debug-break-before",
    llvm::cl::desc("Comma-separated list of pass argument names "
                   "(the `--pass-name` form) before which the pipeline will "
                   "pause for interactive IR editing."),
    llvm::cl::CommaSeparated);

llvm::cl::list<std::string> clBreakAfter(
    "iree-debug-break-after",
    llvm::cl::desc("Comma-separated list of pass argument names "
                   "(the `--pass-name` form) after which the pipeline will "
                   "pause for interactive IR editing."),
    llvm::cl::CommaSeparated);

// Default is empty so we follow the HAL pass-options convention (e.g.
// DumpExecutableSourcesPass `path`): no implicit `/tmp/` path. The factory
// emits a clear error if a break flag is set without an explicit
// --iree-debug-break-file path, instead of silently writing to a hard-coded
// location.
llvm::cl::opt<std::string> clBreakFile(
    "iree-debug-break-file",
    llvm::cl::desc("Path to the file used to exchange IR between the "
                   "compiler and the user during an interactive break. "
                   "Required whenever --iree-debug-break-before or "
                   "--iree-debug-break-after is set."),
    llvm::cl::init(""));

llvm::cl::opt<std::string> clTestPatchFile(
    "iree-debug-break-test-patch-file",
    llvm::cl::desc("[test only] Path to a patch file. When set, each break "
                   "auto-splices from this file and resumes without waiting — "
                   "no stdin or sentinel file is required. Same patch is "
                   "applied at every matching break. Real interactive "
                   "workflows should use the stdin or file mode."),
    llvm::cl::init(""), llvm::cl::Hidden);

llvm::cl::opt<DebugBreakMode> clBreakMode(
    "iree-debug-break-mode",
    llvm::cl::desc("How the compiler waits for a resume decision at a break:"),
    llvm::cl::values(
        clEnumValN(DebugBreakMode::Stdin, "stdin",
                   "Read 'continue' or 'abort' from stdin (default)."),
        clEnumValN(DebugBreakMode::File, "file",
                   "Poll for <break-file>.continue or <break-file>.abort "
                   "sentinel files.")),
    llvm::cl::init(DebugBreakMode::Stdin));

llvm::cl::opt<unsigned> clBreakPollIntervalMs(
    "iree-debug-break-poll-interval-ms",
    llvm::cl::desc("File mode: interval in milliseconds between sentinel "
                   "file polls. Lower values reduce latency at a small CPU "
                   "cost. No effect in stdin mode."),
    llvm::cl::init(100), llvm::cl::Hidden);

// Defaults to one hour. 0 is supported and means "wait forever" (matches
// stdin behavior), but a non-zero default protects agent loops and CI
// drivers from silent hangs when the driver dies or forgets to emit a
// sentinel. Humans at a terminal can pass --iree-debug-break-timeout-sec=0
// for sessions longer than an hour.
llvm::cl::opt<unsigned> clBreakTimeoutSec(
    "iree-debug-break-timeout-sec",
    llvm::cl::desc("File mode: timeout in seconds for a single break's "
                   "poll loop. Default 3600 (one hour). 0 = wait forever, "
                   "matching stdin mode. A non-zero value aborts the "
                   "compile if no sentinel arrives in time."),
    llvm::cl::init(3600), llvm::cl::Hidden);

// Returns the stderr stream used for interactive break banners.
//
// Stderr is the right channel here, even though `--mlir-print-ir-after-all`
// also writes to stderr (so a user redirecting `2>` will see banners and IR
// dumps mixed together): stdout carries the compiler's primary output (a
// .vmfb binary or textual IR depending on flags), and writing banners
// there would corrupt that output. Co-mingling with other stderr
// diagnostics is the lesser evil and matches MLIR convention — every
// other diagnostic in the compiler also lands on stderr.
//
// We bypass MLIR's diagnostic engine because the default handler annotates
// every remark/error with "see current operation: <full module dump>"
// when the context has shouldPrintOpOnDiagnostic=true, which would flood
// the banner with the entire IR at every break.
//
// Kept as a forwarder to `llvm::errs()` rather than a standalone
// `raw_fd_ostream(STDERR_FILENO, …)` so buffered writes interleave cleanly
// with the rest of the compiler's stderr output. If `llvm::errs()` ever
// gets rebound away from stderr, this helper is the single place to swap
// in a direct fd-2 stream.
static inline llvm::raw_ostream &breakLog() { return llvm::errs(); }

// Emits a user-facing error and terminates the process. Used for conditions
// the user can correct (wrong patch path, closed stdin, timeout, explicit
// abort, missing required flag).
//
// Exits via `llvm::report_fatal_error(msg, /*GenCrashDiag=*/false)` rather
// than `std::exit`: that's the LLVM-idiomatic shutdown path (used widely in
// LLVM/MLIR), runs the registered fatal-error handlers, and the
// `GenCrashDiag=false` form skips the bug-report banner / stack trace that
// would otherwise be misleading for a user-triggered condition.
//
// `std::exit` is not the right tool here because PassInstrumentation
// override returns void — we can't propagate a `LogicalResult failure` back
// to the pass manager. Once an unrecoverable user-facing condition is
// reached, the only options are abort-style termination or doing nothing.
// Termination is correct; we just want it without the crash-report styling.
[[noreturn]] static void exitWithUserError(const llvm::Twine &msg) {
  breakLog() << "[iree-debug-break] error: " << msg << "\n";
  breakLog().flush();
  llvm::report_fatal_error(msg, /*GenCrashDiag=*/false);
}

// Warns about any entry in |names| that doesn't resolve to a registered
// pass. Catches typos and flags pipeline-only names that would otherwise
// silently never fire a break. Does not check op-scope compatibility —
// PassRegistryEntry doesn't expose the target op name, and constructing the
// pass just to inspect it is heavier than the warning's value.
static void warnOnUnknownPasses(const llvm::StringSet<> &names,
                                llvm::StringRef flagName) {
  for (const auto &entry : names) {
    llvm::StringRef name = entry.first();
    if (!PassInfo::lookup(name)) {
      breakLog() << "[iree-debug-break] warning: --iree-debug-break-"
                 << flagName << "=" << name
                 << " does not match any registered pass; this break will "
                    "never fire\n";
    }
  }
  breakLog().flush();
}

// Writes |op| to |path| in textual MLIR form. Uses OF_None so the bytes on
// disk match what `op.print` produced — OF_Text would apply CRLF translation
// on Windows, changing checksums for drivers that hash the exchange file.
//
// For non-module ops we dump the enclosing ModuleOp rather than the op
// itself, so cross-references (e.g. a `util.call` to a sibling function)
// resolve when the file is re-parsed and verified standalone. The splice
// in parseAndReplace still targets the specific live op identified by
// op-name + symbol-name; edits the user makes to sibling ops in the dump
// are silently discarded. That trade-off (dump for context, splice for
// safety) is documented in the public header.
static LogicalResult writeOpToFile(Operation *op, llvm::StringRef path) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    breakLog() << "[iree-debug-break] failed to open " << path << ": "
               << ec.message() << "\n";
    return failure();
  }
  Operation *toPrint = op;
  if (!isa<ModuleOp>(op)) {
    if (auto parentModule = op->getParentOfType<ModuleOp>()) {
      toPrint = parentModule;
    }
  }
  toPrint->print(os);
  os << "\n";
  return success();
}

// Builds the qualified symbol path for an op as "@grandparent::@parent::@op",
// stopping at the top-level ModuleOp (which doesn't usually carry a symbol).
// Returns an empty string if the op has no symbol-bearing ancestor at all.
//
// Needed because leaf symbol names aren't necessarily unique across nested
// symbol-bearing scopes: a pipeline can have two ops of the same kind with
// the same leaf symbol name living under different parent symbol scopes.
// Matching by qualified path keeps siblings from colliding when we splice.
static std::string buildSymbolPath(Operation *op) {
  llvm::SmallVector<llvm::StringRef> segments;
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (auto sym = dyn_cast<SymbolOpInterface>(cur)) {
      llvm::StringRef name = sym.getName();
      if (!name.empty()) {
        segments.push_back(name);
      }
    }
    if (isa<ModuleOp>(cur)) {
      break;
    }
  }
  std::string result;
  llvm::raw_string_ostream os(result);
  for (size_t i = segments.size(); i > 0; --i) {
    if (i != segments.size()) {
      os << "::";
    }
    os << "@" << segments[i - 1];
  }
  return result;
}

// Walks |parsedRoot| to find an op matching |liveOp|. Prefers an exact
// match on op-name + qualified symbol path (e.g.
// `hal.executable.variant` + `@sum_dispatch_0::@embedded_elf_x86_64`).
// Falls back to op-name + leaf symbol name, then op-name only — these
// fallbacks let users rename the symbol in the dump and still get their
// body edits spliced. The inherent sym_name on the live op is preserved
// regardless; we emit a stderr note when a fallback is used so the user
// isn't surprised that their rename didn't propagate.
static Operation *findMatchingOp(ModuleOp parsedRoot, Operation *liveOp) {
  llvm::StringRef opName = liveOp->getName().getStringRef();
  llvm::StringRef liveSym;
  if (auto symOp = dyn_cast<SymbolOpInterface>(liveOp)) {
    liveSym = symOp.getName();
  }
  std::string livePath = buildSymbolPath(liveOp);

  Operation *pathMatch = nullptr;
  Operation *symMatch = nullptr;
  Operation *anyMatch = nullptr;
  parsedRoot->walk([&](Operation *candidate) -> WalkResult {
    if (candidate->getName().getStringRef() != opName) {
      return WalkResult::advance();
    }
    if (!anyMatch) {
      anyMatch = candidate;
    }
    if (liveSym.empty()) {
      // Live op has no symbol; first op-name match is the answer.
      return WalkResult::interrupt();
    }
    auto candidateSym = dyn_cast<SymbolOpInterface>(candidate);
    if (!candidateSym || candidateSym.getName() != liveSym) {
      return WalkResult::advance();
    }
    if (!symMatch) {
      symMatch = candidate;
    }
    if (!livePath.empty() && buildSymbolPath(candidate) == livePath) {
      pathMatch = candidate;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (pathMatch) {
    return pathMatch;
  }
  if (symMatch) {
    if (!livePath.empty()) {
      breakLog() << "[iree-debug-break] note: live op " << livePath
                 << " not found by qualified path; falling back to first "
                 << opName << " @" << liveSym
                 << " match. If your module has multiple symbols with the "
                    "same leaf name (e.g. multiple hal.executable.variants),"
                    " the splice may target the wrong one.\n";
      breakLog().flush();
    }
    return symMatch;
  }
  if (anyMatch && !liveSym.empty()) {
    breakLog() << "[iree-debug-break] note: live op @" << liveSym
               << " not found in parsed dump; falling back to first " << opName
               << " match. The live op's symbol name is inherent and won't "
                  "change.\n";
    breakLog().flush();
  }
  return anyMatch;
}

// Parses |path| as a ModuleOp in |liveOp|'s context, finds the op in the
// parsed tree that corresponds to |liveOp| (identity preserved by op-name +
// symbol-name where applicable), transfers its regions and discardable attrs
// into |liveOp|, and runs the MLIR verifier on the spliced result.
//
// For ModuleOp itself we just splice the parsed module's body into the live
// module — the op-name match would also work but the direct path avoids the
// walk and matches the historical behavior. Only discardable attributes are
// copied so any inherent attributes the op type owns aren't stomped.
static LogicalResult parseAndReplace(Operation *liveOp, llvm::StringRef path) {
  MLIRContext *context = liveOp->getContext();
  OwningOpRef<ModuleOp> parsed =
      parseSourceFile<ModuleOp>(path, ParserConfig(context));
  if (!parsed) {
    breakLog() << "[iree-debug-break] failed to re-parse " << path << "\n";
    return failure();
  }

  if (auto liveModule = dyn_cast<ModuleOp>(liveOp)) {
    liveModule.getBodyRegion().takeBody(parsed->getBodyRegion());
    liveModule->setDiscardableAttrs(
        parsed->getOperation()->getDiscardableAttrDictionary());
  } else {
    Operation *match = findMatchingOp(*parsed, liveOp);
    if (!match) {
      breakLog() << "[iree-debug-break] re-parse: no op matching "
                 << liveOp->getName();
      if (auto symOp = dyn_cast<SymbolOpInterface>(liveOp)) {
        breakLog() << " @" << symOp.getName();
      }
      breakLog() << " found in " << path << "\n";
      return failure();
    }
    if (match->getNumRegions() != liveOp->getNumRegions()) {
      breakLog() << "[iree-debug-break] re-parse: region count mismatch ("
                 << match->getNumRegions() << " vs " << liveOp->getNumRegions()
                 << ") for " << liveOp->getName() << " from " << path << "\n";
      return failure();
    }
    for (auto [liveReg, matchReg] :
         llvm::zip(liveOp->getRegions(), match->getRegions())) {
      liveReg.takeBody(matchReg);
    }
    liveOp->setDiscardableAttrs(match->getDiscardableAttrDictionary());
  }
  // parseSourceFile verifies the parsed module in isolation; this verifies
  // the live op after the splice. Catches post-splice inconsistencies and
  // routes diagnostics through the live op tree.
  if (failed(mlir::verify(liveOp))) {
    breakLog() << "[iree-debug-break] verifier rejected spliced "
               << liveOp->getName() << " from " << path << "\n";
    return failure();
  }
  return success();
}

// Outcome of a single wait-for-signal step.
enum class BreakSignal {
  Continue,
  Abort,
  Unrecognized, // stdin mode only; caller should re-prompt.
};

// Blocks on stdin for "continue"/"abort" (or their "c"/"a" shortcuts).
// Returns the parsed decision.
static BreakSignal waitStdin() {
  std::string rawLine;
  if (!std::getline(std::cin, rawLine)) {
    exitWithUserError("stdin closed while waiting for continue/abort");
  }
  llvm::StringRef line = llvm::StringRef(rawLine).rtrim();
  if (line == "continue" || line == "c") {
    return BreakSignal::Continue;
  }
  if (line == "abort" || line == "a") {
    return BreakSignal::Abort;
  }
  breakLog() << "[iree-debug-break] unrecognized input '" << line
             << "'; expected 'continue'/'c' or 'abort'/'a'.\n";
  breakLog().flush();
  return BreakSignal::Unrecognized;
}

// Polls for sentinel files <breakFile>.continue or <breakFile>.abort. The
// first one observed is deleted and its meaning returned. If both exist at
// the same poll tick, `.abort` wins (safer default).
//
// Contract: any sentinel file present when this call starts is consumed as
// a valid signal. The compiler cannot distinguish a legitimately
// pre-created sentinel (driver racing ahead of first break) from a stale
// one left by a previous crashed invocation. Drivers that care about
// avoiding stale-consume should `rm -f <breakFile>.{continue,abort}` at
// their own startup before launching the compiler. A PID-scoped sentinel
// name ('<breakFile>.pid<N>.continue') would let us detect staleness in
// the compiler, but complicates the driver protocol; left as future work.
//
// TODO: Polling is a portable expedient. The hot path here is
// `llvm::sys::fs::exists`, which on POSIX dispatches to a `stat`
// syscall per tick. A real fix would move to inotify on Linux (and
// kqueue on BSD/macOS, ReadDirectoryChangesW on Windows) so we wake
// on the actual filesystem event instead of polling. The polling path
// would remain as a fallback on platforms without a dedicated watcher.
static BreakSignal waitFileSentinels(llvm::StringRef breakFile,
                                     unsigned pollIntervalMs,
                                     unsigned timeoutSec) {
  std::string continuePath = (llvm::Twine(breakFile) + ".continue").str();
  std::string abortPath = (llvm::Twine(breakFile) + ".abort").str();
  auto consumeSentinel = [](llvm::StringRef path) {
    if (std::error_code ec = llvm::sys::fs::remove(path)) {
      breakLog() << "[iree-debug-break] warning: failed to delete sentinel "
                 << path << ": " << ec.message() << "\n";
      breakLog().flush();
    }
  };

  const std::chrono::milliseconds pollInterval(pollIntervalMs);
  const std::chrono::steady_clock::time_point deadline =
      timeoutSec == 0
          ? std::chrono::steady_clock::time_point::max()
          : std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
  while (true) {
    if (llvm::sys::fs::exists(abortPath)) {
      consumeSentinel(abortPath);
      return BreakSignal::Abort;
    }
    if (llvm::sys::fs::exists(continuePath)) {
      consumeSentinel(continuePath);
      return BreakSignal::Continue;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      exitWithUserError(llvm::Twine("timed out after ") +
                        std::to_string(timeoutSec) +
                        "s waiting for a sentinel next to " + breakFile);
    }
    std::this_thread::sleep_for(pollInterval);
  }
}

} // namespace

PassDebugBreakInstrumentation::PassDebugBreakInstrumentation(
    MLIRContext *context, llvm::StringSet<> breakBeforePasses,
    llvm::StringSet<> breakAfterPasses, std::string breakFile,
    std::string patchFile, DebugBreakMode mode, unsigned pollIntervalMs,
    unsigned timeoutSec)
    : context(context),
      priorMultithreadingEnabled(context->isMultithreadingEnabled()),
      breakBeforePasses(std::move(breakBeforePasses)),
      breakAfterPasses(std::move(breakAfterPasses)),
      breakFile(std::move(breakFile)), patchFile(std::move(patchFile)),
      mode(mode), pollIntervalMs(pollIntervalMs), timeoutSec(timeoutSec) {
  // Debug-break is an interactive, single-consumer protocol (one stdin,
  // one exchange file). Disable MLIRContext multithreading for the
  // lifetime of this instrumentation so parallelized nested-op pipelines don't
  // race on stdin and the exchange file. The destructor restores the prior
  // state so a long-lived session (e.g. Python bindings) that reuses the
  // context for later non-debug compiles stays multithreaded.
  if (priorMultithreadingEnabled) {
    context->disableMultithreading();
    breakLog() << "[iree-debug-break] multithreading disabled for the "
                  "lifetime of this pass manager.\n";
    breakLog().flush();
  }
}

PassDebugBreakInstrumentation::~PassDebugBreakInstrumentation() {
  if (priorMultithreadingEnabled) {
    context->enableMultithreading();
  }
}

void PassDebugBreakInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  llvm::StringRef argument = pass->getArgument();
  if (argument.empty() || !breakBeforePasses.contains(argument)) {
    return;
  }
  breakHere(pass, op, "before");
}

void PassDebugBreakInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  llvm::StringRef argument = pass->getArgument();
  if (argument.empty() || !breakAfterPasses.contains(argument)) {
    return;
  }
  breakHere(pass, op, "after");
}

// Appends "<op-name> [@<symbol>]" to |os| as a stable identifier for the
// op the break fired on. Helps users distinguish iterations when a
// function-scope pass fires once per function.
static void printOpIdentifier(llvm::raw_ostream &os, Operation *op) {
  os << op->getName();
  if (auto symOp = dyn_cast<SymbolOpInterface>(op)) {
    os << " @" << symOp.getName();
  }
}

void PassDebugBreakInstrumentation::breakHere(Pass *pass, Operation *op,
                                              llvm::StringRef phase) {
  llvm::StringRef argument = pass->getArgument();

  if (failed(writeOpToFile(op, breakFile))) {
    exitWithUserError(llvm::Twine("failed to write ") + breakFile);
  }

  breakLog() << "[iree-debug-break] BEGIN phase=" << phase
             << " pass=" << argument << " op=";
  printOpIdentifier(breakLog(), op);
  breakLog() << " file=" << breakFile << "\n";
  if (!patchFile.empty()) {
    breakLog() << "[iree-debug-break] auto-patching from " << patchFile << "\n";
    breakLog().flush();
    if (failed(parseAndReplace(op, patchFile))) {
      exitWithUserError(llvm::Twine("auto-patch failed for ") + patchFile);
    }
    breakLog() << "[iree-debug-break] END phase=" << phase
               << " pass=" << argument << "\n";
    breakLog().flush();
    return;
  }
  if (mode == DebugBreakMode::File) {
    breakLog() << "[iree-debug-break] Edit the file, then `touch " << breakFile
               << ".continue` (or `.abort`) to signal.\n";
  } else {
    breakLog() << "[iree-debug-break] Edit the file, then type "
                  "'continue' ('c') or 'abort' ('a') and press Enter.\n";
  }
  breakLog().flush();

  // Loop until either (a) the caller signals 'continue' and the edited file
  // parses+verifies successfully, or (b) the caller signals 'abort'. A failed
  // parse re-prompts without losing the live module state.
  while (true) {
    BreakSignal signal =
        (mode == DebugBreakMode::File)
            ? waitFileSentinels(breakFile, pollIntervalMs, timeoutSec)
            : waitStdin();
    switch (signal) {
    case BreakSignal::Abort:
      exitWithUserError(llvm::Twine("aborted ") + phase + " pass '" + argument +
                        "'");
    case BreakSignal::Unrecognized:
      continue;
    case BreakSignal::Continue:
      break;
    }

    if (succeeded(parseAndReplace(op, breakFile))) {
      break;
    }
    if (mode == DebugBreakMode::File) {
      breakLog() << "[iree-debug-break] re-parse failed — fix " << breakFile
                 << " and `touch " << breakFile
                 << ".continue` again (or `.abort`).\n";
    } else {
      breakLog() << "[iree-debug-break] re-parse failed — fix " << breakFile
                 << " and send 'continue' again, or 'abort' to exit.\n";
    }
    breakLog().flush();
  }

  breakLog() << "[iree-debug-break] END phase=" << phase << " pass=" << argument
             << "\n";
  breakLog().flush();
}

std::unique_ptr<PassDebugBreakInstrumentation>
createPassDebugBreakInstrumentationFromFlags(MLIRContext *context) {
  auto collect = [](const llvm::cl::list<std::string> &flag) {
    llvm::StringSet<> set;
    for (const std::string &name : flag) {
      if (!name.empty()) {
        set.insert(name);
      }
    }
    return set;
  };
  llvm::StringSet<> before = collect(clBreakBefore);
  llvm::StringSet<> after = collect(clBreakAfter);
  if (before.empty() && after.empty()) {
    return nullptr;
  }
  if (clBreakFile.empty()) {
    exitWithUserError(
        "--iree-debug-break-{before,after} requires "
        "--iree-debug-break-file=<path> (no implicit default; pick a "
        "path the driver can read/write)");
  }
  warnOnUnknownPasses(before, "before");
  warnOnUnknownPasses(after, "after");
  return std::make_unique<PassDebugBreakInstrumentation>(
      context, std::move(before), std::move(after), clBreakFile,
      clTestPatchFile, clBreakMode, clBreakPollIntervalMs, clBreakTimeoutSec);
}

} // namespace mlir::iree_compiler
