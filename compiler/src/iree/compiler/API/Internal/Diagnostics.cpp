// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/API/Internal/Diagnostics.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"

// Much of the formatting logic below was adapted from the
// SourceMgrDiagnosticHandler but without the logic for printing actual source
// contents and other adaptations to detach it from a SourceMgr.
// It would be nice to convert the upstream class into something more composable
// and re-use but it is not immediately obvious how to do that given the
// tie-ins to lower level LLVM infra.

namespace mlir::iree_compiler::embed {

namespace {
/// Return a processable CallSiteLoc from the given location.
std::optional<CallSiteLoc> getCallSiteLoc(Location loc) {
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) return callLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return getCallSiteLoc(cast<NameLoc>(loc).getChildLoc());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (auto subLoc : cast<FusedLoc>(loc).getLocations()) {
      if (auto callLoc = getCallSiteLoc(subLoc)) {
        return callLoc;
      }
    }
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<Location> findLocToShow(Location loc) {
  // Recurse into the child locations of some of location types.
  return TypeSwitch<LocationAttr, std::optional<Location>>(loc)
      .Case([&](CallSiteLoc callLoc) -> std::optional<Location> {
        // We recurse into the callee of a call site, as the caller will be
        // emitted in a different note on the main diagnostic.
        return findLocToShow(callLoc.getCallee());
      })
      .Case([&](FileLineColLoc) -> std::optional<Location> { return loc; })
      .Case([&](FusedLoc fusedLoc) -> std::optional<Location> {
        // Fused location is unique in that we try to find a sub-location to
        // show, rather than the top-level location itself.
        for (Location childLoc : fusedLoc.getLocations())
          if (std::optional<Location> showableLoc = findLocToShow(childLoc))
            return showableLoc;
        return std::nullopt;
      })
      .Case([&](NameLoc nameLoc) -> std::optional<Location> {
        return findLocToShow(nameLoc.getChildLoc());
      })
      .Case([&](OpaqueLoc opaqueLoc) -> std::optional<Location> {
        // OpaqueLoc always falls back to a different source location.
        return findLocToShow(opaqueLoc.getFallbackLocation());
      })
      .Case([](UnknownLoc) -> std::optional<Location> {
        // Prefer not to show unknown locations.
        return std::nullopt;
      });
}

}  // namespace

FormattingDiagnosticHandler::FormattingDiagnosticHandler(MLIRContext *ctx,
                                                         Callback callback)
    : ctx(ctx), callback(std::move(callback)) {
  handlerID = ctx->getDiagEngine().registerHandler(std::bind(
      &FormattingDiagnosticHandler::emit, this, std::placeholders ::_1));
}

FormattingDiagnosticHandler::~FormattingDiagnosticHandler() {
  ctx->getDiagEngine().eraseHandler(handlerID);
}

LogicalResult FormattingDiagnosticHandler::emit(Diagnostic &diag) {
  std::string messageAccum;
  llvm::raw_string_ostream os(messageAccum);

  auto appendDiag = [&](Location loc, Twine message,
                        DiagnosticSeverity severity) {
    if (!messageAccum.empty()) {
      os << "\n\n";
    }

    // Extract a file location from this loc.
    auto fileLoc = loc->findInstanceOf<FileLineColLoc>();
    if (fileLoc) {
      os << fileLoc.getFilename().getValue() << ":" << fileLoc.getLine() << ":"
         << fileLoc.getColumn() << ": ";
    } else {
      os << loc << ": ";
    }

    os << message;
  };

  // Assemble location fragments.
  SmallVector<std::pair<Location, StringRef>> locationStack;
  auto addLocToStack = [&](Location loc, StringRef locContext) {
    if (std::optional<Location> showableLoc = findLocToShow(loc))
      locationStack.emplace_back(*showableLoc, locContext);
  };

  // Add locations to display for this diagnostic.
  Location loc = diag.getLocation();
  addLocToStack(loc, /*locContext=*/{});

  // If the diagnostic location was a call site location, add the call stack as
  // well.
  if (auto callLoc = getCallSiteLoc(loc)) {
    // Print the call stack while valid, or until the limit is reached.
    loc = callLoc->getCaller();
    const unsigned callStackLimit = 50;
    for (unsigned curDepth = 0; curDepth < callStackLimit; ++curDepth) {
      addLocToStack(loc, "called from");
      if ((callLoc = getCallSiteLoc(loc)))
        loc = callLoc->getCaller();
      else
        break;
    }
  }

  // If the location stack is empty, use the initial location.
  // Otherwise, use the location stack.
  if (locationStack.empty()) {
    appendDiag(diag.getLocation(), diag.str(), diag.getSeverity());
  } else {
    appendDiag(locationStack.front().first, diag.str(), diag.getSeverity());
    for (auto &it : llvm::drop_begin(locationStack))
      appendDiag(it.first, it.second, DiagnosticSeverity::Note);
  }

  // Append each of the notes.
  for (auto &note : diag.getNotes()) {
    appendDiag(note.getLocation(), note.str(), note.getSeverity());
  }

  // Emit.
  callback(diag.getSeverity(), os.str());

  return success();
}

}  // namespace mlir::iree_compiler::embed
