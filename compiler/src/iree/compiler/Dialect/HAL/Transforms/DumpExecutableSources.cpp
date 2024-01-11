// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/LocationSnapshot.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_DUMPEXECUTABLESOURCESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

static void dumpExecutableToStream(IREE::HAL::ExecutableOp executableOp,
                                   StringRef filePath, llvm::raw_ostream &os) {
  llvm::dbgs() << "I'm dumping a source for: " << executableOp.getName()
               << " with file: " << filePath << "\n";
  mlir::Location location = executableOp.getLoc();
  location.dump();
  OpPrintingFlags flags;
  flags.useLocalScope();
  mlir::generateLocationsFromIR(os, filePath, executableOp, flags);
  os << "\n"; // newline at end of file
}

//===----------------------------------------------------------------------===//
// --iree-hal-dump-executable-sources
//===----------------------------------------------------------------------===//

struct DumpExecutableSourcesPass
    : public IREE::HAL::impl::DumpExecutableSourcesPassBase<
          DumpExecutableSourcesPass> {
  using IREE::HAL::impl::DumpExecutableSourcesPassBase<
      DumpExecutableSourcesPass>::DumpExecutableSourcesPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto moduleName = moduleOp.getName().value_or("module");

    // Help people out and mkdir if needed.
    if (!path.empty() && path != "-") {
      llvm::sys::fs::create_directories(path);
    }

    for (auto executableOp : moduleOp.getOps<IREE::HAL::ExecutableOp>()) {
      // Reset to public visibility so symbol DCE won't drop it on load.
      auto originalVisibility = executableOp.getVisibility();
      executableOp.setVisibility(SymbolTable::Visibility::Public);

      std::string maybePrefix = prefix.empty() ? "" : prefix + "_";

      auto fileName =
          (maybePrefix + moduleName + "_" + executableOp.getName() + ".mlir")
              .str();
      if (path.empty() || path == "-") {
        dumpExecutableToStream(executableOp, fileName, llvm::outs());
      } else {
        auto filePath =
            (path + llvm::sys::path::get_separator() + fileName).str();
        std::string error;
        auto file = mlir::openOutputFile(filePath, &error);
        if (!file) {
          executableOp.emitError()
              << "while dumping to " << path << ": " << error;
          return signalPassFailure();
        }
        dumpExecutableToStream(executableOp, filePath, file->os());
        file->keep();
      }

      // Restore original visibility.
      executableOp.setVisibility(originalVisibility);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
