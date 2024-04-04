// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir::iree_compiler::IREE::Util {

struct DumpModulePass : public DumpModuleBase<DumpModulePass> {
  DumpModulePass(std::string path) { this->path = path; }
  DumpModulePass(const DumpModulePass &pass) {}

  void runOnOperation() override {
    // Ensure the parent paths exist.
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(path));

    // Attempt to open file - should succeed as long as permissions are ok.
    std::string error;
    auto file = mlir::openOutputFile(path, &error);
    if (!file) {
      llvm::errs() << "while dumping to '" << path << "': " << error << "\n";
      return signalPassFailure();
    }

    // If going to binary serialize out and otherwise print as text.
    if (llvm::sys::path::extension(path) == ".mlirbc") {
      BytecodeWriterConfig config;
      if (failed(writeBytecodeToFile(getOperation(), file->os(), config))) {
        llvm::errs() << "failed to serialize module to '" << path << "'\n";
        return signalPassFailure();
      }
    } else {
      OpPrintingFlags flags;
      getOperation().print(file->os(), flags);
    }

    // Keep the temporary file after the write succeeds.
    file->keep();
  }

  Option<std::string> path{
      *this, "path",
      llvm::cl::desc("File path to write the module text or binary into.")};
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createDumpModulePass(std::string path) {
  return std::make_unique<DumpModulePass>(path);
}

} // namespace mlir::iree_compiler::IREE::Util
