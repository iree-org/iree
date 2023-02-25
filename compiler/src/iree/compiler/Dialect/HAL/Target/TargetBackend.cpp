// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"

#include <algorithm>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/FileUtilities.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::IREE::HAL::TargetOptions);

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

void TargetOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory halTargetOptionsCategory(
      "IREE HAL executable target options");

  // This function is called as part of registering the pass
  // TranslateExecutablesPass. Pass registry is also staticly
  // initialized, so targetBackendsFlags needs to be here to be initialized
  // first.
  binder.list<std::string>(
      "iree-hal-target-backends", targets,
      llvm::cl::desc("Target backends for executable compilation."),
      llvm::cl::ZeroOrMore, llvm::cl::cat(halTargetOptionsCategory));

  binder.opt<int>(
      "iree-hal-executable-debug-level", debugLevel,
      llvm::cl::desc("Debug level for executable translation (0-3)"),
      llvm::cl::init(2), llvm::cl::cat(halTargetOptionsCategory));

  binder.opt<std::string>(
      "iree-hal-dump-executable-sources-to", sourceListingPath,
      llvm::cl::desc("Path to write individual hal.executable input "
                     "source listings into (- for stdout)."),
      llvm::cl::cat(halTargetOptionsCategory));

  binder.opt<std::string>(
      "iree-hal-dump-executable-benchmarks-to", executableBenchmarksPath,
      llvm::cl::desc("Path to write standalone hal.executable benchmarks into "
                     "(- for stdout)."),
      llvm::cl::cat(halTargetOptionsCategory));

  binder.opt<std::string>("iree-hal-dump-executable-intermediates-to",
                          executableIntermediatesPath,
                          llvm::cl::desc("Path to write translated executable "
                                         "intermediates (.bc, .o, etc) into."),
                          llvm::cl::cat(halTargetOptionsCategory));

  binder.opt<std::string>(
      "iree-hal-dump-executable-binaries-to", executableBinariesPath,
      llvm::cl::desc(
          "Path to write translated and serialized executable binaries into."),
      llvm::cl::cat(halTargetOptionsCategory));
}

void dumpDataToPath(StringRef path, StringRef baseName, StringRef suffix,
                    StringRef extension, StringRef data) {
  auto fileName = (llvm::join_items("_", baseName, suffix) + extension).str();
  auto fileParts =
      llvm::join_items(llvm::sys::path::get_separator(), path, fileName);
  auto filePath = llvm::sys::path::convert_to_slash(fileParts);
  std::string error;
  auto file = mlir::openOutputFile(filePath, &error);
  if (!file) {
    llvm::errs() << "Unable to dump debug output to " << filePath << "\n";
    return;
  }
  file->os().write(data.data(), data.size());
  file->keep();
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
