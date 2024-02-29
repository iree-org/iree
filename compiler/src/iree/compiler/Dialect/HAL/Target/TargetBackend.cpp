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

namespace mlir::iree_compiler::IREE::HAL {

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
      "iree-hal-dump-executable-files-to", executableFilesPath,
      llvm::cl::desc(
          "Meta flag for all iree-hal-dump-executable-* options. Path to write "
          "executable files (sources, benchmarks, intermediates, binaries) "
          "to."),
      llvm::cl::callback([&](const std::string &path) {
        if (executableSourcesPath.empty())
          executableSourcesPath = path;
        if (executableConfigurationsPath.empty())
          executableConfigurationsPath = path;
        if (executableBenchmarksPath.empty())
          executableBenchmarksPath = path;
        if (executableIntermediatesPath.empty())
          executableIntermediatesPath = path;
        if (executableBinariesPath.empty())
          executableBinariesPath = path;
      }),
      llvm::cl::cat(halTargetOptionsCategory));

  binder.opt<std::string>(
      "iree-hal-dump-executable-sources-to", executableSourcesPath,
      llvm::cl::desc("Path to write individual hal.executable input "
                     "source listings into (- for stdout)."),
      llvm::cl::cat(halTargetOptionsCategory));

  binder.opt<std::string>(
      "iree-hal-dump-executable-configurations-to",
      executableConfigurationsPath,
      llvm::cl::desc("Path to write individual hal.executable input source "
                     "listings into, after translation strategy selection and "
                     "before starting translation (- for stdout)."),
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

SmallVector<std::string>
gatherExecutableTargetNames(IREE::HAL::ExecutableOp executableOp) {
  SmallVector<std::string> targetNames;
  llvm::SmallDenseSet<StringRef> targets;
  executableOp.walk([&](IREE::HAL::ExecutableVariantOp variantOp) {
    auto targetName = variantOp.getTarget().getBackend().getValue();
    if (targets.insert(targetName).second) {
      targetNames.push_back(targetName.str());
    }
  });
  llvm::stable_sort(targetNames);
  return targetNames;
}

SmallVector<std::string> gatherExecutableTargetNames(mlir::ModuleOp moduleOp) {
  SmallVector<std::string> targetNames;
  llvm::stable_sort(targetNames);
  llvm::SmallDenseSet<StringRef> targets;
  moduleOp.walk([&](IREE::HAL::ExecutableOp executableOp) {
    executableOp.walk([&](IREE::HAL::ExecutableVariantOp variantOp) {
      auto targetName = variantOp.getTarget().getBackend().getValue();
      if (targets.insert(targetName).second) {
        targetNames.push_back(targetName.str());
      }
    });
  });
  return targetNames;
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

} // namespace mlir::iree_compiler::IREE::HAL
