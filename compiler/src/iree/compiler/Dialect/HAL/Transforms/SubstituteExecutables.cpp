// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <unordered_map>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Scans |searchPath| for all child files and appends them to |substitutions|.
// The file basename will be treated as an executable name and the path will be
// absolute such that no object path resolution occurs later on.
//
// To support round-tripping with --iree-hal-dump-executable-sources-to= we
// support stripping file names of |prefix| when present.
static LogicalResult scanSearchPath(
    std::string prefix, StringRef searchPath,
    std::unordered_map<std::string, std::string> &substitutions) {
  if (!llvm::sys::fs::is_directory(searchPath)) {
    llvm::errs() << "iree-hal-substitute-executables source path `"
                 << searchPath << "` not found or not a directory\n";
    return failure();
  }

  std::error_code ec;
  for (llvm::sys::fs::directory_iterator dir(searchPath, ec), dir_end;
       dir != dir_end && !ec; dir.increment(ec)) {
    auto childPath = dir->path();
    llvm::sys::fs::file_status status;
    if (llvm::sys::fs::status(childPath, status)) continue;
    switch (status.type()) {
      case llvm::sys::fs::file_type::regular_file:
      case llvm::sys::fs::file_type::symlink_file:
      case llvm::sys::fs::file_type::type_unknown: {
        // File we can access.
        auto childName = llvm::sys::path::stem(childPath);
        if (!childName.empty() && childName != "." && childName != "..") {
          if (childName.starts_with(prefix)) {
            // Strip prefix.
            childName = childName.substr(prefix.size());
          }
          substitutions[std::string(childName)] = childPath;
        }
        break;
      }
      default:
        // Directory/etc we skip.
        break;
    }
  }
  if (ec) {
    llvm::errs()
        << "iree-hal-substitute-executables failed tos can source path `"
        << searchPath << "`: " << llvm::errorCodeToError(ec) << "\n";
    return failure();
  }

  return success();
}

// Loads an MLIR file from the given |filePath| using the HAL object linkage
// mechanism to resolve the file path.
static OwningOpRef<Operation *> loadModuleObject(MLIRContext *context,
                                                 StringRef filePath) {
  Builder builder(context);

  // Wrap the path in an object and try to resolve it to its absolute path.
  auto fileObjectAttr = builder.getAttr<IREE::HAL::ExecutableObjectAttr>(
      builder.getStringAttr(filePath), nullptr);
  auto absPath = fileObjectAttr.getAbsolutePath();
  if (failed(absPath)) {
    llvm::errs()
        << "iree-hal-substitute-executables could not resolve `" << filePath
        << "` using the current --iree-hal-executable-object-search-path=\n";
    return nullptr;
  }

  // Load the module.
  mlir::ParserConfig parserConfig(context);
  return mlir::parseSourceFile(*absPath, parserConfig);
}

// Loads the MLIR at |filePath| and replaces |executableOp| with an executable
// with the same name from the file.
static LogicalResult replaceExecutableOpWithMLIR(
    IREE::HAL::ExecutableOp executableOp, StringRef filePath) {
  // Load the replacement IR. It may have any mix of stuff in it including
  // multiple other executables.
  auto rootOpRef = loadModuleObject(executableOp.getContext(), filePath);
  if (!rootOpRef) return failure();
  IREE::HAL::ExecutableOp replacementOp;
  if (auto moduleOp = dyn_cast<mlir::ModuleOp>(rootOpRef.get())) {
    // We expect a `hal.executable` with the same name as the one we are
    // replacing.
    replacementOp = dyn_cast_or_null<IREE::HAL::ExecutableOp>(
        SymbolTable::lookupSymbolIn(moduleOp, executableOp.getNameAttr()));
  } else {
    // Verify the name matches.
    replacementOp = dyn_cast<IREE::HAL::ExecutableOp>(rootOpRef.get());
    if (replacementOp && replacementOp.getName() != executableOp.getName()) {
      replacementOp = {};
    }
  }
  if (!replacementOp) {
    return rootOpRef.get()->emitError()
           << "iree-hal-substitute-executables expected a hal.executable with "
              "the name `"
           << executableOp.getName() << "` but none was found";
  }

  // We don't currently verify the variants in the executable - we would
  // probably want to do that if this were a user facing feature. We should have
  // 1:1 variants (or at least a superset) in the replacement executable in
  // order to cover all of the targets requested during this compiler
  // invocation.

  // Clone beside the original - it'll transiently have a symbol name conflict.
  OpBuilder(executableOp).clone(*replacementOp);
  executableOp.erase();

  // FYI so it's easy to spot when we're performing a substitution.
  llvm::errs() << "NOTE: hal.executable `" << replacementOp.getName()
               << "` substituted with MLIR source at `" << filePath << "`\n";

  return success();
}

// Drops the implementation of |executableOp| and links against |filePath|.
static LogicalResult externalizeExecutableOp(
    IREE::HAL::ExecutableOp executableOp, StringRef filePath) {
  // Can't support multiple variants on this path. We could allow some magic way
  // to specify the full #hal.executable.objects dictionary but that's a stretch
  // for this developer tool.
  auto variantOps = executableOp.getOps<IREE::HAL::ExecutableVariantOp>();
  if (std::distance(variantOps.begin(), variantOps.end()) != 1) {
    return executableOp.emitError()
           << "iree-hal-substitute-executables pass cannot externalize "
              "executables with multiple variants; try compiling again for "
              "only a single target";
  }
  auto variantOp = *variantOps.begin();
  Builder builder(executableOp.getContext());

  // To create reproducible output we directly load the file inline using the
  // search paths passed in this compiler invocation.
  auto fileObjectAttr = builder.getAttr<IREE::HAL::ExecutableObjectAttr>(
      builder.getStringAttr(filePath), nullptr);
  auto fileContents = fileObjectAttr.loadData();
  if (!fileContents) return failure();

  // Link the referenced object file contents. We fully replace the existing
  // objects in case there were any as this does entire executable replacement -
  // there may have been microkernel libraries or something referenced by the
  // existing module.
  auto dataObjectAttr = builder.getAttr<IREE::HAL::ExecutableObjectAttr>(
      nullptr, DenseIntElementsAttr::get(
                   VectorType::get({static_cast<int64_t>(fileContents->size())},
                                   builder.getI8Type()),
                   ArrayRef(fileContents->data(), fileContents->size())));
  variantOp.setObjectsAttr(builder.getArrayAttr({dataObjectAttr}));

  // Drop the inner module if present (may already be external).
  for (auto moduleOp :
       llvm::make_early_inc_range(variantOp.getOps<mlir::ModuleOp>())) {
    moduleOp.erase();
  }

  // FYI so it's easy to spot when we're performing a substitution.
  llvm::errs() << "NOTE: hal.executable `" << executableOp.getName()
               << "` substituted with object file at `" << filePath << "`\n";

  return success();
}

static LogicalResult substituteExecutableOp(
    IREE::HAL::ExecutableOp executableOp, StringRef filePath) {
  if (filePath.ends_with_insensitive(".mlir") ||
      filePath.ends_with_insensitive(".mlirbc")) {
    return replaceExecutableOpWithMLIR(executableOp, filePath);
  } else {
    return externalizeExecutableOp(executableOp, filePath);
  }
}

class SubstituteExecutablesPass
    : public PassWrapper<SubstituteExecutablesPass, OperationPass<ModuleOp>> {
 public:
  SubstituteExecutablesPass() = default;
  SubstituteExecutablesPass(const SubstituteExecutablesPass &pass) {}
  SubstituteExecutablesPass(ArrayRef<std::string> substitutions) {
    this->substitutions = substitutions;
  }
  SubstituteExecutablesPass(std::string searchPath) {
    this->searchPath = std::move(searchPath);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-substitute-executables";
  }

  StringRef getDescription() const override {
    return "Substitutes hal.executable ops by parsing |substitutions| in "
           "`executable_name=file.xxx` strings.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto moduleName = moduleOp.getName().value_or("module");
    SymbolTable symbolTable(moduleOp);

    // If provided a path then perform a scan of it and append our substitutions
    // list. We'll fail if the path doesn't exist but don't care if no files are
    // present as it just means the user doesn't want to substitute anything.
    std::unordered_map<std::string, std::string> uniqueSubstitutions;
    if (!searchPath.empty()) {
      if (failed(scanSearchPath((moduleName + "_").str(), searchPath,
                                uniqueSubstitutions))) {
        return signalPassFailure();
      }
    }

    // Dedupe substitutions by taking the last flag passed.
    for (const auto &substitution : substitutions) {
      auto [key, value] = StringRef(substitution).split('=');
      if (key.empty() || value.empty()) {
        llvm::errs() << "iree-hal-substitute-executables pass requires "
                        "`executable_name=file.xxx` paths; received malformed "
                        "substitution: `"
                     << substitution << "`\n";
        return signalPassFailure();
      }
      uniqueSubstitutions[std::string(key)] = value;
    }

    if (uniqueSubstitutions.empty()) return;  // no-op

    // Walk each substitution and process the matching executable if found.
    for (auto &[executableName, filePath] : uniqueSubstitutions) {
      auto *op = symbolTable.lookup(executableName);
      if (!op) {
        // Ignore executables that aren't found. We still warn as an FYI and may
        // want to change this to an error depending on how many people run
        // afoul of this. The likely source is changes prior to this pass that
        // change the executable composition of the program and missing
        // executables is the least serious issue (mismatched signatures/etc are
        // harder to detect and more dangerous).
        llvm::errs() << "WARNING: iree-hal-substitute-executables could not "
                        "perform the requested substitution as the executable `"
                     << executableName << "` was not found in the module\n";
        continue;
      } else if (auto executableOp = dyn_cast<IREE::HAL::ExecutableOp>(op)) {
        if (failed(substituteExecutableOp(executableOp, filePath))) {
          return signalPassFailure();
        }
      } else {
        op->emitOpError() << "iree-hal-substitute-executables substitution "
                             "expected a hal.executable";
        return signalPassFailure();
      }
    }
  }

 private:
  ListOption<std::string> substitutions{
      *this, "substitutions",
      llvm::cl::desc(
          "Substitution `executable_name=file.xxx` key-value pairs.")};
  Option<std::string> searchPath{
      *this, "search-path",
      llvm::cl::desc("Path to source executable substitutions from.")};
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createSubstituteExecutablesPass(
    ArrayRef<std::string> substitutions) {
  return std::make_unique<SubstituteExecutablesPass>(substitutions);
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> createSubstituteExecutablesPass(
    std::string searchPath) {
  return std::make_unique<SubstituteExecutablesPass>(std::move(searchPath));
}

static PassRegistration<SubstituteExecutablesPass> pass([] {
  return std::make_unique<SubstituteExecutablesPass>();
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
