// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ModuleUtils.h"

#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler {

std::optional<FileLineColLoc> findFirstFileLoc(Location baseLoc) {
  if (auto loc = dyn_cast<FileLineColLoc>(baseLoc)) {
    return loc;
  }

  if (auto loc = dyn_cast<FusedLoc>(baseLoc)) {
    // Recurse through fused locations.
    for (auto &childLoc : loc.getLocations()) {
      auto childResult = findFirstFileLoc(childLoc);
      if (childResult)
        return childResult;
    }
  } else if (auto loc = dyn_cast<CallSiteLoc>(baseLoc)) {
    // First check caller...
    auto callerResult = findFirstFileLoc(loc.getCaller());
    if (callerResult)
      return callerResult;
    // Then check callee...
    auto calleeResult = findFirstFileLoc(loc.getCallee());
    if (calleeResult)
      return calleeResult;
  } else if (auto loc = dyn_cast<NameLoc>(baseLoc)) {
    auto childResult = findFirstFileLoc(loc.getChildLoc());
    if (childResult)
      return childResult;
  } else if (auto loc = dyn_cast<OpaqueLoc>(baseLoc)) {
    // TODO(scotttodd): Use loc.fallbackLocation()?
  } else if (auto loc = dyn_cast<UnknownLoc>(baseLoc)) {
    // ¯\_(ツ)_/¯
  }

  return std::nullopt;
}

std::string guessModuleName(mlir::ModuleOp moduleOp, StringRef defaultName) {
  std::string moduleName = moduleOp.getName().value_or("").str();
  if (!moduleName.empty())
    return moduleName;
  auto loc = findFirstFileLoc(moduleOp.getLoc());
  if (loc.has_value()) {
    return sanitizeSymbolName(
        llvm::sys::path::stem(loc.value().getFilename()).str());
  } else {
    return defaultName.str();
  }
}

// Renames |op| within |moduleOp| with a new name that is unique within both
// |moduleOp| and |symbolTable|.
static void renameWithDisambiguatedName(Operation *op, Operation *moduleOp,
                                        SymbolTable &symbolTable0,
                                        SymbolTable &symbolTable1,
                                        SymbolTableCollection &symbolTables,
                                        SymbolUserMap &symbolUsers) {
  StringRef originalName = SymbolTable::getSymbolName(op).getValue();

  // This could stand to be rewritten noting that nested symbol refs exist.

  // Iteratively try suffixes until we find one that isn't used.
  // It'd be nice if there was a SymbolTable::getUniqueName or something.
  std::string disambiguatedName;
  int uniqueingCounter = 0;
  do {
    disambiguatedName =
        llvm::formatv("{}_{}", originalName, uniqueingCounter++).str();
  } while (symbolTable0.lookup(disambiguatedName) ||
           symbolTable1.lookup(disambiguatedName));

  mlir::StringAttr nameAttr =
      mlir::StringAttr::get(op->getContext(), disambiguatedName);
  symbolUsers.replaceAllUsesWith(op, nameAttr);
  SymbolTable::setSymbolName(op, disambiguatedName);
}

// Returns true if there are any non-terminator regions in the given |op|.
static bool isOperationEmpty(Operation *op) {
  if (op->getNumRegions() == 0) {
    return true;
  }
  for (auto &region : op->getRegions()) {
    if (!region.empty() && llvm::any_of(region.getOps(), [](Operation &op) {
          return !op.hasTrait<OpTrait::IsTerminator>();
        })) {
      return false;
    }
  }
  return true;
}

LogicalResult mergeModuleInto(Operation *sourceModuleOp,
                              Operation *targetModuleOp,
                              OpBuilder &targetBuilder) {
  // If the source has no ops then we are a no-op.
  if (isOperationEmpty(sourceModuleOp)) {
    return success();
  }

  // Collect operations we'll be moving (so we can mutate the source).
  Block &sourceBlock = sourceModuleOp->getRegion(0).front();
  SmallVector<Operation *> sourceOps(llvm::make_pointer_range(sourceBlock));

  // Fast path: if target is empty just move operations without conflict
  // checks.
  if (isOperationEmpty(targetModuleOp)) {
    for (auto &sourceOp : sourceOps) {
      if (sourceOp->hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }
      // Use moveBefore which works with block->end() (insertion point from
      // setInsertionPointToEnd), then update insertion point for next op.
      sourceOp->moveBefore(targetBuilder.getInsertionBlock(),
                           targetBuilder.getInsertionPoint());
      targetBuilder.setInsertionPointAfter(sourceOp);
    }
    return success();
  }

  // Capture source information we need prior to destructively merging.
  SymbolTable sourceSymbolTable(sourceModuleOp);
  SymbolTable targetSymbolTable(targetModuleOp);

  // Build symbol maps once to avoid O(module_size * rename_count) performance.
  // Note that we do mutate the modules but these stay valid for us so long as
  // we don't erase anything.
  SymbolTableCollection symbolTables;
  SymbolUserMap sourceSymbolUsers(symbolTables, sourceModuleOp);
  SymbolUserMap targetSymbolUsers(symbolTables, targetModuleOp);

  // Resolve conflicts and move the op.
  for (auto &sourceOp : sourceOps) {
    if (sourceOp->hasTrait<OpTrait::IsTerminator>())
      continue;
    if (auto symbolOp = dyn_cast<SymbolOpInterface>(sourceOp)) {
      auto symbolName = symbolOp.getName();

      // Resolve symbol name conflicts.
      if (auto targetOp = targetSymbolTable.lookup(symbolName)) {
        // Only deduplicate stateless functions; initializers and globals have
        // state and side effects so they must be kept separate even if they
        // appear structurally equivalent.
        auto funcOp = dyn_cast<FunctionOpInterface>(sourceOp);
        if (funcOp && !funcOp.getNameAttr().empty() &&
            OperationEquivalence::isEquivalentTo(
                targetOp, sourceOp, OperationEquivalence::exactValueMatch,
                /*markEquivalent=*/nullptr,
                OperationEquivalence::Flags::IgnoreLocations)) {
          // If the two ops are identical then we can ignore the source op and
          // use the existing target op.
          continue;
        }
        if (symbolOp.getVisibility() == SymbolTable::Visibility::Private) {
          // Since the source symbol is private we can rename it as all uses
          // are known to be local to the source module.
          renameWithDisambiguatedName(sourceOp, sourceModuleOp,
                                      sourceSymbolTable, targetSymbolTable,
                                      symbolTables, sourceSymbolUsers);
        } else {
          // The source symbol has 'nested' or 'public' visibility.
          if (SymbolTable::getSymbolVisibility(targetOp) !=
              SymbolTable::Visibility::Private) {
            // Oops! Both symbols are public and we can't safely rename either.
            // If you hit this with ops that you think are safe to rename, mark
            // them private.
            return sourceOp->emitError()
                   << "multiple public symbols with the name: " << symbolName;
          } else {
            // Keep the original name for our new op, rename the target op.
            renameWithDisambiguatedName(targetOp, targetModuleOp,
                                        sourceSymbolTable, targetSymbolTable,
                                        symbolTables, targetSymbolUsers);
          }
        }
      }
      sourceOp->moveBefore(targetBuilder.getInsertionBlock(),
                           targetBuilder.getInsertionPoint());
      targetSymbolTable.insert(sourceOp);
      targetBuilder.setInsertionPointAfter(sourceOp);
    } else {
      sourceOp->moveBefore(targetBuilder.getInsertionBlock(),
                           targetBuilder.getInsertionPoint());
      targetBuilder.setInsertionPointAfter(sourceOp);
    }
  }

  return success();
}

LogicalResult mergeSourceModuleInto(Location loc, StringRef source,
                                    Operation *targetOp,
                                    OpBuilder &targetBuilder) {
  // Parse the module. This will only fail if the compiler was built wrong;
  // we're loading the embedded files from the compiler binary.
  auto sourceModuleRef = mlir::parseSourceString<mlir::ModuleOp>(
      source, targetBuilder.getContext());
  if (!sourceModuleRef) {
    return mlir::emitError(
        loc, "source module failed to parse; ensure dialects are registered");
  }

  // Merge all of the module contents.
  return mergeModuleInto(*sourceModuleRef, targetOp, targetBuilder);
}

LogicalResult writeModule(mlir::ModuleOp moduleOp, StringRef path) {
  // Ensure the parent paths exist.
  llvm::sys::fs::create_directories(llvm::sys::path::parent_path(path));

  // Attempt to open file - should succeed as long as permissions are ok.
  std::string error;
  auto file = mlir::openOutputFile(path, &error);
  if (!file) {
    return mlir::emitError(moduleOp.getLoc())
           << "while dumping to '" << path << "': " << error << "\n";
  }

  // If going to binary serialize out and otherwise print as text.
  if (llvm::sys::path::extension(path) == ".mlirbc") {
    BytecodeWriterConfig config;
    if (failed(mlir::writeBytecodeToFile(moduleOp, file->os(), config))) {
      return mlir::emitError(moduleOp.getLoc())
             << "failed to serialize module to '" << path << "'\n";
    }
  } else {
    OpPrintingFlags flags;
    moduleOp.print(file->os(), flags);
  }

  // Keep the temporary file after the write succeeds.
  file->keep();

  return success();
}

} // namespace mlir::iree_compiler
