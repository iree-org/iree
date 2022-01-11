// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ModuleUtils.h"

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

// Renames |op| within |moduleOp| with a new name that is unique within both
// |moduleOp| and |optionalSymbolTable| (if one is provided).
static void renameWithDisambiguatedName(Operation *op, Operation *moduleOp,
                                        SymbolTable &symbolTable) {
  StringRef originalName = SymbolTable::getSymbolName(op).getValue();

  // This could stand to be rewritten noting that nested symbol refs exist.

  // Iteratively try suffixes until we find one that isn't used.
  // It'd be nice if there was a SymbolTable::getUniqueName or something.
  std::string disambiguatedName;
  int uniqueingCounter = 0;
  do {
    disambiguatedName =
        llvm::formatv("{0}_{1}", originalName, uniqueingCounter++).str();
  } while (symbolTable.lookup(disambiguatedName));

  SymbolTableCollection symbolTables;
  SymbolUserMap symbolUsers(symbolTables, moduleOp);
  mlir::StringAttr nameAttr =
      mlir::StringAttr::get(op->getContext(), disambiguatedName);
  symbolUsers.replaceAllUsesWith(op, nameAttr);
  SymbolTable::setSymbolName(op, disambiguatedName);
}

LogicalResult mergeModuleInto(Operation *sourceOp, Operation *targetOp,
                              SymbolTable &targetSymbolTable,
                              OpBuilder &targetBuilder) {
  // Capture source information we need prior to destructively merging.
  SymbolTable sourceSymbolTable(sourceOp);
  auto &sourceBlock = sourceOp->getRegion(0).front();
  auto sourceOps = llvm::to_vector<8>(
      llvm::map_range(sourceBlock, [&](Operation &op) { return &op; }));

  // Resolve conflicts and move the op.
  for (auto &sourceOp : sourceOps) {
    if (sourceOp->hasTrait<OpTrait::IsTerminator>()) continue;
    if (auto symbolOp = dyn_cast<SymbolOpInterface>(sourceOp)) {
      auto symbolName = symbolOp.getName();

      // Resolve symbol name conflicts.
      if (auto targetOp = targetSymbolTable.lookup(symbolName)) {
        if (symbolOp.getVisibility() == SymbolTable::Visibility::Private) {
          // Private symbols can be safely folded into duplicates or renamed.
          if (OperationEquivalence::isEquivalentTo(
                  targetOp, sourceOp, OperationEquivalence::exactValueMatch,
                  OperationEquivalence::exactValueMatch,
                  OperationEquivalence::Flags::IgnoreLocations)) {
            // Optimization: skip over duplicate private symbols.
            // We could let CSE do this later, but we may as well check here.
            continue;
          } else {
            // Preserve the op but give it a unique name.
            renameWithDisambiguatedName(sourceOp, sourceOp, sourceSymbolTable);
          }
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
            renameWithDisambiguatedName(targetOp, targetOp, targetSymbolTable);
          }
        }
      }
      sourceOp->moveBefore(targetBuilder.getInsertionBlock(),
                           targetBuilder.getInsertionPoint());
      targetSymbolTable.insert(sourceOp);
      targetBuilder.setInsertionPoint(sourceOp);
    }
  }

  return success();
}

LogicalResult mergeSourceModuleInto(Location loc, StringRef source,
                                    Operation *targetOp,
                                    SymbolTable &targetSymbolTable,
                                    OpBuilder &targetBuilder) {
  // Parse the module. This will only fail if the compiler was built wrong;
  // we're loading the embedded files from the compiler binary.
  auto sourceModuleRef =
      mlir::parseSourceString(source, targetBuilder.getContext());
  if (!sourceModuleRef) {
    return mlir::emitError(
        loc, "source module failed to parse; ensure dialects are registered");
  }

  // Merge all of the module contents.
  return mergeModuleInto(*sourceModuleRef, targetOp, targetSymbolTable,
                         targetBuilder);
}

}  // namespace iree_compiler
}  // namespace mlir
