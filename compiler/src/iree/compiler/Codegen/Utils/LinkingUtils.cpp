// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/LinkingUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {

SetVector<IREE::HAL::ExecutableTargetAttr> gatherExecutableTargets(
    ArrayRef<IREE::HAL::ExecutableOp> executableOps) {
  SetVector<IREE::HAL::ExecutableTargetAttr> result;
  for (auto executableOp : executableOps) {
    auto variantOps = llvm::to_vector<4>(
        executableOp.getOps<IREE::HAL::ExecutableVariantOp>());
    for (auto variantOp : variantOps) {
      result.insert(variantOp.getTarget());
    }
  }
  return result;
}

// Renames |op| within |moduleOp| with a new name that is unique within both
// |moduleOp| and |optionalSymbolTable| (if one is provided).
static void renameWithDisambiguatedName(
    Operation *op, Operation *moduleOp,
    DenseMap<StringRef, Operation *> &targetSymbolMap,
    SymbolTable *optionalSymbolTable) {
  StringRef originalName = SymbolTable::getSymbolName(op).getValue();

  // Iteratively try suffixes until we find one that isn't used.
  std::string disambiguatedName;
  int uniqueingCounter = 0;
  do {
    disambiguatedName =
        llvm::formatv("{0}_{1}", originalName, uniqueingCounter++).str();
  } while (
      targetSymbolMap.lookup(disambiguatedName) ||
      (optionalSymbolTable && optionalSymbolTable->lookup(disambiguatedName)));

  SymbolTableCollection symbolTable;
  SymbolUserMap symbolUsers(symbolTable, moduleOp);
  mlir::StringAttr nameAttr =
      mlir::StringAttr::get(op->getContext(), disambiguatedName);
  symbolUsers.replaceAllUsesWith(op, nameAttr);
  SymbolTable::setSymbolName(op, disambiguatedName);
}

// TODO(benvanik): replace with iree/compiler/Utils/ModuleUtils.h version.
// Only difference is one has the symbol map that we don't even need.

// Destructively merges |sourceModuleOp| into |targetModuleOp|.
// |targetSymbolMap| is updated with the new symbols.
//
// If a private symbol in |sourceModuleOp| conflicts with another symbol
// (public or private) tracked in |targetSymbolMap|, it will be renamed.
//
// Fails if a public symbol in |sourceModuleOp| conflicts with another public
// symbol tracked in |targetSymbolMap|.
static LogicalResult mergeModuleInto(
    Operation *sourceModuleOp, Operation *targetModuleOp,
    DenseMap<StringRef, Operation *> &targetSymbolMap) {
  auto &sourceBlock = sourceModuleOp->getRegion(0).front();
  auto &targetBlock = targetModuleOp->getRegion(0).front();
  SymbolTable sourceSymbolTable(sourceModuleOp);
  auto allOps = llvm::to_vector<8>(
      llvm::map_range(sourceBlock, [&](Operation &op) { return &op; }));

  for (auto &op : allOps) {
    if (op->hasTrait<OpTrait::IsTerminator>()) continue;
    if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
      auto symbolName = symbolOp.getName();

      // Resolve symbol name conflicts.
      if (auto targetOp = targetSymbolMap[symbolName]) {
        if (symbolOp.getVisibility() == SymbolTable::Visibility::Private) {
          // Private symbols can be safely folded into duplicates or renamed.
          if (OperationEquivalence::isEquivalentTo(
                  targetOp, op, OperationEquivalence::exactValueMatch,
                  OperationEquivalence::exactValueMatch,
                  OperationEquivalence::Flags::IgnoreLocations)) {
            // Optimization: skip over duplicate private symbols.
            // We could let CSE do this later, but we may as well check here.
            continue;
          } else {
            // Preserve the op but give it a unique name.
            renameWithDisambiguatedName(op, sourceModuleOp, targetSymbolMap,
                                        &sourceSymbolTable);
          }
        } else {
          // The source symbol has 'nested' or 'public' visibility.
          if (SymbolTable::getSymbolVisibility(targetOp) !=
              SymbolTable::Visibility::Private) {
            // Oops! Both symbols are public and we can't safely rename either.
            // If you hit this with ops that you think are safe to rename, mark
            // them private.
            //
            // Note: we could also skip linking between executables with
            // conflicting symbol names. We think such conflicts will be better
            // fixed in other ways, so we'll emit an error until we find a case
            // where that isn't true.
            return op->emitError()
                   << "multiple public symbols with the name: " << symbolName;
          } else {
            // Keep the original name for our new op, rename the target op.
            renameWithDisambiguatedName(targetOp, targetModuleOp,
                                        targetSymbolMap,
                                        /*optionalSymbolTable=*/nullptr);
          }
        }
      }
      targetSymbolMap[SymbolTable::getSymbolName(op).getValue()] = op;
    }
    if (!targetBlock.empty() &&
        targetBlock.back().hasTrait<OpTrait::IsTerminator>()) {
      op->moveBefore(&targetBlock.back());
    } else {
      op->moveBefore(&targetBlock, targetBlock.end());
    }
  }

  // Now that we're done cloning its ops, delete the original target op.
  sourceModuleOp->erase();

  return success();
}

struct SymbolReplacements {
  DenseMap<Attribute, Attribute> executableRefs;
  DenseMap<Attribute, Attribute> variantRefs;
  DenseMap<Attribute, Attribute> exportRefs;
};

// Replaces each usage of an entry point with its original symbol name with a
// new symbol name.
//
// Due to replaceSubElements recursing into symbol refs we need to perform
// replacement in descending symbol ref length; otherwise replacing the
// executable name in `@old_executable::@old_export` would result in
// `@new_executable::@old_export` and an export update would then not match the
// new/old mismatched ref. This means we have to do three walks over the entire
// module in order to do the replacements; not great.
static void replaceEntryPointUses(
    mlir::ModuleOp moduleOp, const SymbolReplacements &symbolReplacements) {
  auto replaceSymbolRefs = [](Operation *rootOp,
                              const DenseMap<Attribute, Attribute> &map) {
    auto allUses = SymbolTable::getSymbolUses(rootOp);
    if (!allUses) return;
    for (auto use : *allUses) {
      auto oldAttr = use.getSymbolRef();
      auto newAttr = map.lookup(oldAttr);
      if (!newAttr) continue;
      auto newDict = use.getUser()->getAttrDictionary().replaceSubElements(
          [&](Attribute attr) -> std::pair<Attribute, WalkResult> {
            if (attr == oldAttr) {
              // Found old->new replacement.
              return {newAttr, WalkResult::skip()};
            } else if (attr.isa<SymbolRefAttr>()) {
              // Don't recurse into symbol refs - we only want to match roots.
              return {attr, WalkResult::skip()};
            }
            // Non-symbol ref attr.
            return {attr, WalkResult::advance()};
          });
      use.getUser()->setAttrs(newDict.cast<DictionaryAttr>());
    }
  };
  replaceSymbolRefs(moduleOp, symbolReplacements.exportRefs);
  replaceSymbolRefs(moduleOp, symbolReplacements.variantRefs);
  replaceSymbolRefs(moduleOp, symbolReplacements.executableRefs);
  for (auto funcLikeOp : moduleOp.getOps<FunctionOpInterface>()) {
    replaceSymbolRefs(funcLikeOp, symbolReplacements.exportRefs);
    replaceSymbolRefs(funcLikeOp, symbolReplacements.variantRefs);
    replaceSymbolRefs(funcLikeOp, symbolReplacements.executableRefs);
  }
}

LogicalResult linkExecutablesInto(
    mlir::ModuleOp moduleOp,
    ArrayRef<IREE::HAL::ExecutableOp> sourceExecutableOps,
    IREE::HAL::ExecutableOp linkedExecutableOp,
    IREE::HAL::ExecutableVariantOp linkedTargetOp,
    std::function<Operation *(mlir::ModuleOp moduleOp)> getInnerModuleFn,
    OpBuilder &builder) {
  int nextEntryPointOrdinal = 0;
  DenseMap<StringRef, Operation *> targetSymbolMap;
  SymbolReplacements symbolReplacements;

  auto linkedTargetBuilder =
      OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());
  auto linkedModuleOp = getInnerModuleFn(linkedTargetOp.getInnerModule());

  // Iterate over all source executable ops, linking as many as we can.
  for (auto sourceExecutableOp : sourceExecutableOps) {
    // Remap root executable refs.
    symbolReplacements.executableRefs[SymbolRefAttr::get(sourceExecutableOp)] =
        SymbolRefAttr::get(linkedExecutableOp);

    auto variantOps = llvm::to_vector<4>(
        sourceExecutableOp.getOps<IREE::HAL::ExecutableVariantOp>());
    for (auto variantOp : variantOps) {
      // Only process compatible targets.
      // TODO(benvanik): allow for grouping when multi-versioning is supported?
      // We could, for example, link all aarch64 variants together and then
      // use function multi-versioning to let LLVM insert runtime switches.
      if (variantOp.getTarget() != linkedTargetOp.getTarget()) continue;

      // Remap variant refs.
      auto oldVariantRefAttr =
          SymbolRefAttr::get(builder.getContext(), sourceExecutableOp.getName(),
                             {SymbolRefAttr::get(variantOp)});
      auto newVariantRefAttr =
          SymbolRefAttr::get(builder.getContext(), linkedExecutableOp.getName(),
                             {SymbolRefAttr::get(linkedTargetOp)});
      symbolReplacements.variantRefs[oldVariantRefAttr] = newVariantRefAttr;

      // Move any constant blocks that need to be preserved for future host
      // translation. There may be duplicates provided but they'll be cleaned
      // up in future passes.
      for (auto constantBlockOp : llvm::make_early_inc_range(
               variantOp.getOps<IREE::HAL::ExecutableConstantBlockOp>())) {
        constantBlockOp->moveBefore(&*linkedTargetBuilder.getInsertionPoint());
      }

      // Clone export ops and queue remapping ordinals and updating
      // symbol refs.
      for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
        auto newExportOp =
            linkedTargetBuilder.create<IREE::HAL::ExecutableExportOp>(
                exportOp.getLoc(), exportOp.getSymNameAttr(),
                builder.getIndexAttr(nextEntryPointOrdinal++),
                exportOp.getLayout(), ArrayAttr{}, IntegerAttr{});
        newExportOp->setDialectAttrs(exportOp->getDialectAttrs());

        // Add to replacement table for fixing up dispatch calls referencing
        // this export.
        auto oldExportRefAttr = SymbolRefAttr::get(
            builder.getContext(), sourceExecutableOp.getName(),
            {SymbolRefAttr::get(variantOp), SymbolRefAttr::get(exportOp)});
        auto newExportRefAttr = SymbolRefAttr::get(
            builder.getContext(), linkedExecutableOp.getName(),
            {SymbolRefAttr::get(linkedTargetOp),
             SymbolRefAttr::get(newExportOp)});
        symbolReplacements.exportRefs[oldExportRefAttr] = newExportRefAttr;
      }

      // Merge the existing module into the new linked module op.
      auto sourceModuleOp = getInnerModuleFn(variantOp.getInnerModule());
      if (failed(mergeModuleInto(sourceModuleOp, linkedModuleOp,
                                 targetSymbolMap))) {
        return failure();
      }

      variantOp.erase();
    }

    if (sourceExecutableOp.getOps<IREE::HAL::ExecutableVariantOp>().empty()) {
      sourceExecutableOp.erase();
    }
  }

  // Update references to @executable::@target::@entry symbols.
  replaceEntryPointUses(moduleOp, symbolReplacements);

  // Remove if we didn't add anything.
  if (linkedTargetOp.getOps<IREE::HAL::ExecutableExportOp>().empty()) {
    linkedTargetOp.erase();
    linkedExecutableOp.erase();
  }

  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
