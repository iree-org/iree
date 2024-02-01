// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/LinkingUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Utils/EquivalenceUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler {

SetVector<IREE::HAL::ExecutableTargetAttr>
gatherExecutableTargets(ArrayRef<IREE::HAL::ExecutableOp> executableOps) {
  SetVector<IREE::HAL::ExecutableTargetAttr> result;
  for (auto executableOp : executableOps) {
    auto variantOps =
        llvm::to_vector(executableOp.getOps<IREE::HAL::ExecutableVariantOp>());
    for (auto variantOp : variantOps) {
      result.insert(variantOp.getTarget());
    }
  }
  return result;
}

// Renames |op| within |moduleOp| with a new name that is unique within both
// |moduleOp| and |optionalSymbolTable| (if one is provided).
static void
renameWithDisambiguatedName(Operation *op, Operation *moduleOp,
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
LogicalResult
mergeModuleInto(Operation *sourceModuleOp, Operation *targetModuleOp,
                DenseMap<StringRef, Operation *> &targetSymbolMap) {
  auto &sourceBlock = sourceModuleOp->getRegion(0).front();
  auto &targetBlock = targetModuleOp->getRegion(0).front();
  SymbolTable sourceSymbolTable(sourceModuleOp);
  auto allOps =
      llvm::map_to_vector<8>(sourceBlock, [&](Operation &op) { return &op; });

  for (auto &sourceOp : allOps) {
    if (sourceOp->hasTrait<OpTrait::IsTerminator>())
      continue;
    if (auto symbolOp = dyn_cast<SymbolOpInterface>(sourceOp)) {
      auto symbolName = symbolOp.getName();

      // Resolve symbol name conflicts.
      if (auto targetOp = targetSymbolMap[symbolName]) {
        if (OperationEquivalence::isEquivalentTo(
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
          renameWithDisambiguatedName(sourceOp, sourceModuleOp, targetSymbolMap,
                                      &sourceSymbolTable);
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
            return sourceOp->emitError()
                   << "multiple public symbols with the name: " << symbolName;
          } else {
            // Keep the original name for our new op, rename the target op.
            renameWithDisambiguatedName(targetOp, targetModuleOp,
                                        targetSymbolMap,
                                        /*optionalSymbolTable=*/nullptr);
          }
        }
      }
      targetSymbolMap[SymbolTable::getSymbolName(sourceOp).getValue()] =
          sourceOp;
    }
    if (!targetBlock.empty() &&
        targetBlock.back().hasTrait<OpTrait::IsTerminator>()) {
      sourceOp->moveBefore(&targetBlock.back());
    } else {
      sourceOp->moveBefore(&targetBlock, targetBlock.end());
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
static void
replaceEntryPointUses(mlir::ModuleOp moduleOp,
                      const SymbolReplacements &symbolReplacements) {
  auto replaceSymbolRefs = [](Operation *rootOp,
                              const DenseMap<Attribute, Attribute> &map) {
    auto allUses = SymbolTable::getSymbolUses(rootOp);
    if (!allUses)
      return;
    for (auto use : *allUses) {
      auto oldAttr = use.getSymbolRef();
      auto newAttr = map.lookup(oldAttr);
      if (!newAttr)
        continue;
      auto newDict = use.getUser()->getAttrDictionary().replace(
          [&](Attribute attr) -> std::pair<Attribute, WalkResult> {
            if (attr == oldAttr) {
              // Found old->new replacement.
              return {newAttr, WalkResult::skip()};
            } else if (llvm::isa<SymbolRefAttr>(attr)) {
              // Don't recurse into symbol refs - we only want to match roots.
              return {attr, WalkResult::skip()};
            }
            // Non-symbol ref attr.
            return {attr, WalkResult::advance()};
          });
      use.getUser()->setAttrs(llvm::cast<DictionaryAttr>(newDict));
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
    SmallVectorImpl<IREE::HAL::ExecutableOp> &sourceExecutableOps,
    IREE::HAL::ExecutableOp linkedExecutableOp,
    IREE::HAL::ExecutableVariantOp linkedTargetOp,
    std::function<LogicalResult(mlir::ModuleOp sourceInnerModule,
                                mlir::ModuleOp linkedInnerModule,
                                DenseMap<StringRef, Operation *> &symbolMap)>
        mergeInnerModuleFn) {
  MLIRContext *context = linkedTargetOp.getContext();
  int nextEntryPointOrdinal = 0;
  DenseMap<StringRef, Operation *> targetSymbolMap;
  SymbolReplacements symbolReplacements;

  auto linkedTargetBuilder =
      OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());

  // Aggregation of all external objects specified on variants used.
  SetVector<Attribute> objectAttrs;

  // Iterate over all source executable ops, linking as many as we can.
  for (auto sourceExecutableOp : sourceExecutableOps) {
    // Remap root executable refs.
    symbolReplacements.executableRefs[SymbolRefAttr::get(sourceExecutableOp)] =
        SymbolRefAttr::get(linkedExecutableOp);

    auto variantOps = llvm::to_vector(
        sourceExecutableOp.getOps<IREE::HAL::ExecutableVariantOp>());
    for (auto variantOp : variantOps) {
      // Only process compatible targets.
      // TODO(benvanik): allow for grouping when multi-versioning is supported?
      // We could, for example, link all aarch64 variants together and then
      // use function multi-versioning to let LLVM insert runtime switches.
      if (variantOp.getTarget() != linkedTargetOp.getTarget())
        continue;

      // Add any required object files to the set we will link in the target.
      if (auto objectsAttr = variantOp.getObjectsAttr()) {
        objectAttrs.insert(objectsAttr.begin(), objectsAttr.end());
      }

      // Remap variant refs.
      auto oldVariantRefAttr =
          SymbolRefAttr::get(context, sourceExecutableOp.getName(),
                             {SymbolRefAttr::get(variantOp)});
      auto newVariantRefAttr =
          SymbolRefAttr::get(context, linkedExecutableOp.getName(),
                             {SymbolRefAttr::get(linkedTargetOp)});
      symbolReplacements.variantRefs[oldVariantRefAttr] = newVariantRefAttr;

      // Move the condition op too. We need to make sure all variant's condition
      // op has the same content.
      auto targetConditionOps =
          linkedTargetOp.getOps<IREE::HAL::ExecutableConditionOp>();
      if (auto sourceConditionOp = variantOp.getConditionOp()) {
        if (targetConditionOps.empty()) {
          sourceConditionOp->moveBefore(
              &*linkedTargetBuilder.getInsertionPoint());
        } else {
          assert(llvm::hasSingleElement(targetConditionOps));
          IREE::HAL::ExecutableConditionOp referenceOp =
              *targetConditionOps.begin();
          if (!isStructurallyEquivalentTo(*sourceConditionOp.getOperation(),
                                          *referenceOp.getOperation())) {
            return variantOp.emitError("contains incompatible condition op");
          }
        }
      } else {
        if (!targetConditionOps.empty()) {
          return variantOp.emitError("should contain a condition op");
        }
      }

      // Move any constant blocks that need to be preserved for future host
      // translation. There may be duplicates provided but they'll be cleaned
      // up in future passes.
      for (auto constantBlockOp :
           llvm::make_early_inc_range(variantOp.getConstantBlockOps())) {
        constantBlockOp->moveBefore(&*linkedTargetBuilder.getInsertionPoint());
        // linkedTargetBuilder.clone(constantBlockOp);
      }

      // Clone export ops and queue remapping ordinals and updating
      // symbol refs.
      for (auto exportOp : variantOp.getExportOps()) {
        auto newExportOp = cast<IREE::HAL::ExecutableExportOp>(
            linkedTargetBuilder.clone(*exportOp));
        newExportOp.setOrdinalAttr(
            linkedTargetBuilder.getIndexAttr(nextEntryPointOrdinal++));

        // Add to replacement table for fixing up dispatch calls referencing
        // this export.
        auto oldExportRefAttr = SymbolRefAttr::get(
            context, sourceExecutableOp.getName(),
            {SymbolRefAttr::get(variantOp), SymbolRefAttr::get(exportOp)});
        auto newExportRefAttr =
            SymbolRefAttr::get(context, linkedExecutableOp.getName(),
                               {SymbolRefAttr::get(linkedTargetOp),
                                SymbolRefAttr::get(newExportOp)});
        symbolReplacements.exportRefs[oldExportRefAttr] = newExportRefAttr;
      }

      // Merge the existing module into the new linked module op.
      if (failed(mergeInnerModuleFn(variantOp.getInnerModule(),
                                    linkedTargetOp.getInnerModule(),
                                    targetSymbolMap))) {
        return failure();
      }

      variantOp.erase();
    }
  }

  // Retain only non-empty source executables. This is necessary to make sure
  // when we scan the source executable list multiple times, we don't access
  // destroyed ones so to avoid data structure corruption.
  int retainSize = 0;
  for (int i = 0, e = sourceExecutableOps.size(); i < e; ++i) {
    IREE::HAL::ExecutableOp executable = sourceExecutableOps[i];
    if (executable.getOps<IREE::HAL::ExecutableVariantOp>().empty()) {
      executable.erase();
    } else {
      sourceExecutableOps[retainSize++] = executable;
    }
  }
  sourceExecutableOps.resize(retainSize);

  // Attach object files from source variants.
  if (!objectAttrs.empty()) {
    linkedTargetOp.setObjectsAttr(
        linkedTargetBuilder.getArrayAttr(objectAttrs.takeVector()));
  }

  // Update references to @executable::@target::@entry symbols.
  replaceEntryPointUses(moduleOp, symbolReplacements);

  // Remove if we didn't add anything.
  if (linkedTargetOp.getExportOps().empty()) {
    linkedTargetOp.erase();
    linkedExecutableOp.erase();
  }

  return success();
}

} // namespace mlir::iree_compiler
