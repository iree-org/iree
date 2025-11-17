// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/EquivalenceUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"

#define DEBUG_TYPE "iree-util-link-modules"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_LINKMODULESPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ModuleIndex - indexes source modules for symbol resolution
//===----------------------------------------------------------------------===//

class ModuleIndex {
public:
  struct SourceModuleInfo;

  // Information about a single symbol in a source module.
  struct SymbolInfo {
    Operation *op;
    std::string qualifiedName; // Owned: "module_a.func_a" or "func_a".
    StringRef localName;       // Points into op's symbol name attribute.
    SourceModuleInfo *sourceModule;
    SymbolTable::Visibility visibility;
    SmallVector<StringRef> requiredSymbols; // In module order.
    bool scanned = false;
  };

  // Information about a source module.
  struct SourceModuleInfo {
    OwningOpRef<ModuleOp> module;
    std::string moduleName; // Empty for anonymous modules.

    // Map: qualified name -> unique_ptr<SymbolInfo>.
    DenseMap<StringRef, std::unique_ptr<SymbolInfo>> symbols;
  };

  explicit ModuleIndex(MLIRContext *context) : context(context) {}

  // Adds a source module from file path.
  LogicalResult addSourceModule(StringRef filename);

  // Adds a source module from already-parsed ModuleOp.
  LogicalResult addSourceModule(OwningOpRef<ModuleOp> module);

  // Adds a directory to search for auto-discovery.
  void addLibraryPath(StringRef path);

  // Finds symbol by qualified name across all source modules.
  SymbolInfo *lookupSymbol(StringRef qualifiedName);

  // Resolves transitive dependencies starting from required symbols.
  // Returns ordered list of symbols to link.
  // symbolLocations maps symbol names to their operation locations for error
  // reporting.
  LogicalResult resolveTransitiveDependencies(
      ArrayRef<StringRef> requiredSymbols,
      SmallVectorImpl<SymbolInfo *> &resolvedSymbols,
      const DenseMap<StringRef, Location> &symbolLocations);

  // Gets all source modules.
  ArrayRef<std::unique_ptr<SourceModuleInfo>> getSourceModules() const {
    return sourceModules;
  }

private:
  // Builds symbol table for a source module.
  // If moduleOp is not provided, uses info.module.get().
  void buildSymbolTable(SourceModuleInfo &info, ModuleOp moduleOp = nullptr);

  // Scans a symbol for its dependencies.
  void scanSymbolDependencies(SymbolInfo *info);

  // Attempts to auto-load a module based on symbol name.
  std::optional<OwningOpRef<ModuleOp>> tryAutoLoadModule(StringRef symbolName);

  // Checks if module contains nested named modules (archive-style).
  bool hasNestedNamedModules(ModuleOp module);

  MLIRContext *context;
  SmallVector<std::unique_ptr<SourceModuleInfo>> sourceModules;
  SmallVector<std::string> libraryPaths;
  DenseMap<StringRef, SymbolInfo *> globalSymbolMap;
};

LogicalResult ModuleIndex::addSourceModule(StringRef filename) {
  // Parse module from file.
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = fileOrErr.getError()) {
    return emitError(UnknownLoc::get(context))
           << "failed to open source module file '" << filename
           << "': " << error.message();
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  ParserConfig config(context, /*verifyAfterParse=*/false);
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, config);
  if (!module) {
    return emitError(UnknownLoc::get(context))
           << "failed to parse source module: '" << filename << "'";
  }

  return addSourceModule(std::move(module));
}

bool ModuleIndex::hasNestedNamedModules(ModuleOp module) {
  for (auto nestedModule : module.getOps<ModuleOp>()) {
    if (nestedModule.getSymName()) {
      return true;
    }
  }
  return false;
}

LogicalResult ModuleIndex::addSourceModule(OwningOpRef<ModuleOp> module) {
  ModuleOp moduleOp = module.get();

  // Check if this is an archive-style module with nested named modules.
  bool isArchive = !moduleOp.getSymName() && hasNestedNamedModules(moduleOp);
  if (isArchive) {
    // Keep the parent module alive first.
    auto outerInfo = std::make_unique<SourceModuleInfo>();
    outerInfo->module = std::move(module);
    ModuleOp outerModule = outerInfo->module.get();

    // Index each nested module separately.
    for (auto nestedModule : outerModule.getOps<ModuleOp>()) {
      if (auto name = nestedModule.getSymName()) {
        auto info = std::make_unique<SourceModuleInfo>();
        info->moduleName = name->str();
        // Don't set info->module - we don't own nested modules.
        // The parent keeps them alive.

        buildSymbolTable(*info, nestedModule);
        sourceModules.push_back(std::move(info));
      }
    }

    // Add the outer module last (it has no symbols of its own).
    sourceModules.push_back(std::move(outerInfo));
  } else {
    // Regular module - index as-is.
    auto info = std::make_unique<SourceModuleInfo>();
    info->module = std::move(module);

    if (auto name = info->module.get().getSymName()) {
      info->moduleName = name->str();
    }

    buildSymbolTable(*info);
    sourceModules.push_back(std::move(info));
  }

  return success();
}

void ModuleIndex::addLibraryPath(StringRef path) {
  libraryPaths.push_back(path.str());
}

void ModuleIndex::buildSymbolTable(SourceModuleInfo &info, ModuleOp moduleOp) {
  ModuleOp module = moduleOp ? moduleOp : info.module.get();

  // Walk top-level operations.
  for (Operation &op : module.getOps()) {
    // Skip nested modules.
    if (isa<ModuleOp>(&op)) {
      continue;
    }

    // Skip globals (too complex for now).
    if (isa<IREE::Util::GlobalOpInterface>(&op)) {
      continue;
    }

    // Skip isolated-from-above ops (except ObjectLike and functions).
    // Functions are isolated but we want to index them.
    if (op.hasTrait<OpTrait::IsIsolatedFromAbove>() &&
        !op.hasTrait<OpTrait::IREE::Util::ObjectLike>() &&
        !isa<FunctionOpInterface>(&op)) {
      continue;
    }

    // Extract symbol name.
    StringRef localName;
    if (auto funcOp = dyn_cast<FunctionOpInterface>(&op)) {
      // Skip initializers which implement FunctionOpInterface but have no name.
      if (isa<IREE::Util::InitializerOpInterface>(&op)) {
        continue;
      }
      if (funcOp.isExternal()) {
        continue; // Skip external declarations.
      }
      localName = funcOp.getName();
    } else if (auto symOp = dyn_cast<SymbolOpInterface>(&op)) {
      localName = symOp.getName();
    } else {
      continue; // Not a symbol.
    }

    // Create SymbolInfo with owned qualified name string.
    auto symbolInfo = std::make_unique<SymbolInfo>();
    symbolInfo->op = &op;
    symbolInfo->localName = localName; // Points to op's attribute storage.
    symbolInfo->sourceModule = &info;

    // Compute qualified name.
    if (!info.moduleName.empty()) {
      symbolInfo->qualifiedName = info.moduleName + "." + localName.str();
    } else {
      symbolInfo->qualifiedName = localName.str();
    }

    // Extract visibility.
    symbolInfo->visibility = SymbolTable::getSymbolVisibility(&op);

    // Add to global map for quick lookup.
    StringRef qualifiedNameRef = symbolInfo->qualifiedName;
    globalSymbolMap[qualifiedNameRef] = symbolInfo.get();

    // For named modules, also add public symbols with local name for external
    // references. Don't add private symbols to avoid conflicts when multiple
    // modules have same private names.
    if (!info.moduleName.empty() &&
        symbolInfo->visibility == SymbolTable::Visibility::Public) {
      globalSymbolMap[localName] = symbolInfo.get();
    }

    // Add to module's symbol map.
    info.symbols[qualifiedNameRef] = std::move(symbolInfo);
  }
}

ModuleIndex::SymbolInfo *ModuleIndex::lookupSymbol(StringRef qualifiedName) {
  auto it = globalSymbolMap.find(qualifiedName);
  if (it != globalSymbolMap.end()) {
    return it->second;
  }
  return nullptr;
}

void ModuleIndex::scanSymbolDependencies(SymbolInfo *info) {
  // Walk the operation and collect all symbol references in order.
  // We manually extract SymbolRefAttr to avoid triggering verification.
  DenseSet<StringRef> seen;

  LLVM_DEBUG(llvm::dbgs() << "Scanning dependencies for: "
                          << info->qualifiedName << "\n");

  // Note: We manually scan attributes instead of using
  // SymbolTable::getSymbolUses() because that triggers verification on
  // incomplete library modules during parsing. This is less efficient but
  // necessary for incremental module loading.
  //
  // For ObjectLike operations (like flow.executable), we need to scan ONLY the
  // operation itself, not its nested regions. ObjectLike operations are
  // self-contained and their internal symbols (like flow.executable.export)
  // should not be treated as external dependencies.

  // Lambda to scan a single operation's attributes.
  auto scanOperation = [&](Operation *op) {
    // Manually scan all attributes for SymbolRefAttr.
    for (NamedAttribute namedAttr : op->getAttrs()) {
      // Check if this attribute is a SymbolRefAttr or contains one.
      std::function<void(Attribute)> scanAttribute = [&](Attribute attrValue) {
        if (auto symRef = dyn_cast<SymbolRefAttr>(attrValue)) {
          StringRef symbolName = symRef.getRootReference().getValue();

          // If this is an unqualified reference and we're in a named module,
          // try to qualify it within the module's scope.
          StringRef qualifiedName = symbolName;
          if (!info->sourceModule->moduleName.empty() &&
              !symbolName.contains('.')) {
            // Unqualified reference in named module - try local scope first.
            SmallString<64> tentativeQualified;
            tentativeQualified += info->sourceModule->moduleName;
            tentativeQualified += '.';
            tentativeQualified += symbolName;
            auto it = globalSymbolMap.find(tentativeQualified);
            if (it != globalSymbolMap.end()) {
              qualifiedName = StringRef(it->second->qualifiedName);
              LLVM_DEBUG(llvm::dbgs() << "  Qualified " << symbolName << " -> "
                                      << qualifiedName << "\n");
            }
          }

          if (seen.insert(qualifiedName).second) {
            info->requiredSymbols.push_back(qualifiedName);
            LLVM_DEBUG(llvm::dbgs() << "  Requires: " << qualifiedName << "\n");
          }
        } else if (auto arrayAttr = dyn_cast<ArrayAttr>(attrValue)) {
          for (Attribute elem : arrayAttr) {
            scanAttribute(elem);
          }
        } else if (auto dictAttr = dyn_cast<DictionaryAttr>(attrValue)) {
          for (NamedAttribute dictEntry : dictAttr) {
            scanAttribute(dictEntry.getValue());
          }
        }
      };
      scanAttribute(namedAttr.getValue());
    }
  };

  // For ObjectLike operations, only scan the operation itself (not regions).
  // For other operations, walk the entire tree.
  if (info->op->hasTrait<OpTrait::IREE::Util::ObjectLike>()) {
    scanOperation(info->op);
  } else {
    info->op->walk([&](Operation *op) {
      scanOperation(op);
      return WalkResult::advance();
    });
  }
}

std::optional<OwningOpRef<ModuleOp>>
ModuleIndex::tryAutoLoadModule(StringRef symbolName) {
  // Extract module prefix from symbol name.
  auto dotPos = symbolName.find('.');
  if (dotPos == StringRef::npos || dotPos == 0) {
    return std::nullopt; // No prefix found.
  }

  StringRef moduleName = symbolName.substr(0, dotPos);

  // Try each library path with both .mlir and .mlirbc extensions.
  for (const auto &libPath : libraryPaths) {
    for (const char *ext : {".mlirbc", ".mlir"}) {
      llvm::SmallString<128> fullPath(libPath);
      llvm::sys::path::append(fullPath, moduleName + ext);

      if (llvm::sys::fs::exists(fullPath)) {
        auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(fullPath);
        if (std::error_code error = fileOrErr.getError()) {
          continue; // Try next path.
        }

        auto sourceMgr = std::make_shared<llvm::SourceMgr>();
        sourceMgr->AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
        ParserConfig config(context);
        OwningOpRef<ModuleOp> module =
            parseSourceFile<ModuleOp>(sourceMgr, config);
        if (module) {
          return module;
        }
      }
    }
  }

  return std::nullopt;
}

LogicalResult ModuleIndex::resolveTransitiveDependencies(
    ArrayRef<StringRef> requiredSymbols,
    SmallVectorImpl<SymbolInfo *> &resolvedSymbols,
    const DenseMap<StringRef, Location> &symbolLocations) {
  // Process each required symbol and its full dependency tree before moving to
  // the next. This ensures deterministic ordering: module_a's symbols and
  // dependencies come before module_b's symbols and dependencies, preserving
  // the original request order.
  DenseSet<StringRef> visited;
  for (StringRef rootSymbol : requiredSymbols) {
    if (visited.contains(rootSymbol)) {
      continue; // Already processed as a dependency of earlier symbol.
    }

    SmallVector<StringRef> worklist = {rootSymbol};
    while (!worklist.empty()) {
      StringRef symbolName = worklist.pop_back_val();
      if (visited.contains(symbolName)) {
        continue; // Already processed.
      }
      visited.insert(symbolName);

      SymbolInfo *info = lookupSymbol(symbolName);
      if (!info) {
        // Try auto-discovery.
        if (auto module = tryAutoLoadModule(symbolName)) {
          if (succeeded(addSourceModule(std::move(*module)))) {
            info = lookupSymbol(symbolName);
          }
        }
      }

      if (!info) {
        // Use the location from symbolLocations if available, otherwise
        // unknown.
        auto it = symbolLocations.find(symbolName);
        Location loc =
            it != symbolLocations.end() ? it->second : UnknownLoc::get(context);
        return emitError(loc)
               << "unresolved external symbol: '" << symbolName << "'";
      }

      // Lazily scan for dependencies if not already done.
      if (!info->scanned) {
        scanSymbolDependencies(info);
        info->scanned = true;
      }

      // Add transitive dependencies to worklist.
      for (StringRef required : info->requiredSymbols) {
        if (!visited.contains(required)) {
          worklist.push_back(required);
        }
      }

      resolvedSymbols.push_back(info);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConflictResolver - detects and resolves private symbol conflicts
//===----------------------------------------------------------------------===//

class ConflictResolver {
public:
  // Represents a rename: oldName -> newName within a source module.
  struct Rename {
    ModuleIndex::SourceModuleInfo *sourceModule;
    StringRef oldName;
    std::string newName;
  };

  ConflictResolver(MLIRContext *context, ModuleOp targetModule)
      : context(context), targetSymbolTable(targetModule),
        equivalenceCache(context) {}

  // Detects conflicts among symbols to link and computes renames.
  // Also generates qualification renames for public symbols from named modules.
  void detectConflicts(ArrayRef<ModuleIndex::SymbolInfo *> symbolsToLink,
                       SmallVectorImpl<Rename> &renames);

  // Applies renames to source modules using AttrTypeReplacer.
  LogicalResult applyRenames(ArrayRef<ModuleIndex::SymbolInfo *> symbolsToLink,
                             ArrayRef<Rename> renames);

private:
  // Generates a unique symbol name by appending _N suffix.
  // Checks both the target symbol table and usedNames set.
  std::string generateUniqueSymbolName(StringRef baseName,
                                       const DenseSet<StringRef> &usedNames) {
    std::string candidate = baseName.str();
    unsigned counter = 0;
    while (targetSymbolTable.lookup(candidate) ||
           usedNames.contains(candidate)) {
      candidate = (baseName + "_" + Twine(counter++)).str();
    }
    return candidate;
  }

  MLIRContext *context;
  SymbolTable targetSymbolTable;
  OperationEquivalenceCache equivalenceCache;
};

void ConflictResolver::detectConflicts(
    ArrayRef<ModuleIndex::SymbolInfo *> symbolsToLink,
    SmallVectorImpl<Rename> &renames) {

  // First, generate qualification renames for public symbols from named
  // modules. Public symbols from named modules use "module.symbol" naming.
  for (ModuleIndex::SymbolInfo *info : symbolsToLink) {
    if (!info->sourceModule->moduleName.empty() &&
        info->visibility == SymbolTable::Visibility::Public) {
      // This public symbol from a named module needs qualification.
      renames.push_back(
          {info->sourceModule, info->localName, info->qualifiedName});
    }
  }

  // Build set of all local names already in use (from symbols being linked).
  DenseSet<StringRef> usedNames;
  for (ModuleIndex::SymbolInfo *info : symbolsToLink) {
    usedNames.insert(info->localName);
  }

  // Track symbols that have been renamed (to skip them in later passes).
  DenseSet<ModuleIndex::SymbolInfo *> renamedSymbols;

  // First, check for conflicts with existing definitions in the target module.
  // If a symbol being linked has the same name as an existing target symbol
  // (and it's not just an external declaration) we need to rename the incoming
  // symbol.
  for (ModuleIndex::SymbolInfo *info : symbolsToLink) {
    Operation *existingOp = targetSymbolTable.lookup(info->localName);
    if (!existingOp) {
      // No conflict with target.
      continue;
    }

    // Check if the existing operation is a definition (not just external decl).
    bool isExistingDef = false;
    if (auto funcOp = dyn_cast<FunctionOpInterface>(existingOp)) {
      isExistingDef = !funcOp.isExternal();
    } else {
      // Non-function symbols are always definitions.
      isExistingDef = true;
    }

    if (isExistingDef) {
      // Conflict: target has a definition, incoming symbol also has a
      // definition. Rename the incoming symbol.
      std::string newName =
          generateUniqueSymbolName(info->localName, usedNames);
      usedNames.insert(newName); // Mark as used.
      renames.push_back(
          {info->sourceModule, info->localName, std::move(newName)});
      renamedSymbols.insert(info); // Track that this symbol was renamed.
    }
  }

  // Build map: local name -> list of SymbolInfo* with that name.
  // Skip symbols that were already renamed in the first pass.
  DenseMap<StringRef, SmallVector<ModuleIndex::SymbolInfo *>> byLocalName;
  for (ModuleIndex::SymbolInfo *info : symbolsToLink) {
    if (renamedSymbols.contains(info)) {
      continue; // Already renamed in target conflict check.
    }
    byLocalName[info->localName].push_back(info);
  }

  // Process each group of symbols with the same local name.
  for (auto &[localName, infos] : byLocalName) {
    if (infos.size() == 1) {
      // No conflict between source modules.
      continue;
    }

    // Collect private symbols from different source modules.
    SmallVector<ModuleIndex::SymbolInfo *> privateSymbols;
    for (ModuleIndex::SymbolInfo *info : infos) {
      if (info->visibility == SymbolTable::Visibility::Private) {
        privateSymbols.push_back(info);
      }
    }
    if (privateSymbols.size() <= 1) {
      // No conflict.
      continue;
    }

    // Group by source module to avoid conflicts within same module.
    DenseMap<ModuleIndex::SourceModuleInfo *,
             SmallVector<ModuleIndex::SymbolInfo *>>
        bySourceModule;
    for (ModuleIndex::SymbolInfo *info : privateSymbols) {
      bySourceModule[info->sourceModule].push_back(info);
    }
    if (bySourceModule.size() == 1) {
      // All from same module, no conflict.
      continue;
    }

    // Sort modules by their appearance in symbolsToLink to ensure deterministic
    // output. This ensures the first module encountered keeps its symbol name.
    // Build index map first to avoid O(n²) complexity during sort.
    DenseMap<ModuleIndex::SourceModuleInfo *, size_t> moduleFirstIndex;
    for (size_t i = 0; i < symbolsToLink.size(); ++i) {
      moduleFirstIndex.try_emplace(symbolsToLink[i]->sourceModule, i);
    }

    SmallVector<std::pair<ModuleIndex::SourceModuleInfo *,
                          SmallVector<ModuleIndex::SymbolInfo *>>>
        sortedModules(bySourceModule.begin(), bySourceModule.end());
    llvm::sort(sortedModules, [&](const auto &a, const auto &b) {
      return moduleFirstIndex[a.first] < moduleFirstIndex[b.first];
    });

    // Keep first occurrence, rename the rest.
    bool isFirst = true;
    for (auto &entry : sortedModules) {
      auto *sourceModule = entry.first;
      auto &moduleInfos = entry.second;
      if (isFirst) {
        // Keep first module's version.
        isFirst = false;
        continue;
      }

      // Rename all symbols from this module.
      for (ModuleIndex::SymbolInfo *info : moduleInfos) {
        // Check structural equivalence with first occurrence.
        // For now, we always rename to be safe (can optimize later).

        // Generate unique name, checking both target and usedNames.
        std::string newName =
            generateUniqueSymbolName(info->localName, usedNames);
        usedNames.insert(newName); // Mark as used.

        renames.push_back({sourceModule, info->localName, std::move(newName)});
      }
    }
  }
}

LogicalResult ConflictResolver::applyRenames(
    ArrayRef<ModuleIndex::SymbolInfo *> symbolsToLink,
    ArrayRef<Rename> renames) {

  // Group renames by source module.
  DenseMap<ModuleIndex::SourceModuleInfo *, SmallVector<Rename>>
      renamesByModule;
  for (const Rename &rename : renames) {
    renamesByModule[rename.sourceModule].push_back(rename);
  }

  // Group symbols to link by source module.
  DenseMap<ModuleIndex::SourceModuleInfo *,
           SmallVector<ModuleIndex::SymbolInfo *>>
      symbolsByModule;
  for (ModuleIndex::SymbolInfo *info : symbolsToLink) {
    symbolsByModule[info->sourceModule].push_back(info);
  }

  // Apply renames per module.
  for (auto &entry : renamesByModule) {
    auto *sourceModule = entry.first;
    auto &moduleRenames = entry.second;

    // Build reverse lookup map: localName -> SymbolInfo* for O(1) lookup.
    DenseMap<StringRef, ModuleIndex::SymbolInfo *> localNameToInfo;
    for (auto &[qualifiedName, symbolInfo] : sourceModule->symbols) {
      localNameToInfo[symbolInfo->localName] = symbolInfo.get();
    }

    // Build replacement map for AttrTypeReplacer.
    DenseMap<StringAttr, StringAttr> replacementMap;
    for (const Rename &rename : moduleRenames) {
      StringAttr oldAttr = StringAttr::get(context, rename.oldName);
      StringAttr newAttr = StringAttr::get(context, rename.newName);
      replacementMap[oldAttr] = newAttr;
    }

    // Setup AttrTypeReplacer.
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](SymbolRefAttr attr) -> std::optional<Attribute> {
          StringAttr rootRef = attr.getRootReference();
          auto it = replacementMap.find(rootRef);
          if (it != replacementMap.end()) {
            if (attr.getNestedReferences().empty()) {
              return SymbolRefAttr::get(it->second);
            } else {
              return SymbolRefAttr::get(it->second, attr.getNestedReferences());
            }
          }
          return std::nullopt;
        });

    // Only walk operations we're linking, not entire module.
    auto symbolsIt = symbolsByModule.find(sourceModule);
    if (symbolsIt != symbolsByModule.end()) {
      for (ModuleIndex::SymbolInfo *info : symbolsIt->second) {
        info->op->walk([&](Operation *op) {
          replacer.replaceElementsIn(op);
          return WalkResult::advance();
        });
      }
    }

    // Rename the operation symbols themselves using O(1) lookup.
    for (const Rename &rename : moduleRenames) {
      auto it = localNameToInfo.find(rename.oldName);
      if (it == localNameToInfo.end()) {
        continue; // Already renamed or not found.
      }
      ModuleIndex::SymbolInfo *info = it->second;

      // Rename the operation.
      if (auto funcOp = dyn_cast<FunctionOpInterface>(info->op)) {
        funcOp.setName(rename.newName);
      } else if (auto symOp = dyn_cast<SymbolOpInterface>(info->op)) {
        symOp.setName(rename.newName);
      }

      // Update SymbolInfo's names to reflect the rename.
      info->qualifiedName = rename.newName;
      info->localName = StringRef(info->qualifiedName);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ModuleLinker - main linking orchestrator
//===----------------------------------------------------------------------===//

class ModuleLinker {
public:
  explicit ModuleLinker(MLIRContext *context)
      : index(context), context(context) {}

  // Adds a source module from file.
  LogicalResult addSourceModule(StringRef filename) {
    return index.addSourceModule(filename);
  }

  // Adds a source module from parsed module.
  LogicalResult addSourceModule(OwningOpRef<ModuleOp> module) {
    return index.addSourceModule(std::move(module));
  }

  // Adds a library search path.
  void addLibraryPath(StringRef path) { index.addLibraryPath(path); }

  // Performs linking on target module.
  LogicalResult link(ModuleOp targetModule);

private:
  ModuleIndex index;
  MLIRContext *context;
};

LogicalResult ModuleLinker::link(ModuleOp targetModule) {
  // Collect external function declarations that need linking.
  SmallVector<FunctionOpInterface> externals;
  targetModule.walk([&](FunctionOpInterface funcOp) {
    if (funcOp.isExternal()) {
      externals.push_back(funcOp);
    }
  });

  // Early exit if nothing to link.
  if (externals.empty()) {
    return success();
  }

  // Resolve transitive dependencies using DFS traversal.
  SmallVector<StringRef> requiredSymbols;
  DenseMap<StringRef, Location> symbolLocations;
  for (auto funcOp : externals) {
    requiredSymbols.push_back(funcOp.getName());
    symbolLocations.insert({funcOp.getName(), funcOp.getLoc()});
  }

  SmallVector<ModuleIndex::SymbolInfo *> resolvedSymbols;
  if (failed(index.resolveTransitiveDependencies(
          requiredSymbols, resolvedSymbols, symbolLocations))) {
    return failure();
  }

  // The resolvedSymbols list is in DFS discovery order from
  // resolveTransitiveDependencies, with each symbol appearing exactly once
  // followed immediately by its transitive dependencies.

  // Detect conflicts and generate all renames (qualification + conflicts).
  ConflictResolver resolver(context, targetModule);
  SmallVector<ConflictResolver::Rename> renames;
  resolver.detectConflicts(resolvedSymbols, renames);

  // Apply all renames in a single unified pass.
  if (!renames.empty()) {
    if (failed(resolver.applyRenames(resolvedSymbols, renames))) {
      return failure();
    }
  }

  // Move operations to target module, preserving natural grouping.
  // Insert new operations after the function that pulled them in for natural
  // grouping.
  SymbolTable targetSymbolTable(targetModule);
  Operation *insertionPoint = nullptr;
  for (ModuleIndex::SymbolInfo *info : resolvedSymbols) {
    // Check if external declaration exists in target.
    Operation *existingDecl = targetSymbolTable.lookup(info->qualifiedName);
    if (auto existingFunc =
            dyn_cast_if_present<FunctionOpInterface>(existingDecl)) {
      if (auto sourceFunc = dyn_cast<FunctionOpInterface>(info->op)) {
        // Verify types match.
        if (existingFunc.getFunctionType() != sourceFunc.getFunctionType()) {
          return emitError(targetModule.getLoc())
                 << "type mismatch for symbol '" << info->qualifiedName
                 << "': expected " << existingFunc.getFunctionType() << ", got "
                 << sourceFunc.getFunctionType();
        }

        // Fill existing declaration in-place (preserves position and
        // visibility).
        existingFunc.getFunctionBody().takeBody(sourceFunc.getFunctionBody());
        info->op->erase();
        insertionPoint = existingFunc;
      } else {
        return emitError(targetModule.getLoc())
               << "symbol type mismatch for '" << info->qualifiedName << "'";
      }
    } else {
      // No existing declaration - insert after last processed operation.
      // These are internal implementation details, so mark as private.
      Operation *op = info->op;
      op->remove();

      if (insertionPoint) {
        // Insert immediately after the last processed operation.
        Block *block = insertionPoint->getBlock();
        block->getOperations().insertAfter(Block::iterator(insertionPoint), op);
      } else {
        // First insertion (no previous operation), append at end.
        OpBuilder::atBlockEnd(targetModule.getBody()).insert(op);
      }

      // Mark new operations as private (internal implementation details).
      SymbolTable::setSymbolVisibility(op, SymbolTable::Visibility::Private);

      insertionPoint = op;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LinkModulesPass
//===----------------------------------------------------------------------===//

struct LinkModulesPass : public impl::LinkModulesPassBase<LinkModulesPass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp targetModule = getOperation();
    ModuleLinker linker(targetModule.getContext());

    // Add library search paths.
    for (const auto &path : libraryPaths) {
      linker.addLibraryPath(path);
    }

    // Load explicit source modules.
    for (const auto &modulePath : linkModules) {
      if (failed(linker.addSourceModule(modulePath))) {
        return signalPassFailure();
      }
    }

    // Perform linking.
    if (failed(linker.link(targetModule))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
