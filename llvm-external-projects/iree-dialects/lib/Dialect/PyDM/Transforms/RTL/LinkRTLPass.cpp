// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "../PassDetail.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.h"
#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h"
#include "iree-dialects/Dialect/PyDM/Transforms/RTL/LinkageAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree_pydm"

using namespace mlir;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
using namespace PYDM;

static StringRef safeModuleName(Operation *op) {
  if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
    auto name = moduleOp.getName();
    return name ? *name : StringRef("");
  }
  return "(unknown module)";
}

namespace {

class LinkIREEPyDMRTLPass : public LinkIREEPyDMRTLBase<LinkIREEPyDMRTLPass> {
public:
  LinkIREEPyDMRTLPass() = default;
  LinkIREEPyDMRTLPass(Optional<SourceBundle> linkRtlSourceBundle)
      : linkRtlSourceBundle(std::move(linkRtlSourceBundle)) {}

private:
  LogicalResult initialize(MLIRContext *context) override {
    SourceBundle localSource;
    if (linkRtlSourceBundle) {
      localSource = *linkRtlSourceBundle;
    } else {
      // Get it from the cli options.
      localSource.asmFilePath = rtlFile;
    }

    if (localSource.asmBlob) {
      // Parse from inline asm.
      auto owningOp = parseSourceString(*localSource.asmBlob, context);
      if (!owningOp)
        return failure();
      rtlModule = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(
          std::move(owningOp));
    } else if (localSource.asmFilePath) {
      // Parse from a file.
      auto owningOp = parseSourceFile(*localSource.asmFilePath, context);
      if (!owningOp)
        return failure();
      rtlModule = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(
          std::move(owningOp));
    } else {
      return emitError(UnknownLoc::get(context))
             << "pass " << getArgument()
             << "must be initialized with an RTL module (did you mean to "
                "add an rtl-file option?)";
    }

    ModuleOp parentModule = rtlModule->get();
    // Walk the module and build a SymbolTable for each sub-module.
    parentModule->walk([&](ModuleOp importModule) {
      if (importModule != parentModule) {
        LLVM_DEBUG(llvm::dbgs() << "Loaded RTL module "
                                << safeModuleName(importModule) << "\n");
        importModules.emplace_back(importModule);
      }
      // We don't need to descend into functions so just skip them.
      return WalkResult::skip();
    });

    return success();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    SymbolTable programSymbolTable(moduleOp);
    for (int i = 0; i < 1000; i++) {
      auto analysis = getAnalysis<LinkageAnalysis>();
      if (!analysis.hasExternFuncs()) {
        LLVM_DEBUG(llvm::dbgs() << "No extern funcs to link.\n");
        if (i == 0) {
          markAllAnalysesPreserved();
        }
        return;
      }

      SetVector<Operation *> externFuncOps(analysis.getExternFuncOps().begin(),
                                           analysis.getExternFuncOps().end());
      while (!externFuncOps.empty()) {
        auto externOp = *externFuncOps.begin();
        if (failed(linkExtern(programSymbolTable, externOp, externFuncOps))) {
          return signalPassFailure();
        }
      }

      getAnalysisManager().invalidate({});
    }

    emitError(moduleOp.getLoc()) << "failed to converge when linking RTL";
    return signalPassFailure();
  }

  LogicalResult linkExtern(SymbolTable &programSymbolTable, Operation *externOp,
                           SetVector<Operation *> &externFuncOps) {
    // First see if we can find a module that defines the symbol.
    StringAttr probeSymbolName = SymbolTable::getSymbolName(externOp);
    for (SymbolTable &importModule : importModules) {
      Operation *probeImport = importModule.lookup(probeSymbolName);
      if (probeImport) {
        LLVM_DEBUG(llvm::dbgs()
                       << "Resolving extern " << probeSymbolName << " from "
                       << safeModuleName(importModule.getOp()) << "\n";);

        if (failed(inlineImportModule(programSymbolTable, importModule,
                                      externFuncOps)))
          return failure();
        return success();
      }
    }

    externOp->emitError() << "could not resolve extern " << probeSymbolName;
    return failure();
  }

  // Inlines an import module into a program module. This is a relatively
  // brute force mechanism and it requires that symbols do not collide (i.e.
  // if the program defined the same name as an RTL export, that would be an
  // error). It is possible to make something smarter but not clear it is
  // necessary, given the limited scope of "linking".
  LogicalResult inlineImportModule(SymbolTable &programSymbolTable,
                                   SymbolTable &importModule,
                                   SetVector<Operation *> &externFuncOps) {
    LLVM_DEBUG(llvm::dbgs() << "+++ Inlining module\n";);
    auto result = importModule.getOp()->walk<WalkOrder::PreOrder>(
        [&](Operation *importOp) -> WalkResult {
          if (importOp == importModule.getOp())
            return WalkResult::advance();
          if (auto symbolImportOp = dyn_cast<SymbolOpInterface>(importOp)) {
            StringAttr name = symbolImportOp.getNameAttr();
            Operation *existing = programSymbolTable.lookup(name);
            if (existing) {
              if (failed(verifyCanImport(existing, importOp)))
                return WalkResult::interrupt();

              LLVM_DEBUG(llvm::dbgs() << "*** Erasing existing import " << name
                                      << "\n";);
              // Bookkeeping.
              externFuncOps.remove(existing);
              programSymbolTable.erase(existing);
            }
            // Clone and insert.
            Operation *importedOp = importOp->clone();
            SymbolTable::setSymbolVisibility(importedOp,
                                             SymbolTable::Visibility::Private);
            programSymbolTable.insert(importedOp);
          } else {
            importOp->emitWarning()
                << "RTL module has non-importable operation ("
                << importOp->getName() << "). Skipping.";
          }
          return WalkResult::skip();
        });
    if (result.wasInterrupted())
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "--- Inlining complete\n";);
    return success();
  }

  LogicalResult verifyCanImport(Operation *existing, Operation *importOp) {
    // Must be the same type of operation.
    if (existing->getName() != importOp->getName()) {
      existing->emitError()
          << "attempt to import RTL operation of different type ("
          << importOp->getName() << " into " << existing->getName() << ")";
      return failure();
    }

    // If a FuncOp, must be an import.
    if (auto symbolOp = dyn_cast<SymbolOpInterface>(existing)) {
      if (!symbolOp.isDeclaration()) {
        return existing->emitError()
               << "cannot import a symbol that is already defined";
      }
    } else {
      return existing->emitError() << "cannot import a non-symbol";
    }

    return success();
  }

  // Really, this is the best option for this kind of thing.
  std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> rtlModule;

  // A SymbolTable for each sub module.
  SmallVector<SymbolTable> importModules;

  // ASM source of RTL modules to link (otherwise will use pass options).
  Optional<SourceBundle> linkRtlSourceBundle;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
PYDM::createLinkIREEPyDMRTLPass(Optional<SourceBundle> linkRtlSourceBundle) {
  return std::make_unique<LinkIREEPyDMRTLPass>(std::move(linkRtlSourceBundle));
}
