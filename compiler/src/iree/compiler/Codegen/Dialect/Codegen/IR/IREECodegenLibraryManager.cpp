// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir::iree_compiler::IREE::Codegen {

/// Helper function that implements the module lookup and validation.
/// The caller is responsible for acquiring the mutex for `libraryModules`.
static FailureOr<ModuleOp> getOrParseTransformLibraryModuleImpl(
    StringRef libraryPath,
    llvm::StringMap<OwningOpRef<ModuleOp>> &libraryModules,
    llvm::function_ref<LogicalResult(OwningOpRef<ModuleOp> &)> loadModuleFn) {

  auto loadedLibrary = libraryModules.find(libraryPath);
  if (loadedLibrary != libraryModules.end()) {
    // Check whether the library already failed to load.
    if (ModuleOp module = loadedLibrary->second.get()) {
      return module;
    }
    return failure();
  }

  // We update the storage for the library regardless of whether parsing
  // succeeds so that other threads don't have to retry.
  OwningOpRef<ModuleOp> &parsedLibrary = libraryModules[libraryPath];

  if (failed(loadModuleFn(parsedLibrary))) {
    return failure();
  }

  if (!parsedLibrary.get()->hasAttr(
          transform::TransformDialect::kWithNamedSequenceAttrName)) {
    parsedLibrary->emitError()
        << "Module without the '"
        << transform::TransformDialect::kWithNamedSequenceAttrName
        << "' attribute is not a transform dialect library";

    // Invalidate the module stored in the library so that this does not
    // succeed on a retry.
    parsedLibrary = nullptr;
    return failure();
  }

  if (!parsedLibrary->getSymName()) {
    parsedLibrary->setSymName("__transform");
  }

  return parsedLibrary.get();
}

FailureOr<ModuleOp>
IREECodegenDialect::getOrLoadTransformLibraryModule(StringRef libraryPath) {
  // Acquire a lock on the map that will release once out of scope.
  std::lock_guard<std::mutex> guard(libraryMutex);
  MLIRContext *ctx = getContext();

  return getOrParseTransformLibraryModuleImpl(
      libraryPath, libraryModules, [=](OwningOpRef<ModuleOp> &parsedLibrary) {
        return transform::detail::parseTransformModuleFromFile(ctx, libraryPath,
                                                               parsedLibrary);
      });
}

FailureOr<ModuleOp> IREECodegenDialect::getOrParseTransformLibraryModule(
    StringRef libraryPath, StringRef libraryMLIRSource) {
  // Acquire a lock on the map that will release once out of scope.
  std::lock_guard<std::mutex> guard(libraryMutex);
  MLIRContext *ctx = getContext();

  return getOrParseTransformLibraryModuleImpl(
      libraryPath, libraryModules, [=](OwningOpRef<ModuleOp> &parsedLibrary) {
        ParserConfig config(ctx);
        parsedLibrary =
            parseSourceString<ModuleOp>(libraryMLIRSource, ctx, libraryPath);
        return success(*parsedLibrary != nullptr);
      });
}

FailureOr<ModuleOp>
IREECodegenDialect::getOrLoadPatchedFuncOpsForDebugging(std::string path) {
  // Acquire a lock on the map that will release once out of scope.
  std::lock_guard<std::mutex> guard(moduleForPatchesMutex);

  auto loadedLibrary = moduleForPatches.find(path);
  if (loadedLibrary != moduleForPatches.end()) {
    // Check whether the library already failed to load.
    if (!(loadedLibrary->second) || !(*(loadedLibrary->second))) {
      return failure();
    }
    return *(loadedLibrary->second);
  }

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> memoryBuffer =
      mlir::openInputFile(path, &errorMessage);
  if (!memoryBuffer) {
    return emitError(
               FileLineColLoc::get(StringAttr::get(getContext(), path), 0, 0))
           << "failed to open transform file: " << errorMessage;
  }
  // Tell sourceMgr about this buffer, the parser will pick it up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  auto moduleOp =
      OwningOpRef<ModuleOp>(parseSourceFile<ModuleOp>(sourceMgr, getContext()));
  if (!moduleOp) {
    // Don't need to emit an error here as the parsing should have already done
    // that.
    return failure();
  }
  if (failed(mlir::verify(*moduleOp))) {
    return failure();
  }

  moduleForPatches[path] = std::move(moduleOp);
  return *moduleForPatches[path];
}

} // namespace mlir::iree_compiler::IREE::Codegen
