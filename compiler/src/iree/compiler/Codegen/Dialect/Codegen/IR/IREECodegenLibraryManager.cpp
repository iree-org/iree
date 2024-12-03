// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"

namespace mlir::iree_compiler::IREE::Codegen {

FailureOr<ModuleOp>
IREECodegenDialect::getOrLoadTransformLibraryModule(std::string libraryPath) {
  // Acquire a lock on the map that will release once out of scope.
  std::lock_guard<std::mutex> guard(libraryMutex);

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

  MLIRContext *ctx = getContext();
  if (failed(transform::detail::parseTransformModuleFromFile(ctx, libraryPath,
                                                             parsedLibrary))) {
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

} // namespace mlir::iree_compiler::IREE::Codegen
