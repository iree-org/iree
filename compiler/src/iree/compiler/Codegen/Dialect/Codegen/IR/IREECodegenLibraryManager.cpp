// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"

namespace mlir::iree_compiler::IREE::Codegen {

FailureOr<ModuleOp>
IREECodegenDialect::getOrLoadTransformLibraryModule(std::string libraryPath) {
  // Acquire a lock on the map that will release once out of scope.
  std::lock_guard<std::mutex> guard(libraryMutex);

  auto loadedLibrary = libraryModules.find(libraryPath);
  if (loadedLibrary != libraryModules.end()) {
    // Check whether the library already failed to load.
    if (!(loadedLibrary->second) || !(*(loadedLibrary->second))) {
      return failure();
    }
    return *(loadedLibrary->second);
  }

  OwningOpRef<ModuleOp> mergedParsedLibraries;
  if (failed(transform::detail::assembleTransformLibraryFromPaths(
          getContext(), SmallVector<std::string>{libraryPath},
          mergedParsedLibraries))) {
    // We update the storage for the library regardless of whether parsing
    // succeeds so that other threads don't have to retry.
    OwningOpRef<ModuleOp> emptyLibrary;
    libraryModules[libraryPath] = std::move(emptyLibrary);
    return failure();
  }

  libraryModules[libraryPath] = std::move(mergedParsedLibraries);
  return *libraryModules[libraryPath];
}

} // namespace mlir::iree_compiler::IREE::Codegen
