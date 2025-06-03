// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h"
#include "mlir/Parser/Parser.h"

namespace mlir::iree_compiler::IREE::ROCM {

FailureOr<ModuleOp> ROCMDialect::getOrLoadBuiltinModule(StringRef path) {
  std::optional<StringRef> maybeBuiltin = builtins.getFile(path);
  if (!maybeBuiltin) {
    return failure();
  }

  // Acquire a lock on the map that will release once out of scope.
  std::lock_guard<std::mutex> guard(builtinMutex);
  MLIRContext *ctx = getContext();

  auto libraryIt = builtinModules.find(path);
  if (libraryIt != builtinModules.end()) {
    // Check whether the library already failed to load.
    if (ModuleOp module = libraryIt->second.get()) {
      return module;
    }
    return failure();
  }

  // We update the storage for the library regardless of whether parsing
  // succeeds so that other threads don't have to retry.
  OwningOpRef<ModuleOp> &parsedLibrary = builtinModules[path];

  parsedLibrary = parseSourceString<mlir::ModuleOp>(maybeBuiltin.value(), ctx);
  if (!parsedLibrary) {
    return failure();
  }

  return parsedLibrary.get();
}

} // namespace mlir::iree_compiler::IREE::ROCM
