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

  // Issue #22842: Avoid doing nontrivial MLIR work (such as parsing) in a
  // critical section. Due to how MLIR threading works, any threaded workload
  // may result in yielding and scheduling another task on the same thread,
  // potentially reentering this code on the same thread, resulting in
  // deadlocks. That is why the code below is structured with two separate
  // critical sections leaving the MLIR parsing itself outside. It was
  // specifically the verifier that was being threaded here, and we could have
  // set verifyAfterParse=false, but that would be unsafely assuming that the
  // verifier would be the only threaded work here.

  {
    // Critical section: check if already found in builtinModules.
    std::lock_guard<std::mutex> guard(builtinMutex);
    auto libraryIt = builtinModules.find(path);
    if (libraryIt != builtinModules.end()) {
      // Check whether the library already failed to load.
      if (ModuleOp module = libraryIt->second.get()) {
        return module;
      }
      return failure();
    }
  }

  // Do the parsing outside of critical sections, so that reentry will not
  // deadlock.
  OwningOpRef<ModuleOp> localModule =
      parseSourceString<mlir::ModuleOp>(maybeBuiltin.value(), getContext(),
                                        /*sourceName=*/path);

  // Critical section: insert into builtinModules.
  std::lock_guard<std::mutex> guard(builtinMutex);
  OwningOpRef<ModuleOp> &destinationModule = builtinModules[path];
  // Insert unconditionally, even if failed to parse: avoid reparsing.
  destinationModule = std::move(localModule);
  // Check if this failed to parse.
  if (!destinationModule) {
    return failure();
  }
  return destinationModule.get();
}

} // namespace mlir::iree_compiler::IREE::ROCM
