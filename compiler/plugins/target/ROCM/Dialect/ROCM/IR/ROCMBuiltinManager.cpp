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

  // Internally (parseSourceString, builtinModules) we use OwningOpRef<ModuleOp>
  // but we need to return a FailureOr<ModuleOp>. This performs the conversion.
  auto failureOr = [](const OwningOpRef<ModuleOp> &m) -> FailureOr<ModuleOp> {
    if (m) {
      return m.get();
    }
    return failure();
  };

  // Issue #22842: Avoid doing nontrivial MLIR work (such as parsing) in a
  // critical section. Due to how MLIR threading works, any threaded workload
  // may result in yielding and scheduling another task on the same thread,
  // potentially reentering this code on the same thread, resulting in
  // deadlocks. That is why the code below is structured with two separate
  // critical sections leaving the MLIR parsing itself outside. It was
  // specifically the verifier that was being threaded here, and we could have
  // set verifyAfterParse=false, but we actually care about the verifier running
  // here, and it is unsafe to assume that it will always be the only threaded
  // thing here.

  {
    // Critical section: check if already found in builtinModules.
    std::lock_guard<std::mutex> guard(builtinMutex);
    auto iter = builtinModules.find(path);
    if (iter != builtinModules.end()) {
      // Check whether the library already failed to load.
      return failureOr(iter->second);
    }
  }

  // Do the parsing outside of critical sections, so that reentry will not
  // deadlock.
  OwningOpRef<ModuleOp> localModule =
      parseSourceString<mlir::ModuleOp>(maybeBuiltin.value(), getContext(),
                                        /*sourceName=*/path);

  // Critical section: insert into builtinModules if not already found.
  std::lock_guard<std::mutex> guard(builtinMutex);
  // Check again if already found: it could have been inserted by another thread
  // while we were parsing.
  auto iter = builtinModules.find(path);
  if (iter != builtinModules.end()) {
    // Check whether the library already failed to load.
    return failureOr(iter->second);
  }
  OwningOpRef<ModuleOp> &insertedModule = builtinModules[path];
  // Insert unconditionally, even if failed to parse: avoid reparsing.
  insertedModule = std::move(localModule);
  // Check if this failed to parse.
  return failureOr(insertedModule);
}

} // namespace mlir::iree_compiler::IREE::ROCM
