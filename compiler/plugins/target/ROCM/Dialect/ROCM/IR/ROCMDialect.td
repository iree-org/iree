// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_DIALECT
#define IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def ROCM_Dialect : Dialect {
  let name = "rocm";
  let cppNamespace = "::mlir::iree_compiler::IREE::ROCM";

  let summary = [{
    A dialect for functionality used by AMD related IREE code generation.
  }];
  let description = [{
    This dialect provides operations and attributes to aid in code generation
    for AMD targets that lower to LLVM.
  }];
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();

    /// Helper for checking whether the requested builtin is present.
    bool hasBuiltin(StringRef name);

    /// Returns the loaded builtin module for the `path`. If the module has
    /// already been loaded in the past, returns the memoized module without
    /// parsing the file again. This function is thead-safe.
    FailureOr<::mlir::ModuleOp>
    getOrLoadBuiltinModule(StringRef path);

    private:

    /// Pseudo-directory for storing references to builtins.
    EmbeddedDataDirectory builtins;

    /// Map from "filename" to module, cached on load. Accesses to this map are
    /// protected by a mutex.
    ::llvm::StringMap<OwningOpRef<::mlir::ModuleOp>> builtinModules;

    /// Lock to control the updating of the builtin modules such that we only
    /// load the module once and can reuse it across all invocations.
    std::mutex builtinMutex;
  }];
}

#endif // IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_DIALECT
