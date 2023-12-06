// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_LINKINGUTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_LINKINGUTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

// Returns a uniqued set of all targets in |executableOps|.
SetVector<IREE::HAL::ExecutableTargetAttr>
gatherExecutableTargets(ArrayRef<IREE::HAL::ExecutableOp> executableOps);

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
                DenseMap<StringRef, Operation *> &targetSymbolMap);

// Links all executables for the current target found in |moduleOp| into
// |linkedExecutableOp|. Functions will be moved into |linkedModuleOp|.
//
// |sourceExecutableOps| will be updated to remove source executable ops once
// they are fully merged into |linkedModuleOp|.
//
// |mergeInnerModuleFn| is a function that specifies how to merge the contents
// in |sourceInnerModule| into the |linkedInnerModule|, and updates the
// |symbolMap| along the way.
LogicalResult linkExecutablesInto(
    mlir::ModuleOp moduleOp,
    SmallVectorImpl<IREE::HAL::ExecutableOp> &sourceExecutableOps,
    IREE::HAL::ExecutableOp linkedExecutableOp,
    IREE::HAL::ExecutableVariantOp linkedTargetOp,
    std::function<LogicalResult(mlir::ModuleOp sourceInnerModule,
                                mlir::ModuleOp linkedInnerModule,
                                DenseMap<StringRef, Operation *> &symbolMap)>
        mergeInnerModuleFn);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_UTILS_LINKINGUTILS_H_
