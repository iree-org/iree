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

// Links all executables for the current target found in |moduleOp| into
// |linkedExecutableOp|. Functions will be cloned into |linkedModuleOp|.
LogicalResult linkExecutablesInto(
    mlir::ModuleOp moduleOp,
    ArrayRef<IREE::HAL::ExecutableOp> sourceExecutableOps,
    IREE::HAL::ExecutableOp linkedExecutableOp,
    IREE::HAL::ExecutableVariantOp linkedTargetOp,
    std::function<Operation *(mlir::ModuleOp moduleOp)> getInnerModuleFn);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_UTILS_LINKINGUTILS_H_
