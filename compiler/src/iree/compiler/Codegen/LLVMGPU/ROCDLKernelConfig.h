// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_ROCDLKERNELCONFIG_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_ROCDLKERNELCONFIG_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler {

LogicalResult initROCDLLaunchConfig(ModuleOp moduleOp);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_ROCDLKERNELCONFIG_H_
