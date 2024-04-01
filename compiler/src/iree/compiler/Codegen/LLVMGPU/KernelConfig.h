// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_KERNELCONFIG_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_KERNELCONFIG_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

int64_t getTargetSharedMemoryLimit(FunctionOpInterface entryPoint);

LogicalResult initGPULaunchConfig(ModuleOp moduleOp);

}  // namespace mlir::iree_compiler
#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_KERNELCONFIG_H_
