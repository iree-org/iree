// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUCONSTRAINTGENERATOR_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUCONSTRAINTGENERATOR_H_

#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler {

/// Register the LLVMGPU constraint generator as an external model on
/// PipelineAttr via the IREEGPUDialect extension.
void registerLLVMGPUConstraintExternalInterfaces(DialectRegistry &registry);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUCONSTRAINTGENERATOR_H_
