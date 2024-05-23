// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_BUFFERIZATIONINTERFACES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_BUFFERIZATIONINTERFACES_H_

#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler {

// Register all interfaces needed for bufferization.
void registerIREEGPUBufferizationInterfaces(DialectRegistry &registry);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_BUFFERIZATIONINTERFACES_H_
