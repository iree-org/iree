// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_BUFFERIZATIONINTERFACES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_BUFFERIZATIONINTERFACES_H_

#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler::IREE::VectorExt {

// Register all interfaces needed for bufferization.
void registerIREEVectorExtBufferizationInterfaces(DialectRegistry &registry);

} // namespace mlir::iree_compiler::IREE::VectorExt

#endif // IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_BUFFERIZATIONINTERFACES_H_
