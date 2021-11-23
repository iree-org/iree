// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_INTERFACES_BUFFERIZATIONINTERFACES_H_
#define IREE_COMPILER_CODEGEN_INTERFACES_BUFFERIZATIONINTERFACES_H_

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {

// Register all interfaces needed for bufferization.
void registerBufferizationInterfaces(DialectRegistry &registry);

// Method to add all the analysis passes for bufferization.
void addPostAnalysisTransformations(
    linalg::comprehensive_bufferize::BufferizationOptions &options);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_INTERFACES_BUFFERIZATIONINTERFACES_H_
