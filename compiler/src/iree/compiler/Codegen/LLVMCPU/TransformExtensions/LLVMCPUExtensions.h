// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_TRANSFORMEXTENSIONS_LLVMCPUEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_TRANSFORMEXTENSIONS_LLVMCPUEXTENSIONS_H_

#include "mlir/Dialect/Transform/IR/TransformDialect.h"

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensionsOps.h.inc"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::iree_compiler {

/// Registers LLVMCPU transformations that require IREE-specific information
/// into the transform dialect.
void registerTransformDialectLLVMCPUExtension(DialectRegistry &registry);

namespace IREE ::transform_dialect {
// Hook to register LLVMCPU transformations to the transform dialect.
class LLVMCPUExtensions
    : public transform::TransformDialectExtension<LLVMCPUExtensions> {
public:
  LLVMCPUExtensions();
};
} // namespace IREE::transform_dialect

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_TRANSFORMEXTENSIONS_LLVMCPUEXTENSIONS_H_
