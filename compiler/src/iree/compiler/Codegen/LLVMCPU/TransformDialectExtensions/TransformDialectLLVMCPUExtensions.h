// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMCPUEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMCPUEXTENSIONS_H_

#include "mlir/Dialect/Transform/IR/TransformDialect.h"

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMCPU/TransformDialectExtensions/TransformDialectLLVMCPUExtensionsOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace iree_compiler {

/// Registers LLVMCPU transformations that require IREE-specific information
/// into the transform dialect.
void registerTransformDialectLLVMCPUExtension(DialectRegistry &registry);

namespace IREE {
namespace transform_dialect {
// Hook to register LLVMCPU transformations to the transform dialect.
class TransformDialectLLVMCPUExtensions
    : public transform::TransformDialectExtension<
          TransformDialectLLVMCPUExtensions> {
 public:
  TransformDialectLLVMCPUExtensions();
};
}  // namespace transform_dialect
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMCPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMCPUEXTENSIONS_H_
