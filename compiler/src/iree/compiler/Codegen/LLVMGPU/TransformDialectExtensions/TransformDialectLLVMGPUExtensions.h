// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMGPUEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMGPUEXTENSIONS_H_

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

namespace mlir {
class DialectRegistry;

namespace scf {
class IfOp;
}  // namespace scf

namespace vector {
class VectorDialect;
class WarpExecuteOnLane0Op;
}  // namespace vector

namespace iree_compiler {

/// Registers Flow transformations that require IREE-specific information into
/// the transform dialect.
void registerTransformDialectLLVMGPUExtension(DialectRegistry &registry);

namespace IREE {
namespace transform_dialect {
// Hook to register LLVMGPU transformations to the transform dialect.
class TransformDialectLLVMGPUExtensions
    : public transform::TransformDialectExtension<
          TransformDialectLLVMGPUExtensions> {
 public:
  TransformDialectLLVMGPUExtensions();
};
}  // namespace transform_dialect
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformDialectExtensions/TransformDialectLLVMGPUExtensionsOps.h.inc"

#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMGPUEXTENSIONS_H_
