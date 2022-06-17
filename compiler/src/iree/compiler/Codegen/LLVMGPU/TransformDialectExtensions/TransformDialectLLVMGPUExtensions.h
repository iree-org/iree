// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMGPUEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMGPUEXTENSIONS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformDialectExtensions/TransformDialectLLVMGPUExtensionsOps.h.inc"

namespace mlir {
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

#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMGPUEXTENSIONS_H_
