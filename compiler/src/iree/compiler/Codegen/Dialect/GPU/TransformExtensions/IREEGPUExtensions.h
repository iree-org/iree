// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS_H_

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

namespace mlir {

class DialectRegistry;

namespace func {
class FuncOp;
} // namespace func

namespace transform {
// Types needed for builders.
class TransformTypeInterface;
} // namespace transform

} // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/TransformExtensions/IREEGPUExtensionsOps.h.inc"

namespace mlir::iree_compiler {

/// Registers transformations for the IREE GPU dialect.
void registerTransformDialectIREEGPUExtension(DialectRegistry &registry);

namespace IREE::transform_dialect {
/// Hook to register common transformations to the transform dialect.
class IREEGPUExtensions
    : public transform::TransformDialectExtension<IREEGPUExtensions> {
public:
  IREEGPUExtensions();
};
} // namespace IREE::transform_dialect

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS_H_
