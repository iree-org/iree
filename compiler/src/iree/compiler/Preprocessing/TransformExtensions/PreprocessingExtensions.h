// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_PREPROCESSING_TRANSFORMEXTENSIONS_PREPROCESSINGEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_PREPROCESSING_TRANSFORMEXTENSIONS_PREPROCESSINGEXTENSIONS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/MatchInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.h.inc"

namespace mlir::iree_compiler {

/// Registers Preprocessing transformations that require IREE-specific
/// information into the transform dialect.
void registerTransformDialectPreprocessingExtension(DialectRegistry &registry);

namespace IREE::transform_dialect {
// Hook to register Preprocessing transformations to the transform dialect.
class PreprocessingExtensions
    : public transform::TransformDialectExtension<PreprocessingExtensions> {
public:
  PreprocessingExtensions();
};
} // namespace IREE::transform_dialect

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_PREPROCESSING_TRANSFORMEXTENSIONS_PREPROCESSINGEXTENSIONS_H_
