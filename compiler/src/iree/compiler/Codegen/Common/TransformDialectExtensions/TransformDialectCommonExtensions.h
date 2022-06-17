// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTCOMMONEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTCOMMONEXTENSIONS_H_

#include "mlir/Dialect/Transform/IR/TransformDialect.h"

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformDialectExtensions/TransformDialectCommonExtensionsOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace iree_compiler {

/// Registers common transformations that require IREE-specific information
/// into the transform dialect.
void registerTransformDialectCommonExtension(DialectRegistry &registry);

namespace IREE {
namespace transform_dialect {
// Hook to register common transformations to the transform dialect.
class TransformDialectCommonExtensions
    : public transform::TransformDialectExtension<
          TransformDialectCommonExtensions> {
 public:
  TransformDialectCommonExtensions();
};
}  // namespace transform_dialect
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTCOMMONEXTENSIONS_H_
