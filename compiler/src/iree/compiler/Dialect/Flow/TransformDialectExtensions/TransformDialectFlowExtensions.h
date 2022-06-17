// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_FLOW_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTFLOWEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_FLOW_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTFLOWEXTENSIONS_H_

#include "mlir/Dialect/Transform/IR/TransformDialect.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/TransformDialectExtensions/TransformDialectFlowExtensionsOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace iree_compiler {

/// Registers Flow transformations that require IREE-specific information into
/// the transform dialect.
void registerTransformDialectFlowExtension(DialectRegistry &registry);

namespace IREE {
namespace transform_dialect {
// Hook to register Flow transformations to the transform dialect.
class TransformDialectFlowExtensions
    : public transform::TransformDialectExtension<
          TransformDialectFlowExtensions> {
 public:
  TransformDialectFlowExtensions();
};
}  // namespace transform_dialect
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_FLOW_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTFLOWEXTENSIONS_H_
