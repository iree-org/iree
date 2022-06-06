// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformOpTraits.h"
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
#include "iree/compiler/Codegen/TransformDialectExtensions/TransformDialectExtensionsOps.h.inc"

namespace mlir {
namespace iree_compiler {

/// Registers transformations that require IREE-specific information into the
/// LinalgTransform dialect.
void registerLinalgTransformDialectExtension(DialectRegistry &registry);

namespace IREE {
namespace transform_dialect {
// Hook to re
class TransformDialectExtensions
    : public transform::TransformDialectExtension<TransformDialectExtensions> {
 public:
  TransformDialectExtensions();
};
}  // namespace transform_dialect
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_H_
