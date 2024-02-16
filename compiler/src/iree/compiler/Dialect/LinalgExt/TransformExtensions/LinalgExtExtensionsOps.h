// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMOPS_LINALGEXTTRANSFORMOPS_H
#define IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMOPS_LINALGEXTTRANSFORMOPS_H

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::scf {
class ForOp;
class ForallOp;
} // namespace mlir::scf

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/TransformExtensions/LinalgExtExtensionsOps.h.inc"

namespace mlir::iree_compiler::IREE::LinalgExt {
class LinalgExtTransformOpsExtension
    : public transform::TransformDialectExtension<
          LinalgExtTransformOpsExtension> {
public:
  LinalgExtTransformOpsExtension();
  void init();
};
} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMOPS_LINALGEXTTRANSFORMOPS_H
