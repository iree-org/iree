// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMOPS_LINALGEXTTRANSFORMOPS_H
#define IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMOPS_LINALGEXTTRANSFORMOPS_H

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace scf {
class ForOp;
class ForallOp;
} // namespace scf
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {
class AttentionOp;
} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/TransformExtensions/LinalgExtExtensionsOps.h.inc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {
class LinalgExtTransformOpsExtension
    : public transform::TransformDialectExtension<
          LinalgExtTransformOpsExtension> {
public:
  LinalgExtTransformOpsExtension();
  void init();
};
} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMOPS_LINALGEXTTRANSFORMOPS_H
