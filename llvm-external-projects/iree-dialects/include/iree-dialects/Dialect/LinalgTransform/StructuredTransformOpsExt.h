// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALG_TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H
#define IREE_DIALECTS_DIALECT_LINALG_TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace linalg {
class LinalgOp;
} // namespace linalg
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h.inc"

namespace transform_ext {
class StructuredTransformOpsExtension
    : public mlir::transform::TransformDialectExtension<
          StructuredTransformOpsExtension> {
public:
  StructuredTransformOpsExtension();
};
} // namespace transform_ext

#endif // IREE_DIALECTS_DIALECT_LINALG_TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H
