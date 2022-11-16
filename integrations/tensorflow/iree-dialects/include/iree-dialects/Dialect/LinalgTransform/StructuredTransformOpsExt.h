// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALG_TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H
#define IREE_DIALECTS_DIALECT_LINALG_TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H

#include "iree-dialects/Transforms/Listener.h"
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

class TrackingListener : public RewriteListener,
                         public transform::TransformState::Extension {
public:
  explicit TrackingListener(transform::TransformState &state)
      : transform::TransformState::Extension(state) {}

  ~TrackingListener() override {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(errorStateChecked && "must check listener error state");
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  }

  void notifyRootReplaced(Operation *op, ValueRange newValues) override;

  void notifyOperationRemoved(Operation *op) override;

  LogicalResult checkErrorState() const {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    errorStateChecked = true;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    return failure(hadErrors);
  }

  /// Remove the mappings between the given operation and any handle that may be
  /// associated with it in the transform op.
  void removeMappings(Operation *op);

private:
  InFlightDiagnostic emitError(Operation *op, const llvm::Twine &message = {}) {
    mayFail(failure());
    return op->emitError(message);
  }

  void mayFail(LogicalResult result) {
    hadErrors |= result.failed();
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    errorStateChecked = false;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
  }

  bool hadErrors = false;

#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  mutable bool errorStateChecked = false;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
};

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
