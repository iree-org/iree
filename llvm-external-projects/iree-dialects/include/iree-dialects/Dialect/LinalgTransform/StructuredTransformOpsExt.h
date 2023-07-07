// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALG_TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H
#define IREE_DIALECTS_DIALECT_LINALG_TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H

#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace linalg {
class LinalgOp;
} // namespace linalg
namespace pdl {
class FoOperationTyperOp;
} // namespace pdl
namespace scf {
class ForOp;
} // namespace scf
namespace transform_ext {
class MatchCallbackOp;
} // namespace transform_ext

/// Matches a C++ callback previously registered under `callbackName` and
/// taking arguments `args`.
/// Unpacks a number of handles `N` (asserts there are exactly `N` matched
/// ops but this could be relaxed if needed). Returns the tuple of handles.
template <int N, typename... MatchingArgs>
auto unpackRegisteredMatchCallback(ImplicitLocOpBuilder &b,
                                   StringRef callbackName,
                                   MatchingArgs... args) {
  SmallVector<Type> matchedTypes(N, transform::AnyOpType::get(b.getContext()));
  auto matchOp = b.create<transform_ext::MatchCallbackOp>(
      matchedTypes, callbackName, std::forward<decltype(args)>(args)...);
  assert(matchOp->getNumResults() == N && "Unexpected number of results");
  std::array<Value, N> a;
  for (int64_t i = 0; i < N; ++i)
    a[i] = matchOp->getResult(i);
  return std::tuple_cat(a);
}

/// A tracking listener for tensor IR that checks for payload replacement
/// errors.
class ErrorCheckingTrackingListener : public transform::TrackingListener {
public:
  using transform::TrackingListener::TrackingListener;

  ~ErrorCheckingTrackingListener() override {
    assert(status.succeeded() && "must check listener error state");
  }

  /// Return "true" if this tracking listener had a failure.
  bool failed() const { return !status.succeeded(); }

  /// Check and return the current error state of this listener. In case of a
  /// failure state, only the most recent error is returned. Afterwards, resets
  /// the error state.
  DiagnosedSilenceableFailure checkAndResetError() {
    DiagnosedSilenceableFailure result(std::move(status));
    status = DiagnosedSilenceableFailure::success();
    return result;
  }

private:
  void notifyPayloadReplacementNotFound(Operation *op,
                                        ValueRange values) override;

  /// The error state of this listener. "Success" indicates that no error
  /// happened so far. Otherwise, the status contains the most recent error.
  DiagnosedSilenceableFailure status = DiagnosedSilenceableFailure::success();
};

} // namespace mlir

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h.inc"

namespace mlir {
namespace transform_ext {
class StructuredTransformOpsExtension
    : public mlir::transform::TransformDialectExtension<
          StructuredTransformOpsExtension> {
public:
  StructuredTransformOpsExtension();
};

} // namespace transform_ext
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALG_TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H
