// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_ERRORCHECKINGTRACKINGLISTENER_H_
#define IREE_COMPILER_CODEGEN_COMMON_ERRORCHECKINGTRACKINGLISTENER_H_

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

namespace mlir::iree_compiler {

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
  void
  notifyPayloadReplacementNotFound(Operation *op, ValueRange values,
                                   DiagnosedSilenceableFailure &&diag) override;

  /// The error state of this listener. "Success" indicates that no error
  /// happened so far. Otherwise, the status contains the most recent error.
  DiagnosedSilenceableFailure status = DiagnosedSilenceableFailure::success();
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_ERRORCHECKINGTRACKINGLISTENER_H_
