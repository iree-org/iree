//===-- TrackingListener.h - Common listener for tracking passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_TRACKINGLISTENER_H
#define IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_TRACKINGLISTENER_H

#include "Transforms/Listener.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformOpInterface.h"

namespace mlir {
namespace linalg {
/// A tracking listener using to perform CSE and canonicalization passes while
/// tracking certain linalg operation handles live in a linalg transform
/// interpreter.
class TrackingListener : public RewriteListener,
                         public transform::TransformState::Extension {
 public:
  TrackingListener(transform::TransformState &state);
  TrackingListener(TrackingListener &&other)
      : transform::TransformState::Extension(
            std::forward<transform::TransformState::Extension>(other)),
        trackedOperationKeys(std::move(other.trackedOperationKeys)),
        hadErrors(other.hadErrors) {
#ifndef NDEBUG
    errorStateChecked = other.errorStateChecked;
    other.errorStateChecked = true;
#endif
  }
  ~TrackingListener() {
#ifndef NDEBUG
    assert(errorStateChecked && "must check listener error state");
#endif  // NDEBUG
  }

  /// When a tracked linalg operation is replaced, try to find a single linalg
  /// op responsible for the replacement values and substitute the handle of the
  /// replaced op for this op.
  void notifyOperationReplaced(Operation *op, ValueRange newValues) override;

  /// When a tracked operation is removed (due to CSE or canonicalization), then
  /// any further transformations on the op are redundant. Remove it from the
  /// tracked operation list.
  void notifyOperationRemoved(Operation *op) override;

  void notifySetPayload(Value handle,
                        ArrayRef<Operation *> operations) override;
  void notifyRemovePayload(Value handle,
                           ArrayRef<Operation *> operations) override;

  /// Emits an error pointing at the given operation. Use this instead of
  /// directly emitting an error on the operation to set the listener into the
  /// error state and thus communicate with its user.
  InFlightDiagnostic emitError(Operation *op, const llvm::Twine &message = {});

  /// Converts the current error state into LogicalResult and clears it.
  LogicalResult checkErrorState() {
    LogicalResult result = failure(hadErrors);
#ifndef NDEBUG
    errorStateChecked = true;
#endif  // NDEBUG
    return result;
  }

 private:
  /// A map from a tracked operation (LinalgOp cannot be used as a key) to its
  /// key in the map.
  DenseMap<Operation *, Value> trackedOperationKeys;
  bool hadErrors = false;
#ifndef NDEBUG
  bool errorStateChecked = false;
#endif  // NDEBUG
};
}  // namespace linalg
}  // namespace mlir

#endif  // IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_TRACKINGLISTENER_H
