// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_SANDBOX_TRANSFORMS_LISTENER_H
#define IREE_LLVM_SANDBOX_TRANSFORMS_LISTENER_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// RewriteListener
//===----------------------------------------------------------------------===//

/// This class represents a listener that can be used to hook on to various
/// rewrite events in an `OpBuilder` or `PatternRewriter`. The class is notified
/// by when:
///
/// - an operation is removed
/// - an operation is inserted
/// - an operation is replaced
/// - a block is created
///
/// Listeners can be used to track IR mutations throughout pattern rewrites.
struct RewriteListener {
  virtual ~RewriteListener();

  /// These are the callback methods that subclasses can choose to implement if
  /// they would like to be notified about certain types of mutations.

  /// Notification handler for when an operation is modified in-place.
  virtual void finalizeRootUpdate(Operation *op) {}

  /// Notification handler for when an operation is inserted into the builder.
  /// op` is the operation that was inserted.
  virtual void notifyOperationInserted(Operation *op) {}

  /// Notification handler for when a block is created using the builder.
  /// `block` is the block that was created.
  virtual void notifyBlockCreated(Block *block) {}

  /// Notification handler for when the specified operation is about to be
  /// replaced with another set of operations. This is called before the uses of
  /// the operation have been replaced with the specific values.
  virtual void notifyRootReplaced(Operation *op, ValueRange newValues) {}

  /// Notification handler for when an the specified operation is about to be
  /// deleted. At this point, the operation has zero uses.
  virtual void notifyOperationRemoved(Operation *op) {}

  /// Notify the listener that a pattern failed to match the given operation,
  /// and provide a callback to populate a diagnostic with the reason why the
  /// failure occurred. This method allows for derived listeners to optionally
  /// hook into the reason why a rewrite failed, and display it to users.
  virtual LogicalResult
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) {
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// ListenerList
//===----------------------------------------------------------------------===//

/// This class contains multiple listeners to which rewrite events can be sent.
class ListenerList : public RewriteListener {
public:
  /// Add a listener to the list.
  void addListener(RewriteListener *listener) { listeners.push_back(listener); }

  /// Send notification of an operation being modified in-place to all
  /// listeners.
  void finalizeRootUpdate(Operation *op) override;
  /// Send notification of an operation being inserted to all listeners.
  void notifyOperationInserted(Operation *op) override;
  /// Send notification of a block being created to all listeners.
  void notifyBlockCreated(Block *block) override;
  /// Send notification that an operation has been replaced to all listeners.
  void notifyRootReplaced(Operation *op, ValueRange newValues) override;
  /// Send notification that an operation is about to be deleted to all
  /// listeners.
  void notifyOperationRemoved(Operation *op) override;
  /// Notify all listeners that a pattern match failed.
  LogicalResult
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override;

private:
  /// The list of listeners to send events to.
  SmallVector<RewriteListener *, 1> listeners;
};

//===----------------------------------------------------------------------===//
// PatternRewriterListener
//===----------------------------------------------------------------------===//

/// This class implements a pattern rewriter with a rewrite listener. Rewrite
/// events are forwarded to the provided rewrite listener.
class PatternRewriterListener : public PatternRewriter, public ListenerList {
public:
  PatternRewriterListener(MLIRContext *context) : PatternRewriter(context) {}

  void finalizeRootUpdate(Operation *op) override {
    ListenerList::finalizeRootUpdate(op);
  }

  /// When an operation is about to be replaced, send out an event to all
  /// attached listeners.
  void replaceOp(Operation *op, ValueRange newValues) override {
    ListenerList::notifyRootReplaced(op, newValues);
    PatternRewriter::replaceOp(op, newValues);
  }

  void notifyOperationInserted(Operation *op) override {
    ListenerList::notifyOperationInserted(op);
  }
  void notifyBlockCreated(Block *block) override {
    ListenerList::notifyBlockCreated(block);
  }
  void notifyOperationRemoved(Operation *op) override {
    ListenerList::notifyOperationRemoved(op);
  }
};

} // namespace mlir

#endif // IREE_LLVM_SANDBOX_TRANSFORMS_LISTENER_H
