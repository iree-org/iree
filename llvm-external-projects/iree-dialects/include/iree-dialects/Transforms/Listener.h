// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_SANDBOX_TRANSFORMS_LISTENER_H
#define IREE_LLVM_SANDBOX_TRANSFORMS_LISTENER_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {

using RewriteListener = RewriterBase::Listener;

//===----------------------------------------------------------------------===//
// ListenerList
//===----------------------------------------------------------------------===//

/// This class contains multiple listeners to which rewrite events can be sent.
class ListenerList : public RewriteListener {
public:
  /// Add a listener to the list.
  void addListener(RewriteListener *listener) { listeners.push_back(listener); }

  /// Send notification of an operation being inserted to all listeners.
  void notifyOperationInserted(Operation *op) override;
  /// Send notification of a block being created to all listeners.
  void notifyBlockCreated(Block *block) override;
  /// Send notification that an operation has been replaced to all listeners.
  void notifyOperationReplaced(Operation *op, ValueRange newValues) override;
  /// Send notification that an operation was modified in-place.
  void notifyOperationModified(Operation *op) override;
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
  PatternRewriterListener(MLIRContext *context) : PatternRewriter(context) {
    setListener(this);
  }

  /// When an operation is about to be replaced, send out an event to all
  /// attached listeners.
  void replaceOp(Operation *op, ValueRange newValues) override {
    ListenerList::notifyOperationReplaced(op, newValues);
    PatternRewriter::replaceOp(op, newValues);
  }

  void notifyOperationModified(Operation *op) override {
    ListenerList::notifyOperationModified(op);
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
