// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Transforms/Listener.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// ListenerList
//===----------------------------------------------------------------------===//

void ListenerList::notifyOperationInserted(Operation *op) {
  for (RewriteListener *listener : listeners)
    listener->notifyOperationInserted(op);
}

void ListenerList::notifyBlockCreated(Block *block) {
  for (RewriteListener *listener : listeners)
    listener->notifyBlockCreated(block);
}

void ListenerList::notifyOperationReplaced(Operation *op,
                                           ValueRange newValues) {
  for (RewriteListener *listener : listeners)
    listener->notifyOperationReplaced(op, newValues);
}

void ListenerList::notifyOperationModified(Operation *op) {
  for (RewriteListener *listener : listeners)
    listener->notifyOperationModified(op);
}

void ListenerList::notifyOperationRemoved(Operation *op) {
  for (RewriteListener *listener : listeners)
    listener->notifyOperationRemoved(op);
}

LogicalResult ListenerList::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  bool failed = false;
  for (RewriteListener *listener : listeners)
    failed |= listener->notifyMatchFailure(loc, reasonCallback).failed();
  return failure(failed);
}

} // namespace mlir
