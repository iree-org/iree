//===- Listener.cpp - Transformation Listener -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transforms/Listener.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// RewriteListener
//===----------------------------------------------------------------------===//

RewriteListener::~RewriteListener() = default;

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

void ListenerList::notifyOperationRemoved(Operation *op) {
  for (RewriteListener *listener : listeners)
    listener->notifyOperationRemoved(op);
}

void ListenerList::notifyMatchFailure(
    Operation *op, function_ref<void(Diagnostic &)> reasonCallback) {
  for (RewriteListener *listener : listeners)
    listener->notifyMatchFailure(op, reasonCallback);
}

} // namespace mlir
