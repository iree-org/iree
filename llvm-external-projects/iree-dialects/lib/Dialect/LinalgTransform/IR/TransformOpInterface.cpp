//===-- TransformOpInterface.cpp - Interface for transform ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgTransform/TransformOpInterface.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// TransformState
//===----------------------------------------------------------------------===//

constexpr const Value transform::TransformState::kTopLevelValue;

transform::TransformState::TransformState(Operation *root) {
  operations[kTopLevelValue].push_back(root);
}

Operation *transform::TransformState::getTopLevel() const {
  return operations.lookup(kTopLevelValue).front();
}

ArrayRef<Operation *> transform::TransformState::getPayloadOps(
    Value value) const {
  auto iter = operations.find(value);
  assert(iter != operations.end() && "unknown handle");
  return iter->getSecond();
}

LogicalResult transform::TransformState::setPayloadOps(
    Value value, ArrayRef<Operation *> targets) {
  assert(value != kTopLevelValue &&
         "attempting to reset the transformation root");

  if (value.use_empty()) return success();

  SmallVector<Operation *> storedTargets(targets.begin(), targets.end());
  bool inserted = operations.insert({value, std::move(storedTargets)}).second;
  assert(inserted && "value is already associated with another list");
  (void)inserted;

  const SmallVector<Operation *> &currentOperationList =
      operations.lookup(value);
  llvm::SmallPtrSet<Operation *, 4> currentOperationSet(
      currentOperationList.begin(), currentOperationList.end());
  for (const auto &kvp : operations) {
    if (kvp.getFirst() == value) continue;
    for (Operation *trackedOp : kvp.getSecond()) {
      if (currentOperationSet.contains(trackedOp)) {
        InFlightDiagnostic diag = trackedOp->emitError()
                                  << "operation tracked by two handles";
        diag.attachNote(value.getLoc()) << "handle";
        diag.attachNote(kvp.getFirst().getLoc()) << "handle";
        return diag;
      }
    }
  }

  for (const auto &keyedExtension : extensions)
    keyedExtension.getSecond()->sendNotifySetPayload(value, targets);

  return success();
}

void transform::TransformState::removePayloadOps(Value value) {
  auto it = operations.find(value);
  if (it == operations.end()) return;

  for (const auto &keyedExtension : extensions)
    keyedExtension.getSecond()->sendNotifyRemovePayload(value, it->getSecond());

  operations.erase(it);
}

void transform::TransformState::updatePayloadOps(
    Value value, function_ref<Operation *(Operation *)> callback) {
  auto it = operations.find(value);
  assert(it != operations.end() && "unknown handle");
  SmallVector<Operation *> &association = it->getSecond();
  SmallVector<Operation *> updated;
  updated.reserve(association.size());

  for (Operation *op : association)
    if (Operation *updatedOp = callback(op)) updated.push_back(updatedOp);

  for (const auto &keyedExtension : extensions)
    keyedExtension.getSecond()->sendNotifyUpdatePayload(value, association,
                                                        updated);

  std::swap(association, updated);
}

LogicalResult transform::TransformState::applyTransform(
    TransformOpInterface transform) {
  transform::TransformResults results(transform->getNumResults());
  if (failed(transform.apply(results, *this))) return failure();

  for (Value target : transform->getOperands()) removePayloadOps(target);

  for (auto en : llvm::enumerate(transform->getResults()))
    if (failed(setPayloadOps(en.value(), results.get(en.index()))))
      return failure();

  return success();
}

// Out-of-line definition to ensure vtable and metadata are emitted to a single
// .o file.
transform::TransformState::Extension::~Extension() {}

//===----------------------------------------------------------------------===//
// TransformResults
//===----------------------------------------------------------------------===//

transform::TransformResults::TransformResults(unsigned numSegments) {
  segments.resize(numSegments,
                  ArrayRef<Operation *>(nullptr, static_cast<size_t>(0)));
}

void transform::TransformResults::set(OpResult value,
                                      ArrayRef<Operation *> ops) {
  unsigned position = value.getResultNumber();
  assert(position < segments.size() &&
         "setting results for a non-existent handle");
  assert(segments[position].data() == nullptr && "results already set");
  unsigned start = operations.size();
  llvm::append_range(operations, ops);
  segments[position] = makeArrayRef(operations).drop_front(start);
}

ArrayRef<Operation *> transform::TransformResults::get(
    unsigned position) const {
  assert(position < segments.size() &&
         "querying results for a non-existent handle");
  assert(segments[position].data() != nullptr && "querying unset results");
  return segments[position];
}

//===----------------------------------------------------------------------===//
// Generated interface implementation.
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgTransform/TransformOpInterface.cpp.inc"
