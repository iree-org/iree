//===-- TrackingListener.cpp - Common listener for tracking passes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgTransform/TrackingListener.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "tracking-listener"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir {
namespace linalg {

/// Find the linalg op that defines all values in range, potentially
/// transitively through tensor casts.
static LinalgOp findSingleLinalgOpDefiningAll(ValueRange range) {
  LinalgOp sourceOp = nullptr;
  for (Value value : range) {
    // See through tensor casts.
    //
    // TODO: we may need some generalization (interfaces?) of this for other
    // operations, especially multi-operand ones to understand which of their
    // operands may be coming from a Linalg op. Or a completely different
    // mechanism of tracking op replacement at creation, or even different
    // patterns that identify the "main" result of a transformation.
    while (auto castOp = value.getDefiningOp<tensor::CastOp>())
      value = castOp.source();

    if (auto currentSourceOp = value.getDefiningOp<LinalgOp>()) {
      if (!sourceOp || sourceOp == currentSourceOp) {
        sourceOp = currentSourceOp;
        continue;
      }

      LLVM_DEBUG(
          DBGS() << "different source linalg ops for replacing one op: \n"
                 << sourceOp << "\n"
                 << currentSourceOp << "\n");
    }
    LLVM_DEBUG(DBGS() << "replacing linalg op with unknown non-linalg op:\n"
                      << *value.getDefiningOp() << "\n");
    return nullptr;
  }
  return sourceOp;
}

/// Find the scf "for" op that defines all values in the range.
static scf::ForOp findSingleForOpDefiningAll(ValueRange range) {
  scf::ForOp forOp = nullptr;
  for (Value value : range) {
    if (auto currentSourceOp = value.getDefiningOp<scf::ForOp>()) {
      if (!forOp || forOp == currentSourceOp) {
        forOp = currentSourceOp;
        continue;
      }
      LLVM_DEBUG(
          DBGS() << "different source scf.for ops when replacing one op\n");
    }

    LLVM_DEBUG(
        DBGS()
        << "could not find a source scf.for when replacing another scf.for\n");
    return nullptr;
  }
  return forOp;
}

// Find a single op that defines all values in the range, optionally
// transitively through other operations in an op-specific way.
static Operation *findSingleDefiningOp(Operation *replacedOp,
                                       ValueRange range) {
  return llvm::TypeSwitch<Operation *, Operation *>(replacedOp)
      .Case<LinalgOp>([&](LinalgOp) -> Operation * {
        return findSingleLinalgOpDefiningAll(range);
      })
      .Case<scf::ForOp>([&](scf::ForOp) -> Operation * {
        return findSingleForOpDefiningAll(range);
      })
      .Default([](Operation *) -> Operation * { return nullptr; });
}

TrackingListener::TrackingListener(transform::TransformState &state)
    : transform::TransformState::Extension(state) {
  for (const auto &pair : getMapping())
    for (Operation *op : pair.second)
      trackedOperationKeys.try_emplace(op, pair.first);
}

void TrackingListener::notifyOperationReplaced(Operation *op,
                                               ValueRange newValues) {
  // Don't attempt to track in error state.
  if (hadErrors) return;

  // Exit early if the op is not tracked.
  auto keyIt = trackedOperationKeys.find(op);
  if (keyIt == trackedOperationKeys.end()) return;
  Value key = keyIt->second;

  Operation *replacement = findSingleDefiningOp(op, newValues);
  if (!replacement) {
    emitError(op) << "could not find replacement for tracked op";
    return;
  }

  LLVM_DEBUG(DBGS() << "replacing tracked " << *op << " with " << *replacement
                    << " for " << key << "\n");
  updatePayloadOps(key, [op, replacement](Operation *tracked) {
    return tracked == op ? replacement : tracked;
  });

  // Update the backwards map. The replacement operation must not be already
  // associated with another key as that would break the bidirectional mapping
  // invariant. Note that operations are pointer-like so we must ensure the
  // absence of accidental reuse of the pointer address with some deleted
  // operation that stayed in this mapping.
  trackedOperationKeys.erase(op);
  bool replaced = trackedOperationKeys.try_emplace(replacement, key).second;
  if (!replaced) {
    InFlightDiagnostic diag =
        emitError(replacement)
        << "replacement operation is already associated with another key";
    diag.attachNote(op->getLoc()) << "replacing this operation";
    diag.attachNote(trackedOperationKeys.lookup(replacement).getLoc())
        << "old key";
    diag.attachNote(key.getLoc()) << "new key";
    return;
  }
}

void TrackingListener::notifyOperationRemoved(Operation *op) {
  // Don't attempt to track in error state.
  if (hadErrors) return;

  auto keyIt = trackedOperationKeys.find(op);
  if (keyIt == trackedOperationKeys.end()) return;
  Value key = keyIt->second;

  LLVM_DEBUG(DBGS() << "removing tracked " << *op << " for " << key << "\n");

  // If a tracked operation is CSE'd, then any further transformations are
  // redundant. Just remove it.
  trackedOperationKeys.erase(op);
  updatePayloadOps(key, [op](Operation *tracked) {
    return tracked != op ? tracked : nullptr;
  });
}

void TrackingListener::notifySetPayload(Value handle,
                                        ArrayRef<Operation *> operations) {
  for (Operation *op : operations) {
    assert(trackedOperationKeys.lookup(op) == Value() &&
           "payload op already associated with another key");
    trackedOperationKeys[op] = handle;
  }
}

void TrackingListener::notifyRemovePayload(Value handle,
                                           ArrayRef<Operation *> operations) {
  for (Operation *op : operations) trackedOperationKeys.erase(op);
}

InFlightDiagnostic TrackingListener::emitError(Operation *op,
                                               const llvm::Twine &message) {
  hadErrors = true;
#ifndef NDEBUG
  errorStateChecked = false;
#endif  // NDEBUG
  return op->emitError(message);
}
}  // namespace linalg
}  // namespace mlir
