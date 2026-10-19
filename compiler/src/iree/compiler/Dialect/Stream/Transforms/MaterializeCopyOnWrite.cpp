// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-materialize-copy-on-write"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_MATERIALIZECOPYONWRITEPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Copy-on-write (🐄)
//===----------------------------------------------------------------------===//

// Materializes a copy for a mutated resource operand on |affinity|. A single
// SSA use does not prove ownership: the resource may borrow external storage.
// Copy elision checks that provenance before removing the clone.
static Value materializeOperandCOW(Location loc, OpOperand &operand,
                                   IREE::Stream::AffinityAttr affinity,
                                   OpBuilder &builder) {
  auto resourceType =
      dyn_cast<IREE::Stream::ResourceType>(operand.get().getType());
  if (!resourceType) {
    return nullptr;
  }

  auto sizeAwareType = cast<IREE::Util::SizeAwareTypeInterface>(resourceType);
  auto size = sizeAwareType.queryValueSize(loc, operand.get(), builder);
  return IREE::Stream::AsyncCloneOp::create(
      builder, loc, resourceType, operand.get(), size, size, affinity);
}

// Materializes a copy for each mutated operand on |tiedOp| as required.
// Returns true if any copy was required and materialized.
static bool materializeTiedOpCOW(IREE::Util::TiedOpInterface tiedOp) {
  bool didChange = false;

  // Any ops we materialize must have the same affinity as their consumer. This
  // ensures the copies we issue happen locally to the consumer.
  IREE::Stream::AffinityAttr affinity;
  if (auto affinityOp =
          dyn_cast<IREE::Stream::AffinityOpInterface>(tiedOp.getOperation())) {
    affinity = affinityOp.getAffinityAttr();
  }

  // Clones each operand that is tied to a result and it may be required.
  OpBuilder builder(tiedOp);
  auto tiedOperandIndices = tiedOp.getTiedResultOperandIndices();
  for (unsigned i = 0; i < tiedOperandIndices.size(); ++i) {
    int64_t operandIdx = tiedOperandIndices[i];
    if (operandIdx == IREE::Util::TiedOpInterface::kUntiedIndex) {
      continue;
    }
    auto &tiedOperand = tiedOp->getOpOperand(operandIdx);

    // If copy was required and materialized, we should forward it to all
    // operands that use the same value.
    if (auto clone = materializeOperandCOW(tiedOp.getLoc(), tiedOperand,
                                           affinity, builder)) {
      Value original = tiedOperand.get();
      tiedOperand.set(clone);
      didChange = true;

      // TODO(#11249): Support in-place collective operations.
      if (!isa<IREE::Stream::AsyncCollectiveOp>(tiedOp)) {
        for (auto &operand : tiedOp->getOpOperands()) {
          if (operand.get() == original) {
            operand.set(clone);
          }
        }
      }
    }
  }

  return didChange;
}

// Materializes copies on writes within |region|.
// Returns true if any copy was required and materialized.
static bool materializeRegionCOW(Region &region) {
  bool didChange = false;
  for (auto &block : region.getBlocks()) {
    for (auto &op : block) {
      if (!op.hasTrait<OpTrait::IREE::Stream::AsyncPhaseOp>()) {
        continue;
      }
      didChange =
          TypeSwitch<Operation *, bool>(&op)
              .Case<IREE::Stream::TensorImportOp, IREE::Stream::TensorExportOp,
                    IREE::Stream::AsyncFillOp, IREE::Stream::AsyncUpdateOp,
                    IREE::Stream::AsyncCopyOp,
                    // TODO(#11249): special case collectives for in-place.
                    // We don't want to clone the send buffer.
                    IREE::Stream::AsyncCollectiveOp,
                    IREE::Stream::AsyncDispatchOp, IREE::Stream::AsyncCallOp,
                    IREE::Stream::AsyncExecuteOp,
                    IREE::Stream::AsyncConcurrentOp>(
                  [&](auto op) { return materializeTiedOpCOW(op); })
              .Default(false) ||
          didChange;
    }
  }
  return didChange;
}

//===----------------------------------------------------------------------===//
// -iree-stream-materialize-copy-on-write
//===----------------------------------------------------------------------===//

// Inserts copies for tied resource operands that may be mutated. This may
// introduce copies that are not required for correctness; the
// --iree-stream-elide-async-copies pass removes them when it can prove that
// the source storage is owned or otherwise safe to reuse.
//
// Conceptually this work is performed in two phases: copy insertion and copy
// elision. This pass inserts copies at all mutation sites regardless of whether
// they are required, effectively disabling ties as a mechanism for in-place
// updates but ensuring correct execution semantics. Afterward a dataflow
// analysis pass is run to identify which copies can be elided based on use-def
// chains (including ones spanning the CFG). Though this process can lead to
// additional copies it is easier to ensure that each pass works independently
// and also makes it easy to disable copy elision to ferret out issues.
struct MaterializeCopyOnWritePass
    : IREE::Stream::impl::MaterializeCopyOnWritePassBase<
          MaterializeCopyOnWritePass> {
  void runOnOperation() override {
    bool didChange = false;
    getOperation()->walk([&](Region *region) {
      didChange = materializeRegionCOW(*region) || didChange;
    });
    // TODO(benvanik): run canonicalization patterns inline if anything changed.
    (void)didChange;
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
