// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-materialize-copy-on-write"

namespace mlir::iree_compiler::IREE::Stream {
namespace {

//===----------------------------------------------------------------------===//
// Copy-on-write (🐄)
//===----------------------------------------------------------------------===//

// Returns true if the given |operand| value does not need a copy on write.
// This is a conservative check and will return false ("not safe to elide") in
// many cases that otherwise don't need a copy. The
// --iree-stream-elide-async-copies pass will do a whole-program analysis and
// remove the copies we insert here when possible.
static bool isSafeToElideCOW(Value operand, IREE::Stream::ResourceType type) {
  // Can't do anything with block args without analysis - we don't know if the
  // value they carry is the last user (move semantics).
  if (llvm::isa<BlockArgument>(operand))
    return false;

  // If our value is a constant then we need to ensure that we aren't
  // tied to a constant operand. If we are we need to clone to a
  // non-constant value. We could make this work in cases where constants are
  // being initialized, however those are best modeled as transfer operations
  // where no mutations will occur on the constant transfer target.
  if (type.getLifetime() == IREE::Stream::Lifetime::Constant)
    return false;

  // If there's more than one user we can't make a local decision. It's
  // expensive to query relative operation order within a block and within a
  // region the lifetime of values may vary - all things we can't tell here.
  Operation *firstUser = nullptr;
  for (Operation *user : operand.getUsers()) {
    if (firstUser == nullptr)
      firstUser = user;
    else if (firstUser != user)
      return false;
  }

  // We are the only user and the value is contained entirely within the
  // current region. We by construction know we do not need to worry.
  return true;
}

// Materializes a copy for a mutated |operand| on |affinity| if required.
// If it's determined that eliding the copy is safe it will be omitted.
// Returns a clone operation result if the copy was required and materialized,
// and nullptr otherwise.
static Value materializeOperandCOW(Location loc, OpOperand &operand,
                                   IREE::Stream::AffinityAttr affinity,
                                   OpBuilder &builder) {
  // If we can safely elide the copy early we do so here to avoid adding too
  // much IR. Anything that requires wider analysis (CFG, across functions, etc)
  // has to wait until a subsequent pass.
  auto resourceType =
      dyn_cast<IREE::Stream::ResourceType>(operand.get().getType());
  if (!resourceType)
    return nullptr;
  if (isSafeToElideCOW(operand.get(), resourceType))
    return nullptr;

  // Materialize a clone operation just for the operand provided.
  auto sizeAwareType =
      llvm::cast<IREE::Util::SizeAwareTypeInterface>(resourceType);
  auto size = sizeAwareType.queryValueSize(loc, operand.get(), builder);
  return builder.create<IREE::Stream::AsyncCloneOp>(
      loc, resourceType, operand.get(), size, size, affinity);
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
    affinity = affinityOp.getAffinity();
  }

  // Clones each operand that is tied to a result and it may be required.
  OpBuilder builder(tiedOp);
  auto tiedOperandIndices = tiedOp.getTiedResultOperandIndices();
  for (unsigned i = 0; i < tiedOperandIndices.size(); ++i) {
    int64_t operandIdx = tiedOperandIndices[i];
    if (operandIdx == IREE::Util::TiedOpInterface::kUntiedIndex)
      continue;
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
          if (operand.get() == original)
            operand.set(clone);
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
      if (!op.hasTrait<OpTrait::IREE::Stream::AsyncPhaseOp>())
        continue;
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

// Applies a relatively simple heuristic to insert copies where they _may_ be
// required. This may introduce copies that are not required for the sake of
// ensuring correctness. Intended to be paired with
// -iree-stream-elide-async-copies.
//
// Conceptually this work is performed in two phases: copy insertion and copy
// elision. This pass inserts copies at all mutation sites regardless of whether
// they are required, effectively disabling ties as a mechanism for in-place
// updates but ensuring correct execution semantics. Afterward a dataflow
// analysis pass is run to identify which copies can be elided based on use-def
// chains (including ones spanning the CFG). Though this process can lead to
// additional copies it is easier to ensure that each pass works independently
// and also makes it easy to disable copy elision to ferret out issues.
class MaterializeCopyOnWritePass
    : public MaterializeCopyOnWriteBase<MaterializeCopyOnWritePass> {
public:
  MaterializeCopyOnWritePass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

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

std::unique_ptr<OperationPass<>> createMaterializeCopyOnWritePass() {
  return std::make_unique<MaterializeCopyOnWritePass>();
}

} // namespace mlir::iree_compiler::IREE::Stream
