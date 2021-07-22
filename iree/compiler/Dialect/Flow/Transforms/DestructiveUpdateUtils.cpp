// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- DestructiveUpdateUtilss.cpp - Utils to rewrite destructive updates--===//
//
// Implementation to rewrite Linalg on tensors destructive updates into updates
// through memory.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-linalg-rewrite-destructive-updates"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Detect the pattern:
// %d0 = for  (iter_args %e0 = %0)
//   ...
//   %dk = for ((iter_args %ek = %e{k-1}))
//     ...
//       %dn = destructive-update-op (%en)
//       yield %dn
//     ...
//     yield %dk
//   yield %dk
struct SpecialTerminatorOpCapture {
  Value initValue;
  // For now, must be scf::ForOps.
  SmallVector<Operation *, 4> loops;
  // For now, must be a SubTensorInsertOp.
  Operation *rootDestructiveUpdate;
  bool readOnly = false;
  bool writeOnly = false;
};

// TODO(nicolasvasilache): Use some interface instead of op names directly.
static bool hasDestructiveUpdateSubTensorUses(
    BlockArgument arg, SpecialTerminatorOpCapture &capture) {
  SmallVector<Operation *> reads;
  SmallVector<tensor::InsertSliceOp> writes;
  for (OpOperand &u : arg.getUses()) {
    if (auto subTensorInsertOp =
            dyn_cast<tensor::InsertSliceOp>(u.getOwner())) {
      writes.push_back(subTensorInsertOp);
    } else {
      reads.push_back(u.getOwner());
    }
  }
  // For now, only allow exactly a single SubTensorInsertOp that must be
  // dominated by all SubTensorOp.
  if (writes.size() != 1) return false;
  // Small local dominance computation.
  DominanceInfo domInfo(writes.front()->getParentOp());
  for (auto read : reads) {
    LLVM_DEBUG(llvm::dbgs() << "read: " << *read << "\n");
    if (!domInfo.properlyDominates(read, writes.front())) {
      LLVM_DEBUG(llvm::dbgs() << "non-destructive use-def: " << *read
                              << " does not properly dominate "
                              << *(writes.front().getOperation()) << "\n");
      return false;
    }
  }

  capture.readOnly = writes.empty();
  capture.writeOnly = reads.empty();
  capture.rootDestructiveUpdate = writes.front();
  LLVM_DEBUG(llvm::dbgs() << "readOnly: " << capture.readOnly
                          << " writeOnly: " << capture.writeOnly << "\n");
  return true;
}

// Determine whether `tensor` is produced by a destructive update of another
// tensor. When successful, fill a SpecialTerminatorOpCapture struct that
// captures the relevant (distributed) pieces of IR that for the destructive
// update pattern. Iteratively traverse an (imperfectly nested) loop nest such
// as:
//
// ```
// %d0 = for  (iter_args %e0 = %0)
//   ...
//   %dk = for ((iter_args %ek = %e{k-1}))
//     ...
//     %dn = destructive-update-op (%en)
//     yield %dn
//     ...
//   yield %dk
// ```
//
// to determine whether `d0` is produced by a scf::ForOp with destructive
// update semantics.
//
// Return the value into which the destructive update occurs.
// Return nullptr if `tensor` is not a destructive update of some other tensor
// value.
static Value isADestructiveUpdatePattern(Value tensor,
                                         SpecialTerminatorOpCapture &capture) {
  Value returnValue;
  while (auto scfForOp = dyn_cast_or_null<scf::ForOp>(tensor.getDefiningOp())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Step destructive update pattern: " << scfForOp << "\n");
    // Capture the loop.
    capture.loops.push_back(scfForOp);
    // Analyze the iterArg at the proper position.
    unsigned idx = tensor.cast<OpResult>().getResultNumber();
    BlockArgument regionArg = *(scfForOp.getRegionIterArgs().begin() + idx);
    // Set return value if not yet set.
    if (!returnValue) returnValue = *(scfForOp.getIterOperands().begin() + idx);

    // Case 1: zero use -> no destructive update.
    if (regionArg.use_empty()) return nullptr;

    // Case 2: multiple uses from an scf::ForOp then this must be used only by
    // SubTensorOp / SubTensorInsertOp with proper dominance.
    if (!regionArg.hasOneUse()) {
      if (!hasDestructiveUpdateSubTensorUses(regionArg, capture))
        return nullptr;
      return returnValue;
    }

    assert(regionArg.hasOneUse());
    LLVM_DEBUG(llvm::dbgs() << "one use analysis: " << regionArg << "\n");
    OpOperand *operand = regionArg.getUses().begin().getOperand();
    auto innerForOp = dyn_cast<scf::ForOp>(operand->getOwner());
    // Case 3a: Single use which is not an scf::ForOp, it may still be a
    // single SubTensor / SubTensorInsertOp.
    if (!innerForOp) {
      if (!hasDestructiveUpdateSubTensorUses(regionArg, capture))
        return nullptr;
      return returnValue;
    }

    // Case 3b: Single use which is an scf::ForOp: `innerIterArgIdx` is the
    // candidate result and iterArg number.
    unsigned innerIterArgIdx =
        operand->getOperandNumber() - innerForOp.getNumControlOperands();
    Value innerForOpResultTensor = innerForOp.getResult(innerIterArgIdx);
    Value yieldValue =
        scfForOp.region().front().getTerminator()->getOperand(idx);

    // Check that the return position of dk and the yield position of dk
    // agree (in the loop structure below). This avoids ping-pong effects
    // between operands, yields and results.
    //
    // %d0 = for  (iter_args %e0 = %0)
    //   ...
    //   %dk = for ((iter_args %ek = %e{k-1}))
    //     ...
    //     %dn = destructive-update-op (%en)
    //     yield %dn
    //     ...
    //   yield %dk
    LLVM_DEBUG(llvm::dbgs()
               << "innerForOpResultTensor: " << innerForOpResultTensor << "\n"
               << "yieldValue: " << yieldValue << "\n"
               << "step in: " << (innerForOpResultTensor == yieldValue)
               << "\n");
    if (innerForOpResultTensor != yieldValue) return nullptr;

    // Prepare for the next level with the innerForOp's result at position
    // `innerIterArgIdx`.
    tensor = innerForOp.getResult(innerIterArgIdx);
    LLVM_DEBUG(llvm::dbgs() << "next tensor: " << tensor << "\n");
  }
  return nullptr;
}

/// Convert `subtensor %t [offsets][sizes][strides] -> %st` to a
/// flow.dispatch.tensor.load.
static LogicalResult propagateSubTensorOp(OpBuilder &b,
                                          tensor::ExtractSliceOp op) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  auto loadOp = op.source().getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
  if (!loadOp) {
    BlockArgument val = op.source().dyn_cast<BlockArgument>();
    while (val) {
      auto forOp = dyn_cast<scf::ForOp>(val.getOwner()->getParentOp());
      // val is a block argument but not to an scf::ForOp -> bail.
      if (!forOp) return failure();
      unsigned idx = val.getArgNumber() - 1;  // accounting for IV arg.
      Value iterOperand = *(forOp.getIterOperands().begin() + idx);
      loadOp = iterOperand.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
      val = iterOperand.dyn_cast<BlockArgument>();
    }
  }
  if (!loadOp) return failure();

  Value loaded = b.create<IREE::Flow::DispatchTensorLoadOp>(
      op.getLoc(), op.getResult().getType(), loadOp.source(), op.offsets(),
      op.sizes(), op.strides(), op.static_offsets(), op.static_sizes(),
      op.static_strides());

  op.getResult().replaceAllUsesWith(loaded);
  op.erase();
  return success();
}

static LogicalResult rewriteSubTensorInsertInPlace(OpBuilder &b,
                                                   tensor::InsertSliceOp op,
                                                   Value target) {
  LLVM_DEBUG(llvm::dbgs() << "RewriteSubTensorInsertInPlace: "
                          << *(op.getOperation()) << "\n");
  if (!op.getResult().hasOneUse()) {
    return op.emitError("not a single use operation");
  }

  Operation *user = *(op.getResult().getUsers().begin());
  if (isa<scf::YieldOp>(user)) {
    auto dest = op.dest();
    if (!dest.isa<BlockArgument>()) {
      return op.emitError("dest is not a argument to the loop");
    }
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);

    // Kills the SSA use-def chain.
    op.replaceAllUsesWith(dest);

    b.create<IREE::Flow::DispatchTensorStoreOp>(
        op.getLoc(), op.source(), target, op.offsets(), op.sizes(),
        op.strides(), op.static_offsets(), op.static_sizes(),
        op.static_strides());

    return success();
  }
  return failure();
}

// Return true if any control flow is found in the DispatchWorkgroupsOp besides
// scf::ForOp.
static bool hasNonScfForControlFlow(
    IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  return dispatchOp
      .walk([&](Operation *op) {
        if (isa<BranchOpInterface>(op) || isa<RegionBranchOpInterface>(op)) {
          if (!isa<scf::ForOp>(op) &&
              !isa<IREE::Flow::DispatchWorkgroupsOp>(op))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

// Rewrite specific SubTensor / SubTensorInsert ops that match a "destructive
// tensor update" pattern, by an inplace update at `binding` and `offset1, using
// hal.interface.*.tensor.tile ops.
// This serves as a step in jumping the abstraction gap between transformed
// "linalg on tensors" IR and the buffer world.
// This is possible because we control the production of such patterns in IREE
// and can take the necessary shortcuts wrt inplace semantics.
// In the future it is reasonable to expect special IR constructs to capture
// some of the destructive update patterns,
//
// Assumptions/Invariants on "Control the Production of Such Patterns"
// ===================================================================
// 1. Input tensors may not participate in a destructive update pattern.
// 2. Init and output tensors may participate in a destructive update pattern.
// 3. No init or output tensor backing storage aliases with any other tensor
//    storage.
// 4. SubTensorOp/SubTensorInsertOp are the only ops that can extract/insert
//    from/into tensors.
// 5. All SubTensorOp/SubTensorInsertOp must have been introduced by Linalg
//    tiling on tensors.
// 6. Such tilings that result in yielded tensors across loops may only tile
//    parallel Linalg iterators atm.
// 7. (Future) Allow non-parallel Linalg iterators tiling and ensure first-read
//    or writeOnly by construction.
//
// Note: the assumptions/invariants above are subject to changing ordering of
// passes. When dispatch region and hal.interfaces are created on the linalg on
// buffers path, these are all assumptions. In the future, when dispatch regions
// and hal.interfaces are created post-transformations on the linalg on tensors
// path some assumptions will become invariants.
//
// For now, the following destructive update patterns are rewritten.
//
// Coming from an `InterfaceLoadTensorOp`
// ======================================
// ```
//   %0 = hal.interface.load.tensor @x[offsetx]
//   ...
//   %1 = destructive_update(%0)
//   ...
//   use_of(%1) // e.g. hal.interface.store.tensor %1 @y[offsety]
// ```
// is rewritten into:
// ```
//   %0 = hal.interface.load.tensor @x[offsetx]
//   ...
//   inplace_update @binding[offset]
//   %2 = hal.interface.load.tensor  @binding[offset]
//   ...
//   use_of(%2) // e.g. hal.interface.store.tensor %2 @y[offsety]
// ```
//
// This is a typical pattern that appears after tiling Linalg ops on tensors
// with operands that come from hal.interface.
//
// Coming from a `LinalgOp`
// =========================
// ```
//   %0 = linalg-op
//   ...
//   %1 = destructive_update(%0) // only subtensor_inserts into %0
//   ...
//   use_of(%1) // e.g. hal.interface.store.tensor %1 @y
// ```
// is rewritten into:
// ```
//   %0 = linalg-op
//   ...
//   inplace_update @binding[offset]
//   %2 = hal.interface.load.tensor @binding[offset]
//   ...
//   hal.interface.store.tensor %2 @y[offsety]
// ```
// This is a typical pattern that appears after tileAndFuse ops with operands
// produced by other linalg ops. In this case, tile and fuse leaves %0 behind
// because it is the op that materializes the full tensor. This could be
// replaced by a notional "tensor.undef" and the compute would become a dead
// value.
// The rewrite breaks the use-def chain for %0 and may result in the linalg-op
// being DCE'd.
//
// Other rewrites:
// ===============
// Furthermore, when `@binding` == `@y` and `offset` == `offsety` and `...`
// contains no aliasing read/write to either `@binding[offset]` or `@y[offsety]`
// the following:
// ```
//   %2 = hal.interface.load.tensor @binding[offset]
//   ...
//   hal.interface.store.tensor %2 @y[offsety]
// ```
// is elided.
// This should probably become a dedicated pass based on core alias analysis,
// when the latter becomes available.
static LogicalResult rewriteDestructiveUpdateInPlace(
    OpBuilder &b, SpecialTerminatorOpCapture &capture, Value target) {
  Operation *outermostProducingOp = (capture.loops.empty())
                                        ? capture.rootDestructiveUpdate
                                        : capture.loops.front();
  LLVM_DEBUG(llvm::dbgs() << "outermost producing: " << *outermostProducingOp
                          << "\n");

  // Try to rewrite inplace.
  if (failed(rewriteSubTensorInsertInPlace(
          b, cast<tensor::InsertSliceOp>(capture.rootDestructiveUpdate),
          target))) {
    return failure();
  }

  if (scf::ForOp loopOp = dyn_cast<scf::ForOp>(outermostProducingOp))
    loopOp.walk(
        [&](tensor::ExtractSliceOp op) { (void)propagateSubTensorOp(b, op); });

  return success();
}

// TODO(nicolasvasilache): generalize to more than naive "top of the region
// consecutive ops". Probably better to wait until core alias analysis is
// upstreamed.
// TODO(nicolasvasilache): interfaces.
static bool hasInterleavedAliases(IREE::Flow::DispatchTensorLoadOp loadOp,
                                  IREE::Flow::DispatchTensorStoreOp storeOp) {
  Block *bLoad = loadOp.getOperation()->getBlock();
  Block *bStore = loadOp.getOperation()->getBlock();
  if (!isa<IREE::Flow::DispatchWorkgroupsOp>(bLoad->getParentOp()) ||
      !isa<IREE::Flow::DispatchWorkgroupsOp>(bStore->getParentOp()) ||
      bLoad->getParentOp() != bStore->getParentOp())
    return true;

  if (storeOp.getOperation()->getPrevNode() != loadOp) return true;

  return false;
}

LogicalResult rewriteLinalgDestructiveUpdates(
    IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  // Bail on any control-flow for now.
  if (hasNonScfForControlFlow(dispatchOp)) {
    return success();
  }

  MLIRContext *context = dispatchOp->getContext();
  OpBuilder b(context);
  SmallVector<IREE::Flow::DispatchTensorStoreOp> processedStores;
  // For each tensor store op, look for destructive updates and replace the
  // destructive pattern by a custom inplace update pattern.
  bool fail = dispatchOp
                  .walk([&](IREE::Flow::DispatchTensorStoreOp op) {
                    SpecialTerminatorOpCapture capture;
                    capture.initValue = op.value();
                    Value sourceValue =
                        isADestructiveUpdatePattern(capture.initValue, capture);
                    if (!sourceValue || !isa_and_nonnull<tensor::InsertSliceOp>(
                                            capture.rootDestructiveUpdate))
                      return WalkResult::advance();
                    if (failed(rewriteDestructiveUpdateInPlace(b, capture,
                                                               op.target()))) {
                      return WalkResult::interrupt();
                    }
                    processedStores.push_back(op);
                    return WalkResult::advance();
                  })
                  .wasInterrupted();
  if (fail) return failure();

  for (auto op : processedStores) {
    op.erase();
  }

  // Non-default canonicalization patterns.
  // TODO(nicolasvasilache): add Linalg tiling canonicalization patterns,
  // affineminscf and others as needed.
  OwningRewritePatternList canonicalizationPatterns(context);
  scf::ForOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  (void)applyPatternsAndFoldGreedily(dispatchOp,
                                     std::move(canonicalizationPatterns));
  return success();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
