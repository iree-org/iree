// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- LinalgRewriteDestructiveUpdates.cpp - Pass for destructive updates--===//
//
// Pass to rewrite Linalg on tensors destructive updates into updates through
// memory.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-bufferize"

namespace mlir {
namespace iree_compiler {

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

// TODO: Use some interface instead of op names directly.
static bool hasDestructiveUpdateSubTensorUses(
    Value v, SpecialTerminatorOpCapture &capture) {
  SmallVector<SubTensorOp, 4> reads;
  SmallVector<SubTensorInsertOp, 4> writes;
  for (auto &u : v.getUses()) {
    if (auto subTensorOp = dyn_cast<SubTensorOp>(u.getOwner())) {
      reads.push_back(subTensorOp);
      continue;
    }
    if (auto subTensorInsertOp = dyn_cast<SubTensorInsertOp>(u.getOwner())) {
      writes.push_back(subTensorInsertOp);
      continue;
    }
    if (auto dimOp = dyn_cast<DimOp>(u.getOwner())) {
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "found non-destructive update pattern use: "
                            << *(u.getOwner()) << "\n");
    return false;
  }
  // For now, only allow exactly a single SubTensorInsertOp that must be
  // dominated by all SubTensorOp.
  if (writes.size() != 1) return false;
  // Small local dominance computation.
  DominanceInfo domInfo(writes.front()->getParentOp());
  for (auto read : reads) {
    LLVM_DEBUG(llvm::dbgs() << "read: " << *read.getOperation() << "\n");
    if (!domInfo.properlyDominates(read.getOperation(), writes.front())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "non-destructive use-def: " << *(read.getOperation())
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
// update pattern. There are 2 cases.
//
// Simple case
// ===========
//
// Just detect a SubTensorInsertOp.
//
// Loop case
// =========
//
// Iteratively traverse an (imperfectly nested) loop nest such as:
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
  // Simple case: no loops and directly a tensorInsertOp.
  if (auto tensorInsertOp =
          dyn_cast_or_null<SubTensorInsertOp>(tensor.getDefiningOp())) {
    capture.rootDestructiveUpdate = tensorInsertOp;
    return tensorInsertOp.dest();
  }

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
/// hal.interface.load.tensor.tile.
static LogicalResult propagateSubTensorOp(OpBuilder &b, SubTensorOp op) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  auto loadOp = op.source().getDefiningOp<IREE::HAL::InterfaceLoadTensorOp>();
  if (!loadOp) {
    BlockArgument val = op.source().dyn_cast<BlockArgument>();
    while (val) {
      auto forOp = dyn_cast<scf::ForOp>(val.getOwner()->getParentOp());
      // val is a block argument but not to an scf::ForOp -> bail.
      if (!forOp) return failure();
      unsigned idx = val.getArgNumber() - 1;  // accounting for IV arg.
      Value iterOperand = *(forOp.getIterOperands().begin() + idx);
      loadOp = iterOperand.getDefiningOp<IREE::HAL::InterfaceLoadTensorOp>();
      val = iterOperand.dyn_cast<BlockArgument>();
    }
  }
  if (!loadOp) return failure();

  Value loaded = b.create<IREE::HAL::InterfaceLoadTensorTileOp>(
      op.getLoc(), op.getResult().getType(), loadOp.binding(), loadOp.offset(),
      op.offsets(), op.sizes(), op.strides(), op.static_offsets(),
      op.static_sizes(), op.static_strides());
  op.getResult().replaceAllUsesWith(loaded);
  op.erase();
  return success();
}

static LogicalResult rewriteSubTensorInsertInPlace(OpBuilder &b,
                                                   SubTensorInsertOp op,
                                                   SymbolRefAttr binding,
                                                   Value offset) {
  LLVM_DEBUG(llvm::dbgs() << "RewriteSubTensorInsertInPlace: "
                          << *(op.getOperation()) << "\n");
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  auto dest = op.dest();

  // Sanity check for insert into an initArg that is immediately yielded.
  if (!dest.isa<BlockArgument>() || !op.getResult().hasOneUse() ||
      !isa<scf::YieldOp>(op.getResult().getUses().begin()->getOwner())) {
    LLVM_DEBUG(llvm::dbgs() << "Rewrite failed: no single scf::YieldOp use\n");
    return failure();
  }

  // Kills the SSA use-def chain.
  op.replaceAllUsesWith(dest);
  b.create<IREE::HAL::InterfaceStoreTensorTileOp>(
      op.getLoc(), TypeRange{}, op.source(), binding, offset, op.offsets(),
      op.sizes(), op.strides(), op.static_offsets(), op.static_sizes(),
      op.static_strides());
  return success();
}

// Return true if any control flow is found in the FuncOp besides scf::ForOp.
static bool hasNonScfForControlFlow(FuncOp funcOp) {
  return funcOp
      .walk([&](Operation *op) {
        if (isa<BranchOpInterface>(op) || isa<RegionBranchOpInterface>(op)) {
          if (!isa<scf::ForOp>(op)) return WalkResult::interrupt();
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
static LogicalResult rewriteDestructiveUpdateInPlace(OpBuilder &b, Value v,
                                                     SymbolRefAttr binding,
                                                     Value offset) {
  SpecialTerminatorOpCapture capture;
  capture.initValue = v;
  Value sourceValue = isADestructiveUpdatePattern(capture.initValue, capture);

  // No destructive update semantics, bail.
  if (!sourceValue || !capture.rootDestructiveUpdate) return failure();

  Operation *outermostProducingOp = (capture.loops.empty())
                                        ? capture.rootDestructiveUpdate
                                        : capture.loops.front();
  LLVM_DEBUG(llvm::dbgs() << "outermost producing: " << *outermostProducingOp
                          << "\n");

  // Helper to clone the load right after `outermostProducingOp` and use it in
  // `op`.
  auto buildAndUseLoad = [&](SymbolRefAttr bindingAttr, Value offsetValue) {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(outermostProducingOp);
    // TODO: no attributes other than `bindingAttr` on hal.interface.load.tensor
    // atm. Do we need other attributes propagated?
    Value newLoad = b.create<IREE::HAL::InterfaceLoadTensorOp>(
        outermostProducingOp->getLoc(), v.getType(), bindingAttr, offsetValue);
    // TODO: this brutally replaces all uses by the result of this load.
    // In practice we may want more recompute and we may have lost information.
    // Revisit this when things are morefleshed out.
    v.replaceAllUsesWith(newLoad);
  };

  // `sourceValue`-specific determination of binding and offset for inplace
  // update.
  // TODO: when the destructively updated value comes from a LoadTensorOp, we
  // need to decide whether to perform an update "into" the init tensor (i.e.
  // the loadOp binding + offset) or directly into the output.
  // In the fullness of time, this is dependent on whether or not the
  // destructive update can guarantee that any particular subtensor of the
  // result is updated only once.
  // This is guaranteed for loops that come from tiling "parallel" Linalg
  // iterators.
  // Reduction iterators are subject to additional first-read/last-write
  // considerations, usually derived from traditional memory-based dependence
  // analysis.
  // For now we assume that Linalg on tensors only tiles and fuses across
  // parallel iterators, which allows reading the proper init value and updating
  // the result.
  SymbolRefAttr bindingAttr =
      TypeSwitch<Operation *, SymbolRefAttr>(sourceValue.getDefiningOp())
          .Case<IREE::HAL::InterfaceLoadTensorOp>(
              // [&](auto op) { return op.binding(); })
              [&](auto op) { return binding; })
          .Case<linalg::LinalgOp>([&](auto op) { return binding; })
          .Default([](Operation *) { return nullptr; });
  Value offsetValue =
      TypeSwitch<Operation *, Value>(sourceValue.getDefiningOp())
          .Case<IREE::HAL::InterfaceLoadTensorOp>(
              // [&](auto op) { return op.offset(); })
              [&](auto op) { return offset; })
          .Case<linalg::LinalgOp>([&](auto op) { return offset; })
          .Default([](Operation *) { return nullptr; });

  // TODO: support more cases as needed.
  if (!bindingAttr) return failure();

  // Try to rewrite inplace.
  if (failed(rewriteSubTensorInsertInPlace(
          b, cast<SubTensorInsertOp>(capture.rootDestructiveUpdate),
          bindingAttr, offsetValue)))
    return failure();

  // Reload the value produced inplace right after the inplace update.
  buildAndUseLoad(bindingAttr, offsetValue);

  if (scf::ForOp loopOp = dyn_cast<scf::ForOp>(outermostProducingOp))
    loopOp.walk([&](SubTensorOp op) { propagateSubTensorOp(b, op); });

  return success();
}

// TODO: generalize to more than naive "top of the function consecutive ops".
// Probably better to wait until core alias analysis is upstreamed.
// TODO: interfaces.
static bool hasInterleavedAliases(IREE::HAL::InterfaceLoadTensorOp loadOp,
                                  IREE::HAL::InterfaceStoreTensorOp storeOp) {
  Block *bLoad = loadOp.getOperation()->getBlock();
  Block *bStore = loadOp.getOperation()->getBlock();
  if (!isa<FuncOp>(bLoad->getParentOp()) ||
      !isa<FuncOp>(bStore->getParentOp()) ||
      bLoad->getParentOp() != bStore->getParentOp())
    return true;

  if (storeOp.getOperation()->getPrevNode() != loadOp) return true;

  return false;
}

namespace {
struct LinalgRewriteDestructiveUpdates
    : public PassWrapper<LinalgRewriteDestructiveUpdates, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREEDialect, linalg::LinalgDialect, scf::SCFDialect,
                    StandardOpsDialect>();
  }
  void runOnFunction() override;
};
}  // namespace

void LinalgRewriteDestructiveUpdates::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Bail on any control-flow for now.
  if (hasNonScfForControlFlow(funcOp)) return signalPassFailure();

  MLIRContext *context = &getContext();
  OpBuilder b(context);
  // For each tensor store op, look for destructive updates and replace the
  // destructive pattern by a custom inplace update pattern.
  funcOp.walk([&](IREE::HAL::InterfaceStoreTensorOp op) {
    if (failed(rewriteDestructiveUpdateInPlace(b, op.operand(), op.binding(),
                                               op.offset()))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // For each tensor store op, redundant load/store optimization.
  funcOp.walk([&](IREE::HAL::InterfaceStoreTensorOp storeOp) {
    auto loadOp = dyn_cast_or_null<IREE::HAL::InterfaceLoadTensorOp>(
        storeOp.operand().getDefiningOp());

    // Bail if there exists an interleaved aliasing.
    if (!loadOp || hasInterleavedAliases(loadOp, storeOp)) return;

    // Bail if this is not a simple forwarding.
    // TODO: Handle more advanced forwarding, but we may need to do it
    // earlier while we still have SSA use-def chains.
    // I.e. revisit later cases such as:
    // ```
    //   inplace_update_tiles_rooted_at @legacy_io::@arg0, offset = %c0
    //   %2 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 :
    //     tensor<2x4xf32>
    //   hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0
    //     {operand_result_index = 3 : i32} : tensor<2x4xf32>
    //   return
    // ```
    // where the inplace update could be done directly in @legacy_io::@ret0,
    // offset = %c0.
    if (loadOp.binding() != storeOp.binding() ||
        loadOp.offset() != storeOp.offset())
      return;

    storeOp.erase();
  });

  // Non-default canonicalization patterns.
  // TODO: add Linalg tiling canonicalization patterns, affineminscf and others
  // as needed.
  OwningRewritePatternList canonicalizationPatterns;
  scf::ForOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  applyPatternsAndFoldGreedily(funcOp, std::move(canonicalizationPatterns));
}

std::unique_ptr<OperationPass<FuncOp>>
createLinalgRewriteDestructiveUpdatesPass() {
  return std::make_unique<LinalgRewriteDestructiveUpdates>();
}

static PassRegistration<LinalgRewriteDestructiveUpdates> pass(
    "iree-codegen-linalg-rewrite-destructive-updates",
    "Test the rewrite of destructive update patterns to inplace update form.",
    [] { return std::make_unique<LinalgRewriteDestructiveUpdates>(); });
}  // namespace iree_compiler
}  // namespace mlir
