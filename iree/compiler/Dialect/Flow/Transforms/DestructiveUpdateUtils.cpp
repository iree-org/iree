// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- DestructiveUpdateUtils.cpp - Utils to rewrite destructive updates --===//
//
// Implementation to rewrite Linalg on tensors destructive updates into updates
// through memory.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
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
  // For now, must be scf.for ops.
  SmallVector<Operation *, 4> loops;
  // For now, must be a tensor.insert_slice op.
  Operation *rootDestructiveUpdate;
  bool readOnly = false;
  bool writeOnly = false;
};

// TODO(nicolasvasilache): Use some interface instead of op names directly.
static bool hasDestructiveUpdateUses(BlockArgument arg,
                                     SpecialTerminatorOpCapture &capture) {
  SmallVector<Operation *> reads;
  SmallVector<Operation *> writes;
  for (OpOperand &u : arg.getUses()) {
    TypeSwitch<Operation *, void>(u.getOwner())
        .Case<linalg::LinalgOp, linalg_ext::LinalgExtOp>(
            [&](auto linalgLikeOp) {
              if (linalgLikeOp.isOutputTensor(&u)) {
                writes.push_back(linalgLikeOp);
              } else {
                reads.push_back(linalgLikeOp);
              }
            })
        .Case<tensor::InsertSliceOp>([&](tensor::InsertSliceOp sliceOp) {
          if (sliceOp.dest() == u.get()) {
            writes.push_back(sliceOp);
          } else {
            reads.push_back(sliceOp);
          }
        })
        .Default([&](Operation *op) { reads.push_back(op); });
  }
  // For now, only allow exactly a single tensor.insert_slice op that must be
  // dominated by all tensor.extract_slice ops.
  if (writes.size() != 1) return false;
  // Small local dominance computation.
  DominanceInfo domInfo(writes.front()->getParentOp());
  for (auto read : reads) {
    LLVM_DEBUG(llvm::dbgs() << "read: " << *read << "\n");
    if (!domInfo.properlyDominates(read, writes.front())) {
      LLVM_DEBUG(llvm::dbgs() << "non-destructive use-def: " << *read
                              << " does not properly dominate "
                              << *(writes.front()) << "\n");
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
    // tensor.extract_slice / tensor.insert_slice op with proper dominance.
    if (!regionArg.hasOneUse()) {
      if (!hasDestructiveUpdateUses(regionArg, capture)) return nullptr;
      return returnValue;
    }

    assert(regionArg.hasOneUse());
    LLVM_DEBUG(llvm::dbgs() << "one use analysis: " << regionArg << "\n");
    OpOperand *operand = regionArg.getUses().begin().getOperand();
    auto innerForOp = dyn_cast<scf::ForOp>(operand->getOwner());
    // Case 3a: Single use which is not an scf::ForOp, it may still be a
    // single tensor.extract_slice / tensor.insert_slice op.
    if (!innerForOp) {
      if (!hasDestructiveUpdateUses(regionArg, capture)) return nullptr;
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

/// Given a `sliceOp` taking a slice from a `sourceOp`, computes the `offsets`,
/// `sizes`, and `strides` relative to `sourceOp`'s source.
LogicalResult computeSliceOffsetsSizesStrides(
    OffsetSizeAndStrideOpInterface sourceOp,
    OffsetSizeAndStrideOpInterface sliceOp,
    SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides, OpBuilder &builder) {
  // Special case where the source op has no offsets/sizes/strides. That means
  // the full range.
  if (sourceOp.offsets().empty() && sourceOp.static_offsets().empty() &&
      sourceOp.sizes().empty() && sourceOp.static_sizes().empty() &&
      sourceOp.strides().empty() && sourceOp.static_strides().empty()) {
    offsets = sliceOp.getMixedOffsets();
    sizes = sliceOp.getMixedSizes();
    strides = sliceOp.getMixedStrides();
    return success();
  }

  // Require the source op to have all-one strides This makes sure we can just
  // use the slice op's strides as the final strides.
  if (!llvm::all_of(
          sourceOp.static_strides().getAsValueRange<IntegerAttr>(),
          [](const APInt &stride) { return stride.getZExtValue() == 1; }))
    return failure();

  MLIRContext *context = sliceOp.getContext();
  Location loc = sliceOp.getLoc();
  AffineExpr sym0, sym1;
  bindSymbols(context, sym0, sym1);
  auto addMap = AffineMap::get(0, 2, {sym0 + sym1}, context);

  auto getStrideValue = [&](OpFoldResult attrOrValue) {
    if (auto val = attrOrValue.dyn_cast<Value>()) return val;
    auto attr = attrOrValue.get<Attribute>().cast<IntegerAttr>();
    auto cstOp = builder.create<arith::ConstantIndexOp>(loc, attr.getInt());
    return cstOp.getResult();
  };

  // The new offsets should be adding source offsets and slice offsets, as the
  // sliceOp is relative to the sourceOp.
  offsets.clear();
  for (auto pair :
       llvm::zip(sourceOp.getMixedOffsets(), sliceOp.getMixedOffsets())) {
    auto applyOp = builder.create<AffineApplyOp>(
        loc, addMap,
        ValueRange{getStrideValue(std::get<0>(pair)),
                   getStrideValue(std::get<1>(pair))});
    offsets.push_back(applyOp.getResult());
  }
  sizes = sliceOp.getMixedSizes();
  strides = sliceOp.getMixedStrides();

  return success();
}

/// Flos tensor.extract_slice ops on top of flow.dispatch.tensor.load ops into
/// new flow.dispatch.tensor.load ops.
static LogicalResult foldExtractSliceOp(OpBuilder &b,
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

  SmallVector<OpFoldResult> offsets, sizes, strides;
  if (failed(computeSliceOffsetsSizesStrides(loadOp, op, offsets, sizes,
                                             strides, b)))
    return failure();

  Value loaded = b.create<IREE::Flow::DispatchTensorLoadOp>(
      op.getLoc(), op.getResult().getType().cast<RankedTensorType>(),
      loadOp.source(), offsets, sizes, strides);

  op.getResult().replaceAllUsesWith(loaded);
  op.erase();
  return success();
}

template <typename OpTy>
static LogicalResult rewriteDestructiveUpdateInPlace(
    OpBuilder &b, DispatchTensorStoreOp storeOp, OpTy linalgLikeOp,
    Value target) {
  LLVM_DEBUG(llvm::dbgs() << "RewriteDestructiveUpdateInPlace: "
                          << *linalgLikeOp.getOperation() << "\n");
  if (!linalgLikeOp->hasOneUse()) {
    return linalgLikeOp.emitError("not a single use operation");
  }

  OpOperand &use = *(linalgLikeOp->use_begin());
  if (isa<scf::YieldOp>(use.getOwner())) {
    OpResult usedResult = use.get().cast<OpResult>();
    Value dest =
        linalgLikeOp.getOutputOperand(usedResult.getResultNumber())->get();
    if (!dest || !dest.isa<BlockArgument>()) {
      return linalgLikeOp.emitError("dest is not a argument to the loop");
    }
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(linalgLikeOp);

    // Kills the SSA use-def chain.
    usedResult.replaceAllUsesWith(dest);

    b.create<IREE::Flow::DispatchTensorStoreOp>(linalgLikeOp.getLoc(),
                                                usedResult, target);

    return success();
  }
  return failure();
}

/// Rewrites destructive in-place updates with the update operation being
/// tensor.insert_slice.
template <>
LogicalResult rewriteDestructiveUpdateInPlace<tensor::InsertSliceOp>(
    OpBuilder &b, DispatchTensorStoreOp storeOp,
    tensor::InsertSliceOp insertSliceOp, Value target) {
  LLVM_DEBUG(llvm::dbgs() << "RewriteDestructiveUpdateInPlace: "
                          << *insertSliceOp.getOperation() << "\n");
  if (!insertSliceOp->hasOneUse()) {
    return insertSliceOp.emitError("not a single use operation");
  }

  OpOperand &use = *(insertSliceOp->use_begin());
  if (isa<scf::YieldOp>(use.getOwner())) {
    OpResult usedResult = use.get().cast<OpResult>();
    Value dest = insertSliceOp.dest();
    if (!dest || !dest.isa<BlockArgument>()) {
      return insertSliceOp.emitError("dest is not a argument to the loop");
    }
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(insertSliceOp);

    // Kills the SSA use-def chain.
    usedResult.replaceAllUsesWith(dest);

    SmallVector<OpFoldResult> offsets, sizes, strides;
    if (failed(computeSliceOffsetsSizesStrides(storeOp, insertSliceOp, offsets,
                                               sizes, strides, b)))
      return failure();

    b.create<IREE::Flow::DispatchTensorStoreOp>(insertSliceOp->getLoc(),
                                                insertSliceOp.source(), target,
                                                offsets, sizes, strides);

    return success();
  }
  return failure();
}

// Return true if any control flow is found in the DispatchWorkgroupsOp besides
// scf::ForOp.
static bool hasNonScfForControlFlow(Operation *regionOp) {
  return regionOp
      ->walk([&](Operation *op) {
        if (isa<BranchOpInterface>(op) || isa<RegionBranchOpInterface>(op)) {
          if (!isa<scf::ForOp>(op) &&
              !isa<IREE::Flow::DispatchWorkgroupsOp>(op))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static LogicalResult rewriteDestructiveUpdateInPlace(
    OpBuilder &b, DispatchTensorStoreOp storeOp,
    SpecialTerminatorOpCapture &capture, Value target) {
  Operation *outermostProducingOp = (capture.loops.empty())
                                        ? capture.rootDestructiveUpdate
                                        : capture.loops.front();
  LLVM_DEBUG(llvm::dbgs() << "outermost producing: " << *outermostProducingOp
                          << "\n");

  // Try to rewrite inplace.
  auto status =
      TypeSwitch<Operation *, LogicalResult>(capture.rootDestructiveUpdate)
          .Case<linalg::LinalgOp, linalg_ext::LinalgExtOp,
                tensor::InsertSliceOp>([&](auto op) {
            return rewriteDestructiveUpdateInPlace(b, storeOp, op, target);
          })
          .Default([&](Operation *) { return failure(); });
  if (failed(status)) return failure();

  if (scf::ForOp loopOp = dyn_cast<scf::ForOp>(outermostProducingOp))
    loopOp.walk(
        [&](tensor::ExtractSliceOp op) { (void)foldExtractSliceOp(b, op); });

  return success();
}

LogicalResult rewriteLinalgDestructiveUpdates(Operation *regionOp) {
  // Bail on any control-flow for now.
  if (hasNonScfForControlFlow(regionOp)) return success();

  MLIRContext *context = regionOp->getContext();
  OpBuilder b(context);
  SmallVector<IREE::Flow::DispatchTensorStoreOp> processedStores;
  // For each tensor store op, look for destructive updates and replace the
  // destructive pattern by a custom inplace update pattern.
  auto walkResult = regionOp->walk([&](IREE::Flow::DispatchTensorStoreOp op) {
    SpecialTerminatorOpCapture capture;
    capture.initValue = op.value();
    Value sourceValue = isADestructiveUpdatePattern(capture.initValue, capture);
    if (!sourceValue) return WalkResult::advance();
    if (failed(rewriteDestructiveUpdateInPlace(b, op, capture, op.target()))) {
      return WalkResult::interrupt();
    }
    processedStores.push_back(op);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) return failure();

  for (auto op : processedStores) op.erase();

  // Non-default canonicalization patterns.
  // TODO(nicolasvasilache): add Linalg tiling canonicalization patterns,
  // affineminscf and others as needed.
  OwningRewritePatternList canonicalizationPatterns(context);
  scf::ForOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  (void)applyPatternsAndFoldGreedily(regionOp,
                                     std::move(canonicalizationPatterns));
  return success();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
