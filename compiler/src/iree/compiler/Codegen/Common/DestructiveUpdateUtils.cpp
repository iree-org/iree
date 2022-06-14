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

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-linalg-rewrite-destructive-updates"

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
  // For now, must be scf.for ops.
  SmallVector<scf::ForOp, 4> loops;
  OpOperand *rootDestructiveUpdate;
  bool readOnly = false;
  bool writeOnly = false;
  // `flow.dispatch.tensor.store` that is is written into.
  // Could be empty for dead uses.
  Optional<IREE::Flow::DispatchTensorStoreOp> dest = llvm::None;
};

// TODO(nicolasvasilache): Use some interface instead of op names directly.
static bool hasDestructiveUpdateUses(BlockArgument arg,
                                     SpecialTerminatorOpCapture &capture) {
  SmallVector<OpOperand *> reads;
  SmallVector<OpOperand *> writes;
  for (OpOperand &u : arg.getUses()) {
    Operation *user = u.getOwner();
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(user)) {
      if (linalgOp.isOutputTensor(&u)) {
        writes.push_back(&u);
      } else {
        reads.push_back(&u);
      }
    } else if (auto linalgExtOp =
                   dyn_cast<IREE::LinalgExt::LinalgExtOp>(user)) {
      if (linalgExtOp.isOutputTensor(&u)) {
        writes.push_back(&u);
      } else {
        reads.push_back(&u);
      }
    } else if (auto sliceOp = dyn_cast<tensor::InsertSliceOp>(user)) {
      if (sliceOp.dest() == u.get()) {
        writes.push_back(&u);
      } else {
        reads.push_back(&u);
      }
    } else {
      reads.push_back(&u);
    }
  }
  // For now, only allow exactly a single tensor.insert_slice op that must be
  // dominated by all tensor.extract_slice ops.
  if (writes.size() != 1) return false;
  // Small local dominance computation.
  DominanceInfo domInfo(writes.front()->getOwner()->getParentOp());
  for (auto read : reads) {
    LLVM_DEBUG(llvm::dbgs() << "read: " << *(read->getOwner()) << "\n");
    if (!domInfo.properlyDominates(read->getOwner(),
                                   writes.front()->getOwner())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "non-destructive use-def: " << *(read->getOwner())
                 << " does not properly dominate "
                 << *(writes.front()->getOwner()) << "\n");
      return false;
    }
  }

  capture.readOnly = writes.empty();
  capture.writeOnly = reads.empty();
  capture.rootDestructiveUpdate = writes.front();
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
        scfForOp.getRegion().front().getTerminator()->getOperand(idx);

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

/// Folds tensor.extract_slice ops on top of flow.dispatch.tensor.load ops into
/// new flow.dispatch.tensor.load ops.
static LogicalResult foldExtractSliceOp(OpBuilder &b,
                                        tensor::ExtractSliceOp op) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  auto sourceOp = op.source().getDefiningOp();
  if (!sourceOp) {
    BlockArgument val = op.source().dyn_cast<BlockArgument>();
    while (val) {
      auto forOp = dyn_cast<scf::ForOp>(val.getOwner()->getParentOp());
      // val is a block argument but not to an scf::ForOp -> bail.
      if (!forOp) return failure();
      unsigned idx = val.getArgNumber() - 1;  // accounting for IV arg.
      Value iterOperand = *(forOp.getIterOperands().begin() + idx);
      sourceOp = iterOperand.getDefiningOp();
      val = iterOperand.dyn_cast<BlockArgument>();
    }
  }
  if (!sourceOp) return failure();

  Operation *replacement = nullptr;
  if (auto loadOp = dyn_cast<IREE::Flow::DispatchTensorLoadOp>(sourceOp)) {
    SmallVector<OpFoldResult> offsets, sizes, strides;
    Location loc = op.getLoc();
    if (failed(foldOffsetsSizesAndStrides(b, loc, loadOp, op,
                                          loadOp.getDroppedDims(), offsets,
                                          sizes, strides))) {
      return failure();
    }

    replacement = b.create<IREE::Flow::DispatchTensorLoadOp>(
        op.getLoc(), op.getType(), loadOp.source(), loadOp.source_dims(),
        offsets, sizes, strides);
  } else if (auto initTensorOp = dyn_cast<linalg::InitTensorOp>(sourceOp)) {
    replacement = b.create<linalg::InitTensorOp>(
        op.getLoc(), op.getMixedSizes(), op.getType().getElementType());
  }

  if (!replacement) return failure();

  op->replaceAllUsesWith(replacement->getResults());
  op.erase();
  return success();
}

template <typename OpTy>
static LogicalResult rewriteDestructiveUpdateInPlace(
    OpBuilder &b, OpTy linalgLikeOp, OpOperand *operand,
    Optional<IREE::Flow::DispatchTensorStoreOp> storeOp) {
  LLVM_DEBUG(llvm::dbgs() << "RewriteDestructiveUpdateInPlace: "
                          << *linalgLikeOp.getOperation() << "\n");
  Optional<unsigned> resultNumber = llvm::None;
  for (auto outputOperand : llvm::enumerate(linalgLikeOp.getOutputOperands())) {
    if (outputOperand.value() == operand) {
      resultNumber = outputOperand.index();
      break;
    }
  }
  if (!resultNumber) {
    return linalgLikeOp.emitOpError("operand is not a out operand");
  }
  Value result = linalgLikeOp->getResult(resultNumber.getValue());
  if (result.use_empty()) return success();
  if (!result.hasOneUse()) {
    return linalgLikeOp.emitError("not a single use result");
  }

  OpOperand &use = *(result.use_begin());
  if (isa<scf::YieldOp>(use.getOwner())) {
    Value dest = operand->get();
    if (!dest || !dest.isa<BlockArgument>()) {
      return linalgLikeOp.emitError("dest is not a argument to the loop");
    }
    // Kills the SSA use-def chain.
    result.replaceAllUsesWith(dest);

    if (storeOp) {
      // If the result is the actual return of the dispatch, create the matching
      // store.
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointAfter(linalgLikeOp);

      b.create<IREE::Flow::DispatchTensorStoreOp>(
          linalgLikeOp.getLoc(), result, storeOp->target(),
          storeOp->target_dims(), storeOp->getMixedOffsets(),
          storeOp->getMixedSizes(), storeOp->getMixedStrides());
    }
    return success();
  }
  return failure();
}

/// Rewrites destructive in-place updates with the update operation being
/// tensor.insert_slice.
template <>
LogicalResult rewriteDestructiveUpdateInPlace<tensor::InsertSliceOp>(
    OpBuilder &b, tensor::InsertSliceOp insertSliceOp, OpOperand *operand,
    Optional<IREE::Flow::DispatchTensorStoreOp> storeOp) {
  if (insertSliceOp.use_empty()) return success();
  LLVM_DEBUG(llvm::dbgs() << "RewriteDestructiveUpdateInPlace: "
                          << *insertSliceOp.getOperation() << "\n");
  if (operand->get() != insertSliceOp.dest()) {
    return insertSliceOp.emitOpError("expected operand to be the dest");
  }
  if (!insertSliceOp->hasOneUse()) {
    return insertSliceOp.emitOpError("not a single use operation");
  }

  OpOperand &use = *(insertSliceOp->use_begin());
  if (isa<scf::YieldOp>(use.getOwner())) {
    OpResult usedResult = use.get().cast<OpResult>();
    Value dest = insertSliceOp.dest();
    if (!dest || !dest.isa<BlockArgument>()) {
      return insertSliceOp.emitError("dest is not a argument to the loop");
    }
    // Kills the SSA use-def chain.
    usedResult.replaceAllUsesWith(dest);

    if (storeOp) {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPoint(insertSliceOp);

      SmallVector<OpFoldResult> offsets, sizes, strides;
      Location loc = insertSliceOp->getLoc();
      if (failed(foldOffsetsSizesAndStrides(
              b, loc, storeOp.getValue(), insertSliceOp,
              storeOp->getDroppedDims(), offsets, sizes, strides))) {
        return failure();
      }

      b.create<IREE::Flow::DispatchTensorStoreOp>(
          insertSliceOp->getLoc(), insertSliceOp.source(), storeOp->target(),
          storeOp->target_dims(), offsets, sizes, strides);
    }

    return success();
  }
  return failure();
}

// Return true if any control flow is found in the workgroup besides scf::ForOp.
static bool hasNonScfForControlFlow(func::FuncOp funcOp) {
  return funcOp
      ->walk([&](Operation *op) {
        if (isa<BranchOpInterface>(op) || isa<RegionBranchOpInterface>(op)) {
          if (!isa<scf::ForOp, scf::IfOp>(op) && !isa<linalg::LinalgOp>(op))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static LogicalResult rewriteDestructiveUpdateInPlace(
    OpBuilder &b, SpecialTerminatorOpCapture &capture) {
  scf::ForOp outermostProducingOp = capture.loops.front();
  LLVM_DEBUG(llvm::dbgs() << "outermost producing: " << *outermostProducingOp
                          << "\n");

  // Try to rewrite inplace.
  auto status = TypeSwitch<Operation *, LogicalResult>(
                    capture.rootDestructiveUpdate->getOwner())
                    .Case<linalg::LinalgOp, IREE::LinalgExt::LinalgExtOp,
                          tensor::InsertSliceOp>([&](auto op) {
                      return rewriteDestructiveUpdateInPlace(
                          b, op, capture.rootDestructiveUpdate, capture.dest);
                    })
                    .Default([&](Operation *) { return failure(); });
  if (failed(status)) return failure();

  outermostProducingOp.walk(
      [&](tensor::ExtractSliceOp op) { (void)foldExtractSliceOp(b, op); });

  return success();
}

LogicalResult rewriteLinalgDestructiveUpdates(func::FuncOp funcOp) {
  // Bail on any control-flow for now.
  if (hasNonScfForControlFlow(funcOp)) return success();

  MLIRContext *context = funcOp->getContext();
  OpBuilder b(context);
  SmallVector<IREE::Flow::DispatchTensorStoreOp> processedStores;
  // For each tensor store op, look for destructive updates and replace the
  // destructive pattern by a custom inplace update pattern.
  llvm::MapVector<Value, SpecialTerminatorOpCapture> destructiveUpdates;
  funcOp.walk([&](IREE::Flow::DispatchTensorStoreOp op) {
    SpecialTerminatorOpCapture capture;
    capture.initValue = op.value();
    Value sourceValue = isADestructiveUpdatePattern(capture.initValue, capture);
    if (!sourceValue || capture.loops.empty()) return;
    capture.dest = op;
    destructiveUpdates[op.value()] = capture;
  });

  // Check if the other returns of the destructive update loops are also
  // destuctive updates. These come from values that are dead outside dispatch,
  // so have no stores, but are still needed for use within the dispatch.
  llvm::MapVector<Value, SpecialTerminatorOpCapture>
      deadOutputDestructiveUpdates;
  for (auto &it : destructiveUpdates) {
    scf::ForOp outerMostLoop = it.second.loops.front();
    for (auto result : outerMostLoop.getResults()) {
      if (result == it.second.initValue) continue;
      SpecialTerminatorOpCapture newCapture;
      newCapture.initValue = result;
      Value sourceValue = isADestructiveUpdatePattern(result, newCapture);
      if (!sourceValue || newCapture.loops.empty()) continue;
      deadOutputDestructiveUpdates[result] = newCapture;
    }
  }

  // Rewrite destructive updates that are eventually published using
  // `flow.dispatch.tensor`.
  for (auto &it : destructiveUpdates) {
    if (failed(rewriteDestructiveUpdateInPlace(b, it.second))) {
      return failure();
    }
    processedStores.push_back(it.second.dest.getValue());
  };
  for (auto op : processedStores) op.erase();

  // Rewrite destructive updates that are dead outside the dispatch.
  for (auto &it : deadOutputDestructiveUpdates) {
    if (failed(rewriteDestructiveUpdateInPlace(b, it.second))) {
      return failure();
    }
  }

  // Non-default canonicalization patterns.
  // TODO(nicolasvasilache): add Linalg tiling canonicalization patterns,
  // affineminscf and others as needed.
  RewritePatternSet canonicalizationPatterns(context);
  scf::ForOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  return applyPatternsAndFoldGreedily(funcOp,
                                      std::move(canonicalizationPatterns));
}

}  // namespace iree_compiler
}  // namespace mlir
