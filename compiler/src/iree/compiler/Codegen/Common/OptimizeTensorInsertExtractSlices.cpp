// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-codegen-optimize-tensor-insert-extract-slices"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_OPTIMIZETENSORINSERTEXTRACTSLICESPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class OptimizeTensorInsertExtractSlicesPass final
    : public impl::OptimizeTensorInsertExtractSlicesPassBase<
          OptimizeTensorInsertExtractSlicesPass> {
  using impl::OptimizeTensorInsertExtractSlicesPassBase<
      OptimizeTensorInsertExtractSlicesPass>::
      OptimizeTensorInsertExtractSlicesPassBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};

// Check if this insertion is loop invariant except it's source.
// We would also be okay as long as the destination is loop invariant,
// but we would have to do some cloning, so we don't do it here.
static bool canBeHoisted(LoopLikeOpInterface loopLike,
                         SubsetInsertionOpInterface insertion) {
  // Do not move terminators.
  if (insertion->hasTrait<OpTrait::IsTerminator>())
    return false;

  // Walk the nested operations and check that all used values are either
  // defined outside of the loop or in a nested region, but not at the level of
  // the loop body.
  auto walkFn = [&](Operation *child) {
    for (OpOperand &operand : child->getOpOperands()) {
      // Ignore values defined in a nested region.
      if (insertion->isAncestor(operand.get().getParentRegion()->getParentOp()))
        continue;
      if (!loopLike.isDefinedOutsideOfLoop(operand.get()) &&
          &operand != &insertion.getSourceOperand()) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  };
  return !insertion->walk(walkFn).wasInterrupted();
}

/// Return the newly created loop op (that has extra iter_args) or the original
/// loop op if nothing was hoisted.
static LoopLikeOpInterface
hoistLoopInvariantSubsetAtIterArg(RewriterBase &rewriter,
                                  LoopLikeOpInterface loopLike, int64_t idx) {
  // Get subset insertion of this yielded arg.
  auto insertion = loopLike.getYieldedValues()[idx]
                       .getDefiningOp<SubsetInsertionOpInterface>();
  if (!insertion) {
    return loopLike;
  }
  if (!canBeHoisted(loopLike, insertion)) {
    return loopLike;
  }

  bool changed = true;
  // The below `while` loop is a WAR for the core issue that is unidentified.
  // To avoid infinite loops, limit number of iterations to 10.
  int numIterations = 0;
  while (changed && numIterations < 10) {
    numIterations++;
    changed = false;
    // Get all subset extraction uses of this iter_arg and try to hoist them
    // out of the loop.
    for (Operation *op : loopLike.getRegionIterArgs()[idx].getUsers()) {
      auto extraction = dyn_cast<SubsetExtractionOpInterface>(op);
      if (!extraction) {
        continue;
      }

      // Check if this extraction is operating on the same subset as the
      // insertion.
      bool equivalent = extraction.operatesOnEquivalentSubset(
          insertion, [](Value v1, Value v2) {
            // The callback to this method checks if the given two values are
            // aliasing tensors/buffers from which the subset slice comes from.
            // For our case, we only care if the slices are same, so we can
            // always return true.
            return true;
          });

      if (!equivalent) {
        continue;
      }

      // Hoist out the extraction/insertion ops.
      NewYieldValuesFn newYieldValuesFn =
          [&](OpBuilder &b, Location loc,
              ArrayRef<BlockArgument> innerNewBBArgs) -> SmallVector<Value> {
        return {insertion.getSourceOperand().get()};
      };

      // replaceInitOperandUsesInLoop is set to true S.T we will use new IV
      // instead of hoisted out extract.
      FailureOr<LoopLikeOpInterface> newLoop =
          loopLike.replaceWithAdditionalYields(
              rewriter, extraction.getResult(),
              /*replaceInitOperandUsesInLoop=*/true, newYieldValuesFn);
      if (failed(newLoop)) {
        return loopLike;
      }
      loopLike = *newLoop;

      BlockArgument iterArg = loopLike.getRegionIterArgs()[idx];
      OpResult loopResult = loopLike.getTiedLoopResult(iterArg);
      OpResult newLoopResult = loopLike.getLoopResults()->back();
      rewriter.moveOpBefore(extraction, loopLike);

      // Hoist the extraction/insertion ops.
      extraction.getSourceOperand().set(
          loopLike.getTiedLoopInit(iterArg)->get());

      // Clone the insertion to outside the not removing the final insertion, as
      // it still can be used by other extraction ops loop.
      rewriter.setInsertionPointAfter(loopLike);
      SubsetInsertionOpInterface newInsertion =
          cast<SubsetInsertionOpInterface>(
              rewriter.clone(*insertion.getOperation()));

      rewriter.replaceAllUsesWith(loopResult,
                                  newInsertion.getUpdatedDestination());
      newInsertion.getSourceOperand().set(newLoopResult);

      // loopLike changed, restart the check.
      changed = true;
      break;
    }
  }

  return loopLike;
}

/// The task of loop invariant subset hoisting as transformation is to find
/// a subset being used by a loop, which is "loop invariant", i.e. the loop
/// always works on that subset, instead of the whole set. Example:
///
/// for %i = 0 to 128 iter_args(%t = %init) {
///   %a = extract_slice %t[0, 0][8, 8]
///   %b = extract_slice %t2[0, %i][8, 8]
///   %c = add %a, %b
///   %out = %insert_slice %t[0, 0][8, 8]
///   yield %out
/// }
///
/// In this example, the loop is only operating on a loop invariant subset
/// of %t, which allows us to hoist out the extract_slice/insert_slice out
/// of the loop, and pass the subset as an iter_arg.
///
/// %a = extract_slice %init[0, 0][8, 8]
/// %loop = for %i = 0 to 128 iter_args(%t = %a) {
///   %b = extract_slice %t2[0, %i][8, 8]
///   %c = add %t, %b
///   yield %c
/// }
/// %out = insert_slice %loop into %init[0, 0][8, 8]
///
/// This hoisting only works when we are working on the same subset of the same
/// tensor, because the complement of the subset could have been updated,
/// but we don't know about it, so we need to preserve it.
///
/// However, if the destination of the insertion is a loop invariant tensor,
/// we do not need to preserve the complement of the subset, so we can still do
/// the hoisting. Example:
///
/// for %i = 0 to 128 iter_args(%t = %init) {
///   %a = extract_slice %t[0, 0][8, 8]
///   %b = extract_slice %t2[0, %i][8, 8]
///   %c = add %a, %b
///   %out = %insert_slice %init2[0, 0][8, 8]
///   yield %out
/// }
///
/// %a = extract_slice %init[0, 0][8, 8]
/// %loop = for %i = 0 to 128 iter_args(%t = %a) {
///   %b = extract_slice %t2[0, %i][8, 8]
///   %c = add %t, %b
///   yield %c
/// }
/// %out = insert_slice %loop into %init2[0, 0][8, 8]
///
/// The function implements the later transformation.
///
/// TODO (Groverkss): Improve upstream subset hoisting to account for this. I
/// think there is a more general way to handle this.
void hoistSubsetWithLoopInvariantTensor(RewriterBase &rewriter,
                                        LoopLikeOpInterface loopLike) {
  for (int64_t i = 0;
       i < static_cast<int64_t>(loopLike.getRegionIterArgs().size()); ++i) {
    loopLike = hoistLoopInvariantSubsetAtIterArg(rewriter, loopLike, i);
  }
}

void moveLoopInvariantCodeFromGenericOps(Operation *op) {
  // linalg.generic operations are also loop-like, but they don't have
  // LoopLikeOpInterface implemented for them.
  op->walk([&](linalg::GenericOp genericOp) {
    moveLoopInvariantCode(
        &genericOp.getBodyRegion(),
        [&](Value value, Region *) {
          return !genericOp->isAncestor(value.getParentRegion()->getParentOp());
        },
        [&](Operation *op, Region *) {
          return !isa<linalg::IndexOp>(op) && isMemoryEffectFree(op) &&
                 isSpeculatable(op);
        },
        [&](Operation *op, Region *) { op->moveBefore(genericOp); });
  });
}

namespace {
struct CastLikeExtractSliceOpFolder final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!tensor::isCastLikeExtractSliceOp(sliceOp) ||
        sliceOp.getSourceType() != sliceOp.getResultType()) {
      return failure();
    }
    rewriter.replaceOp(sliceOp, sliceOp.getSource());
    return success();
  }
};

struct CastLikeInsertSliceOpFolder final
    : OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!tensor::isCastLikeInsertSliceOp(sliceOp) ||
        sliceOp.getSourceType() != sliceOp.getResultType()) {
      return failure();
    }
    rewriter.replaceOp(sliceOp, sliceOp.getSource());
    return success();
  }
};

/// Folds IR resembling:
/// ```
///   %20 = vector.transfer_write %19, %16[%c0], %17 {in_bounds = [true]}
//      : vector<128xf16>, tensor<?xf16>
//    %21 = vector.transfer_read %20[%c0], %cst_2, %17
///     : tensor<?xf16>, vector<128xf16>
/// ```
/// into a simpler masked vector.transfer_read.
/// After bufferization, this generally removes the need for materializing the
/// write to memory.
// TODO: Consider upstreaming
struct FoldMaskedTransferRAW : OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    // Fail to match if the read doesn't have pure tensor semantics.
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    // Try to get the producing write op.
    auto writeOp =
        dyn_cast_or_null<vector::TransferWriteOp>(op.getBase().getDefiningOp());
    // Fail to match if the write doesn't have pure tensor semantics.
    if (!writeOp || !writeOp.hasPureTensorSemantics()) {
      return failure();
    }

    Value valToStore = writeOp.getValueToStore();
    // Fail to match if the in/out types are different
    if (valToStore.getType() != op.getType()) {
      return failure();
    }

    // Work only with trivial or equal indices.
    if ((llvm::any_of(op.getIndices(),
                      [](Value v) { return !isZeroInteger(v); }) ||
         llvm::any_of(writeOp.getIndices(),
                      [](Value v) { return !isZeroInteger(v); })) &&
        (op.getIndices() != writeOp.getIndices()))
      return failure();

    // Work only with minor identity mappings.
    if (!op.getPermutationMap().isMinorIdentity() ||
        !writeOp.getPermutationMap().isMinorIdentity()) {
      return failure();
    }

    TypedValue<VectorType> wMask = writeOp.getMask();
    Value rPad = op.getPadding();

    // Match only if the write and read op are masked and have the same mask.
    if (!wMask || (wMask != op.getMask())) {
      return failure();
    }

    // NOTE[FoldMaskedTransferRAW]: since masking is not supported on shaped
    // types with vector element types (see `verifyTransferOp` in upstream MLIR
    // VectorOps.cpp), and the write op has a mask, it can be assumed `rPad`
    // never has a vector type. But for sanity add an assert in case things
    // change upstream.
    assert(!isa<VectorType>(rPad.getType()) &&
           "search `NOTE[FoldMaskedTransferRAW]` in "
           "GenericVectorization.cpp::FoldMaskedTransferRAW for information");

    // Materialize the padding with a constant.
    auto padVal = rewriter.create<vector::SplatOp>(rPad.getLoc(),
                                                   valToStore.getType(), rPad);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, wMask, valToStore, padVal);
    return success();
  }
};
} // namespace

// Find the earliest insertion point in the block for the given operation.
static Operation *getEarliestInsertionPointInsideBlock(Block *block,
                                                       Operation *op) {

  Operation *currInsertionPoint = &(*block->getOperations().begin());
  DominanceInfo dominanceInfo(currInsertionPoint);

  for (auto operand : op->getOperands()) {
    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      continue;
    }
    Operation *defOp = operand.getDefiningOp();
    if (!dominanceInfo.dominates(defOp, currInsertionPoint)) {
      currInsertionPoint = defOp;
    }
  }
  return currInsertionPoint;
}

void OptimizeTensorInsertExtractSlicesPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(funcOp->getContext());

  // TODO: This is a temporary hack enabled for bufferization to
  // get rid of empty buffers.
  // Tracked here: https://github.com/llvm/llvm-project/issues/122869
  funcOp.walk([&](tensor::ExtractSliceOp extractSliceOp) {
    Block *currBlock = extractSliceOp.getOperation()->getBlock();
    auto latestInsertionPoint =
        getEarliestInsertionPointInsideBlock(currBlock, extractSliceOp);
    extractSliceOp->moveAfter(latestInsertionPoint);
  });

  funcOp.walk([&](scf::ForOp forOp) { moveLoopInvariantCode(forOp); });
  LDBG("after hoisting loop invariant code\n" << funcOp);

  moveLoopInvariantCodeFromGenericOps(funcOp);
  LDBG("after hoisting loop invariant code out of generic ops\n" << funcOp);

  // TODO: walking in some reverse / inside-out order would be more efficient
  // and would capture more cases.
  funcOp.walk(
      [&](scf::ForOp forOp) { hoistLoopInvariantSubsets(rewriter, forOp); });
  LDBG("after hoisting loop invariant subsets\n" << funcOp);

  funcOp.walk([&](scf::ForOp forOp) {
    hoistSubsetWithLoopInvariantTensor(rewriter, forOp);
  });
  LDBG("after hoisting subset loop invariant tensors" << funcOp);

  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateVectorTransferTensorSliceTransforms(patterns);
  scf::ForOp::getCanonicalizationPatterns(patterns, context);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
  if (foldIdentitySlices) {
    patterns.add<CastLikeExtractSliceOpFolder>(context);
    patterns.add<CastLikeInsertSliceOpFolder>(context);
  }
  // Apply masked transfer_write + transfer_read folding to avoid spurious
  // (future) roundtrips to memory.
  // TODO: consider upstreaming.
  patterns.add<FoldMaskedTransferRAW>(context);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  LDBG("after folding tensor.extract_slice and vector.transfer_read Ops \n"
       << funcOp);
}

} // namespace
} // namespace mlir::iree_compiler
