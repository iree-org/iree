// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "llvm/Support/DebugLog.h"
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
  if (insertion->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }

  // Walk the nested operations and check that all used values are either
  // defined outside of the loop or in a nested region, but not at the level of
  // the loop body.
  auto walkFn = [&](Operation *child) {
    for (OpOperand &operand : child->getOpOperands()) {
      // Ignore values defined in a nested region.
      if (insertion->isAncestor(
              operand.get().getParentRegion()->getParentOp())) {
        continue;
      }
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

namespace {
struct CastLikeExtractSliceOpFolder final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;

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
  using Base::Base;

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

/// Folds a transfer_read that reads from the result of a transfer_write on
/// the same region (Read-After-Write) into arithmetic on the written value,
/// the original tensor, the masks, and the read's padding.
///
/// The general semantics are:
///
///   written_tensor[i] = wMask[i] ? valToStore[i] : original[i]
///   result[i]         = rMask[i] ? written_tensor[i] : rPad
///
/// Which gives:
///   result = select(rMask, select(wMask, valToStore, original),
///   broadcast(rPad))
///
/// Special cases avoid emitting unnecessary IR:
///   - No wMask (unmasked write): wMask is implicitly all-true, inner select
///     collapses to valToStore.
///   - No rMask (unmasked read): rMask is implicitly all-true, outer select
///     collapses away.
///   - wMask == rMask: the original tensor is never needed (anywhere rMask is
///     true, wMask is also true), so the inner select collapses to valToStore.
///
/// After bufferization, this generally removes the need for materializing the
/// write to memory.
// TODO: Consider upstreaming
struct FoldTransferRAW : OpRewritePattern<vector::TransferReadOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (!readOp.hasPureTensorSemantics()) {
      return failure();
    }

    auto writeOp = dyn_cast_if_present<vector::TransferWriteOp>(
        readOp.getBase().getDefiningOp());
    if (!writeOp || !writeOp.hasPureTensorSemantics()) {
      return failure();
    }

    Value valToStore = writeOp.getValueToStore();
    if (valToStore.getType() != readOp.getType()) {
      return failure();
    }

    // Work only with trivial or equal indices.
    if ((llvm::any_of(readOp.getIndices(),
                      [](Value v) { return !isZeroInteger(v); }) ||
         llvm::any_of(writeOp.getIndices(),
                      [](Value v) { return !isZeroInteger(v); })) &&
        (readOp.getIndices() != writeOp.getIndices())) {
      return failure();
    }

    if (!readOp.getPermutationMap().isMinorIdentity() ||
        !writeOp.getPermutationMap().isMinorIdentity()) {
      return failure();
    }

    TypedValue<VectorType> wMask = writeOp.getMask();
    TypedValue<VectorType> rMask = readOp.getMask();

    // Build the inner value: select(wMask, valToStore, original).
    // When wMask is absent (unmasked write) or wMask == rMask (original is
    // never accessed), this simplifies to just valToStore.
    Value inner = valToStore;
    bool needsOriginal = wMask && wMask != rMask;
    if (needsOriginal) {
      Value originalRead = vector::TransferReadOp::create(
          rewriter, readOp.getLoc(), readOp.getType(), writeOp.getBase(),
          readOp.getIndices(), readOp.getPermutationMap(), readOp.getPadding(),
          /*mask=*/Value(), readOp.getInBoundsAttr());
      inner = arith::SelectOp::create(rewriter, readOp.getLoc(), wMask,
                                      valToStore, originalRead);
    }

    if (!rMask) {
      rewriter.replaceOp(readOp, inner);
      return success();
    }

    // Build the outer value: select(rMask, inner, broadcast(rPad)).
    // When rMask is absent (unmasked read), the result is just inner.
    Value rPad = readOp.getPadding();
    assert(!isa<VectorType>(rPad.getType()) &&
           "masked transfers on vector element types are not supported; see "
           "verifyTransferOp in upstream MLIR VectorOps.cpp");
    Value padVal = vector::BroadcastOp::create(rewriter, rPad.getLoc(),
                                               valToStore.getType(), rPad);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(readOp, rMask, inner, padVal);
    return success();
  }
};

/// Folds transfer_read(tensor.empty).
///
/// Since tensor.empty has unspecified contents, reading from it produces
/// an unspecified value, which is exactly the semantics of ub.poison.
/// Out of bounds means that pad is used.
///
///   Case 1 — fully in-bounds, no mask:
///     %e = tensor.empty() : tensor<128xf16>
///     %r = vector.transfer_read %e[%c0], %pad {in_bounds = [true]}
///   ->
///     %r = ub.poison : vector<128xf16>
///
///   Case 2 — fully in-bounds, masked:
///     %e = tensor.empty() : tensor<128xf16>
///     %r = vector.transfer_read %e[%c0], %pad, %mask {in_bounds = [true]}
///   ->
///     %poison = ub.poison : vector<128xf16>
///     %bcast  = vector.broadcast %pad : f16 to vector<128xf16>
///     %r = arith.select %mask, %poison, %bcast
///
///   Case 3 — not fully in-bounds, no mask:
///     %e = tensor.empty() : tensor<100xf16>
///     %r = vector.transfer_read %e[%c0], %pad
///                              : tensor<100xf16>, vector<128xf16>
///   ->
///     %r = vector.broadcast %pad : f16 to vector<128xf16>
///
///   Case 4 — not fully in-bounds, masked:
///     %e = tensor.empty() : tensor<100xf16>
///     %r = vector.transfer_read %e[%c0], %pad, %mask
///                              : tensor<100xf16>, vector<128xf16>
///   ->
///     %r = vector.broadcast %pad : f16 to vector<128xf16>
struct FoldTransferReadOfEmptyTensor
    : OpRewritePattern<vector::TransferReadOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (!op.getBase().getDefiningOp<tensor::EmptyOp>()) {
      return failure();
    }

    if (!op.getPermutationMap().isMinorIdentity()) {
      return failure();
    }

    bool fullyInBounds =
        llvm::all_of(op.getInBoundsValues(), [](bool v) { return v; });
    TypedValue<VectorType> mask = op.getMask();

    if (mask && fullyInBounds) {
      // Masked, fully in-bounds: mask-on lanes read unspecified contents
      // (poison), mask-off lanes produce the padding value.
      Value rPad = op.getPadding();
      assert(!isa<VectorType>(rPad.getType()) &&
             "masked transfers on vector element types are not supported; "
             "see verifyTransferOp in upstream MLIR VectorOps.cpp");
      Value poison = ub::PoisonOp::create(rewriter, op.getLoc(), op.getType());
      Value padVal = vector::BroadcastOp::create(rewriter, rPad.getLoc(),
                                                 op.getType(), rPad);
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, poison, padVal);
      return success();
    }

    if (!mask && fullyInBounds) {
      // Unmasked, fully in-bounds: every lane reads unspecified contents.
      rewriter.replaceOp(
          op, ub::PoisonOp::create(rewriter, op.getLoc(), op.getType()));
      return success();
    }

    // Not fully in-bounds (with or without mask): out-of-bounds lanes
    // produce pad, and in-bounds lanes read unspecified contents from
    // tensor.empty, so we may choose pad for those too.
    Value rPad = op.getPadding();
    rewriter.replaceOp(op, vector::BroadcastOp::create(rewriter, rPad.getLoc(),
                                                       op.getType(), rPad));
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
  mlir::FunctionOpInterface funcOp = getOperation();
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

  moveLoopInvariantCodeFromGuaranteedLoops(funcOp);
  LDBG() << "after hoisting loop invariant code\n" << funcOp << "\n";

  // TODO: walking in some reverse / inside-out order would be more efficient
  // and would capture more cases.
  funcOp.walk(
      [&](scf::ForOp forOp) { hoistLoopInvariantSubsets(rewriter, forOp); });
  LDBG() << "after hoisting loop invariant subsets\n" << funcOp;

  funcOp.walk([&](scf::ForOp forOp) {
    hoistSubsetWithLoopInvariantTensor(rewriter, forOp);
  });
  LDBG() << "after hoisting subset loop invariant tensors" << funcOp;

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
  patterns.add<FoldTransferRAW, FoldTransferReadOfEmptyTensor>(context);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  LDBG() << "after folding tensor.extract_slice and vector.transfer_read Ops \n"
         << funcOp;
}

} // namespace
} // namespace mlir::iree_compiler
