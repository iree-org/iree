// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-codegen-optimize-tensor-insert-extract-slices"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

namespace {

class OptimizeTensorInsertExtractSlicesPass
    : public OptimizeTensorInsertExtractSlicesBase<
          OptimizeTensorInsertExtractSlicesPass> {
public:
  using OptimizeTensorInsertExtractSlicesBase::
      OptimizeTensorInsertExtractSlicesBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};

/// Checks whether the given op can be hoisted by checking that
/// - the op and none of its contained operations depend on values inside of the
///   loop (by means of calling definedOutside).
/// - the op has no side-effects.
static bool canBeHoisted(Operation *op,
                         function_ref<bool(OpOperand &)> condition) {
  // Do not move terminators.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;

  // Walk the nested operations and check that all used values are either
  // defined outside of the loop or in a nested region, but not at the level of
  // the loop body.
  auto walkFn = [&](Operation *child) {
    for (OpOperand &operand : child->getOpOperands()) {
      // Ignore values defined in a nested region.
      if (op->isAncestor(operand.get().getParentRegion()->getParentOp()))
        continue;
      if (!condition(operand))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  return !op->walk(walkFn).wasInterrupted();
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
  // Check if this insertion is loop invariant except it's source.
  // We would also be okay as long as the destination is loop invariant,
  // but we would have to do some cloning, so we don't do it here.
  if (!canBeHoisted(insertion, [&](OpOperand &operand) {
        return loopLike.isDefinedOutsideOfLoop(operand.get()) ||
               &operand == &insertion.getSourceOperand();
      })) {
    return loopLike;
  }

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
          // We don't care if they are operating on the same tensor.
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
    FailureOr<LoopLikeOpInterface> newLoop =
        loopLike.replaceWithAdditionalYields(
            rewriter, extraction.getResult(),
            /*replaceInitOperandUsesInLoop=*/true, newYieldValuesFn);
    if (failed(newLoop))
      return loopLike;
    loopLike = *newLoop;

    BlockArgument iterArg = loopLike.getRegionIterArgs()[idx];
    OpResult loopResult = loopLike.getTiedLoopResult(iterArg);
    OpResult newLoopResult = loopLike.getLoopResults()->back();
    rewriter.moveOpBefore(extraction, loopLike);

    // Hoist the extraction/insertion ops
    extraction.getSourceOperand().set(loopLike.getTiedLoopInit(iterArg)->get());

    // Clone the insertion to outside the not removing the final insertion, as
    // it still can be used by other extraction ops. loop.
    rewriter.setInsertionPointAfter(loopLike);
    SubsetInsertionOpInterface newInsertion = cast<SubsetInsertionOpInterface>(
        rewriter.clone(*insertion.getOperation()));

    rewriter.replaceAllUsesWith(loopResult,
                                newInsertion.getUpdatedDestination());
    newInsertion.getSourceOperand().set(newLoopResult);
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

void OptimizeTensorInsertExtractSlicesPass::runOnOperation() {
  auto funcOp = getOperation();
  linalg::hoistRedundantVectorTransfers(cast<func::FuncOp>(funcOp));
  IRRewriter rewriter(funcOp->getContext());
  // TODO: walking in some reverse / inside-out order would be more efficient
  // and would capture more cases.
  funcOp.walk(
      [&](scf::ForOp forOp) { hoistLoopInvariantSubsets(rewriter, forOp); });
  LDBG("after hoisting loop invariant subsets\n" << funcOp);

  funcOp.walk([&](scf::ForOp forOp) {
    hoistSubsetWithLoopInvariantTensor(rewriter, forOp);
  });
  LDBG("after hoisting subset loop invariant tensors" << funcOp);
  vector::transferOpflowOpt(rewriter, funcOp);
  MLIRContext *context = &getContext();

  LDBG("after hoisting redundant transfers on tensors\n" << funcOp);

  RewritePatternSet patterns(context);
  populateVectorTransferTensorSliceTransforms(patterns);
  scf::ForOp::getCanonicalizationPatterns(patterns, context);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  LDBG("after folding tensor.extract_slice and vector.transfer_read Ops \n"
       << funcOp);
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createOptimizeTensorInsertExtractSlicesPass() {
  return std::make_unique<OptimizeTensorInsertExtractSlicesPass>();
}

} // namespace mlir::iree_compiler
