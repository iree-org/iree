// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir {
namespace iree_compiler {

// Return true if all the uses of op are either Store/transfer_write.
// There can be SubviewOp users as long as all its users are also
// StoreOp/transfer_write. If return true it also fills out the uses, if it
// returns false uses is unchanged.
static bool allUsesAreStores(Operation* op, std::vector<Operation*>& uses) {
  std::vector<Operation*> opUses;
  for (OpOperand& use : op->getUses()) {
    Operation* useOp = use.getOwner();
    if (isa<vector::TransferWriteOp, memref::StoreOp>(useOp) ||
        (isa<memref::SubViewOp>(useOp) && allUsesAreStores(useOp, opUses))) {
      opUses.push_back(useOp);
      continue;
    }
    return false;
  }
  uses.insert(uses.end(), opUses.begin(), opUses.end());
  return true;
}

// Track temporary allocations that are never read from. If this is the case
// it means both the allocations and associated stores can be removed.
static void eraseDeadAllocAndStores(func::FuncOp funcOp) {
  std::vector<Operation*> opToErase;
  funcOp.walk([&](memref::AllocOp op) {
    if (allUsesAreStores(op, opToErase)) {
      opToErase.push_back(op.getOperation());
    }
  });
  for (Operation* op : opToErase) {
    op->erase();
  }
}

namespace {

// Pattern to canonialize tranpose where only one dimension is not unit
// dimension. In this case the transpose is a no-op and should be simplified
// before getting to the conversion to llvm/spirv.
// TODO(thomasraoux): This should be moved in
// `populateCastAwayVectorLeadingOneDimPatterns` but might need more discussion
// on the semantic of transpose in this case.
class TransposeUnitDimToShapeCast
    : public OpRewritePattern<vector::TransposeOp> {
 public:
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter& rewriter) const override {
    unsigned numNonUnitSrcDim =
        llvm::count_if(op.getSourceVectorType().getShape(),
                       [](int64_t dim) { return dim != 1; });
    if (numNonUnitSrcDim > 1) return failure();
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        op, op.getResultVectorType(), op.getVector());
    return success();
  }
};

static void loopInvariantCodeMotion(func::FuncOp funcOp) {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  funcOp.walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
}

struct OptimizeVectorTransferPass
    : public OptimizeVectorTransferBase<OptimizeVectorTransferPass> {
  OptimizeVectorTransferPass(bool flatten) : flatten(flatten) {}
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    // Generate vector.shape_cast for dropping leading one dimensions in vector
    // ops. This increases the chance that we can forward more transfer writes
    // to transfer reads.
    {
      RewritePatternSet patterns(&getContext());
      mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, &getContext());
      patterns.add<TransposeUnitDimToShapeCast>(&getContext());
      mlir::vector::
          populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
              patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Workaround, run loop invariant code motion before hoist redudant vector
    // transfer to workaround a bug upstream.
    // TODO(thomasraoux): Remove it once the fix is merged.
    loopInvariantCodeMotion(funcOp);
    linalg::hoistRedundantVectorTransfers(funcOp);
    IRRewriter rewriter(funcOp->getContext());
    vector::transferOpflowOpt(rewriter, funcOp);

    // Move bitcast inwards from loop region boundaries to increase chances to
    // cancel them.
    {
      RewritePatternSet patterns(&getContext());
      vector::populateBubbleVectorBitCastOpPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Second stage of patterns to flatten transfer ops.
    if (flatten) {
      RewritePatternSet patterns(&getContext());
      mlir::vector::populateVectorTransferDropUnitDimsPatterns(patterns);
      mlir::vector::populateFlattenVectorTransferPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    // Delete potential dead alloc and associated ops after store to load
    // forwarding.
    eraseDeadAllocAndStores(funcOp);
  }
  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    // `flatten` may have been set to `true` in the constructor already.
    // The |= is so we preserve that rather than overwrite it with the default
    // value `false` of `optionFlatten`.
    flatten |= optionFlatten;
    return success();
  }

 private:
  bool flatten;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeVectorTransferPass(
    bool flatten) {
  return std::make_unique<OptimizeVectorTransferPass>(flatten);
}

}  // namespace iree_compiler
}  // namespace mlir
