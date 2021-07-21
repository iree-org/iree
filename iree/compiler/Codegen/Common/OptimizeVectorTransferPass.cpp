// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
static void eraseDeadAllocAndStores(FuncOp funcOp) {
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
    unsigned numNonUnitSrcDim = llvm::count_if(
        op.getVectorType().getShape(), [](int64_t dim) { return dim != 1; });
    if (numNonUnitSrcDim > 1) return failure();
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getResultType(),
                                                     op.vector());
    return success();
  }
};

struct OptimizeVectorTransferPass
    : public OptimizeVectorTransferBase<OptimizeVectorTransferPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    // Generate vector.shape_cast for dropping leading one dimensions in vector
    // ops. This increases the chance that we can forward more transfer writes
    // to transfer reads.
    OwningRewritePatternList patterns(&getContext());
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    patterns.add<TransposeUnitDimToShapeCast>(&getContext());
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    vector::transferOpflowOpt(funcOp);
    // Delete potential dead alloc and associated ops after store to load
    // forwarding.
    eraseDeadAllocAndStores(funcOp);
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createOptimizeVectorTransferPass() {
  return std::make_unique<OptimizeVectorTransferPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
