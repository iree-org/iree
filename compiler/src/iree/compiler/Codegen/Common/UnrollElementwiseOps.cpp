// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

namespace {

// Unrolls N-D elementwise vector operations to 1-D by iterating over all
// leading dimensions and extracting/inserting 1-D subvectors.
struct UnrollElementwiseOps final : RewritePattern {
  UnrollElementwiseOps(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op) ||
        op->getNumResults() != 1) {
      return failure();
    }

    Location loc = op->getLoc();
    VectorType dstVecTy = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!dstVecTy || dstVecTy.getRank() <= 1) {
      return failure();
    }
    ArrayRef<int64_t> originalSize = dstVecTy.getShape();

    Value result = ub::PoisonOp::create(rewriter, loc, dstVecTy);
    auto subVecTy =
        VectorType::get({originalSize.back()}, dstVecTy.getElementType());

    SmallVector<int64_t> tileShape(dstVecTy.getRank() - 1, 1);
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalSize.drop_back(), tileShape)) {
      SmallVector<Value> operands;
      for (Value val : op->getOperands()) {
        // Extract subvector if the operand is a vector. This is to handle
        // things like arith.select which take a scalar conditional but are
        // otherwise elementwise.
        if (isa<VectorType>(val.getType())) {
          val = vector::ExtractOp::create(rewriter, loc, val, offsets);
        }
        operands.push_back(val);
      }

      Operation *clonedOp = clone(rewriter, op, subVecTy, operands);
      Value subResult = clonedOp->getResult(0);
      result =
          vector::InsertOp::create(rewriter, loc, subResult, result, offsets);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateUnrollElementwiseOpsPatterns(RewritePatternSet &patterns) {
  patterns.add<UnrollElementwiseOps>(patterns.getContext());
}

} // namespace mlir::iree_compiler
