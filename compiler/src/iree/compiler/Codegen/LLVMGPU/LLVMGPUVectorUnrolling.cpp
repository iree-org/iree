// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORUNROLLINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

struct UnrollElementwisePattern : public RewritePattern {
  UnrollElementwisePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return failure();

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
      // Extract from each operand.
      SmallVector<Value> operands;
      for (Value val : op->getOperands()) {
        Value extracted =
            vector::ExtractOp::create(rewriter, loc, val, offsets);
        operands.push_back(extracted);
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

struct LLVMGPUVectorUnrollingPass final
    : impl::LLVMGPUVectorUnrollingPassBase<LLVMGPUVectorUnrollingPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // Unroll vector.transfer_read/write operations to 1-D vector operations.
    // Also try to lower them to vector.load/vector.store if possible.
    VectorTransferToSCFOptions vectorToSCFOptions;
    vectorToSCFOptions.enableFullUnroll();
    populateVectorToSCFConversionPatterns(patterns, vectorToSCFOptions);
    memref::populateFoldMemRefAliasOpPatterns(patterns);
    vector::populateVectorTransferLoweringPatterns(patterns);

    vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
    vector::InsertOp::getCanonicalizationPatterns(patterns, ctx);

    patterns.add<UnrollElementwisePattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace mlir::iree_compiler
