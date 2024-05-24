// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_FUSEANDHOISTPARALLELLOOPSPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct FuseAndHoistParallelLoopsPass final
    : impl::FuseAndHoistParallelLoopsPassBase<FuseAndHoistParallelLoopsPass> {
  void runOnOperation() override;
};
} // namespace

struct FuseForalls final : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto sliceParent = sliceOp->getParentOfType<scf::ForallOp>();
    if (!sliceParent) {
      return failure();
    }

    auto producerForall = sliceOp.getSource().getDefiningOp<scf::ForallOp>();
    if (!producerForall) {
      return failure();
    }

    // TODO: Allow extracting multiple uses within the same consumer loop. Still
    // single producer single consumer loop, but multiple uses within the
    // consumer.
    if (!producerForall->hasOneUse()) {
      return failure();
    }

    return fuseForallIntoSlice(rewriter, producerForall, sliceParent, sliceOp);
  }
};

void FuseAndHoistParallelLoopsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);

  // These two patterns are run to a fixed point, allowing fusion within
  // potentially nested loops, hoisting from said loops, and continued fusion.
  patterns.add<FuseForalls>(context);
  populateForallLoopHoistingPattern(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
