// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
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

void OptimizeTensorInsertExtractSlicesPass::runOnOperation() {
  auto funcOp = getOperation();
  linalg::hoistRedundantVectorTransfers(funcOp);
  IRRewriter rewriter(funcOp->getContext());
  // TODO: walking in some reverse / inside-out order would be more efficient
  // and would capture more cases.
  funcOp.walk(
      [&](scf::ForOp forOp) { hoistLoopInvariantSubsets(rewriter, forOp); });
  vector::transferOpflowOpt(rewriter, funcOp);
  MLIRContext *context = &getContext();

  LDBG("after hoisting redundant transfers on tensors\n" << funcOp);

  RewritePatternSet patterns(context);
  populateVectorTransferTensorSliceTransforms(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  LDBG("after folding tensor.extract_slice and vector.transfer_read Ops \n"
       << funcOp);
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createOptimizeTensorInsertExtractSlicesPass() {
  return std::make_unique<OptimizeTensorInsertExtractSlicesPass>();
}

} // namespace mlir::iree_compiler
