// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-peel"

namespace mlir {
namespace iree_compiler {
namespace {
// Gathers tiled loops that aren't distribution loops from previous tiling
// stages.
void collectLoopsToPeel(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                        SmallVectorImpl<scf::ForOp> &loopsToPeel) {
  if (!iree_compiler::getLoweringConfig(linalgOp)) return;
  if (!linalgOp) return;

  auto maxNumLoopsToPeel = linalgOp.getNumLoops();
  Operation *currentOp = linalgOp;
  for (int i = 0; i < maxNumLoopsToPeel; ++i) {
    currentOp = currentOp->getParentOfType<scf::ForOp>();
    auto loop = llvm::cast_or_null<scf::ForOp>(currentOp);
    if (!loop || iree_compiler::isTiledAndDistributedLoop(loop)) {
      break;
    }
    loopsToPeel.push_back(loop);
  }

  std::reverse(loopsToPeel.begin(), loopsToPeel.end());
}

class LLVMCPUPeelPass : public LLVMCPUPeelBase<LLVMCPUPeelPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    scf::SCFDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUPeelPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  SmallVector<linalg::LinalgOp> candidates;
  funcOp.walk([&](linalg::LinalgOp op) { candidates.push_back(op); });
  for (auto linalgOp : candidates) {
    LLVM_DEBUG(llvm::dbgs() << "candidate: " << linalgOp << "\n");

    IRRewriter rewriter(context);
    IRRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(linalgOp);

    SmallVector<scf::ForOp> loopsToPeel;
    collectLoopsToPeel(rewriter, linalgOp, loopsToPeel);
    linalg::peelLoops(rewriter, loopsToPeel);
  }

  RewritePatternSet patterns(context);
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
    return signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUPeelPass() {
  return std::make_unique<LLVMCPUPeelPass>();
}
}  // namespace iree_compiler
}  // namespace mlir
