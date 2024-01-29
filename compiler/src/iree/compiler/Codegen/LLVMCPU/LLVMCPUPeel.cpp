// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-peel"

namespace mlir::iree_compiler {

namespace {

// Gathers tiled loops that aren't distribution loops from previous tiling
// stages.
void collectLoopsToPeel(Operation *op,
                        llvm::SmallSetVector<scf::ForOp, 8> &loopsToPeel) {
  if (!iree_compiler::getLoweringConfig(op))
    return;

  int maxNumLoopsToPeel = TypeSwitch<Operation *, int>(op)
                              .Case<linalg::LinalgOp>([](auto linalgOp) {
                                return linalgOp.getNumLoops();
                              })
                              .Case<tensor::PackOp>([](auto packOp) {
                                return packOp.getSourceRank();
                              })
                              .Default([](auto) { return 0; });
  for (int i = 0; i < maxNumLoopsToPeel; ++i) {
    op = op->getParentOfType<scf::ForOp>();
    auto loop = llvm::cast_or_null<scf::ForOp>(op);
    if (!loop || iree_compiler::isTiledAndDistributedLoop(loop))
      break;

    LLVM_DEBUG(llvm::dbgs() << "Loop to peel:\n" << *op << "\n");
    loopsToPeel.insert(loop);
  }
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

  llvm::SmallSetVector<scf::ForOp, 8> uniqueLoopsToPeel;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp, tensor::PackOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Gather loops to peel from candidate op:\n"
                              << *op << "\n");
      collectLoopsToPeel(op, uniqueLoopsToPeel);
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Peeling loops\n");
  // Visiting the loops in outer-to-inner order will prevent loops nested in
  // partial iterations to be peeled again.
  SmallVector<scf::ForOp, 8> outerToInnerLoopsToPeel(uniqueLoopsToPeel.rbegin(),
                                                     uniqueLoopsToPeel.rend());
  IRRewriter rewriter(context);
  linalg::peelLoops(rewriter, outerToInnerLoopsToPeel);

  LLVM_DEBUG(llvm::dbgs() << "Canonicalizing loops\n");
  RewritePatternSet patterns(context);
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
    return signalPassFailure();
  }
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUPeelPass() {
  return std::make_unique<LLVMCPUPeelPass>();
}

} // namespace mlir::iree_compiler
