// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/LLVMCPUPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
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
void collectLoopsToPeel(RewriterBase &rewriter, Operation *op,
                        SmallVectorImpl<scf::ForOp> &loopsToPeel) {
  if (!iree_compiler::getLoweringConfig(op)) return;

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
    if (!loop || iree_compiler::isTiledAndDistributedLoop(loop)) break;
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
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp, tensor::PackOp>(op)) {
      candidates.push_back(op);
    }
  });
  for (auto op : candidates) {
    LLVM_DEBUG(llvm::dbgs() << "candidate: " << op << "\n");

    IRRewriter rewriter(context);
    IRRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(op);

    SmallVector<scf::ForOp> loopsToPeel;
    collectLoopsToPeel(rewriter, op, loopsToPeel);
    linalg::peelLoops(rewriter, loopsToPeel);
  }

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
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUPeelPass() {
  return std::make_unique<LLVMCPUPeelPass>();
}
}  // namespace iree_compiler
}  // namespace mlir
