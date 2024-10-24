// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-promote-matmul-to-fit-mma"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUPROMOTEMATMULTOFITMMAPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

class LLVMGPUPromoteMatmulToFitMMAPass final
    : public impl::LLVMGPUPromoteMatmulToFitMMAPassBase<
          LLVMGPUPromoteMatmulToFitMMAPass> {
public:
  using impl::LLVMGPUPromoteMatmulToFitMMAPassBase<
      LLVMGPUPromoteMatmulToFitMMAPass>::LLVMGPUPromoteMatmulToFitMMAPassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }

  void padWithZeroValue(RewriterBase &rewriter, linalg::LinalgOp op,
                        ArrayRef<int64_t> padToMultipleOf) const {
    LLVM_DEBUG(llvm::dbgs() << "candidate: " << op << "\n");
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    SmallVector<int64_t> paddingDims =
        llvm::to_vector(llvm::seq<int64_t>(padToMultipleOf.size()));

    SmallVector<Attribute> paddingValueAttributes;
    for (auto &operand : op->getOpOperands()) {
      auto elemType = getElementTypeOrSelf(operand.get().getType());
      paddingValueAttributes.push_back(rewriter.getZeroAttr(elemType));
    }

    auto options =
        linalg::LinalgPaddingOptions()
            .setPaddingDimensions(paddingDims)
            .setPaddingValues(paddingValueAttributes)
            .setPadToMultipleOf(padToMultipleOf)
            .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);

    FailureOr<linalg::LinalgOp> result =
        linalg::padAndHoistLinalgOp(rewriter, op, options);
    if (failed(result)) {
      LLVM_DEBUG(llvm::dbgs() << "failed to pad op " << op << "\n");
    }
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();

    SmallVector<linalg::LinalgOp> candidates;
    funcOp->walk([&](linalg::LinalgOp op) {
      if (linalg::isaContractionOpInterface(op)) {
        candidates.push_back(op);
      }
    });

    IRRewriter rewriter(ctx);
    for (linalg::LinalgOp op : candidates) {
      auto config = dyn_cast_or_null<IREE::GPU::LoweringConfigAttr>(
          getLoweringConfig(op));
      if (!config) {
        continue;
      }

      SmallVector<int64_t> wgTiles = config.getStaticTilingLevelSizes(
          static_cast<unsigned>(IREE::GPU::TilingLevel::Workgroup), op);
      SmallVector<int64_t> redTiles = config.getStaticTilingLevelSizes(
          static_cast<unsigned>(IREE::GPU::TilingLevel::Reduction), op);

      // Populate padding dimensions to maximum of possible tile sizes.
      SmallVector<int64_t> padToMultipleOf(op.getNumLoops(), 1);
      for (auto [wgTile, redTile, padMultiple] :
           llvm::zip_equal(wgTiles, redTiles, padToMultipleOf)) {
        padMultiple = std::max({wgTile, redTile, padMultiple});
      }
      SmallVector<int64_t> paddingDimensions =
          llvm::to_vector(llvm::seq<int64_t>(op.getNumLoops()));

      padWithZeroValue(rewriter, op, padToMultipleOf);
    }

    {
      RewritePatternSet patterns(ctx);
      linalg::populateSwapExtractSliceWithFillPatterns(patterns);
      linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
      ctx->getLoadedDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      tensor::PadOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
