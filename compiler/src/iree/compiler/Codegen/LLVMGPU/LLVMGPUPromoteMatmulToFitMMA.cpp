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
  explicit LLVMGPUPromoteMatmulToFitMMAPass(
      const LLVMGPUMatmulPadOption &option) {
    this->targetDimensions.setValue(option);
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }

  void padWithZeroValue(RewriterBase &rewriter, linalg::LinalgOp op,
                        utils::IteratorType targetIterType, bool nofold) const {
    LLVM_DEBUG(llvm::dbgs() << "candidate: " << op << "\n");
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    SmallVector<int64_t> paddingDims;
    for (auto [index, iterType] : llvm::enumerate(op.getIteratorTypesArray())) {
      if (iterType == targetIterType) {
        paddingDims.push_back(index);
      }
    }

    SmallVector<bool> packPaddings(op.getNumDpsInputs(), nofold);

    // One is enough because they will essentially be padded to corresponding
    // tile sizes, which should be multiple of MMA shapes.
    SmallVector<int64_t> padToMultipleOf(paddingDims.size(), 1);
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
            .setPackPaddings(packPaddings)
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

    // Preserve the innermost tensor.pad ops (i.e., pad for reduction dims), so
    // we can kick canonicalization patterns to fold outer tensor.pad ops away.
    bool nofold = false;
    utils::IteratorType targetIterType = utils::IteratorType::parallel;
    switch (targetDimensions) {
    case LLVMGPUMatmulPadOption::ParallelDims:
      LLVM_DEBUG(llvm::dbgs() << "padding parallel dims\n");
      targetIterType = utils::IteratorType::parallel;
      nofold = false;
      break;
    case LLVMGPUMatmulPadOption::ReductionDims:
      LLVM_DEBUG(llvm::dbgs() << "padding reduction dims\n");
      targetIterType = utils::IteratorType::reduction;
      nofold = true;
      break;
    default: // Unreachable.
      assert(false);
      break;
    };

    SmallVector<linalg::LinalgOp> candidates;
    funcOp->walk([&](linalg::LinalgOp op) {
      if (linalg::isaContractionOpInterface(op)) {
        candidates.push_back(op);
      }
    });

    IRRewriter rewriter(ctx);
    for (auto op : candidates) {
      padWithZeroValue(rewriter, op, targetIterType, nofold);
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

    // XXX(hanchung): This is needed for pad op fusion, which will remove
    // outer pad ops. I.e., it mainly wants to remove first pad op in the
    // pad->extract_slice->pad chain, while the canonicalization pattern can
    // only recognize slice->pad->slice->pad.
    {
      SmallVector<tensor::PadOp> padOps;
      funcOp.walk([&](tensor::PadOp op) { padOps.push_back(op); });
      for (auto op : padOps) {
        auto srcExtractSliceOp =
            op.getSource().getDefiningOp<tensor::ExtractSliceOp>();
        if (!srcExtractSliceOp) {
          continue;
        }
        auto producerPadOp =
            srcExtractSliceOp.getSource().getDefiningOp<tensor::PadOp>();
        if (!producerPadOp) {
          continue;
        }
        auto src = producerPadOp.getSource()
                       .getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
        if (!src) {
          continue;
        }

        rewriter.setInsertionPointAfter(src);
        SmallVector<OpFoldResult> sizes =
            tensor::getMixedSizes(rewriter, op.getLoc(), src);
        SmallVector<OpFoldResult> offsets(sizes.size(),
                                          rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> strides(sizes.size(),
                                          rewriter.getIndexAttr(1));
        auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
            op.getLoc(), src.getResult(), offsets, sizes, strides);
        rewriter.startOpModification(op);
        producerPadOp.getSourceMutable().assign(extractSliceOp.getResult());
        rewriter.finalizeOpModification(op);
      }

      RewritePatternSet patterns(ctx);
      tensor::PadOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUPromoteMatmulToFitMMAPass(LLVMGPUMatmulPadOption option) {
  return std::make_unique<LLVMGPUPromoteMatmulToFitMMAPass>(option);
}

} // namespace mlir::iree_compiler
