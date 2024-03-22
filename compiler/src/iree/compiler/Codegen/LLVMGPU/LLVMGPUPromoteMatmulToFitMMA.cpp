// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-promote-matmul-to-fit-mma"

namespace mlir::iree_compiler {

namespace {

/// Fold `tensor.pad(cst, tensor.extract*(linalg.fill(cst)))` into
/// `linalg.fill(cst, empty)` when the padding constant and the fill constant
/// are the same.
/// This seems generally desirable as a folding but may be too intrusive, so we
/// only apply it selectively for now.
// TODO: atm hardcoded on linalg.fill but we could take any result of any
// generic that yields a constant in that result.
// TODO(hanchung): Refactor it to a common place. This is copied from
// Codegen/Common/TransformExtensions/CommonExtensions.cpp
struct FoldFillIntoPad : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const final {
    Operation *currentOp = padOp.getSource().getDefiningOp();
    auto maybeExtractSlice =
        dyn_cast_or_null<tensor::ExtractSliceOp>(currentOp);
    while (currentOp && maybeExtractSlice) {
      currentOp = maybeExtractSlice.getSource().getDefiningOp();
      maybeExtractSlice = dyn_cast_or_null<tensor::ExtractSliceOp>(currentOp);
    }
    auto fillOp = dyn_cast_or_null<linalg::FillOp>(currentOp);
    if (!fillOp) {
      return rewriter.notifyMatchFailure(
          padOp, "not coming from a linalg.fill op via tensor.extract_slice*");
    }

    Value padValue = padOp.getConstantPaddingValue();
    RankedTensorType resultType = padOp.getResultType();
    if (!padValue ||
        getAsOpFoldResult(padValue) !=
            getAsOpFoldResult(fillOp.getDpsInputOperand(0)->get())) {
      return rewriter.notifyMatchFailure(
          padOp, "not a constant value matching the fill value");
    }

    Location loc = padOp.getLoc();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(rewriter, loc, padOp),
        resultType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(padOp, padValue,
                                                emptyOp.getResult());

    return success();
  }
};

class LLVMGPUPromoteMatmulToFitMMAPass
    : public LLVMGPUPromoteMatmulToFitMMABase<
          LLVMGPUPromoteMatmulToFitMMAPass> {
private:
  LLVMGPUMatmulPadOption option = LLVMGPUMatmulPadOption::ParallelDims;

public:
  explicit LLVMGPUPromoteMatmulToFitMMAPass(LLVMGPUMatmulPadOption option)
      : option(option) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }

  void pad(RewriterBase &rewriter, linalg::LinalgOp op,
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

    FailureOr<IREE::Codegen::LoweringConfigAttr> loweringConfig =
        getLoweringConfig(op);
    if (succeeded(loweringConfig)) {
      options =
          options.setSmallestStaticBounds(loweringConfig->getTileSizeVals(0));
    }

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
    switch (option) {
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
      pad(rewriter, op, targetIterType, nofold);
    }

    {
      RewritePatternSet patterns(ctx);
      patterns.insert<FoldFillIntoPad>(ctx);
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
        auto src =
            op.getSource().getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
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
        op.getSourceMutable().assign(extractSliceOp.getResult());
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
