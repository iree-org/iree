// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-decompose-pack-unpack-ops"

namespace mlir {
namespace iree_compiler {
namespace {

struct LowerPackPattern : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::LowerPackResult> res = linalg::lowerPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to pad + expand + transpose");
    }
    return success();
  }
};

struct LowerUnPackPattern : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::LowerUnPackOpResult> res =
        linalg::lowerUnPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to pad + expand + transpose");
    }
    return success();
  }
};

struct GeneralizeTransposeOp : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    FailureOr<linalg::GenericOp> res =
        linalg::generalizeNamedOp(rewriter, linalgOp);
    if (failed(res)) return failure();
    return success();
  }
};

struct FoldExpandUnitShapeIntoExtractSliceOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto expandShape = op.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandShape) return failure();
    auto unitDims = op.getDroppedDims();
    for (auto indices : expandShape.getReassociationIndices()) {
      int nonUnit = 0;
      for (auto idx : indices)
        if (!unitDims.test(idx))
          nonUnit++;
      if (nonUnit != 1) return failure();
    }
    rewriter.replaceOp(op, expandShape.getSrc());
    return success();
  }
};

struct DecomposePackUnPackOpsPass
    : public DecomposePackUnPackOpsBase<DecomposePackUnPackOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, func::FuncDialect, arith::ArithDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void DecomposePackUnPackOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  {
    RewritePatternSet patterns(ctx);
    patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern,
                 linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(ctx, 10);
    patterns.add<LowerPackPattern, LowerUnPackPattern, GeneralizeTransposeOp>(
        ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<FoldExpandUnitShapeIntoExtractSliceOp>(ctx);
    linalg::populateFoldUnitExtentDimsViaSlicesPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createDecomposePackUnPackOpsPass() {
  return std::make_unique<DecomposePackUnPackOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
