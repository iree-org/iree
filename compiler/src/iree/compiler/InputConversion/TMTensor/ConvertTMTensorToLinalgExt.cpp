// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/InputConversion/TMTensor/PassDetail.h"
#include "iree/compiler/InputConversion/TMTensor/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"

namespace mlir {
namespace iree_compiler {
namespace TMTensor {

namespace {

template <typename SrcOpTy, typename TargetOpTy>
struct TMTensorOpConversion : public OpRewritePattern<SrcOpTy> {
  using OpRewritePattern<SrcOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOpTy srcOp,
                                PatternRewriter &rewriter) const override {
    OperationState state(srcOp->getLoc(), TargetOpTy::getOperationName(),
                         srcOp->getOperands(), srcOp->getResultTypes(),
                         srcOp->getAttrs(), srcOp->getSuccessors());
    for (Region &srcRegion : srcOp->getRegions()) {
      Region *targetRegion = state.addRegion();
      rewriter.inlineRegionBefore(srcRegion, *targetRegion,
                                  targetRegion->begin());
    }
    Operation *targetOp = rewriter.create(state);
    rewriter.replaceOp(srcOp, targetOp->getResults());
    return success();
  }
};

struct ScatterOpConversion
    : public OpRewritePattern<mlir::torch::TMTensor::ScatterOp> {
  using OpRewritePattern<mlir::torch::TMTensor::ScatterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::torch::TMTensor::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto indicesTy = op.getIndicesType();
    if (!indicesTy.hasRank()) return failure();

    if (indicesTy.isDynamicDim(indicesTy.getRank() - 1)) {
      return rewriter.notifyMatchFailure(op, "number of indices is unknown");
    }

    auto numIndices = indicesTy.getShape().back();
    llvm::SmallVector<int64_t> dimMap(numIndices);
    for (int i = 0; i < numIndices; i++) dimMap[i] = i;

    auto scatterOp = rewriter.create<IREE::LinalgExt::ScatterOp>(
        op.getLoc(), op->getResultTypes(), op.getInputs(), op.getOutputs(),
        dimMap, op.getUniqueIndices());

    rewriter.inlineRegionBefore(op.getRegion(), scatterOp.getRegion(),
                                scatterOp.getRegion().begin());
    rewriter.replaceOp(op, scatterOp->getResults());
    return success();
  }
};
}  // namespace

namespace {

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertTMTensorToLinalgExtPass
    : public ConvertTMTensorToLinalgExtBase<ConvertTMTensorToLinalgExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

#define INSERT_TMTENSOR_CONVERSION_PATTERN(Op)                               \
  patterns.add<                                                              \
      TMTensorOpConversion<mlir::torch::TMTensor::Op, IREE::LinalgExt::Op>>( \
      context);

    INSERT_TMTENSOR_CONVERSION_PATTERN(YieldOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(ScanOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(SortOp);

#undef INSERT_TMTENSOR_CONVERSION_PATTERN

    patterns.add<ScatterOpConversion>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTMTensorToLinalgExtPass() {
  return std::make_unique<ConvertTMTensorToLinalgExtPass>();
}

}  // namespace TMTensor
}  // namespace iree_compiler
}  // namespace mlir
