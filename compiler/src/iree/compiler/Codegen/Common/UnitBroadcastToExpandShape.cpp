// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {
#define GEN_PASS_DEF_UNITBROADCASTTOEXPANDSHAPEPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

struct UnitBroadCastToExpandShape final
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {

    if (!IREE::LinalgExt::isBroadcastingOp(genericOp)) {
      return failure();
    }
    auto loc = genericOp.getLoc();
    auto inputTensor = genericOp.getInputs().front();
    auto outputTensor = genericOp.getOutputs().front();
    auto outputType = cast<RankedTensorType>(outputTensor.getType());
    auto inputShape = cast<RankedTensorType>(inputTensor.getType()).getShape();
    auto outputShape =
        cast<RankedTensorType>(outputTensor.getType()).getShape();

    int broadcastRank = outputShape.size() - inputShape.size();
    assert(broadcastRank > 0 && "expected output rank > input rank");

    // Check that the shape of broadcasted dim is 1.
    for (int i = 0; i < broadcastRank; i++) {
      if (outputShape[i] != 1) {
        return failure();
      }
    }

    SmallVector<ReassociationIndices> reassociationIdx(inputShape.size());
    for (auto i = 0; i < outputShape.size(); i++) {
      if (i <= broadcastRank) {
        reassociationIdx.front().push_back(i);
      } else {
        reassociationIdx[i - broadcastRank].push_back(i);
      }
    }

    auto expandShapeOp = rewriter
                             .create<tensor::ExpandShapeOp>(
                                 loc, outputType, inputTensor, reassociationIdx)
                             .getResult();
    rewriter.replaceOp(genericOp, expandShapeOp);
    return success();
  }
};

struct UnitBroadCastToExpandShapePass
    : public impl::UnitBroadCastToExpandShapePassBase<
          UnitBroadCastToExpandShapePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto fn = getOperation();
    RewritePatternSet UnitBroadCastToExpandShapePatterns(&getContext());
    UnitBroadCastToExpandShapePatterns.insert<UnitBroadCastToExpandShape>(
        fn.getContext());
    if (failed(applyPatternsAndFoldGreedily(
            fn, std::move(UnitBroadCastToExpandShapePatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace mlir::iree_compiler
