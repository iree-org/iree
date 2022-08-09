// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/ConvertTensorToFlow.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
struct ConvertTensorToFlowPass
    : public ConvertTensorToFlowBase<ConvertTensorToFlowPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Flow::FlowDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = getOperation()->getContext();
    RewritePatternSet convertToFlowPatterns(ctx);
    populateTensorToFlowConversionPatterns(ctx, convertToFlowPatterns);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(
        convertToFlowPatterns);
    IREE::Flow::TensorReshapeOp::getCanonicalizationPatterns(
        convertToFlowPatterns, ctx);
    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(convertToFlowPatterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createConvertTensorToFlowPass() {
  return std::make_unique<ConvertTensorToFlowPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
