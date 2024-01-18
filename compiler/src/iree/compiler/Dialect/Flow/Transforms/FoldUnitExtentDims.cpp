// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- FoldUnitExtentDims.cpp - Pass to fold unit extent dims of tensors -===//
//
// Light weight wrapper to call the patterns to fold unit extent dims with
// IREE control.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {
struct FoldUnitExtentDimsPass
    : public FoldUnitExtentDimsBase<FoldUnitExtentDimsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void FoldUnitExtentDimsPass::runOnOperation() {
  Operation *funcOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet foldUnitDimsPatterns(context);
  linalg::ControlDropUnitDims options;
  auto defaultFn = options.controlFn;
  options.controlFn = [&](Operation *op) {
    // Ignore operations already in dispatches.
    if (!isNonNullAndOutsideDispatch(op)) {
      return SmallVector<unsigned>{};
    }
    return defaultFn(op);
  };
  linalg::populateFoldUnitExtentDimsPatterns(foldUnitDimsPatterns, options);
  linalg::populateMoveInitOperandsToInputPattern(foldUnitDimsPatterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp,
                                          std::move(foldUnitDimsPatterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFoldUnitExtentDimsPass() {
  return std::make_unique<FoldUnitExtentDimsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
