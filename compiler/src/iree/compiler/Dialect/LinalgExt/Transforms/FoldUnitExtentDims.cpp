// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_FOLDUNITEXTENTDIMSPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct FoldUnitExtentDimsPass final
    : impl::FoldUnitExtentDimsPassBase<FoldUnitExtentDimsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::LinalgExt::IREELinalgExtDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

} // namespace

void FoldUnitExtentDimsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  linalg::ControlDropUnitDims options;
  IREE::LinalgExt::populateFoldUnitExtentDimsPatterns(patterns, options);
  walkAndApplyPatterns(getOperation(), std::move(patterns));
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
