// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVWebGPUTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {
namespace {
struct WGSLExpandExtendedMultiplicationPass final
    : WGSLExpandExtendedMultiplicationBase<
          WGSLExpandExtendedMultiplicationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    spirv::populateSPIRVExpandExtendedMultiplicationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createWGSLExpandExtendedMultiplicationPass() {
  return std::make_unique<WGSLExpandExtendedMultiplicationPass>();
}

}  // namespace mlir::iree_compiler
