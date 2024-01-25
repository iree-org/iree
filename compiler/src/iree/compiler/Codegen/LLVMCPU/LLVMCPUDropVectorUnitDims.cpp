// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-drop-vector-unit-dims"

namespace mlir::iree_compiler {
namespace {
class LLVMCPUDropVectorUnitDimsPass
    : public LLVMCPUDropVectorUnitDimsBase<LLVMCPUDropVectorUnitDimsPass> {
public:
  using LLVMCPUDropVectorUnitDimsBase::LLVMCPUDropVectorUnitDimsBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUDropVectorUnitDimsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  // Apply transfer ops write to read forwarding and dead transfer write
  // optimizations.
  IRRewriter rewriter(ctx);
  vector::transferOpflowOpt(rewriter, funcOp);

  RewritePatternSet patterns(ctx);
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  vector::populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
      patterns);
  vector::populateVectorTransferDropUnitDimsPatterns(patterns);
  vector::populateDropUnitDimWithShapeCastPatterns(patterns);
  vector::InsertOp::getCanonicalizationPatterns(patterns, ctx);
  vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUDropVectorUnitDimsPass() {
  return std::make_unique<LLVMCPUDropVectorUnitDimsPass>();
}

} // namespace mlir::iree_compiler
