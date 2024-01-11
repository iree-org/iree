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

#define DEBUG_TYPE "iree-llvmcpu-vector-shape-cast-lowering"

namespace mlir::iree_compiler {
namespace {
class LLVMCPUVectorShapeCastLoweringPass
    : public LLVMCPUVectorShapeCastLoweringBase<
          LLVMCPUVectorShapeCastLoweringPass> {
public:
  using LLVMCPUVectorShapeCastLoweringBase::LLVMCPUVectorShapeCastLoweringBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVectorShapeCastLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  RewritePatternSet patterns(ctx);
  vector::populateVectorShapeCastLoweringPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUVectorShapeCastLoweringPass() {
  return std::make_unique<LLVMCPUVectorShapeCastLoweringPass>();
}

} // namespace mlir::iree_compiler
