// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-vector-bitcast-lowering"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUVECTORBITCASTLOWERINGPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {
class LLVMCPUVectorBitCastLoweringPass
    : public impl::LLVMCPUVectorBitCastLoweringPassBase<
          LLVMCPUVectorBitCastLoweringPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVectorBitCastLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  RewritePatternSet patterns(ctx);
  vector::populateVectorBitCastLoweringPatterns(patterns, /*targetRank=*/1);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
} // namespace
} // namespace mlir::iree_compiler
