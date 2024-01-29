// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-vector-transfer-lowering"

namespace mlir::iree_compiler {
namespace {
class LLVMCPUVectorTransferLoweringPass
    : public LLVMCPUVectorTransferLoweringBase<
          LLVMCPUVectorTransferLoweringPass> {
public:
  using LLVMCPUVectorTransferLoweringBase::LLVMCPUVectorTransferLoweringBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVectorTransferLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  {
    RewritePatternSet patterns(&getContext());
    mlir::vector::populateFlattenVectorTransferPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorTransferLoweringPatterns(patterns,
                                                   /*maxTransferRank=*/1);
    auto vectorTransferToSCFOptions =
        VectorTransferToSCFOptions().enableFullUnroll();
    populateVectorToSCFConversionPatterns(patterns, vectorTransferToSCFOptions);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
}
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUVectorTransferLoweringPass() {
  return std::make_unique<LLVMCPUVectorTransferLoweringPass>();
}

} // namespace mlir::iree_compiler
