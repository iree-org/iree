// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-vector-transpose-lowering"

namespace mlir::iree_compiler {
namespace {

static bool has16x16Transpose(mlir::FunctionOpInterface funcOp) {
  bool res = false;
  funcOp.walk([&](vector::TransposeOp op) {
    auto srcGtOneDims = isTranspose2DSlice(op);
    if (failed(srcGtOneDims))
      return WalkResult::advance();
    VectorType srcType = op.getSourceVectorType();
    int64_t m = srcType.getDimSize(std::get<0>(srcGtOneDims.value()));
    int64_t n = srcType.getDimSize(std::get<1>(srcGtOneDims.value()));
    if (m == 16 && n == 16) {
      res = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return res;
}

class LLVMCPUVectorTransposeLoweringPass
    : public LLVMCPUVectorTransposeLoweringBase<
          LLVMCPUVectorTransposeLoweringPass> {
public:
  using LLVMCPUVectorTransposeLoweringBase::LLVMCPUVectorTransposeLoweringBase;
  LLVMCPUVectorTransposeLoweringPass(bool lowerVectorTransposeToAVX2) {
    this->lowerVectorTransposeToAVX2 = lowerVectorTransposeToAVX2;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVectorTransposeLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  auto vectorTransformOptions =
      vector::VectorTransformsOptions().setVectorTransposeLowering(
          vector::VectorTransposeLowering::Shuffle1D);
  if (has16x16Transpose(funcOp)) {
    vectorTransformOptions.setVectorTransposeLowering(
        vector::VectorTransposeLowering::Shuffle16x16);
  }

  constexpr unsigned kSpecializedBenefit = 10;
  constexpr unsigned kNarrowTypeEmulationBenefit = 20;

  RewritePatternSet patterns(ctx);
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(patterns,
                                                  vectorTransformOptions);
  vector::populateVectorTransposeNarrowTypeRewritePatterns(
      patterns, kNarrowTypeEmulationBenefit);

  if (lowerVectorTransposeToAVX2) {
    auto avx2LoweringOptions =
        x86vector::avx2::LoweringOptions().setTransposeOptions(
            x86vector::avx2::TransposeLoweringOptions()
                .lower4x8xf32()
                .lower8x8xf32());
    x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
        patterns, avx2LoweringOptions, kSpecializedBenefit);
  }
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUVectorTransposeLoweringPass(bool lowerVectorTransposeToAVX2) {
  return std::make_unique<LLVMCPUVectorTransposeLoweringPass>(
      lowerVectorTransposeToAVX2);
}

} // namespace mlir::iree_compiler
