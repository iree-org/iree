// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-vector-lowering"

namespace mlir::iree_compiler {

static bool has16x16Transpose(func::FuncOp funcOp) {
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

namespace {
/// Pass to lower Vector ops before conversion to LLVM.
class LLVMCPUVectorLoweringPass
    : public LLVMCPUVectorLoweringBase<LLVMCPUVectorLoweringPass> {
public:
  using LLVMCPUVectorLoweringBase::LLVMCPUVectorLoweringBase;
  LLVMCPUVectorLoweringPass(const LLVMCPUVectorLoweringPassOptions &options) {
    this->splitVectorTransfersTo = options.splitVectorTransfersTo;
    this->lowerVectorTransposeToAVX2 = options.lowerVectorTransposeToAVX2;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVectorLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  // Per-function lowering pipeline.
  auto vectorTransposeLowering = vector::VectorTransposeLowering::Shuffle1D;
  auto vectorMultiReductionLowering =
      vector::VectorMultiReductionLowering::InnerReduction;
  auto vectorContractLowering = vector::VectorContractLowering::OuterProduct;
  auto vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);
  auto vectorTransformOptions =
      vector::VectorTransformsOptions()
          .setVectorTransposeLowering(vectorTransposeLowering)
          .setVectorTransformsOptions(vectorContractLowering)
          .setVectorMultiReductionLowering(vectorMultiReductionLowering)
          .setVectorTransferSplit(vectorTransferSplit);

  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorGatherLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After applying patterns for vector.gather Ops ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Lower high level vector operations like contract or multidim reduce ops
  // to lower level vector ops.
  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateVectorContractLoweringPatterns(
        patterns, vectorTransformOptions,
        /*benefit=*/1,
        /*disableOuterProductLowering=*/false);
    // This pattern will transform vector loads whose elements are used in a
    // scalar fashion into scalar loads. This will let scalar loads to be folded
    // into broadcast/arithmetic operations and reduce register pressure.
    vector::populateScalarVectorTransferLoweringPatterns(
        patterns, /*benefit=*/1, /*allowMultipleUses=*/true);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vectorMultiReductionLowering);
    populateVectorTransferFullPartialPatterns(patterns, vectorTransformOptions);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After lowering high level vector ops to lower level "
                    "vector ops ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Make sure we remove redundant vector ops (e.g., vector tranposes) before we
  // lower them and can't be optimized away anymore.
  // TODO (dcaballe): We should run full canonicalization here.
  {
    RewritePatternSet patterns(ctx);
    vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
    vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransposeOp::getCanonicalizationPatterns(patterns, ctx);
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

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After lowering vector transfers to SCF ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Lowering for vector.transpose ops.
  {
    if (has16x16Transpose(funcOp)) {
      vectorTransformOptions.setVectorTransposeLowering(
          vector::VectorTransposeLowering::Shuffle16x16);
    }
    RewritePatternSet patterns(ctx);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateVectorTransposeLoweringPatterns(patterns,
                                                    vectorTransformOptions);
    if (lowerVectorTransposeToAVX2) {
      auto avx2LoweringOptions =
          x86vector::avx2::LoweringOptions().setTransposeOptions(
              x86vector::avx2::TransposeLoweringOptions()
                  .lower4x8xf32()
                  .lower8x8xf32());
      x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
          patterns, avx2LoweringOptions, /*benefit=*/10);
    }
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After lowering vector transpose ops ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // 'vector.shape_cast' are very expensive operations that are even generated
  // by some of the lowerings above (e.g., transpose lowering). There are
  // chances to cancel them out if they are not lowered too early so we lower
  // them at the very end of the pass.
  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorShapeCastLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
}
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUVectorLoweringPass() {
  return std::make_unique<LLVMCPUVectorLoweringPass>();
}
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUVectorLoweringPass(
    const LLVMCPUVectorLoweringPassOptions &options) {
  return std::make_unique<LLVMCPUVectorLoweringPass>(options);
}
} // namespace mlir::iree_compiler
