// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-gpu-tensorcore-vectorization"

namespace mlir::iree_compiler {

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void vectorizeLinalgOps(mlir::FunctionOpInterface funcOp) {
  MLIRContext *context = funcOp.getContext();
  IRRewriter rewriter(context);
  IREE::LinalgExt::LinalgTransformationFilter f(
      StringAttr::get(context, getVectorizeMarker()));

  funcOp.walk([&](Operation *op) {
    if (failed(f.checkAndNotify(rewriter, op)) ||
        !isa<linalg::FillOp, linalg::GenericOp, linalg::ContractionOpInterface>(
            op)) {
      return WalkResult::advance();
    }
    (void)linalg::vectorize(rewriter, op);
    return WalkResult::advance();
  });
}

static void populateVectorUnrollPatterns(RewritePatternSet &patterns,
                                         bool useMmaSyncShape) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract)
      return std::nullopt;
    return gpuMmaUnrollOrder(contract);
  };
  auto getNativeShape = [useMmaSyncShape](Operation *op) {
    if (useMmaSyncShape)
      return getMmaNativeVectorSize(op);
    return getWmmaNativeVectorSize(op);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getNativeShape)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

namespace {
struct LLVMGPUTensorCoreVectorizationPass
    : public LLVMGPUTensorCoreVectorizationBase<
          LLVMGPUTensorCoreVectorizationPass> {
  LLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType tensorCoreType)
      : tensorCoreType(tensorCoreType) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    LLVM_DEBUG({
      llvm::dbgs() << "LLVMGPUTensorCoreVectorizationPass runOnOperation():\n";
      funcOp->dump();
    });

    MLIRContext *context = &getContext();
    {
      // Step 1(a). Vectorize (linalg to vector).
      vectorizeLinalgOps(funcOp);
      RewritePatternSet contractionPatterns(context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          contractionPatterns);
      vector::populateVectorReductionToContractPatterns(contractionPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(contractionPatterns)))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "\nAfter populateVectorizationPatterns:\n";
        funcOp->dump();
      });

      // Step 1(b). Fold arithmetic extensions into vector contraction ops.
      // Linalg to vector conversion introduces arithmetic extensions on the
      // operands of vector contraction ops for mixed precision computation.
      // This pattern folds the arithmetic extensions into the vector.contract.
      RewritePatternSet foldArithExtPatterns(context);
      vector::populateFoldArithExtensionPatterns(foldArithExtPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(foldArithExtPatterns)))) {
        return signalPassFailure();
      }

      // Step 2. Fold consumer add ops into the contraction op itself.
      RewritePatternSet canonicalizationPatterns(context);
      vector::ContractionOp::getCanonicalizationPatterns(
          canonicalizationPatterns, context);
      populateCombineVectorTransferReadBroadcastPatterns(
          canonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs()
            << "\nAfter populateCombineVectorTransferReadBroadcastPatterns:\n";
        funcOp->dump();
      });

      // Step 3. Prepare vector operations to be lowered to native tensor core
      // operations (nvgpu.mmasync, nvgpu.ldmatrix).
      if (tensorCoreType == GPUTensorCoreType::MMA_SYNC) {
        RewritePatternSet vectorContractPatterns(funcOp.getContext());
        mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(
            vectorContractPatterns);
        mlir::populatePrepareVectorToMMAPatterns(vectorContractPatterns,
                                                 /*useMMASync=*/true);
        if (failed(applyPatternsAndFoldGreedily(
                getOperation(), std::move(vectorContractPatterns)))) {
          return signalPassFailure();
        }
      }
      LLVM_DEBUG({
        llvm::dbgs()
            << "\nAfter populateCastAwayVectorLeadingOneDimPatterns and "
               "populatePrepareVectorToMMAPatterns:\n";
        funcOp->dump();
      });

      bool useMmaSyncShape = tensorCoreType == GPUTensorCoreType::MMA_SYNC;
      // Step 4. Break and unroll warp tile size to native math and load sizes.
      RewritePatternSet vectorUnrollPatterns(context);
      populateVectorUnrollPatterns(vectorUnrollPatterns, useMmaSyncShape);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorUnrollPatterns)))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "\nAfter populateVectorUnrollPattern:\n";
        funcOp->dump();
      });
    }
  }

private:
  GPUTensorCoreType tensorCoreType;
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType tensorCoreType) {
  return std::make_unique<LLVMGPUTensorCoreVectorizationPass>(tensorCoreType);
}

} // namespace mlir::iree_compiler
