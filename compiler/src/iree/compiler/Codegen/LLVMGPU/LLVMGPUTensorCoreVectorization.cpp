// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-gpu-tensorcore-vectorization"

using mlir::iree_compiler::IREE::LinalgExt::LinalgVectorizationPattern;
using mlir::iree_compiler::IREE::LinalgExt::VectorizationPatterns;

namespace mlir {
namespace iree_compiler {

/// Pattern to fold arithmetic extensions on floating point data types into
/// vector contraction operations. linalg.matmul introduces arithmetic
/// extensions on its operands. Please mlir snippets below for more details.
/// ```mlir
///   "linalg.matmul"(%lhs, %rhs, %acc) ({
///      ^bb0(%arg1: f16, %arg2: f16, %arg3: f32):
///        %lhs_f32 = "arith.extf"(%arg1) : (f16) -> f32
///        %rhs_f32 = "arith.extf"(%arg2) : (f16) -> f32
///        %mul = "arith.mulf"(%lhs_f32, %rhs_f32) : (f32, f32) -> f32
///        %acc = "arith.addf"(%arg3, %mul) : (f32, f32) -> f32
///        "linalg.yield"(%acc) : (f32) -> ()
///     })
/// ```
/// This restricts the native usage of mixed precision NVIDIA Ampere Tensor
/// Cores, i.e, `mma.sync.*.f32.f16.f16.f32` and `mma.sync.*.f32.bf16.bf16.f32`.
/// This pattern folds the arithmetic extensions into the vector contraction and
/// enables the usage of native mixed precision Tensor Core instructions.
struct FoldArithExtIntoContractionOp
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    auto lhsDefOp = contractOp.getLhs().getDefiningOp();
    auto rhsDefOp = contractOp.getRhs().getDefiningOp();

    if (!isa<arith::ExtFOp>(lhsDefOp) || !isa<arith::ExtFOp>(rhsDefOp)) {
      return rewriter.notifyMatchFailure(
          contractOp, "no arith::ExtFOp on contract operands");
    }

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, lhsDefOp->getOperand(0), rhsDefOp->getOperand(0),
        contractOp.getAcc(), contractOp.getIndexingMapsAttr(),
        contractOp.getIteratorTypesAttr());

    return success();
  }
};

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(RewritePatternSet &patterns) {
  IREE::LinalgExt::LinalgTransformationFilter f(
      StringAttr::get(patterns.getContext(), getVectorizeMarker()));
  IREE::LinalgExt::LinalgVectorizationOptions vectorizationOptions;
  VectorizationPatterns<linalg::FillOp, linalg::GenericOp>::insert(
      patterns, vectorizationOptions, f);
  patterns.add<LinalgVectorizationPattern>(
      patterns.getContext(), vectorizationOptions,
      f.addOpFilter<linalg::ContractionOpInterface>());
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);
}

static void populateVectorUnrollPatterns(RewritePatternSet &patterns,
                                         bool useMmaSyncShape) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract) return std::nullopt;
    return gpuMmaUnrollOrder(contract);
  };
  auto getNativeShape = [useMmaSyncShape](Operation *op) {
    if (useMmaSyncShape) return getMmaNativeVectorSize(op);
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
      RewritePatternSet vectorizationPatterns(context);
      populateVectorizationPatterns(vectorizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorizationPatterns)))) {
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
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType tensorCoreType) {
  return std::make_unique<LLVMGPUTensorCoreVectorizationPass>(tensorCoreType);
}

}  // namespace iree_compiler
}  // namespace mlir
