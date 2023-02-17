// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using mlir::iree_compiler::IREE::LinalgExt::LinalgVectorizationPattern;
using mlir::iree_compiler::IREE::LinalgExt::VectorizationPatterns;

namespace mlir {
namespace iree_compiler {

/// Flag defined in Passes.cpp.
extern llvm::cl::opt<bool> llvmgpuUseMMASync;

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

static Optional<SmallVector<int64_t>> getGPUTensorCoreNativeVectorSize(
    Operation *op) {
  if (llvmgpuUseMMASync) return getMmaNativeVectorSize(op);
  return getWmmaNativeVectorSize(op);
}

static void populateVectorUnrollPatterns(RewritePatternSet &patterns) {
  auto unrollOrder = [](Operation *op) -> Optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract) return std::nullopt;
    return gpuMmaUnrollOrder(contract);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getGPUTensorCoreNativeVectorSize)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

namespace {
struct LLVMGPUTensorCoreVectorizationPass
    : public LLVMGPUTensorCoreVectorizationBase<
          LLVMGPUTensorCoreVectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    {
      // Step 1. Vectorize.
      RewritePatternSet vectorizationPatterns(context);
      populateVectorizationPatterns(vectorizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorizationPatterns)))) {
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

      // Step 3. Prepare vector operations to be lowered to native tensor core
      // operations (nvgpu.mmasync, nvgpu.ldmatrix).
      if (llvmgpuUseMMASync) {
        RewritePatternSet vectorContractPatterns(funcOp.getContext());
        mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(
            vectorContractPatterns);
        mlir::populatePrepareVectorToMMAPatterns(vectorContractPatterns,
                                                 llvmgpuUseMMASync);
        if (failed(applyPatternsAndFoldGreedily(
                getOperation(), std::move(vectorContractPatterns)))) {
          return signalPassFailure();
        }
      }

      // Step 4. Break and unroll warp tile size to native math and load sizes.
      RewritePatternSet vectorUnrollPatterns(context);
      populateVectorUnrollPatterns(vectorUnrollPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorUnrollPatterns)))) {
        return signalPassFailure();
      }
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUTensorCoreVectorizationPass() {
  return std::make_unique<LLVMGPUTensorCoreVectorizationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
