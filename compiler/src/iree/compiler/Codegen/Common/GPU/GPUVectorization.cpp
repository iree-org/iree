// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/GPU/CommonGPUPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using mlir::iree_compiler::IREE::LinalgExt::LinalgVectorizationPattern;
using mlir::iree_compiler::IREE::LinalgExt::VectorizationPatterns;

#define DEBUG_TYPE "iree-codegen-gpu-vectorization"

namespace mlir {
namespace iree_compiler {

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(RewritePatternSet &patterns,
                                          int64_t maxVectorSize) {
  MLIRContext *ctx = patterns.getContext();
  IREE::LinalgExt::LinalgTransformationFilter f(
      {StringAttr::get(ctx, getWorkgroupKTiledMarker()),
       StringAttr::get(ctx, getVectorizeMarker())},
      std::nullopt);
  f.setMatchByDefault();
  // When vectorizing if some ops didn't get tiled we may end up with large
  // vectors being created that will later explode code size. If we have any
  // vectors larger than what would fit in register skip vectorization.
  f.addFilter([maxVectorSize](Operation *op) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) return success();
    int64_t maxFlatVecSize = 1;
    for (OpOperand &operand : linalgOp->getOpOperands()) {
      auto type = llvm::dyn_cast<ShapedType>(operand.get().getType());
      if (!type) continue;
      if (!type.hasStaticShape()) return failure();
      maxFlatVecSize = std::max(maxFlatVecSize, type.getNumElements());
    }
    return success(maxFlatVecSize <= maxVectorSize);
  });

  IREE::LinalgExt::LinalgVectorizationOptions vectorizationOptions;
  vectorizationOptions.setVectorizeGatherAccesses(true);
  VectorizationPatterns<linalg::FillOp, linalg::GenericOp,
                        linalg::Conv1DNwcWcfOp, linalg::Conv1DNcwFcwOp,
                        linalg::TransposeOp>::insert(patterns,
                                                     vectorizationOptions, f);
  patterns.add<linalg::CopyVectorizationPattern>(ctx);
  patterns.add<LinalgVectorizationPattern>(
      ctx, vectorizationOptions,
      f.addOpFilter<linalg::ContractionOpInterface>());
  linalg::populatePadOpVectorizationPatterns(patterns);
}

namespace {
struct GPUVectorizationPass
    : public GPUVectorizationBase<GPUVectorizationPass> {
  GPUVectorizationPass(bool generateContract, int64_t maxVectorSize) {
    this->generateContract = generateContract;
    this->maxVectorSize = maxVectorSize;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Pre-process convolution ops.
    RewritePatternSet decompositionPattern(context);
    linalg::populateDecomposeConvolutionPatterns(decompositionPattern);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(decompositionPattern)))) {
      return signalPassFailure();
    }

    RewritePatternSet vectorizationPatterns(context);
    populateVectorizationPatterns(vectorizationPatterns, maxVectorSize);
    if (generateContract) {
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          vectorizationPatterns);
      vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
    }
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorizationPatterns)))) {
      return signalPassFailure();
    }

    linalg::hoistRedundantVectorTransfersOnTensor(funcOp);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGPUVectorizationPass(
    bool generateContract, int64_t maxVectorSize) {
  return std::make_unique<GPUVectorizationPass>(generateContract,
                                                maxVectorSize);
}

}  // namespace iree_compiler
}  // namespace mlir
