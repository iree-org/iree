// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvmgpu-vectorization"

namespace mlir {
namespace iree_compiler {

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  linalg::LinalgVectorizationOptions opt;
  linalg::LinalgTransformationFilter f(
      {StringAttr::get(ctx, getWorkgroupKTiledMarker()),
       StringAttr::get(ctx, getVectorizeMarker())},
      llvm::None);
  f.setMatchByDefault();
  linalg::VectorizationPatterns<linalg::FillOp, linalg::GenericOp>::insert(
      patterns, opt, f);
  patterns.add<linalg::CopyVectorizationPattern>(ctx);
  patterns.add<linalg::LinalgVectorizationPattern>(
      ctx, f.addOpFilter<linalg::ContractionOpInterface>(), opt);
}

namespace {
struct LLVMGPUVectorizationPass
    : public LLVMGPUVectorizationBase<LLVMGPUVectorizationPass> {
  LLVMGPUVectorizationPass(bool generateContract) {
    this->generateContract = generateContract;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet vectorizationPatterns(context);
    populateVectorizationPatterns(vectorizationPatterns);
    if (generateContract) {
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          vectorizationPatterns);
      vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
    }
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorizationPatterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorizationPass(
    bool generateContract) {
  return std::make_unique<LLVMGPUVectorizationPass>(generateContract);
}

}  // namespace iree_compiler
}  // namespace mlir
