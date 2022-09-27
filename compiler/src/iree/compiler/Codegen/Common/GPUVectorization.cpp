// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
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

using mlir::iree_compiler::IREE::LinalgExt::LinalgVectorizationPattern;
using mlir::iree_compiler::IREE::LinalgExt::VectorizationPatterns;

#define DEBUG_TYPE "iree-codegen-gpu-vectorization"

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
  VectorizationPatterns<linalg::FillOp, linalg::GenericOp,
                        linalg::Conv1DNwcWcfOp>::insert(patterns, opt, f);
  patterns.add<linalg::CopyVectorizationPattern>(ctx);
  patterns.add<LinalgVectorizationPattern>(
      ctx, f.addOpFilter<linalg::ContractionOpInterface>(), opt);
}

namespace {
struct GPUVectorizationPass
    : public GPUVectorizationBase<GPUVectorizationPass> {
  GPUVectorizationPass(bool generateContract) {
    this->generateContract = generateContract;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Pre-process convolution ops.
    RewritePatternSet decompositionPattern(funcOp.getContext());
    linalg::LinalgTransformationFilter f(
        {StringAttr::get(context, getWorkgroupKTiledMarker())},
        StringAttr::get(context, getVectorizeMarker()));
    f.setMatchByDefault();
    linalg::populateDecomposeConvolutionPatterns(decompositionPattern, f);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(decompositionPattern))))
      return signalPassFailure();

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

std::unique_ptr<OperationPass<func::FuncOp>> createGPUVectorizationPass(
    bool generateContract) {
  return std::make_unique<GPUVectorizationPass>(generateContract);
}

}  // namespace iree_compiler
}  // namespace mlir
