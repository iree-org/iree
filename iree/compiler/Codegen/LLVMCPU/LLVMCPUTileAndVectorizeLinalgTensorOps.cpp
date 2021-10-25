// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-to-llvm-tile-and-pad-workgroups"

namespace mlir {
namespace iree_compiler {

namespace {
// Could just be linalg::TilingPattern with a ContractionOpInterface filter, but
// that is always templated on an op.
struct TileWorkgroups : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileWorkgroups(MLIRContext *context, linalg::LinalgTilingOptions options,
                 linalg::LinalgTransformationFilter marker)
      : LinalgBaseTilingPattern(context, options, marker) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto contractionOp = dyn_cast<linalg::ContractionOpInterface>(op);
    if (!contractionOp) return failure();

    linalg::TiledLinalgOp tiledLinalgOp;
    if (failed(Base::matchAndRewriteBase(op, rewriter, tiledLinalgOp))) {
      return failure();
    }
    rewriter.replaceOp(op, tiledLinalgOp.tensorResults);
    return success();
  }
};

}  // namespace

namespace {
struct LLVMCPUTileAndVectorizePass
    : public LLVMCPUTileAndVectorizeBase<LLVMCPUTileAndVectorizePass> {
  LLVMCPUTileAndVectorizePass(bool vectorize = true)
      : lowerToVectors(vectorize) {}
  LLVMCPUTileAndVectorizePass(const LLVMCPUTileAndVectorizePass &pass) {
    lowerToVectors = pass.lowerToVectors;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;

 private:
  bool lowerToVectors;
};
}  // namespace

void LLVMCPUTileAndVectorizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  // First level of tiling patterns {
  {
    OwningRewritePatternList l1patterns(&getContext());
    l1patterns.insert<TileWorkgroups>(
        context,
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) -> SmallVector<Value, 4> {
              return getTileSizes(builder, op,
                                  static_cast<unsigned>(TilingLevel::L1Tiles));
            }),
        linalg::LinalgTransformationFilter(
            Identifier::get(getWorkgroupMarker(), context),
            Identifier::get(getWorkgroupL1TileMarker(), context)));

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(l1patterns)))) {
      return signalPassFailure();
    }
  }

  // Apply canoncalization
  {
    OwningRewritePatternList canonicalizationPatterns(&getContext());
    linalg::populateLinalgTilingCanonicalizationPatterns(
        canonicalizationPatterns);
    memref::DimOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                               context);
    scf::populateSCFForLoopCanonicalizationPatterns(canonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      return signalPassFailure();
    }
  }

  // Second level of tiling patterns{
  {
    OwningRewritePatternList l2patterns(&getContext());
    l2patterns.insert<TileWorkgroups>(
        context,
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) -> SmallVector<Value, 4> {
              return getTileSizes(
                  builder, op, static_cast<unsigned>(TilingLevel::VectorTiles));
            }),
        linalg::LinalgTransformationFilter(
            Identifier::get(getWorkgroupL1TileMarker(), context),
            Identifier::get(getVectorizeMarker(), context)));

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(l2patterns)))) {
      return signalPassFailure();
    }
  }
  // Apply canoncalization
  {
    OwningRewritePatternList canonicalizationPatterns(&getContext());
    linalg::populateLinalgTilingCanonicalizationPatterns(
        canonicalizationPatterns);
    memref::DimOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                               context);
    scf::populateSCFForLoopCanonicalizationPatterns(canonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      return signalPassFailure();
    }
  }

  if (!lowerToVectors) {
    return;
  }

  // Apply vectorization patterns.
  {
    OwningRewritePatternList vectorizationPatterns(&getContext());
    linalg::insertVectorizationPatterns<linalg::ContractionOpInterface,
                                        linalg::CopyOp, linalg::FillOp>(
        vectorizationPatterns, linalg::LinalgVectorizationOptions(),
        linalg::LinalgTransformationFilter(
            Identifier::get(getVectorizeMarker(), context)));
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorizationPatterns)))) {
      return signalPassFailure();
    }
  }

  {
    // Fold consumer add ops into the contraction op itself.
    RewritePatternSet canonicalizationPatterns(context);
    vector::ContractionOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                       context);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPatterns));
  }

  // Apply vector specific operation lowering.
  {
    vector::VectorTransformsOptions vectorTransformsOptions =
        vector::VectorTransformsOptions().setVectorTransformsOptions(
            vector::VectorContractLowering::OuterProduct);
    OwningRewritePatternList vectorContractLoweringPatterns(&getContext());
    vectorContractLoweringPatterns
        .insert<ContractionOpToOuterProductOpLowering,
                ContractionOpToMatmulOpLowering, ContractionOpLowering>(
            vectorTransformsOptions, context);
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorContractLoweringPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorContractLoweringPatterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<FuncOp>> createLLVMCPUTileAndVectorizePass(
    bool lowerToVectors) {
  return std::make_unique<LLVMCPUTileAndVectorizePass>(lowerToVectors);
}

}  // namespace iree_compiler
}  // namespace mlir
