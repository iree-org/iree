// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/TransformUtils.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"
#include "iree/compiler/Conversion/PassDetail.h"
#include "iree/compiler/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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
struct LinalgToLLVMTilePadAndVectorizeWorkgroupsPass
    : public LinalgToLLVMTilePadAndVectorizeWorkgroupsBase<
          LinalgToLLVMTilePadAndVectorizeWorkgroupsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

void LinalgToLLVMTilePadAndVectorizeWorkgroupsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  // First level of tiling patterns {
  {
    OwningRewritePatternList l1patterns(&getContext());
    l1patterns.insert<TileWorkgroups>(
        context,
        linalg::LinalgTilingOptions()
            .setTileSizeComputationFunction(
                [](OpBuilder &builder,
                   Operation *operation) -> SmallVector<Value, 4> {
                  return getTileSizes(
                      builder, operation,
                      static_cast<unsigned>(TilingLevel::Level1Tiles));
                })
            .setPaddingValueComputationFunction(
                [](OpBuilder &b, OpOperand &op) -> Value {
                  auto t = getElementTypeOrSelf(op.get().getType());
                  return b.create<ConstantOp>(op.getOwner()->getLoc(), t,
                                              b.getZeroAttr(t));
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
    canonicalizationPatterns.add<linalg::AffineMinSCFCanonicalizationPattern>(
        &getContext());
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      return signalPassFailure();
    }
  }
  // Second level of tiling patterns{
  {
    OwningRewritePatternList l1patterns(&getContext());
    l1patterns.insert<TileWorkgroups>(
        context,
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder,
               Operation *operation) -> SmallVector<Value, 4> {
              return getTileSizes(
                  builder, operation,
                  static_cast<unsigned>(TilingLevel::Level2Tiles));
            }),
        linalg::LinalgTransformationFilter(
            Identifier::get(getWorkgroupL1TileMarker(), context),
            Identifier::get(getVectorizeMarker(), context)));

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
    canonicalizationPatterns.add<linalg::AffineMinSCFCanonicalizationPattern>(
        &getContext());
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      return signalPassFailure();
    }
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

  // TODO: This should be a folding of Add into Contract in core but while
  // they live in different dialects, it is not possible without unnatural
  // dependencies.
  funcOp.walk([&](Operation *op) {
    if (auto contract = canonicalizeContractionAdd(op))
      op->replaceAllUsesWith(contract);
  });
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
    vector::populateVectorTransferLoweringPatterns(
        vectorContractLoweringPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorContractLoweringPatterns)))) {
      return signalPassFailure();
    }
  }
  //
  // Hosit hierarchical tiling indexing and other loop invariant transfer
  // ops computation.
  //
  // Programmatic controlled lowering of vector.transfer only.
  {
    VectorTransferToSCFOptions vectorToSCFOptions =
        VectorTransferToSCFOptions().setUnroll(true);
    OwningRewritePatternList vectorToLoopsPatterns(&getContext());
    populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                          vectorToSCFOptions);
    // Hosit hierarchical tiling indexing and other loop invariant transfer
    // ops computation.
    linalg::hoistRedundantVectorTransfers(funcOp);

    memref::populateFoldSubViewOpPatterns(vectorToLoopsPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorToLoopsPatterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<FuncOp>>
createLinalgToLLVMTilePadAndVectorizeWorkgroupsPass() {
  return std::make_unique<LinalgToLLVMTilePadAndVectorizeWorkgroupsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
