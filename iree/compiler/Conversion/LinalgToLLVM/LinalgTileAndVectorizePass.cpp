// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-to-llvm-tile-and-vectorize"

namespace mlir {
namespace iree_compiler {

namespace {
template <typename LinalgOpTy>
struct TileWorkgroups : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileWorkgroups(MLIRContext *context, linalg::LinalgTilingOptions options,
                 linalg::LinalgMarker marker, PatternBenefit benefit = 1)
      : Base(LinalgOpTy::getOperationName(), context, options, marker,
             benefit) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> tensorResults;
    if (failed(Base::matchAndRewriteBase(op, rewriter, tensorResults)) ||
        !tensorResults.empty()) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

namespace {
struct TileAndVectorizeWorkgroups
    : public PassWrapper<TileAndVectorizeWorkgroups, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AffineDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnFunction() override;
};
}  // namespace

void TileAndVectorizeWorkgroups::runOnFunction() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();

  OwningRewritePatternList l1patterns, l2patterns;
  CPUKernelDispatch cpuKernelDispatch;

  // First level of tiling patterns. (workgroups memory)
  l1patterns.insert<TileWorkgroups<linalg::MatmulOp>,
                    TileWorkgroups<linalg::BatchMatmulOp>>(
      context,
      linalg::LinalgTilingOptions().setTileSizeComputationFunction(
          [&cpuKernelDispatch](OpBuilder &builder,
                               Operation *operation) -> SmallVector<Value, 4> {
            return TileSizeFn::get<TilingLevel::Level1Tiles>(
                cpuKernelDispatch, builder, operation);
          }),
      linalg::LinalgMarker(
          Identifier::get(getWorkgroupMarker(), context),
          Identifier::get(getWorkgroupL1TileMarker(), context)));

  // Second level of tiling patterns. (workgroups memroey -> vectors)
  l2patterns.insert<TileWorkgroups<linalg::MatmulOp>,
                    TileWorkgroups<linalg::BatchMatmulOp>>(
      context,
      linalg::LinalgTilingOptions().setTileSizeComputationFunction(
          [&cpuKernelDispatch](OpBuilder &builder,
                               Operation *operation) -> SmallVector<Value, 4> {
            return TileSizeFn::get<TilingLevel::Level2Tiles>(
                cpuKernelDispatch, builder, operation);
          }),
      linalg::LinalgMarker(Identifier::get(getWorkgroupL1TileMarker(), context),
                           Identifier::get(getVectorizeMarker(), context)));

  // Apply tiling.
  applyPatternsAndFoldGreedily(funcOp, std::move(l1patterns));
  applyPatternsAndFoldGreedily(funcOp, std::move(l2patterns));

  // Apply canonicalization.
  OwningRewritePatternList canonicalizationPatterns;
  canonicalizationPatterns.insert<AffineMinCanonicalizationPattern>(context);
  AffineApplyOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  AffineMinOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  SubViewOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  applyPatternsAndFoldGreedily(funcOp, std::move(canonicalizationPatterns));

  // Apply vectorization.
  OwningRewritePatternList vectorizationPatterns;
  vectorizationPatterns
      .insert<linalg::LinalgVectorizationPattern<linalg::MatmulOp>,
              linalg::LinalgVectorizationPattern<linalg::BatchMatmulOp>>(
          context,
          linalg::LinalgMarker(Identifier::get(getVectorizeMarker(), context)));
  applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));

  // Apply vector specific operation lowering.
  vector::VectorTransformsOptions vectorTransformsOptions =
      vector::VectorTransformsOptions().setVectorTransformsOptions(
          vector::VectorContractLowering::OuterProduct);
  OwningRewritePatternList vectorContractLoweringPatterns;
  vectorContractLoweringPatterns
      .insert<ContractionOpToOuterProductOpLowering,
              ContractionOpToMatmulOpLowering, ContractionOpLowering>(
          vectorTransformsOptions, context);
  applyPatternsAndFoldGreedily(funcOp,
                               std::move(vectorContractLoweringPatterns));

  // Programmatic controlled lowering of vector.transfer only.
  VectorTransferToSCFOptions vectorToSCFOptions =
      VectorTransferToSCFOptions().setUnroll(true);
  OwningRewritePatternList vectorToLoopsPatterns;
  populateVectorToSCFConversionPatterns(vectorToLoopsPatterns, context,
                                        vectorToSCFOptions);
  // Hosit hierarchical tiling indexing and other loop invariant transfer ops
  // computation.
  linalg::hoistViewAllocOps(funcOp);
  linalg::hoistRedundantVectorTransfers(funcOp);

  // TODO(ataei): Move this to common vector dialect patterns.
  populateStdLegalizationPatternsForSPIRVLowering(context,
                                                  vectorToLoopsPatterns);
  applyPatternsAndFoldGreedily(funcOp, std::move(vectorToLoopsPatterns));
}

std::unique_ptr<FunctionPass> createLinalgTileAndVectorizeWorkgroupsPass() {
  return std::make_unique<TileAndVectorizeWorkgroups>();
}

static PassRegistration<TileAndVectorizeWorkgroups> pass(
    "iree-codegen-linalg-to-llvm-workgroups-vectorization-pass",
    "Tile and vectorize llvm workgroups",
    [] { return std::make_unique<TileAndVectorizeWorkgroups>(); });

}  // namespace iree_compiler
}  // namespace mlir
