// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileAndDistribute.cpp -----------------------------------------===//
//
// This pass tiles and distributes Linalg ops with buffer semantics to
// invocations.
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/SPIRVPasses.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;

#define DEBUG_TYPE "iree-spirv-tile-and-distribute"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Invocation tiling patterns
//===----------------------------------------------------------------------===//

/// Patterns for third level tiling to target invocations.
static void populateTilingToInvocationPatterns(
    RewritePatternSet &patterns,
    const linalg::TileSizeComputationFunction &computeFn) {
  auto getThreadProcInfoFn = [](OpBuilder &builder, Location loc,
                                ArrayRef<Range> parallelLoopRanges) {
    return getGPUProcessorIdsAndCounts<gpu::ThreadIdOp, gpu::BlockDimOp>(
        builder, loc, parallelLoopRanges.size());
  };
  linalg::LinalgLoopDistributionOptions distributionOptions;
  distributionOptions.procInfo = getThreadProcInfoFn;

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(computeFn)
                           .setDistributionOptions(distributionOptions);

  MLIRContext *context = patterns.getContext();
  auto marker = StringAttr::get(context, getTileReductionMarker());
  auto filter = IREE::LinalgExt::LinalgTransformationFilter(
                    ArrayRef<StringAttr>(), marker)
                    .setMatchByDefault();

  patterns.add<IREE::LinalgExt::LinalgTilingPattern>(context, tilingOptions,
                                                     filter);
  patterns.add<IREE::LinalgExt::TilingInterfaceTilingPattern>(
      context, tilingOptions, filter);
}

//====---------------------------------------------------------------------===//
// Reduction tiling patterns
//====---------------------------------------------------------------------===//

static void populateTilingReductionPatterns(
    RewritePatternSet &patterns,
    const linalg::TileSizeComputationFunction &computeFn) {
  auto filter = IREE::LinalgExt::LinalgTransformationFilter(
      StringAttr::get(patterns.getContext(), getTileReductionMarker()),
      std::nullopt);

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(computeFn);

  TilingPatterns<linalg::BatchMatmulOp, linalg::MatmulOp>::insert(
      patterns, tilingOptions, filter);
  filter.addFilter([](Operation *op) {
    return success(isa<linalg::ConvolutionOpInterface>(op));
  });
  patterns.add<IREE::LinalgExt::LinalgTilingPattern>(patterns.getContext(),
                                                     tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and distributing Linalg ops with
/// buffer semantics.
class SPIRVTileAndDistributePass
    : public SPIRVTileAndDistributeBase<SPIRVTileAndDistributePass> {
 public:
  SPIRVTileAndDistributePass() = default;
  SPIRVTileAndDistributePass(const SPIRVTileAndDistributePass &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

//====---------------------------------------------------------------------===//
// Main pass implementation
//====---------------------------------------------------------------------===//

void SPIRVTileAndDistributePass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  if (!isEntryPoint(funcOp)) return;

  auto threadTileComputeFn = getSPIRVTileSizeComputeFn(funcOp, 1);
  if (failed(threadTileComputeFn)) return signalPassFailure();
  auto reductionTileComputeFn = getSPIRVTileSizeComputeFn(funcOp, 2);
  if (failed(reductionTileComputeFn)) return signalPassFailure();

  {  // Tile and distribute to invocations.
    RewritePatternSet invocationTilingPatterns(context);
    populateTilingToInvocationPatterns(invocationTilingPatterns,
                                       *threadTileComputeFn);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(invocationTilingPatterns)))) {
      funcOp.emitOpError() << "failure in tiling";
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to invocations ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);

    SmallVector<int64_t> numWorkgroups = getStaticNumWorkgroups(funcOp);
    populateFoldAffineMinInDistributedLoopsPatterns(canonicalizationPatterns,
                                                    numWorkgroups);

    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      // TODO(#4759): This does not converge after the max number of iterations.
      // It indicates that some pattern upstream is generating ops even when the
      // pattern failed to match. Not related to correctness, but would be good
      // to figure out and fix.
      // return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling canonicalization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {  // Tile reduction dimensions.
    RewritePatternSet reductionTilingPatterns(context);
    populateTilingReductionPatterns(reductionTilingPatterns,
                                    *reductionTileComputeFn);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(reductionTilingPatterns)))) {
      funcOp.emitOpError() << "failing in tile reduction";
      return signalPassFailure();
    }

    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(canonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      funcOp.emitOpError() << "failing canonicalizing after tile reduction";
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling reduction dimensions  ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVTileAndDistributePass() {
  return std::make_unique<SPIRVTileAndDistributePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
