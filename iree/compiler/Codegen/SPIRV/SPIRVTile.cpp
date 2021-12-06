// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTile.cpp ------------------------------------------------------===//
//
// This pass tiles and Linalg ops with tensor semantics to invocations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-tile"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Tiling patterns
//===----------------------------------------------------------------------===//

/// Populates `patterns` with patterns that tiles convolution/matmul ops with
/// markers.
static void populateTilingReductionPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  auto getTileSizeFn = [&](OpBuilder &builder, Operation *op) {
    return getTileSizes(builder, op, 2);
  };
  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getTileSizeFn);
  auto marker = Identifier::get(getTileReductionMarker(), context);
  auto filter = linalg::LinalgTransformationFilter({marker}, llvm::None);

  patterns.insert<linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
                  linalg::LinalgTilingPattern<linalg::Conv2DNhwcHwcfOp>,
                  linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwcHwcOp>,
                  linalg::LinalgTilingPattern<linalg::MatmulOp>>(
      context, tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {

class SPIRVTilePass final : public SPIRVTileBase<SPIRVTilePass> {
 public:
  SPIRVTilePass() = default;
  SPIRVTilePass(const SPIRVTilePass &pass) = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FuncOp funcOp = getOperation();

    SmallVector<Operation *> computeOps;
    SmallVector<LoopTilingAndDistributionInfo> loopInfos;
    if (failed(getComputeOps(funcOp, computeOps, loopInfos))) {
      return signalPassFailure();
    }

    {  // Tile to invocations.
      auto consumerOp = dyn_cast<linalg::LinalgOp>(computeOps.back());

      OpBuilder builder(context);
      SmallVector<int64_t> tileSizes = getTileSizes(consumerOp, 1);
      auto identityLoopOrder =
          llvm::to_vector<4>(llvm::seq<int64_t>(0, tileSizes.size()));

      FailureOr<linalg::TileLoopNest> loopNest =
          linalg::tileConsumerAndFuseProducers(builder, consumerOp, tileSizes,
                                               identityLoopOrder);
      if (failed(loopNest)) return signalPassFailure();

      consumerOp->replaceAllUsesWith(loopNest->getRootOpReplacementResults());

      // We don't distribute here; instead, it will be done in a later step
      // after bufferization. So add attributes to the tiled loop nest to
      // indicate that they should be distributed to invocations.
      ArrayRef<scf::ForOp> loops = loopNest->getLoopOps();
      assert(loops.size() <= kNumGPUDims);
      const char *attrName = getSPIRVDistributeAttrName();
      for (int i = loops.size() - 1, dim = 0; i >= 0; --i) {
        loops[i]->setAttr(attrName, builder.getIndexAttr(dim++));
      }

      LLVM_DEBUG({
        llvm::dbgs() << "--- After tiling to invocations ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    {
      RewritePatternSet patterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      // Pulling in upstream scf.for and affine.min canonicalization patterns.
      // They work on tiled (but not distributed) loops.
      scf::populateSCFForLoopCanonicalizationPatterns(patterns);
      // Pulling in IREE scf.for and affine.min canonicalization patterns.
      // They work on tiled and distributed loops.
      populateFoldAffineMinInDistributedLoopsPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After tiling canonicalization ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    {
      // Set markers to drive tiling reduction dimensions.
      OpBuilder builder(context);
      auto marker = builder.getStringAttr(getTileReductionMarker());
      funcOp.walk([&](linalg::LinalgOp op) {
        if (isa<linalg::ContractionOpInterface>(*op) ||
            isa<linalg::ConvolutionOpInterface>(*op)) {
          op->setAttr("__internal_linalg_transform__", marker);
        }
      });
    }

    {  // Tile reduction dimensions.
      RewritePatternSet tilingPatterns(&getContext());
      populateTilingReductionPatterns(tilingPatterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(tilingPatterns)))) {
        return signalPassFailure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "--- After tiling reduction dimensions  ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    {
      RewritePatternSet patterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      // Pulling in upstream scf.for and affine.min canonicalization patterns.
      // They work on tiled (but not distributed) loops. We only tiled reduction
      // loops previously so this should be fine.
      scf::populateSCFForLoopCanonicalizationPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "--- After tiling canonicalization  ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createSPIRVTilePass() {
  return std::make_unique<SPIRVTilePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
