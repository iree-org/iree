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

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
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
  auto marker = StringAttr::get(context, getTileReductionMarker());
  auto filter = linalg::LinalgTransformationFilter({marker}, llvm::None);

  linalg::TilingPatterns<linalg::BatchMatmulOp, linalg::Conv2DNhwcHwcfOp,
                         linalg::DepthwiseConv2DNhwcHwcOp,
                         linalg::MatmulOp>::insert(patterns, tilingOptions,
                                                   filter);
}

struct ConcretizePadResultShape final : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    ReifiedRankedShapedTypeDims resultShapes;
    auto interface = cast<ReifyRankedShapedTypeOpInterface>(*padOp);
    if (failed(interface.reifyResultShapes(rewriter, resultShapes))) {
      return rewriter.notifyMatchFailure(
          padOp, "failed to reify tensor.pad op result shape");
    }

    auto oldType = padOp.getResultType();

    SmallVector<int64_t> staticShape;
    staticShape.reserve(oldType.getRank());
    for (Value v : resultShapes.front()) {
      IntegerAttr cstAttr;
      if (matchPattern(v, m_Constant(&cstAttr))) {
        staticShape.push_back(cstAttr.getValue().getZExtValue());
      } else {
        return rewriter.notifyMatchFailure(padOp,
                                           "found non-constant dim size");
      }
    }

    rewriter.startRootUpdate(padOp);
    padOp.result().setType(RankedTensorType::get(
        staticShape, oldType.getElementType(), oldType.getEncoding()));
    rewriter.finalizeRootUpdate(padOp);
    return success();
  };
};

//===----------------------------------------------------------------------===//
// Tiling configuration
//===----------------------------------------------------------------------===//

/// Returns a tile size configuration to split the given original tile `sizes`
/// into two even halves.
static SmallVector<int64_t> halveTileSizes(ArrayRef<int64_t> sizes) {
  SmallVector<int64_t> halfSizes(sizes.size(), 0);

  // Drop all non tiled dimensions at the end.
  while (!sizes.empty() && sizes.back() == 0) sizes = sizes.drop_back();

  // Find the first dimension that can be halved. If it is not the last tiled
  // dimension, we can just use it. Otherwise, only halve if we can still
  // maintain at least 4 elements per tile for vectorization.
  for (int i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == 0 || sizes[i] % 2 != 0) continue;
    if (i != sizes.size() - 1 || sizes[i] >= 8) halfSizes[i] = sizes[i] / 2;
    break;
  }

  return halfSizes;
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

    {
      RewritePatternSet patterns(context);
      tensor::populateSplitPaddingPatterns(patterns);
      scf::populateIfRegionExpansionPatterns(patterns);
      patterns.add<ConcretizePadResultShape>(context);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After splitting padding cases ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    SmallVector<Operation *> computeOps;
    SmallVector<scf::IfOp, 1> ifOps;
    funcOp.walk([&ifOps](scf::IfOp ifOp) { ifOps.push_back(ifOp); });
    if (ifOps.empty()) {
      SmallVector<LoopTilingAndDistributionInfo> loopInfos;
      if (failed(getComputeOps(funcOp, computeOps, loopInfos))) {
        return signalPassFailure();
      }
      while (computeOps.size() > 1) computeOps.erase(computeOps.begin());
    } else {
      if (ifOps.size() > 1) {
        funcOp.emitError("expected to contain no more than one scf.if ops");
        return signalPassFailure();
      }

      for (Operation &op : llvm::reverse(*ifOps.front().thenBlock())) {
        if (isa<linalg::LinalgOp, IREE::LinalgExt::TiledOpInterface>(op)) {
          computeOps.push_back(&op);
          break;
        }
      }
      if (Block *elseBlock = ifOps.front().elseBlock()) {
        for (Operation &op : llvm::reverse(*elseBlock)) {
          if (isa<linalg::LinalgOp, IREE::LinalgExt::TiledOpInterface>(op)) {
            computeOps.push_back(&op);
            break;
          }
        }
      }
    }
    assert(computeOps.size() <= 2);

    for (int i = 0; i < computeOps.size(); ++i) {  // Tile to invocations.
      auto consumerOp = dyn_cast<linalg::LinalgOp>(computeOps[i]);

      OpBuilder builder(context);
      SmallVector<int64_t> tileSizes = getTileSizes(consumerOp, 1);
      auto identityLoopOrder =
          llvm::to_vector<4>(llvm::seq<int64_t>(0, tileSizes.size()));

      FailureOr<linalg::TileLoopNest> loopNest =
          linalg::tileConsumerAndFuseProducers(builder, consumerOp, tileSizes,
                                               identityLoopOrder);
      if (failed(loopNest)) return signalPassFailure();

      consumerOp->replaceAllUsesWith(loopNest->getRootOpReplacementResults());
      computeOps[i] = loopNest->getRootOp();

      // We don't distribute here; instead, it will be done in a later step
      // after bufferization. So add attributes to the tiled loop nest to
      // indicate that they should be distributed to invocations.
      ArrayRef<scf::ForOp> loops = loopNest->getLoopOps();
      assert(loops.size() <= kNumGPUDims);
      const char *attrName = getSPIRVDistributeAttrName();
      for (int i = loops.size() - 1, dim = 0; i >= 0; --i) {
        loops[i]->setAttr(attrName, builder.getIndexAttr(dim++));
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to invocations ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    if (!ifOps.empty()) {  // Tile padded case again to save registers
      auto consumerOp = dyn_cast<linalg::LinalgOp>(computeOps.back());

      SmallVector<int64_t> tileSizes =
          halveTileSizes(getTileSizes(consumerOp, 1));
      LLVM_DEBUG({
        llvm::dbgs() << "Tiling padded case again using tile size: [";
        llvm::interleaveComma(tileSizes, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      if (llvm::any_of(tileSizes, [](int64_t size) { return size != 0; })) {
        OpBuilder builder(context);
        auto identityLoopOrder =
            llvm::to_vector<4>(llvm::seq<int64_t>(0, tileSizes.size()));
        FailureOr<linalg::TileLoopNest> loopNest =
            linalg::tileConsumerAndFuseProducers(builder, consumerOp, tileSizes,
                                                 identityLoopOrder);

        if (failed(loopNest)) return signalPassFailure();

        consumerOp->replaceAllUsesWith(loopNest->getRootOpReplacementResults());
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling padded case ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      RewritePatternSet patterns(context);
      patterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
          context, [](tensor::ExtractSliceOp) { return false; });
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After fusing padding into consumers ---\n";
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
      // Pulling in scf.if canonicalization patterns to remove scf.if ops with
      // static true/false conditions.
      scf::IfOp::getCanonicalizationPatterns(patterns, context);
      // Pulling in IREE scf.for and affine.min canonicalization patterns.
      // They work on tiled and distributed loops.
      populateFoldAffineMinInDistributedLoopsPatterns(patterns);
      // Pulling in flow.dispatch.tensor.load/store op canonicalization
      // patterns. Tiling can generate dim ops taking them as operands.
      IREE::Flow::populateFlowDispatchCanonicalizationPatterns(patterns,
                                                               context);
      patterns.add<ConcretizePadResultShape>(context);
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
      RewritePatternSet patterns(context);
      patterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
          context, [](tensor::ExtractSliceOp) { return false; });
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After fusing padding into consumers ---\n";
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
      // Pulling in flow.dispatch.tensor.load/store op canonicalization
      // patterns. Tiling can generate dim ops taking them as operands.
      IREE::Flow::populateFlowDispatchCanonicalizationPatterns(patterns,
                                                               context);
      patterns.add<ConcretizePadResultShape>(context);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

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
