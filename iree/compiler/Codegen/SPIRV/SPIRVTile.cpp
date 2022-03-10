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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
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

/// Gets the given `attrOrValue` as an index value by creating constant ops
/// for attributes.
static Value getAsIndexValue(OpFoldResult attrOrValue, OpBuilder &builder,
                             Location loc) {
  IntegerAttr attr;
  if (Value val = attrOrValue.dyn_cast<Value>()) {
    if (val.getType().isIndex()) return val;
    matchPattern(val, m_Constant(&attr));
  } else {
    attr = attrOrValue.get<Attribute>().cast<IntegerAttr>();
  }
  return builder.createOrFold<arith::ConstantIndexOp>(
      loc, attr.getValue().getSExtValue());
}

namespace {

/// Concretizes tensor.pad op's result shape if its source op implements
/// OffsetSizeAndStrideOpInterface. For example, pad(extract_slice).
struct ConcretizePadResultShape final : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // If the result shape is already static, then nothing to do.
    if (padOp.getResultType().hasStaticShape()) return failure();

    int rank = padOp.getResultType().getRank();
    SmallVector<int64_t> staticShape;
    staticShape.reserve(rank);

    auto sourceIfxOp = dyn_cast_or_null<OffsetSizeAndStrideOpInterface>(
        padOp.source().getDefiningOp());
    if (!sourceIfxOp) return failure();

    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> source = sourceIfxOp.getMixedSizes();
    SmallVector<OpFoldResult> highPad = padOp.getMixedHighPad();

    MLIRContext *context = padOp.getContext();
    Location loc = padOp.getLoc();

    AffineExpr sym0, sym1, sym2;
    bindSymbols(context, sym0, sym1, sym2);
    auto addMap = AffineMap::get(0, 3, {sym0 + sym1 + sym2}, context);

    SmallVector<Value, 3> valueSizes;
    for (int dimIndex = 0; dimIndex < rank; ++dimIndex) {
      valueSizes.clear();
      valueSizes.push_back(getAsIndexValue(lowPad[dimIndex], rewriter, loc));
      valueSizes.push_back(getAsIndexValue(source[dimIndex], rewriter, loc));
      valueSizes.push_back(getAsIndexValue(highPad[dimIndex], rewriter, loc));

      // The pad op's result shape is low padding + source size + high padding.
      // Try to see if we can get a constant number by composing and
      // canonicalizing the result. We use affine mechanisms here because
      // generating arithmetic add ops over dim ops won't work, given they are
      // SSA values that would need invoking other patterns to simplify. We
      // cannot invoke patterns in patterns.
      AffineMap map = addMap;
      fullyComposeAffineMapAndOperands(&map, &valueSizes);
      canonicalizeMapAndOperands(&map, &valueSizes);

      auto cstExpr = map.getResult(0).dyn_cast<AffineConstantExpr>();
      // Specially handle the case where we have both dimensions and symbols and
      // they map to the same value, e.g.:
      //   affine_map<(d0, s0) -> (d0 - s0 + 4)>(%v, %v).
      // Due to the restrictions over dimensions and symbols, the above won't
      // simplify. Try to change dimensions for symbols for such cases.
      if (!cstExpr && llvm::is_splat(valueSizes)) {
        int numDims = map.getNumDims();
        int numSyms = map.getNumSymbols();
        DenseMap<AffineExpr, AffineExpr> dimToSymMap;
        for (int i = 0; i < numDims; ++i) {
          dimToSymMap[rewriter.getAffineDimExpr(i)] =
              rewriter.getAffineSymbolExpr(numSyms + i);
        }
        map = map.replace(dimToSymMap, /*numResultDims=*/0,
                          /*numResultSyms=*/numDims + numSyms);

        canonicalizeMapAndOperands(&map, &valueSizes);
        cstExpr = map.getResult(0).dyn_cast<AffineConstantExpr>();
      }
      if (!cstExpr) return failure();

      staticShape.push_back(cstExpr.getValue());
    }

    auto resultType = RankedTensorType::get(
        staticShape, padOp.getResultType().getElementType(),
        padOp.getResultType().getEncoding());

    rewriter.updateRootInPlace(padOp,
                               [&]() { padOp.result().setType(resultType); });
    return success();
  }
};

}  // namespace

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

    // Try to find computation ops which we will use as anchor to tile and fuse
    // again. If there are `scf.if` ops, we have both a fast and slow paths for
    // padding handling. Then we need to scan both regions to discover such
    // computation ops so that we can tile and fuse both regions.
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

    // Now tile the last computation op to invocations and fuse all operand
    // computation ops into the materialized loop nest.
    for (Operation *computeOp : computeOps) {
      auto consumerOp = dyn_cast<linalg::LinalgOp>(computeOp);

      OpBuilder builder(context);
      SmallVector<int64_t> tileSizes = getTileSizes(consumerOp, 1);
      auto identityLoopOrder =
          llvm::to_vector<4>(llvm::seq<int64_t>(0, tileSizes.size()));

      FailureOr<linalg::TileLoopNest> loopNest =
          linalg::tileConsumerAndFuseProducers(builder, consumerOp, tileSizes,
                                               identityLoopOrder, llvm::None);
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
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to invocations ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {  // Fuse `tensor.pad` op inside the materalized loop nest too.
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

    {  // Canonicalize.
      RewritePatternSet patterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      // Pulling in upstream scf.for and affine.min canonicalization patterns.
      // They work on tiled (but not distributed) loops.
      scf::populateSCFForLoopCanonicalizationPatterns(patterns);
      // Pulling in IREE scf.for and affine.min canonicalization patterns.
      // They work on tiled and distributed loops.
      populateFoldAffineMinInDistributedLoopsPatterns(patterns);
      // Pulling in flow.dispatch.tensor.load op canonicalization patterns.
      // Tiling can generate dim ops taking them as operands.
      IREE::Flow::DispatchTensorLoadOp::getCanonicalizationPatterns(patterns,
                                                                    context);
      patterns.add<ConcretizePadResultShape>(context);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After tiling canonicalization ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    {  // Set markers to drive tiling reduction dimensions.
      OpBuilder builder(context);
      auto marker = builder.getStringAttr(getTileReductionMarker());
      funcOp.walk([&](linalg::LinalgOp op) {
        if (isa<linalg::ContractionOpInterface>(*op) ||
            isa<linalg::ConvolutionOpInterface>(*op)) {
          op->setAttr(linalg::LinalgTransforms::kLinalgTransformMarker, marker);
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

    {  // Fuse `tensor.pad` op inside the materalized loop nest too.
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

    {  // Canonicalize.
      RewritePatternSet patterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      // Pulling in upstream scf.for and affine.min canonicalization patterns.
      // They work on tiled (but not distributed) loops. We only tiled reduction
      // loops previously so this should be fine.
      scf::populateSCFForLoopCanonicalizationPatterns(patterns);
      // Pulling in flow.dispatch.tensor.load op canonicalization patterns.
      // Tiling can generate dim ops taking them as operands.
      IREE::Flow::DispatchTensorLoadOp::getCanonicalizationPatterns(patterns,
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
