// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

struct TileAndDistributeToWorkgroupsUsingForallOpPass
    : public TileAndDistributeToWorkgroupsUsingForallOpBase<
          TileAndDistributeToWorkgroupsUsingForallOpPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

} // namespace

/// Find the lowering config to use for getting the tile sizes.
// TODO: For now this is taking the "last op" in the dispatch, but
// ideally this should take the "root op" that gets tiled and everything
// gets fused with it. For now to keep consistent with the legacy
// tile-and-distribute it is still looking for the "last compute operation".
struct TilingInfo {
  Operation *tilableOp;
  SmallVector<OpFoldResult> tileSizes;
  SmallVector<int64_t> interchange;
};

static FailureOr<TilingInfo>
getTiledAndDistributionInfo(RewriterBase &rewriter,
                            ArrayRef<Operation *> computeOps) {
  Operation *tilableOp = nullptr;
  for (Operation *op : llvm::reverse(computeOps)) {
    if (getLoweringConfig(op)) {
      tilableOp = op;
      break;
    }
  }
  if (!tilableOp) {
    // There is no lowering config. Return `null`.
    return TilingInfo{nullptr, {}, {}};
  }

  IREE::Codegen::LoweringConfigAttrInterface tilableOpConfig =
      getLoweringConfig(tilableOp);
  if (!tilableOpConfig) {
    return tilableOp->emitOpError("unable to find configuration of root op to "
                                  "define workgroup count region");
  }
  auto tileSizes = llvm::map_to_vector(
      tilableOpConfig.getWorkgroupTileSizes(),
      [&](int64_t t) -> OpFoldResult { return rewriter.getIndexAttr(t); });
  SmallVector<int64_t> interchange = tilableOpConfig.getWorkgroupInterchange();

  return TilingInfo{tilableOp, tileSizes, interchange};
}

/// Helper function to return the mapping attribute to use given the tile sizes
static SmallVector<Attribute> getMapping(MLIRContext *context,
                                         ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<Attribute> mapping;
  mapping.reserve(tileSizes.size());
  for (auto tileSize : llvm::reverse(tileSizes)) {
    if (isConstantIntValue(tileSize, 0)) {
      continue;
    }
    uint64_t currSize = mapping.size();
    switch (currSize) {
    case 0:
    case 1:
    case 2:
      mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
          context, IREE::Codegen::symbolizeWorkgroupId(currSize).value()));
      break;
    default:
      mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
          context, IREE::Codegen::WorkgroupId::IdZ, currSize - 2));
    }
  }
  return llvm::to_vector(llvm::reverse(mapping));
}

static SmallVector<int64_t> getStaticNumWorkgroups(RewriterBase &rewriter,
                                                   scf::ForallOp loop) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loop);

  SmallVector<OpFoldResult> lowerBound = loop.getMixedLowerBound();
  SmallVector<OpFoldResult> upperBound = loop.getMixedUpperBound();
  SmallVector<OpFoldResult> step = loop.getMixedStep();

  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr numItersExpr = (s1 - s0).ceilDiv(s2);
  SmallVector<OpFoldResult> numIters = llvm::map_to_vector(
      llvm::zip_equal(lowerBound, upperBound, step), [&](auto it) {
        return affine::makeComposedFoldedAffineApply(
            rewriter, loop.getLoc(), numItersExpr,
            {std::get<0>(it), std::get<1>(it), std::get<2>(it)});
      });
  SmallVector<Value> dynamicNumIters;
  SmallVector<int64_t> staticNumIters;
  dispatchIndexOpFoldResults(numIters, dynamicNumIters, staticNumIters);
  return staticNumIters;
}

void TileAndDistributeToWorkgroupsUsingForallOpPass::runOnOperation() {
  auto funcOp = getOperation();
  auto *context = &getContext();
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);

  IRRewriter rewriter(context);
  FailureOr<TilingInfo> tilingInfo =
      getTiledAndDistributionInfo(rewriter, computeOps);
  if (failed(tilingInfo)) {
    return signalPassFailure();
  }
  auto tilableOp = dyn_cast_or_null<TilingInterface>(tilingInfo->tilableOp);
  if (!tilableOp) {
    // Did not find a tileable op. So do nothing.
    return;
  }

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tilingInfo->tileSizes);
  tilingOptions.setInterchange(tilingInfo->interchange);
  tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

  tilingOptions.setMapping(getMapping(context, tilingInfo->tileSizes));

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);
  // TODO: For now use the default tile and fuse control function. That needs
  // to be modified to allow for returning the values of the producer when
  // needed.
  rewriter.setInsertionPoint(tilableOp);

  FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
      tileConsumerAndFuseProducersUsingSCF(rewriter, tilableOp,
                                           tileAndFuseOptions);
  if (failed(tileAndFuseResult)) {
    funcOp.emitOpError("tile and fuse greedily failed");
    return signalPassFailure();
  }
  for (auto [origValue, replacement] : tileAndFuseResult->replacements) {
    rewriter.replaceAllUsesWith(origValue, replacement);
  }

  // Cleanup patterns for tile and distribute
  {
    SmallVector<int64_t> staticNumIters;
    if (!tileAndFuseResult->loops.empty()) {
      assert(tileAndFuseResult->loops.size() == 1 &&
             "expected only a single scf.forall loop");
      auto forallLoop = cast<scf::ForallOp>(tileAndFuseResult->loops.front());
      staticNumIters = getStaticNumWorkgroups(rewriter, forallLoop);
    }
    RewritePatternSet patterns(context);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    populateFoldAffineMinInDistributedLoopsPatterns(patterns, staticNumIters);
    context->getOrLoadDialect<tensor::TensorDialect>()
        ->getCanonicalizationPatterns(patterns);
    context->getOrLoadDialect<IREE::LinalgExt::IREELinalgExtDialect>()
        ->getCanonicalizationPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("tiling canonicalization failed");
      return signalPassFailure();
    }
  }

  return;
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTileAndDistributeToWorkgroupsUsingForallOpPass() {
  return std::make_unique<TileAndDistributeToWorkgroupsUsingForallOpPass>();
}

} // namespace mlir::iree_compiler
