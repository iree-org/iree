// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-tile-using-forall"

namespace mlir::iree_compiler {
namespace {
class TileUsingForallPass : public TileUsingForallBase<TileUsingForallPass> {
public:
  TileUsingForallPass(int64_t tilingLevel = -1) {
    this->tilingLevel.setValue(tilingLevel);
  }

  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override;
};

} // namespace

void TileUsingForallPass::runOnOperation() {
  if (tilingLevel == -1) {
    LLVM_DEBUG(llvm::dbgs() << "tilingLevel not set, skip tiling\n");
    return;
  }
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<IREE::Codegen::LoweringConfigAttr> loweringConfig =
      getLoweringConfig(computeOps);
  if (failed(loweringConfig)) {
    LLVM_DEBUG(llvm::dbgs() << "can't find lowering_config, skip tiling\n");
    return;
  }

  IRRewriter rewriter(ctx);
  auto first = dyn_cast<TilingInterface>(computeOps.back());
  SmallVector<OpFoldResult> mixedNumThreads;
  SmallVector<OpFoldResult> mixedTileSizes =
      llvm::map_to_vector(loweringConfig->getTileSizeVals(tilingLevel),
                          [&](int64_t size) -> OpFoldResult {
                            return rewriter.getIndexAttr(size);
                          });

  Attribute bX = gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimX);
  Attribute bY = gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimY);
  Attribute bZ = gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimZ);
  auto mapping = rewriter.getArrayAttr({bZ, bY, bX});

  rewriter.setInsertionPoint(first);
  FailureOr<linalg::ForallTilingResult> maybeTilingResult = failure();
  if (!mixedNumThreads.empty()) {
    maybeTilingResult =
        linalg::tileToForallOp(rewriter, first, mixedNumThreads, mapping);
  } else {
    maybeTilingResult = linalg::tileToForallOpUsingTileSizes(
        rewriter, first, mixedTileSizes, mapping);
  }
  if (failed(maybeTilingResult)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to tile using forall op");
    return signalPassFailure();
  }
  rewriter.replaceOp(first, maybeTilingResult->tileOp->getResults());

  {
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(ctx);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    ctx->getLoadedDialect<tensor::TensorDialect>()->getCanonicalizationPatterns(
        patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTileUsingForallPass(int32_t tilingLevel) {
  return std::make_unique<TileUsingForallPass>(tilingLevel);
}

} // namespace mlir::iree_compiler
