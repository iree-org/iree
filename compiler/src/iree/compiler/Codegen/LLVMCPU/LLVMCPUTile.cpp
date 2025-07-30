// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-tile"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUTILEPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

/// This pass tiles all the TilingInterface operations. The `tilingLevel` must
/// be specified. It picks the `tilingLevel`-th list as tiling sizes from
/// lowering_config.
struct LLVMCPUTilePass : impl::LLVMCPUTilePassBase<LLVMCPUTilePass> {
  using impl::LLVMCPUTilePassBase<LLVMCPUTilePass>::LLVMCPUTilePassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};

void LLVMCPUTilePass::runOnOperation() {
  if (tilingLevel == -1) {
    LDBG() << "tilingLevel not set, skip tiling";
    return;
  }
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  for (auto computeOp : computeOps) {
    auto op = dyn_cast<TilingInterface>(computeOp);
    if (!op || op.getLoopIteratorTypes().empty())
      continue;

    // For now do not tile `tensor.pad` operations. The `tensor.pad`
    // operations might be those introduced by the padding-based
    // codegeneration strategy. Those are not meant to be tiled again.
    // Need a better way for handling this, but this works for now.
    if (isa<tensor::PadOp>(computeOp))
      continue;

    IREE::Codegen::LoweringConfigAttrInterface maybeLoweringConfig =
        getLoweringConfig(op);
    if (!maybeLoweringConfig) {
      LDBG() << "can't find lowering_config, skip tiling";
      continue;
    }
    if (!maybeLoweringConfig.hasTilingLevel(tilingLevel)) {
      LDBG() << "target tiling level does not exist";
      continue;
    }

    LDBG() << "candidate: " << op;
    if (skipRootOp && maybeLoweringConfig.hasWorkgroupTilingLevel()) {
      LDBG() << "skip tiling on the root op";
      continue;
    }

    auto tileSizesAttr = dyn_cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        getLoweringConfig(op).getTilingLevelAttr(tilingLevel));
    SmallVector<int64_t> tileSizes(tileSizesAttr.getSizes());
    SmallVector<bool> tileScalableFlags(tileSizesAttr.getScalableFlags());
    scf::SCFTilingOptions tilingOptions;
    setSCFTileSizes(tilingOptions, op, std::move(tileSizes),
                    std::move(tileScalableFlags));
    if (llvm::all_of(tileSizes, [](int64_t v) { return v == 0; })) {
      LDBG() << "tiling sizes are all zeros, skip tiling";
      continue;
    }

    IRRewriter rewriter(context);
    scf::SCFTilingOptions options{};
    setSCFTileSizes(options, op, std::move(tileSizes),
                    std::move(tileScalableFlags));
    FailureOr<scf::SCFTilingResult> tiledResults =
        scf::tileUsingSCF(rewriter, op, options);
    if (failed(tiledResults))
      continue;
    rewriter.replaceOp(op, tiledResults->replacements);
  }

  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    LDBG() << "----- cleanup failed -----";
    return signalPassFailure();
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTilePass(int64_t tilingLevel, bool skipRootOp) {
  LLVMCPUTilePassOptions options;
  options.tilingLevel = tilingLevel;
  options.skipRootOp = skipRootOp;
  return std::make_unique<LLVMCPUTilePass>(options);
}

} // namespace mlir::iree_compiler
