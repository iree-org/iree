// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-padding-level"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYPADDINGLEVELPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUApplyPaddingLevelPass final
    : impl::GPUApplyPaddingLevelPassBase<GPUApplyPaddingLevelPass> {
  using GPUApplyPaddingLevelPassBase::GPUApplyPaddingLevelPassBase;
  void runOnOperation() override;
};
} // namespace

static llvm::SmallDenseSet<TilingInterface>
getTiledOps(Operation *funcOp, IREE::GPU::TilingLevel tilingLevel) {
  llvm::SmallDenseSet<TilingInterface> targets;
  unsigned opaqueLevel = llvm::to_underlying(tilingLevel);
  funcOp->walk([&](TilingInterface target) {
    // TODO: This would probably be easier with a lowering config interface
    // method that checks whether a particular level is tiled.
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(target)) {
      if (loweringConfig.hasTilingLevel(opaqueLevel)) {
        targets.insert(target);
      }
    }
  });
  return targets;
}

static LogicalResult applyPaddingLevel(RewriterBase &rewriter,
                                       TilingInterface tilingInterfaceOp,
                                       IREE::GPU::TilingLevel tilingLevel) {
  SmallVector<int64_t> tileSizes =
      getLoweringConfig(tilingInterfaceOp)
          .getStaticTilingLevelSizes(llvm::to_underlying(tilingLevel),
                                     tilingInterfaceOp);

  // Pad the tile sizes with zero.
  int64_t numLoops = tilingInterfaceOp.getLoopIteratorTypes().size();
  if (tileSizes.size() > numLoops) {
    tilingInterfaceOp.emitWarning("tileSizes.size() > numLoops");
    return failure();
  }
  while (tileSizes.size() < numLoops) {
    tileSizes.push_back(0);
  }

  SmallVector<int64_t> padSizes = llvm::map_to_vector(
      tileSizes, [](int64_t tileSize) { return tileSize == 0 ? 1 : tileSize; });

  SmallVector<int64_t> paddingDims =
      llvm::to_vector(llvm::seq<int64_t>(0, numLoops));

  auto options =
      linalg::LinalgPaddingOptions()
          .setPaddingDimensions(paddingDims)
          .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None)
          .setPadToMultipleOf(padSizes);

  if (auto linalgOp =
          dyn_cast<linalg::LinalgOp>(tilingInterfaceOp.getOperation())) {
    linalg::LinalgOp paddedOp;
    SmallVector<Value> newResults;
    SmallVector<tensor::PadOp> padOps;
    if (failed(linalg::rewriteAsPaddedOp(rewriter, linalgOp, options, paddedOp,
                                         newResults, padOps))) {
      linalgOp.emitWarning("failed to pad ops");
      return failure();
    }
    rewriter.replaceOp(linalgOp, paddedOp);
  } else {
    tilingInterfaceOp.emitWarning("not a linalg op");
    return failure();
  }

  return success();
}

void GPUApplyPaddingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  llvm::SmallDenseSet<TilingInterface> targetOps =
      getTiledOps(funcOp, tilingLevel);

  IRRewriter rewriter(funcOp);
  for (TilingInterface op : targetOps) {
    // If some op does not get padded, that is fine for now.
    (void)applyPaddingLevel(rewriter, op, tilingLevel);
  }
}

} // namespace mlir::iree_compiler
