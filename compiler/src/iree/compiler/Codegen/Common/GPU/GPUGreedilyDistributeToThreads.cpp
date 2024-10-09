// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUGREEDILYDISTRIBUTETOTHREADSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct GPUGreedilyDistributeToThreadsPass final
    : impl::GPUGreedilyDistributeToThreadsPassBase<
          GPUGreedilyDistributeToThreadsPass> {
  void runOnOperation() override;
};
} // namespace

/// Helper to tile and greedily fuse the given operation to threads. This uses
/// the iree_gpu.derived_thread_config logic internally to determine tile sizes
/// to use. This does not yield any fused operation and only replaces the tiling
/// root.
///
/// If tiling fails this returns silently (tiling is best effort). Later
/// verification steps will throw an error if distribution does not occur.
static void tileToThreads(RewriterBase &rewriter,
                          TilingInterface tilingInterfaceOp) {
  rewriter.setInsertionPoint(tilingInterfaceOp);
  auto configAttr =
      IREE::GPU::DerivedThreadConfigAttr::get(rewriter.getContext());
  SmallVector<OpFoldResult> tileSizes = configAttr.getTilingLevelSizes(
      rewriter, llvm::to_underlying(IREE::GPU::TilingLevel::Thread),
      tilingInterfaceOp);

  // Pad the tile sizes with zero.
  auto zero = rewriter.getIndexAttr(0);
  int64_t numLoops = tilingInterfaceOp.getLoopIteratorTypes().size();
  if (tileSizes.size() > numLoops) {
    return;
  }
  while (tileSizes.size() < numLoops) {
    tileSizes.push_back(zero);
  }

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

  // Use a "descending" relative thread mapping (ordered from slowest to
  // fastest varying ID). For example, [linear_dim_n, ..., linear_dim_0].
  SmallVector<Attribute> mapping;
  int idx = 0;
  for (auto size : tileSizes) {
    if (!isConstantIntValue(size, 0)) {
      unsigned mappingId =
          static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx++;
      mapping.push_back(gpu::GPUThreadMappingAttr::get(
          rewriter.getContext(), static_cast<gpu::MappingId>(mappingId)));
    }
  }
  tilingOptions.setMapping(llvm::to_vector(llvm::reverse(mapping)));

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);

  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    // Always fuse but never yield a replacement.
    return scf::SCFTileAndFuseOptions::ControlFnResult{
        /*yieldProducerReplacement=*/false};
  };
  tileAndFuseOptions.setFusionControlFn(controlFn);

  FailureOr<scf::SCFTileAndFuseResult> tiledResults =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                tileAndFuseOptions);
  if (failed(tiledResults)) {
    return;
  }

  // Perform the replacement of the tiling root.
  for (OpResult res : tilingInterfaceOp->getResults()) {
    if (auto replacement = tiledResults->replacements.lookup(res)) {
      rewriter.replaceAllUsesWith(res, replacement);
    }
  }

  if (tilingInterfaceOp->use_empty()) {
    rewriter.eraseOp(tilingInterfaceOp);
  }
}

/// Recursively process the given region and tile all tilable operations not
/// present within an `scf.forall` with thread/warp/lane mapping.
static void processRegion(RewriterBase &rewriter, Region *region) {
  // Process the region blocks in reverse.
  for (Block &block : llvm::reverse(region->getBlocks())) {
    // Save a reversed list of operations within the block. Ops will be
    // greedily tiled + fused in reverse so that if a producer can be fused
    // with a consumer we only distribute the producer once via fusion.
    SmallVector<Operation *> targetOps =
        llvm::map_to_vector(llvm::reverse(block.getOperations()),
                            [](Operation &op) { return &op; });
    // Skip all unused ops (possibly from tiling).
    for (Operation *op : targetOps) {
      if (op->use_empty()) {
        continue;
      }
      // Skip all operations contained within an `scf.forall` mapped to threads
      // warps or lanes. These are already distributed and fine to leave as is.
      if (auto forall = dyn_cast<scf::ForallOp>(op)) {
        if (forallOpHasMappingType<gpu::GPUThreadMappingAttr,
                                   gpu::GPUWarpMappingAttr,
                                   IREE::GPU::LaneIdAttr>(forall)) {
          continue;
        }
      }

      // If an op implements the tiling interface, try to greedily tile + fuse.
      if (auto tilableOp = dyn_cast<TilingInterface>(op)) {
        tileToThreads(rewriter, tilableOp);
        continue;
      }

      // Else recursively process all nested operations.
      for (auto &region : op->getRegions()) {
        processRegion(rewriter, &region);
      }
    }
  }
}

void GPUGreedilyDistributeToThreadsPass::runOnOperation() {
  auto funcOp = getOperation();

  IRRewriter rewriter(funcOp->getContext());
  for (auto &region : funcOp->getRegions()) {
    processRegion(rewriter, &region);
  }
}

} // namespace mlir::iree_compiler
