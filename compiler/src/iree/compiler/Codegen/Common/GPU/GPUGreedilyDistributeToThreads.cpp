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

  // Perform the replacement of tiled and fused values.
  for (OpResult res : tilingInterfaceOp->getResults()) {
    if (auto replacement = tiledResults->replacements.lookup(res)) {
      rewriter.replaceAllUsesWith(res, replacement);
    }
  }

  if (tilingInterfaceOp->use_empty()) {
    rewriter.eraseOp(tilingInterfaceOp);
  }
}

static void processRegion(RewriterBase &rewriter, Region *region) {
  for (Block &block : llvm::reverse(region->getBlocks())) {
    SmallVector<Operation *> targetOps =
        llvm::map_to_vector(llvm::reverse(block.getOperations()),
                            [](Operation &op) { return &op; });
    for (Operation *op : targetOps) {
      if (op->use_empty()) {
        continue;
      }
      if (auto forall = dyn_cast<scf::ForallOp>(op)) {
        if (forallOpHasMappingType<gpu::GPUThreadMappingAttr,
                                   gpu::GPUWarpMappingAttr,
                                   IREE::GPU::LaneIdAttr>(forall)) {
          continue;
        }
      }

      if (auto tilableOp = dyn_cast<TilingInterface>(op)) {
        tileToThreads(rewriter, tilableOp);
        continue;
      }

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
