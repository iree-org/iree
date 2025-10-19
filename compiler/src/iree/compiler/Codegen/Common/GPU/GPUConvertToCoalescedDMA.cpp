// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-convert-to-coalesced-dma"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCONVERTTOCOALESCEDDMAPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Get subgroup tile sizes from the lowering config.
static SmallVector<OpFoldResult> getSubgroupTileSizes(OpBuilder &builder,
                                                      Operation *op) {
  auto loweringConfig = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  if (!loweringConfig) {
    return {};
  }

  SmallVector<OpFoldResult> tileSizes = loweringConfig.getTilingLevelSizes(
      builder, llvm::to_underlying(IREE::GPU::TilingLevel::Subgroup), op);
  return tileSizes;
}

/// Create GPU warp mapping attributes for the given rank.
static SmallVector<Attribute> getWarpMapping(MLIRContext *ctx, int64_t rank) {
  SmallVector<Attribute> mapping;
  for (int64_t i = 0; i < rank; ++i) {
    auto mappingId = static_cast<gpu::MappingId>(
        static_cast<int>(gpu::MappingId::LinearDim0) + (rank - 1 - i));
    mapping.push_back(gpu::GPUWarpMappingAttr::get(ctx, mappingId));
  }
  return mapping;
}

/// Create GPU thread mapping attributes for the given rank.
static SmallVector<Attribute> getThreadMapping(MLIRContext *ctx, int64_t rank) {
  SmallVector<Attribute> mapping;
  for (int64_t i = 0; i < rank; ++i) {
    auto mappingId = static_cast<gpu::MappingId>(
        static_cast<int>(gpu::MappingId::LinearDim0) + (rank - 1 - i));
    mapping.push_back(gpu::GPUThreadMappingAttr::get(ctx, mappingId));
  }
  return mapping;
}

/// Get thread tile sizes from the lowering config.
static SmallVector<OpFoldResult> getThreadTileSizes(OpBuilder &builder,
                                                    Operation *op) {
  auto loweringConfig = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  if (!loweringConfig) {
    return {};
  }

  SmallVector<OpFoldResult> tileSizes = loweringConfig.getTilingLevelSizes(
      builder, llvm::to_underlying(IREE::GPU::TilingLevel::Thread), op);
  return tileSizes;
}

/// Callback for tiled loop generation with subgroup tiling.
static scf::SCFTilingOptions getSubgroupTilingOptions(OpBuilder &builder,
                                                      Operation *op) {
  scf::SCFTilingOptions options;
  options.setTileSizeComputationFunction(
      [](OpBuilder &b, Operation *op) { return getSubgroupTileSizes(b, op); });

  // Use forall loop type
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

  // Set the mapping to warp-level
  auto gatherOp = dyn_cast<IREE::LinalgExt::GatherOp>(op);
  if (gatherOp) {
    auto initType = cast<RankedTensorType>(gatherOp.getOutput().getType());
    int64_t rank = initType.getRank();
    options.setMapping(getWarpMapping(builder.getContext(), rank));
  }

  return options;
}

struct GPUConvertToCoalescedDMAPass final
    : impl::GPUConvertToCoalescedDMAPassBase<GPUConvertToCoalescedDMAPass> {
  using GPUConvertToCoalescedDMAPassBase::GPUConvertToCoalescedDMAPassBase;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    // Tile linalg_ext.gather operations to subgroup level only
    SmallVector<IREE::LinalgExt::GatherOp> gatherOps;
    funcOp->walk([&](IREE::LinalgExt::GatherOp gatherOp) {
      gatherOps.push_back(gatherOp);
    });

    for (auto gatherOp : gatherOps) {
      scf::SCFTilingOptions options =
          getSubgroupTilingOptions(rewriter, gatherOp);

      rewriter.setInsertionPoint(gatherOp);
      FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
          rewriter, cast<TilingInterface>(gatherOp.getOperation()), options);

      if (failed(tilingResult)) {
        continue;
      }

      // Replace the original operation with the tiled version
      rewriter.replaceOp(gatherOp, tilingResult->replacements);
    }

    // Second pass: Tile inner gather ops to thread level using tiling framework
    funcOp->walk([&](IREE::LinalgExt::GatherOp gatherOp) {
      // Get thread tile sizes
      SmallVector<OpFoldResult> threadTileSizes =
          getThreadTileSizes(rewriter, gatherOp);
      if (threadTileSizes.empty()) {
        return WalkResult::advance();
      }

      auto initType = cast<RankedTensorType>(gatherOp.getOutput().getType());
      int64_t rank = initType.getRank();

      // Create thread-level tiling options
      scf::SCFTilingOptions threadOptions;
      threadOptions.setTileSizeComputationFunction(
          [threadTileSizes](OpBuilder &b, Operation *op) {
            return threadTileSizes;
          });
      threadOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
      threadOptions.setMapping(getThreadMapping(context, rank));

      rewriter.setInsertionPoint(gatherOp);
      FailureOr<scf::SCFTilingResult> threadTilingResult = scf::tileUsingSCF(
          rewriter, cast<TilingInterface>(gatherOp.getOperation()),
          threadOptions);

      if (failed(threadTilingResult)) {
        return WalkResult::advance();
      }

      // Find the forall op
      scf::ForallOp forallOp = nullptr;
      for (auto loop : threadTilingResult->loops) {
        if (auto fop = dyn_cast<scf::ForallOp>(loop.getOperation())) {
          forallOp = fop;
          break;
        }
      }
      if (!forallOp) {
        return WalkResult::advance();
      }

      // Find the tiled gather inside the forall
      IREE::LinalgExt::GatherOp innerGather = nullptr;
      forallOp->walk([&](IREE::LinalgExt::GatherOp op) {
        innerGather = op;
        return WalkResult::interrupt();
      });

      if (innerGather) {
        // Get the shared_outs block argument
        Block *forallBody = forallOp.getBody();
        Value sharedOut = forallBody->getArguments()[rank];

        // Find and remove any existing parallel_insert_slice in the in_parallel
        // region
        auto inParallelOp =
            cast<scf::InParallelOp>(forallBody->getTerminator());
        Block &inParallelBlock = inParallelOp.getRegion().front();

        SmallVector<tensor::ParallelInsertSliceOp> toErase;
        for (auto &op : inParallelBlock) {
          if (auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op)) {
            toErase.push_back(insertOp);
          }
        }

        // Create the DMA op in the in_parallel region
        rewriter.setInsertionPointToStart(&inParallelBlock);
        auto dmaOp = rewriter.create<IREE::GPU::CoalescedGatherDMAOp>(
            innerGather.getLoc(), sharedOut.getType(), innerGather.getIndices(),
            innerGather.getSource(), sharedOut);

        // Erase the parallel_insert_slice ops
        for (auto insertOp : toErase) {
          rewriter.eraseOp(insertOp);
        }

        // Replace the inner gather with the DMA result
        rewriter.replaceOp(innerGather, dmaOp.getResult());
      }

      // Replace the original gather with the tiled version
      rewriter.replaceOp(gatherOp, threadTilingResult->replacements);
      return WalkResult::advance();
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler
