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
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Helper function to validate and extract constant tile sizes from
// OpFoldResults
static FailureOr<SmallVector<int64_t>>
getConstantTileSizes(ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<int64_t> staticTileSizes;
  for (OpFoldResult ofr : tileSizes) {
    std::optional<int64_t> tileSize = getConstantIntValue(ofr);
    if (!tileSize) {
      return failure();
    }
    staticTileSizes.push_back(*tileSize);
  }
  return staticTileSizes;
}

// Helper function to create GPU mapping attributes for forall loops
static SmallVector<Attribute>
createGPUMappingAttrs(MLIRContext *ctx, int64_t rank, bool useWarpMapping) {
  SmallVector<Attribute> mapping;
  for (int64_t i = 0; i < rank; ++i) {
    auto mappingId = static_cast<gpu::MappingId>(
        static_cast<int>(gpu::MappingId::LinearDim0) + (rank - 1 - i));
    if (useWarpMapping) {
      mapping.push_back(gpu::GPUWarpMappingAttr::get(ctx, mappingId));
    } else {
      mapping.push_back(gpu::GPUThreadMappingAttr::get(ctx, mappingId));
    }
  }
  return mapping;
}

// Helper function to cast indices to index type if needed
static Value castIndicesToIndexType(OpBuilder &rewriter, Location loc,
                                    Value indices) {
  auto indicesType = cast<RankedTensorType>(indices.getType());
  if (indicesType.getElementType().isIndex()) {
    return indices;
  }
  auto indexIndicesType =
      RankedTensorType::get(indicesType.getShape(), rewriter.getIndexType());
  return rewriter.create<arith::IndexCastOp>(loc, indexIndicesType, indices);
}

// Tiles linalg_ext.gather ops with two-level nested scf.forall structure:
// 1. Outer forall with warp-level (subgroup) mapping
// 2. Inner forall with thread-level mapping containing coalesced_gather_dma
//
// Example transformation:
//   %result = linalg_ext.gather
//       ins(%source, %indices : tensor<1024xf32>, tensor<128x16xindex>)
//       outs(%init : tensor<128x16xf32>)
//
// After (with subgroup tiles [8, 16] and thread tiles [32, 1]):
//   %result = scf.forall (%wg_i, %wg_j) in (128, 16) step (8, 16)
//       shared_outs(%wg_out = %init) -> (tensor<128x16xf32>) {
//     %indices_wg_slice = tensor.extract_slice %indices[%wg_i, %wg_j] [8, 16]
//     %dest_wg_slice = tensor.extract_slice %wg_out[%wg_i, %wg_j] [8, 16]
//
//     %inner_result = scf.forall (%sg_i, %sg_j) in (8, 16) step (32, 1)
//         shared_outs(%sg_out = %dest_wg_slice) -> (tensor<8x16xf32>) {
//       scf.forall.in_parallel {
//         iree_gpu.coalesced_gather_dma %indices_wg_slice, %source into %sg_out
//       }
//     } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
//
//     scf.forall.in_parallel {
//       tensor.parallel_insert_slice %inner_result into %wg_out[%wg_i, %wg_j]
//     }
//   } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
//
struct ConvertGatherOpToCoalescedDMA
    : public OpRewritePattern<IREE::LinalgExt::GatherOp> {
  using OpRewritePattern<IREE::LinalgExt::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    Location loc = gatherOp.getLoc();

    // Get lowering config
    auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(gatherOp);
    if (!loweringConfig) {
      return rewriter.notifyMatchFailure(
          gatherOp, "missing lowering config for gather op");
    }

    // Get and validate subgroup (warp-level) tile sizes for outer forall
    auto subgroupTileSizes = loweringConfig.getTilingLevelSizes(
        rewriter, llvm::to_underlying(IREE::GPU::TilingLevel::Subgroup),
        gatherOp);
    if (subgroupTileSizes.empty()) {
      return rewriter.notifyMatchFailure(
          gatherOp, "missing subgroup tile sizes in lowering config");
    }

    auto staticSubgroupTileSizes = getConstantTileSizes(subgroupTileSizes);
    if (failed(staticSubgroupTileSizes)) {
      return rewriter.notifyMatchFailure(
          gatherOp, "subgroup tile sizes must be constant");
    }

    // Get and validate thread tile sizes for inner forall
    auto threadTileSizes = loweringConfig.getTilingLevelSizes(
        rewriter, llvm::to_underlying(IREE::GPU::TilingLevel::Thread),
        gatherOp);
    if (threadTileSizes.empty()) {
      return rewriter.notifyMatchFailure(
          gatherOp, "missing thread tile sizes in lowering config");
    }

    auto staticThreadTileSizes = getConstantTileSizes(threadTileSizes);
    if (failed(staticThreadTileSizes)) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "thread tile sizes must be constant");
    }

    // Get operands and validate types
    Value source = gatherOp.getSource();
    Value indices = gatherOp.getIndices();
    Value init = gatherOp.getOutput();

    auto initType = cast<RankedTensorType>(init.getType());
    int64_t rank = initType.getRank();

    if (!initType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          gatherOp, "init tensor must have static dimensions");
    }

    if (rank != staticSubgroupTileSizes->size() ||
        rank != staticThreadTileSizes->size()) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "tile sizes must match tensor rank");
    }

    // Verify that subgroup tile sizes divide the dimensions evenly
    for (int64_t i = 0; i < rank; ++i) {
      if (initType.getDimSize(i) % (*staticSubgroupTileSizes)[i] != 0) {
        return rewriter.notifyMatchFailure(
            gatherOp, "subgroup tile size must divide dimension size evenly");
      }
    }

    // Create outer forall with warp mapping
    SmallVector<OpFoldResult> outerLowerBounds(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> outerUpperBounds;
    for (int64_t dimSize : initType.getShape()) {
      outerUpperBounds.push_back(rewriter.getIndexAttr(dimSize));
    }

    SmallVector<Attribute> outerMapping =
        createGPUMappingAttrs(rewriter.getContext(), rank,
                              /*useWarpMapping=*/true);
    auto outerForallOp = rewriter.create<scf::ForallOp>(
        loc, outerLowerBounds, outerUpperBounds, subgroupTileSizes, init,
        rewriter.getArrayAttr(outerMapping));

    // Build the body of outer forall
    OpBuilder::InsertionGuard guard(rewriter);
    Block *outerBody = outerForallOp.getBody();
    rewriter.setInsertionPointToStart(outerBody);

    SmallVector<Value> outerIVs(outerBody->getArguments().begin(),
                                outerBody->getArguments().begin() + rank);
    Value outerSharedOut = outerBody->getArguments()[rank];

    // Extract slices for the workgroup tile
    SmallVector<OpFoldResult> outerOffsets(outerIVs.begin(), outerIVs.end());
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto indicesWgSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, indices, outerOffsets, subgroupTileSizes, strides);

    auto destWgSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, outerSharedOut, outerOffsets, subgroupTileSizes, strides);

    // Cast indices to index type if needed
    Value indexIndices = castIndicesToIndexType(rewriter, loc, indicesWgSlice);

    // Create inner forall with thread mapping
    SmallVector<OpFoldResult> innerLowerBounds(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> innerUpperBounds = subgroupTileSizes;

    SmallVector<Attribute> innerMapping =
        createGPUMappingAttrs(rewriter.getContext(), rank,
                              /*useWarpMapping=*/false);
    auto innerForallOp = rewriter.create<scf::ForallOp>(
        loc, innerLowerBounds, innerUpperBounds, threadTileSizes,
        destWgSlice.getResult(), rewriter.getArrayAttr(innerMapping));

    // Build the body of inner forall with DMA op
    Block *innerBody = innerForallOp.getBody();
    rewriter.setInsertionPointToStart(innerBody);

    Value innerSharedOut = innerBody->getArguments()[rank];
    auto innerInParallelOp =
        cast<scf::InParallelOp>(innerBody->getTerminator());
    Block &innerInParallelBlock = innerInParallelOp.getRegion().front();
    rewriter.setInsertionPointToStart(&innerInParallelBlock);

    rewriter.create<IREE::GPU::CoalescedGatherDMAOp>(
        loc, innerSharedOut.getType(), indexIndices, source, innerSharedOut);

    // Insert the result of inner forall back into outer forall's output.
    rewriter.setInsertionPointAfter(innerForallOp);
    auto outerInParallelOp =
        cast<scf::InParallelOp>(outerBody->getTerminator());
    Block &outerInParallelBlock = outerInParallelOp.getRegion().front();
    rewriter.setInsertionPointToStart(&outerInParallelBlock);

    rewriter.create<tensor::ParallelInsertSliceOp>(
        loc, innerForallOp.getResult(0), outerSharedOut, outerOffsets,
        subgroupTileSizes, strides);

    rewriter.replaceOp(gatherOp, outerForallOp.getResults());
    return success();
  }
};

struct GPUConvertToCoalescedDMAPass final
    : impl::GPUConvertToCoalescedDMAPassBase<GPUConvertToCoalescedDMAPass> {
  using GPUConvertToCoalescedDMAPassBase::GPUConvertToCoalescedDMAPassBase;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertGatherOpToCoalescedDMA>(context);
    walkAndApplyPatterns(funcOp, std::move(patterns));
  }
};

} // namespace

} // namespace mlir::iree_compiler
