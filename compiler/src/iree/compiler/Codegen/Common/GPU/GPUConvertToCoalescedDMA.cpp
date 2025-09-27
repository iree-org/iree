// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-convert-to-coalesced-dma"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCONVERTTOCOALESCEDDMAPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

// Tiles linalg_ext.gather ops according to the subgroup tiling level in the
// lowering config and converts them to scf.forall with coalesced_gather_dma.
//
// Before:
//   %result = linalg_ext.gather
//       ins(%source, %indices : tensor<128xf32>, tensor<64xi32>)
//       outs(%init : tensor<64xf32>)
//
// After (with subgroup tile size [32]):
//   %result = scf.forall (%iv) in (64) step (32)
//       shared_outs(%shared = %init) -> (tensor<64xf32>) {
//     %indices_slice = tensor.extract_slice %indices[%iv] [32] [1]
//     %indices_idx = arith.index_cast %indices_slice : tensor<32xi32> to
//     tensor<32xindex> scf.forall.in_parallel {
//       %dma = iree_gpu.coalesced_gather_dma %indices_idx, %source, %shared
//     }
//   } {mapping = [#gpu.warp<linear_dim_0>]}
//
struct ConverGatherOpToCoalescedDMA
    : public OpRewritePattern<IREE::LinalgExt::GatherOp> {
  using OpRewritePattern<IREE::LinalgExt::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    // The GatherOp must be marked with a lowering config.
    auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(gatherOp);
    if (!loweringConfig) {
      return failure();
    }

    auto subgroupTileSizes = loweringConfig.getTilingLevelSizes(
        rewriter, llvm::to_underlying(IREE::GPU::TilingLevel::Subgroup),
        gatherOp);
    if (subgroupTileSizes.empty()) {
      return failure();
    }

    // Actual tiling to CoalescedDMA:
    Value source = gatherOp.getInputs()[0];
    Value indices = gatherOp.getInputs()[1];
    Value init = gatherOp.getOutputs()[0];

    auto initType = cast<RankedTensorType>(init.getType());
    int64_t rank = initType.getRank();

    SmallVector<OpFoldResult> tileSizes;
    SmallVector<OpFoldResult> lowerBounds;
    SmallVector<OpFoldResult> upperBounds;
    SmallVector<OpFoldResult> steps;

    Location loc = gatherOp.getLoc();
    for (int64_t i = 0; i < rank; ++i) {
      OpFoldResult tileSize = subgroupTileSizes[i];
      tileSizes.push_back(tileSize);
      lowerBounds.push_back(rewriter.getIndexAttr(0));
      upperBounds.push_back(rewriter.getIndexAttr(initType.getDimSize(i)));
      steps.push_back(tileSize);
    }

    SmallVector<Attribute> mapping;
    for (int64_t i = 0; i < rank; ++i) {
      mapping.push_back(gpu::GPUWarpMappingAttr::get(
          rewriter.getContext(),
          static_cast<gpu::MappingId>(
              static_cast<int>(gpu::MappingId::LinearDim0) + (rank - 1 - i))));
    }

    auto forallOp =
        rewriter.create<scf::ForallOp>(loc, lowerBounds, upperBounds, steps,
                                       init, rewriter.getArrayAttr(mapping));

    Block *body = forallOp.getBody();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(body);

    SmallVector<Value> ivs(body->getArguments().begin(),
                           body->getArguments().begin() + rank);
    Value sharedOut = body->getArguments()[rank];

    SmallVector<OpFoldResult> offsets;
    for (Value iv : ivs) {
      offsets.push_back(iv);
    }
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto extractedIndices = rewriter.create<tensor::ExtractSliceOp>(
        loc, indices, offsets, tileSizes, strides);

    auto indicesType = cast<RankedTensorType>(extractedIndices.getType());
    auto indexIndicesType =
        RankedTensorType::get(indicesType.getShape(), rewriter.getIndexType());
    Value indexIndices = rewriter.create<arith::IndexCastOp>(
        loc, indexIndicesType, extractedIndices.getResult());

    auto inParallelOp = cast<scf::InParallelOp>(body->getTerminator());

    Block &inParallelBlock = inParallelOp.getRegion().front();
    rewriter.setInsertionPointToStart(&inParallelBlock);

    rewriter.create<IREE::GPU::CoalescedGatherDMAOp>(
        loc, sharedOut.getType(), indexIndices, source, sharedOut);

    rewriter.replaceOp(gatherOp, forallOp.getResults());

    return success();
  }
};

struct GPUConvertToCoalescedDMAPass final
    : impl::GPUConvertToCoalescedDMAPassBase<GPUConvertToCoalescedDMAPass> {
  using GPUConvertToCoalescedDMAPassBase::GPUConvertToCoalescedDMAPassBase;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    LLVM_DEBUG(llvm::dbgs()
               << "Running GPUConvertToCoalescedDMAPass on function: "
               << funcOp.getName() << "\n");

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<ConverGatherOpToCoalescedDMA>(context);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "Pattern application failed\n");
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "GPUConvertToCoalescedDMAPass completed\n");
  }
};

} // namespace

} // namespace mlir::iree_compiler
