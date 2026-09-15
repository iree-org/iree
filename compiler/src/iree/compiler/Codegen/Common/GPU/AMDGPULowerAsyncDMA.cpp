// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree/compiler/Codegen/Common/GPU/GPUNestedLayoutUtils.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/Repeated.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-amdgpu-lower-async-dma"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_AMDGPULOWERASYNCDMAPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

using namespace IREE::VectorExt;

namespace {

static IREE::Codegen::XORShuffleAttr getDestXorSwizzleAttr(Value dest) {
  while (auto viewOp = dest.getDefiningOp<ViewLikeOpInterface>()) {
    dest = viewOp.getViewSource();
  }
  if (auto hintOp = dest.getDefiningOp<IREE::Codegen::SwizzleHintOp>()) {
    return dyn_cast<IREE::Codegen::XORShuffleAttr>(hintOp.getSwizzle());
  }
  return {};
}

struct LowerAsyncDMA final : OpRewritePattern<IREE::GPU::AsyncDMAOp> {
  LowerAsyncDMA(MLIRContext *context, Value threadId, int64_t subgroupSize,
                int64_t numThreads)
      : OpRewritePattern(context), threadId(threadId),
        subgroupSize(subgroupSize), numThreads(numThreads) {}

  LogicalResult matchAndRewrite(IREE::GPU::AsyncDMAOp op,
                                PatternRewriter &rewriter) const override {
    LDBG() << "Processing AsyncDMAOp: " << op;

    // Currently, only support lowering of operations with all-true in_bounds.
    // TODO(#23782): Support cases where some dimensions are out-of-bounds.
    if (std::optional<ArrayAttr> inBounds = op.getInBounds();
        inBounds && llvm::any_of(*inBounds, [](Attribute attr) {
          return !cast<BoolAttr>(attr).getValue();
        })) {
      return rewriter.notifyMatchFailure(
          op, "requires all in_bounds values to be true");
    }

    // Only handle scalar-index (contiguous) case for now.
    if (op.hasGatherIndices()) {
      return rewriter.notifyMatchFailure(op, "gather indices not supported");
    }

    // Validate source is fat_raw_buffer memref.
    auto sourceType = dyn_cast<MemRefType>(op.getSource().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "source is not a memref");
    }
    if (!hasAMDGPUFatRawBufferAddressSpace(sourceType)) {
      return rewriter.notifyMatchFailure(
          op, "source does not have fat_raw_buffer address space");
    }
    // For now, only support contiguous memrefs.
    // TODO(#23782): Relax this constraint.
    if (!sourceType.areTrailingDimsContiguous(sourceType.getRank())) {
      return rewriter.notifyMatchFailure(op, "source memref not contiguous");
    }

    // Validate dest is workgroup memref.
    auto destType = dyn_cast<MemRefType>(op.getDest().getType());
    if (!destType) {
      return rewriter.notifyMatchFailure(op, "dest is not a memref");
    }
    if (!hasSharedMemoryAddressSpace(destType)) {
      return rewriter.notifyMatchFailure(
          op, "dest does not have workgroup address space");
    }

    // For now, only support contiguous memrefs.
    // TODO(#23782): Relax this constraint.
    if (!destType.areTrailingDimsContiguous(destType.getRank())) {
      return rewriter.notifyMatchFailure(op,
                                         "destination memref not contiguous");
    }

    // Get GPU target and DMA sizes.
    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!target) {
      return rewriter.notifyMatchFailure(op, "no GPU target attribute");
    }

    ArrayRef<int64_t> dmaSizes;
    if (DenseI64ArrayAttr dmaSizesAttr = target.getWgp().getDmaSizes()) {
      dmaSizes = dmaSizesAttr.asArrayRef();
    }
    if (dmaSizes.empty()) {
      return rewriter.notifyMatchFailure(op, "no DMA sizes available");
    }

    // Derive the transfer shape from the transfer size attached to the
    // async_dma rather than the memref shape, so dynamic dest memrefs are
    // supported.
    auto transferVectorType = cast<VectorType>(op.getTransferType());
    SmallVector<int64_t> transferShape(transferVectorType.getShape());

    int64_t elementBitWidth = destType.getElementTypeBitWidth();
    int64_t destRank = destType.getRank();
    MLIRContext *context = op.getContext();

    IREE::Codegen::XORShuffleAttr destSwizzle =
        getDestXorSwizzleAttr(op.getDest());
    std::optional<int64_t> swizzleAccessElems;
    if (destSwizzle) {
      // When dest has a swizzle_hint, only use DMA sizes where elementsPerDMA
      // fits within one swizzle access block. This ensures gather_to_lds writes
      // contiguous dest elements that all map to contiguous source elements
      // under the XOR permutation.
      swizzleAccessElems = destSwizzle.getAccessElementCount();
    }

    FailureOr<NestedLayoutAttr> dmaLayoutOrFailure =
        getGlobalLoadDMALayout(context, transferShape, numThreads, subgroupSize,
                               elementBitWidth, dmaSizes, swizzleAccessElems);
    if (failed(dmaLayoutOrFailure)) {
      return rewriter.notifyMatchFailure(op, "failed to compute DMA layout");
    }
    NestedLayoutAttr dmaLayout = *dmaLayoutOrFailure;

    // Delinearize thread ID for DMA layout.
    SmallVector<Value> warpIndices, threadIndices;
    if (failed(populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
                                            dmaLayout, warpIndices,
                                            threadIndices))) {
      return rewriter.notifyMatchFailure(op, "failed to delinearize thread ID");
    }

    // Compute the element tile shape and transfer type.
    int64_t transferElements = llvm::product_of(dmaLayout.getElementTile());
    VectorType transferType =
        VectorType::get({transferElements}, destType.getElementType());

    // Get distributed shape for iteration.
    SmallVector<int64_t> distributedShape = dmaLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(dmaLayout);

    // Get permutation map (or identity).
    AffineMap permutationMap =
        AffineMap::getMultiDimIdentityMap(destRank, context);
    if (auto mapAttr = op.getPermutationMapAttr()) {
      permutationMap = mapAttr.getValue();
    }
    AffineMap destIdentityMap =
        AffineMap::getMultiDimIdentityMap(destRank, context);

    // Build zero thread indices for dest (uniform across subgroup).
    Location loc = op.getLoc();
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    llvm::Repeated<Value> zeroThreadIndices(destRank, c0);

    // Emit gather_to_lds for each batch/outer tile offset.
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distributedShape, tileShape)) {

      // Dest indices: uniform (zero thread contribution).
      SmallVector<Value> destBaseIndices(op.getDestIndices());
      SmallVector<Value> dstIndices = getTransferIndicesFromNestedLayout(
          rewriter, destBaseIndices, offsets, dmaLayout, destIdentityMap,
          warpIndices, zeroThreadIndices);

      if (!destSwizzle) {
        // Source indices: divergent (include thread contribution).
        SmallVector<Value> srcBaseIndices(op.getSourceIndices());
        SmallVector<Value> srcIndices = getTransferIndicesFromNestedLayout(
            rewriter, srcBaseIndices, offsets, dmaLayout, permutationMap,
            warpIndices, threadIndices);

        amdgpu::GatherToLDSOp::create(rewriter, loc, op.getSource(), srcIndices,
                                      op.getDest(), dstIndices,
                                      TypeAttr::get(transferType));
        continue;
      }
      // Swizzled source indices: compute per-lane tile-relative offset,
      // apply swizzle (self-inverse), delinearize, map to source coords.
      // Zero base because the swizzle operates on tile-relative positions;
      // source base indices are added via the permutation map loop below.
      llvm::Repeated<Value> zeroBase(destRank, c0);
      SmallVector<Value> multiDimLane = getTransferIndicesFromNestedLayout(
          rewriter, zeroBase, offsets, dmaLayout, destIdentityMap, warpIndices,
          threadIndices);
      Value flatLane = affine::AffineLinearizeIndexOp::create(
          rewriter, loc, multiDimLane, transferShape, /*disjoint=*/true);
      Value swizzledFlat = applyInverseXorSwizzleToDMASourceOffset(
          rewriter, loc, flatLane, destSwizzle, op.getDest());

      auto coords = affine::AffineDelinearizeIndexOp::create(
          rewriter, loc, swizzledFlat, transferShape);

      SmallVector<Value> srcIndices(op.getSourceIndices());
      for (auto [i, dim] : llvm::enumerate(permutationMap.getResults())) {
        unsigned pos = cast<AffineDimExpr>(dim).getPosition();
        srcIndices[pos] = arith::AddIOp::create(rewriter, loc, srcIndices[pos],
                                                coords.getResult(i));
      }
      amdgpu::GatherToLDSOp::create(rewriter, loc, op.getSource(), srcIndices,
                                    op.getDest(), dstIndices,
                                    TypeAttr::get(transferType));
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  Value threadId;
  int64_t subgroupSize;
  int64_t numThreads;
};

struct AMDGPULowerAsyncDMAPass final
    : impl::AMDGPULowerAsyncDMAPassBase<AMDGPULowerAsyncDMAPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
        getWorkgroupSize(funcOp);
    if (!maybeWorkgroupSize) {
      return;
    }

    std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
    if (!subgroupSize) {
      return;
    }
    int64_t numThreads = llvm::product_of(*maybeWorkgroupSize);

    // Create linear thread ID.
    IRRewriter rewriter(funcOp);
    rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    Location loc = funcOp.getLoc();

    SmallVector<int64_t> workgroupSize(*maybeWorkgroupSize);
    workgroupSize.resize(3, 1);
    SmallVector<Value> threadGrid = {
        rewriter.createOrFold<gpu::ThreadIdOp>(loc, gpu::Dimension::z),
        rewriter.createOrFold<gpu::ThreadIdOp>(loc, gpu::Dimension::y),
        rewriter.createOrFold<gpu::ThreadIdOp>(loc, gpu::Dimension::x)};
    std::reverse(workgroupSize.begin(), workgroupSize.end());
    Value linearThreadId = affine::AffineLinearizeIndexOp::create(
        rewriter, loc, threadGrid, workgroupSize, /*disjoint=*/true);

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerAsyncDMA>(&getContext(), linearThreadId, *subgroupSize,
                                numThreads);

    walkAndApplyPatterns(funcOp, std::move(patterns));

    // Verify all AsyncDMAOps were lowered.
    WalkResult result = funcOp.walk([&](IREE::GPU::AsyncDMAOp op) {
      op.emitOpError("failed to lower async_dma to gather_to_lds");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
