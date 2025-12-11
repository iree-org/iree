// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>
#include <optional>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-amdgpu-lower-coalesced-dma-to-gather-lds"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_AMDGPULOWERCOALESCEDDMATOGATHERLDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

static LogicalResult verifyThreadMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mappingAttr = forallOp.getMapping();
  if (!mappingAttr) {
    return failure();
  }

  // Verify that all mappings are thread mappings.
  return success(llvm::all_of(
      *mappingAttr,
      llvm::IsaPred<gpu::GPUThreadMappingAttr, IREE::GPU::LaneIdAttr>));
}

static LogicalResult
verifyMemoryLayoutContiguous(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                             PatternRewriter &rewriter) {
  // Important: check that destination memref is contiguous.
  auto destMemRefType = cast<MemRefType>(dmaOp.getInit().getType());

  if (!destMemRefType.areTrailingDimsContiguous(1)) {
    return rewriter.notifyMatchFailure(
        dmaOp,
        "destination memref does not have contiguous trailing dimension");
  }

  return success();
}

// A segment of the innermost dimension that will be transferred
// using a specific DMA size.
struct TransferSegment {
  int64_t elementsPerLane; // Number of elements each lane transfers.
  int64_t numTransfers;    // Number of full transfers for this segment.
  int64_t startOffset;     // Element offset in innermost dimension.
};

struct LowerCoalescedGatherDMAPattern final
    : public OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp> {
  using Base::Base;

  LowerCoalescedGatherDMAPattern(MLIRContext *context,
                                 ArrayRef<int64_t> targetDmaSizes)
      : OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp>(context),
        targetDmaSizes(targetDmaSizes) {}

  LogicalResult matchAndRewrite(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                                PatternRewriter &rewriter) const override {
    LDBG() << "Processing CoalescedGatherDMAOp: " << dmaOp;

    auto forallOp = dmaOp->getParentOfType<scf::ForallOp>();
    if (!forallOp) {
      return rewriter.notifyMatchFailure(
          dmaOp, "coalesced_gather_dma not inside scf.forall");
    }

    if (failed(verifyThreadMapping(forallOp))) {
      return rewriter.notifyMatchFailure(
          dmaOp, "forall does not have proper thread mapping");
    }

    if (failed(verifyMemoryLayoutContiguous(dmaOp, rewriter))) {
      return failure();
    }

    Value source = dmaOp.getSource();
    Value dest = dmaOp.getInit();

    auto sourceType = cast<MemRefType>(source.getType());

    Type elementType = sourceType.getElementType();
    int64_t elementBits = sourceType.getElementTypeBitWidth();
    LDBG() << "  Element type: " << elementType;
    LDBG() << "  Element bits: " << elementBits;

    int64_t innermostDimSize = sourceType.getShape().back();
    int64_t transferSizeBits = innermostDimSize * elementBits;
    LDBG() << "  Source innermost dimension size: " << innermostDimSize;
    LDBG() << "  Transfer size in bits: " << transferSizeBits;

    std::optional<int64_t> subgroupSize =
        getSubgroupSize(dmaOp->getParentOfType<FunctionOpInterface>());
    if (!subgroupSize.has_value()) {
      return rewriter.notifyMatchFailure(
          dmaOp, "unable to determine subgroup size from forall");
    }
    LDBG() << "Subgroup size: " << *subgroupSize;

    // Build transfer segments: try to use larger DMA sizes first, then smaller
    // sizes to cover the remainder.
    //
    // Example: subgroup_size=32, dma_sizes=[32,128], innermost_dim=<160xf32>
    //   issue 2 transfers for 160 elements:
    //   * Transfer 1: 128-bit DMA, 32*4=128 elements/transfer, 160/128=1
    //   transfer
    //   * Transfer 2: 32-bit DMA, 32*1=32 elements/transfer
    auto sortedDmaSizes = llvm::to_vector_of<int64_t>(targetDmaSizes);
    llvm::sort(sortedDmaSizes, std::greater<>());

    SmallVector<TransferSegment> segments;
    int64_t remainingElements = innermostDimSize;
    int64_t currentOffset = 0;

    for (int64_t dmaSize : sortedDmaSizes) {
      // Calculate elements per lane for this DMA size.
      if (dmaSize % elementBits != 0)
        continue;
      int64_t elementsPerLane = dmaSize / elementBits;

      // Calculate total elements per transfer (all lanes combined).
      int64_t totalElementsPerTransfer = *subgroupSize * elementsPerLane;

      // Calculate how many full transfers we can do with this DMA size.
      int64_t numTransfers = remainingElements / totalElementsPerTransfer;
      if (numTransfers > 0) {
        segments.push_back({elementsPerLane, numTransfers, currentOffset});
        LDBG() << "Segment: " << dmaSize << "-bit DMA, " << numTransfers
               << " transfers, elementsPerLane=" << elementsPerLane
               << ", startOffset=" << currentOffset;

        int64_t coveredElements = numTransfers * totalElementsPerTransfer;
        remainingElements -= coveredElements;
        currentOffset += coveredElements;
      }

      if (remainingElements == 0)
        break;
    }

    if (remainingElements != 0) {
      return rewriter.notifyMatchFailure(
          dmaOp, "innermost dimension cannot be covered by any combination "
                 "of supported DMA transfer sizes");
    }

    auto destType = cast<MemRefType>(dest.getType());
    ArrayRef<int64_t> destShape = destType.getShape();
    LDBG() << "Destination rank: " << destShape.size();

    OperandRange indices = dmaOp.getIndices();
    size_t numIndexDims = indices.size();
    LDBG() << "Number of index dimensions: " << numIndexDims;

    // Actually create the GatherToLDS ops to perform the transfer.
    rewriter.setInsertionPoint(dmaOp);

    TypedValue<IndexType> laneId = dmaOp.getLane();
    Location loc = dmaOp.getLoc();

    // Build tile sizes for outer dimensions: [1, 1, ..., 1, innermost].
    // We handle the innermost dimension separately via segments.
    SmallVector<int64_t> outerTileSizes(destShape.size(), 1);
    outerTileSizes.back() = innermostDimSize;

    // Precompute laneOffset for each unique segment type. This allows the
    // values to be reused across all rows while maintaining row-major order.
    SmallVector<Value> segmentLaneOffsets;
    for (const TransferSegment &segment : segments) {
      Value laneOffset =
          arith::MulIOp::create(rewriter, loc, laneId,
                                arith::ConstantIndexOp::create(
                                    rewriter, loc, segment.elementsPerLane));
      segmentLaneOffsets.push_back(laneOffset);
    }

    // Iterate over each row, then segments within each row.
    for (const SmallVector<int64_t> &outerOffsets :
         StaticTileOffsetRange(destShape, outerTileSizes)) {
      // For each segment, generate the transfers.
      for (auto [segmentIdx, segment] : llvm::enumerate(segments)) {
        int64_t elementsPerLane = segment.elementsPerLane;
        int64_t totalElementsPerTransfer = *subgroupSize * elementsPerLane;
        auto transferType = VectorType::get({elementsPerLane}, elementType);
        Value laneOffset = segmentLaneOffsets[segmentIdx];

        // Generate transfers for this segment.
        for (int64_t transferIdx = 0; transferIdx < segment.numTransfers;
             ++transferIdx) {
          int64_t innerOffset =
              segment.startOffset + transferIdx * totalElementsPerTransfer;

          SmallVector<Value> srcIndices;
          SmallVector<Value> dstIndices;

          for (auto [dim, outerOffset] : llvm::enumerate(outerOffsets)) {
            // Build source index for this dimension.
            Value srcIdx;
            Value outerOffsetVal =
                arith::ConstantIndexOp::create(rewriter, loc, outerOffset);
            if (dim < numIndexDims) {
              // This dimension has an index memref - load from it.
              srcIdx = memref::LoadOp::create(rewriter, loc, indices[dim],
                                              ValueRange{outerOffsetVal});
            }

            // For the innermost dimension, compute the full offset.
            if (dim == destShape.size() - 1) {
              Value innerOffsetVal =
                  arith::ConstantIndexOp::create(rewriter, loc, innerOffset);
              if (srcIdx) {
                // Had an index memref - add inner offset to loaded index.
                srcIdx = arith::AddIOp::create(rewriter, loc, srcIdx,
                                               innerOffsetVal);
              } else {
                // No index memref - use inner offset directly.
                srcIdx = innerOffsetVal;
              }
              srcIdx = arith::AddIOp::create(rewriter, loc, srcIdx, laneOffset);
              // Reuse innerOffsetVal for destination index.
              dstIndices.push_back(innerOffsetVal);
            } else if (!srcIdx) {
              // Non-innermost dimension without index memref.
              srcIdx = outerOffsetVal;
              dstIndices.push_back(outerOffsetVal);
            } else {
              // Dimension with index memref (non-innermost).
              dstIndices.push_back(outerOffsetVal);
            }
            srcIndices.push_back(srcIdx);
          }

          amdgpu::GatherToLDSOp::create(rewriter, loc, source, srcIndices, dest,
                                        dstIndices,
                                        TypeAttr::get(transferType));
        }
      }
    }

    rewriter.eraseOp(dmaOp);
    return success();
  }

private:
  ArrayRef<int64_t> targetDmaSizes;
};

namespace {
struct AMDGPULowerCoalescedDMAToGatherLDSPass final
    : impl::AMDGPULowerCoalescedDMAToGatherLDSPassBase<
          AMDGPULowerCoalescedDMAToGatherLDSPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    LDBG() << "GPU Target attribute: " << target;
    if (!target) {
      LDBG() << "Missing GPU target attribute, pass will fail";
      // Don't fail if no target attribute - just skip the pass.
      return;
    }

    // dma_sizes is optional - if not specified, skip the size validation.
    ArrayRef<int64_t> dmaSizes;
    if (auto dmaSizesAttr = target.getWgp().getDmaSizes()) {
      dmaSizes = dmaSizesAttr.asArrayRef();
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerCoalescedGatherDMAPattern>(context, dmaSizes);

    walkAndApplyPatterns(funcOp, std::move(patterns));

#ifndef NDEBUG
    // Verify all CoalescedGatherDMAOps were lowered. Currently, we require all
    // ops to be successfully lowered. In the future, a fallback lowering path
    // (e.g., using global_load) could handle ops that don't match the pattern.
    WalkResult result = funcOp.walk([&](IREE::GPU::CoalescedGatherDMAOp op) {
      op.emitOpError("failed to lower coalesced_gather_dma op");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      return signalPassFailure();
    }
#endif // NDEBUG
  }
};
} // namespace

} // namespace mlir::iree_compiler
