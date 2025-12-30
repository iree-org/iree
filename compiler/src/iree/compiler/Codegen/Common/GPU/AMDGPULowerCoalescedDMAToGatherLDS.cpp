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

/// Returns true if the memref type is fully contiguous, i.e., there are no
/// gaps between elements across all dimensions. A memref is fully contiguous
/// when the strides form a row-major layout where each stride equals the
/// product of all inner dimension sizes.
static bool isFullyContiguousMemRef(MemRefType memrefType) {
  // A memref is fully contiguous if all dimensions are contiguous.
  return memrefType.getNumContiguousTrailingDims() == memrefType.getRank();
}

/// Converts a linear index to multi-dimensional indices for a given shape.
/// Uses row-major (C-style) layout where the last dimension varies fastest.
///
/// Example: shape = [4, 3, 2], linearIdx = 7
///   strides = [6, 2, 1]
///   indices[0] = 7 / 6 = 1, remainder = 7 % 6 = 1
///   indices[1] = 1 / 2 = 0, remainder = 1 % 2 = 1
///   indices[2] = 1 / 1 = 1
///   Result: [1, 0, 1]
static SmallVector<int64_t> linearToMultiIndex(int64_t linearIdx,
                                               ArrayRef<int64_t> shape) {
  SmallVector<int64_t> indices(shape.size());
  for (int64_t i = shape.size() - 1; i >= 0; --i) {
    indices[i] = linearIdx % shape[i];
    linearIdx /= shape[i];
  }
  return indices;
}

/// A segment of elements that will be transferred using a specific DMA size.
///
/// For non-contiguous memrefs (row-wise transfer):
///   - Segments are computed per-row based on the innermost dimension size.
///   - `startOffset` is the element offset within the innermost dimension.
///   - Each row is processed independently.
///
/// For fully contiguous memrefs (linearized transfer):
///   - The entire memref is treated as a 1D array of `totalElements` size.
///   - `startOffset` is the linear element offset from the memref start.
///   - Segments can span multiple rows since there are no gaps between rows.
///   - This allows using larger DMA transfers (e.g., 128-bit instead of 32-bit)
///     more efficiently by covering more elements per transfer.
///
/// In both cases, larger DMA sizes are prioritized to minimize the number of
/// transfers. The total elements covered must exactly match the memref size
/// to avoid copying excess elements.
///
/// Example for contiguous memref with shape <4x32xf32>, subgroup_size=64:
///   Total elements = 128, using 128-bit DMA (4 elements/lane):
///   - elementsPerTransfer = 64 * 4 = 256, but we only have 128 elements
///   - Fall back to 32-bit DMA (1 element/lane):
///   - elementsPerTransfer = 64 * 1 = 64
///   - numTransfers = 128 / 64 = 2 transfers covering all elements
struct TransferSegment {
  int64_t elementsPerLane;     // Number of elements each lane transfers.
  int64_t numTransfers;        // Number of DMA operations for this segment.
  int64_t startOffset;         // Element offset from segment start.
  int64_t elementsPerTransfer; // Total elements moved per transfer
                               // (subgroupSize * elementsPerLane).
};

/// Computes transfer segments for a given number of elements.
/// Prioritizes larger DMA sizes to minimize the number of transfers.
/// Returns failure if the elements cannot be fully covered by the available
/// DMA sizes.
static FailureOr<SmallVector<TransferSegment>>
computeTransferSegments(int64_t totalElements, int64_t elementBits,
                        int64_t subgroupSize, ArrayRef<int64_t> dmaSizes) {
  // Sort DMA sizes in descending order to prioritize larger transfers.
  SmallVector<int64_t> sortedDmaSizes(dmaSizes);
  llvm::sort(sortedDmaSizes, std::greater<>());

  SmallVector<TransferSegment> segments;
  int64_t remainingElements = totalElements;
  int64_t currentOffset = 0;

  for (int64_t dmaSize : sortedDmaSizes) {
    // Calculate elements per lane for this DMA size.
    if (dmaSize % elementBits != 0)
      continue;
    int64_t elementsPerLane = dmaSize / elementBits;

    // Calculate total elements per transfer (all lanes combined).
    int64_t elementsPerTransfer = subgroupSize * elementsPerLane;

    // Calculate how many full transfers we can do with this DMA size.
    int64_t numTransfers = remainingElements / elementsPerTransfer;
    if (numTransfers > 0) {
      segments.push_back(
          {elementsPerLane, numTransfers, currentOffset, elementsPerTransfer});
      LDBG() << "Segment: " << dmaSize << "-bit DMA, " << numTransfers
             << " transfers, elementsPerLane=" << elementsPerLane
             << ", startOffset=" << currentOffset
             << ", elementsPerTransfer=" << elementsPerTransfer;

      int64_t coveredElements = numTransfers * elementsPerTransfer;
      remainingElements -= coveredElements;
      currentOffset += coveredElements;
    }

    if (remainingElements == 0)
      break;
  }

  if (remainingElements != 0) {
    return failure();
  }

  return segments;
}

/// Generates source and destination indices for a GatherToLDS operation.
///
/// For each dimension, computes:
/// - dstIdx: The destination offset for this dimension.
/// - srcIdx: The source offset, which may come from:
///   1. An index memref (for gather dimensions)
///   2. The destination offset directly (for non-gather dimensions)
///   3. For the innermost dimension: adds laneOffset for lane-parallel access.
///
/// \param rewriter The pattern rewriter for creating IR.
/// \param loc The location for created operations.
/// \param dimOffsets Static offsets for each dimension.
/// \param innermostOffset The offset within the innermost dimension.
/// \param laneOffset Runtime lane offset (lane_id * elements_per_lane).
/// \param indices Index memrefs for gather dimensions (may be empty).
/// \param destRank The rank of the destination memref.
/// \returns A pair of (srcIndices, dstIndices).
static std::pair<SmallVector<Value>, SmallVector<Value>>
generateGatherIndices(PatternRewriter &rewriter, Location loc,
                      ArrayRef<int64_t> dimOffsets, int64_t innermostOffset,
                      Value laneOffset, OperandRange indices,
                      int64_t destRank) {
  SmallVector<Value> srcIndices;
  SmallVector<Value> dstIndices;
  size_t numIndexDims = indices.size();

  for (auto [dim, offset] : llvm::enumerate(dimOffsets)) {
    bool isInnermost = (static_cast<int64_t>(dim) == destRank - 1);
    int64_t offsetValue = isInnermost ? innermostOffset : offset;
    Value offsetVal =
        arith::ConstantIndexOp::create(rewriter, loc, offsetValue);

    // Build source index for this dimension.
    Value srcIdx;
    if (dim < numIndexDims) {
      // This dimension has an index memref - load from it.
      Value indexOffsetVal =
          arith::ConstantIndexOp::create(rewriter, loc, offset);
      srcIdx = memref::LoadOp::create(rewriter, loc, indices[dim],
                                      ValueRange{indexOffsetVal});
    }

    if (isInnermost) {
      // For innermost dimension: add base offset and lane offset.
      if (srcIdx) {
        srcIdx = arith::AddIOp::create(rewriter, loc, srcIdx, offsetVal);
      } else {
        srcIdx = offsetVal;
      }
      srcIdx = arith::AddIOp::create(rewriter, loc, srcIdx, laneOffset);
      dstIndices.push_back(offsetVal);
    } else if (!srcIdx) {
      // Non-innermost dimension without index memref.
      srcIdx = offsetVal;
      dstIndices.push_back(offsetVal);
    } else {
      // Dimension with index memref (non-innermost).
      dstIndices.push_back(offsetVal);
    }
    srcIndices.push_back(srcIdx);
  }

  return {srcIndices, dstIndices};
}

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
    auto destType = cast<MemRefType>(dest.getType());

    Type elementType = sourceType.getElementType();
    int64_t elementBits = sourceType.getElementTypeBitWidth();
    LDBG() << "  Element type: " << elementType;
    LDBG() << "  Element bits: " << elementBits;

    ArrayRef<int64_t> destShape = destType.getShape();
    int64_t innermostDimSize = destShape.back();
    LDBG() << "  Destination innermost dimension size: " << innermostDimSize;
    LDBG() << "  Destination rank: " << destShape.size();

    std::optional<int64_t> subgroupSize =
        getSubgroupSize(dmaOp->getParentOfType<FunctionOpInterface>());
    if (!subgroupSize.has_value()) {
      return rewriter.notifyMatchFailure(
          dmaOp, "unable to determine subgroup size from forall");
    }
    LDBG() << "Subgroup size: " << *subgroupSize;

    OperandRange indices = dmaOp.getIndices();
    size_t numIndexDims = indices.size();
    LDBG() << "Number of index dimensions: " << numIndexDims;

    // Check if destination is fully contiguous for optimized linearized
    // transfer.
    bool useLinearizedTransfer = isFullyContiguousMemRef(destType);
    LDBG() << "  Destination is fully contiguous: " << useLinearizedTransfer;

    // Compute total elements for segment calculation.
    // For contiguous dest: use total elements across all dimensions.
    // For non-contiguous: use innermost dimension only (row-wise).
    int64_t totalElements =
        useLinearizedTransfer
            ? std::accumulate(destShape.begin(), destShape.end(), int64_t(1),
                              std::multiplies<int64_t>())
            : innermostDimSize;
    LDBG() << "  Total elements for segmentation: " << totalElements;

    // Compute transfer segments using the helper function.
    FailureOr<SmallVector<TransferSegment>> segmentsOrFailure =
        computeTransferSegments(totalElements, elementBits, *subgroupSize,
                                targetDmaSizes);
    if (failed(segmentsOrFailure)) {
      return rewriter.notifyMatchFailure(
          dmaOp, "cannot cover elements with any combination "
                 "of supported DMA transfer sizes");
    }
    SmallVector<TransferSegment> segments = std::move(*segmentsOrFailure);

    // Set up for code generation.
    rewriter.setInsertionPoint(dmaOp);
    TypedValue<IndexType> laneId = dmaOp.getLane();
    Location loc = dmaOp.getLoc();
    int64_t destRank = destShape.size();

    // Precompute laneOffset for each segment type.
    SmallVector<Value> segmentLaneOffsets;
    for (const TransferSegment &segment : segments) {
      Value laneOffset =
          arith::MulIOp::create(rewriter, loc, laneId,
                                arith::ConstantIndexOp::create(
                                    rewriter, loc, segment.elementsPerLane));
      segmentLaneOffsets.push_back(laneOffset);
    }

    if (useLinearizedTransfer) {
      // Linearized transfer path: treat dest as 1D array, transfers can span
      // multiple rows.
      LDBG() << "Using linearized transfer path";

      for (auto [segmentIdx, segment] : llvm::enumerate(segments)) {
        VectorType transferType =
            VectorType::get({segment.elementsPerLane}, elementType);
        Value laneOffset = segmentLaneOffsets[segmentIdx];

        for (int64_t transferIdx = 0; transferIdx < segment.numTransfers;
             ++transferIdx) {
          int64_t linearOffset =
              segment.startOffset + transferIdx * segment.elementsPerTransfer;

          // Convert linear offset to multi-dimensional indices.
          SmallVector<int64_t> multiDimOffsets =
              linearToMultiIndex(linearOffset, destShape);

          // The innermost offset from linearToMultiIndex is already correct.
          int64_t innermostOffset = multiDimOffsets.back();
          auto [srcIndices, dstIndices] = generateGatherIndices(
              rewriter, loc, multiDimOffsets, innermostOffset, laneOffset,
              indices, destRank);

          amdgpu::GatherToLDSOp::create(rewriter, loc, source, srcIndices, dest,
                                        dstIndices,
                                        TypeAttr::get(transferType));
        }
      }
    } else {
      // Row-wise transfer path: process each row independently.
      LDBG() << "Using row-wise transfer path";

      // Build tile sizes for outer dimensions: [1, 1, ..., 1, innermost].
      SmallVector<int64_t> outerTileSizes(destShape.size(), 1);
      outerTileSizes.back() = innermostDimSize;

      for (const SmallVector<int64_t> &outerOffsets :
           StaticTileOffsetRange(destShape, outerTileSizes)) {
        for (auto [segmentIdx, segment] : llvm::enumerate(segments)) {
          VectorType transferType =
              VectorType::get({segment.elementsPerLane}, elementType);
          Value laneOffset = segmentLaneOffsets[segmentIdx];

          for (int64_t transferIdx = 0; transferIdx < segment.numTransfers;
               ++transferIdx) {
            int64_t innerOffset =
                segment.startOffset + transferIdx * segment.elementsPerTransfer;

            auto [srcIndices, dstIndices] =
                generateGatherIndices(rewriter, loc, outerOffsets, innerOffset,
                                      laneOffset, indices, destRank);

            amdgpu::GatherToLDSOp::create(rewriter, loc, source, srcIndices,
                                          dest, dstIndices,
                                          TypeAttr::get(transferType));
          }
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
