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

/// A segment of elements that will be transferred using a specific DMA size.
///
/// The pass linearizes contiguous trailing dimensions and iterates over
/// non-contiguous outer dimensions. For each linearized block:
///   - `startOffset` is the element offset within the linearized portion.
///   - `elementsPerTransfer` is subgroupSize * elementsPerLane.
///   - `numTransfers` is how many DMA operations use this segment's DMA size.
///
/// Larger DMA sizes are prioritized to minimize the number of transfers.
/// The total elements covered must exactly match the linearized block size.
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
    if (dmaSize % elementBits != 0) {
      continue;
    }
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

    if (remainingElements == 0) {
      break;
    }
  }

  if (remainingElements != 0) {
    return failure();
  }

  return segments;
}

/// Generates source and destination indices for a GatherToLDS operation.
///
/// Index computation rules for each dimension:
///
/// | Condition        | dstIdx    | srcIdx                         |
/// |------------------|-----------|--------------------------------|
/// | Gather dimension | dimOffset | load(indices[dim][dimOffset])  |
/// | Non-gather       | dimOffset | dimOffset                      |
///
/// Where:
///   - dimOffset: position in this dimension from delinearization
///   - indices[dim]: index memref mapping dest positions to source positions
///
/// Lane-parallel access is achieved by incorporating the lane offset into the
/// linearized offset BEFORE delinearization. This ensures correct distribution
/// across all dimensions, handles overflow naturally, and provides proper
/// per-element mapping for gather operations.
static std::pair<SmallVector<Value>, SmallVector<Value>>
generateGatherIndices(PatternRewriter &rewriter, Location loc,
                      ValueRange dimOffsets, OperandRange indices) {
  SmallVector<Value> srcIndices;
  SmallVector<Value> dstIndices;
  size_t numIndexDims = indices.size();

  for (auto [dim, dimOffset] : llvm::enumerate(dimOffsets)) {
    // Destination always uses the dimension offset directly.
    dstIndices.push_back(dimOffset);

    // Build source index for this dimension.
    Value srcIdx;
    if (dim < numIndexDims) {
      // Gather dimension: load the source index from the index memref.
      // The index memref maps destination positions to source positions.
      // Lane offset is already incorporated into dimOffset via delinearization.
      srcIdx = memref::LoadOp::create(rewriter, loc, indices[dim],
                                      ValueRange{dimOffset});
    } else {
      // Non-gather dimension: source and dest use the same offset.
      // Lane offset is already incorporated via delinearization.
      srcIdx = dimOffset;
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
    LDBG() << "  Destination shape: " << destShape.size() << "D";

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

    // Compute how many trailing dimensions to linearize.
    // We can linearize dimensions that are contiguous in the destination
    // memref. Gather indices don't affect this - they determine how we compute
    // source indices, but the destination layout determines what we can
    // linearize.
    //
    // Example: For dest <2x4x128> fully contiguous:
    //   - numLinearDims = 3 -> linearize all dims (1024 elements)
    // Example: For dest <2x4x128> with only dims 1-2 contiguous:
    //   - numLinearDims = 2 -> linearize dims 1-2 (512 elements)
    int64_t destRank = destShape.size();
    int64_t numLinearDims = destType.getNumContiguousTrailingDims();
    // Always linearize at least the innermost dimension.
    numLinearDims = std::max<int64_t>(numLinearDims, 1);
    LDBG() << "  Number of linear dims: " << numLinearDims;

    // Verify all linearized dimensions are static.
    for (int64_t i = destRank - numLinearDims; i < destRank; ++i) {
      if (ShapedType::isDynamic(destShape[i])) {
        return rewriter.notifyMatchFailure(
            dmaOp, "dynamic dimension in linearized portion");
      }
    }

    // Compute total elements in the linearized portion (last numLinearDims
    // dims).
    int64_t linearSize = 1;
    for (int64_t i = destRank - numLinearDims; i < destRank; ++i) {
      linearSize *= destShape[i];
    }
    LDBG() << "  Linear size for segmentation: " << linearSize;

    // Compute transfer segments using the helper function.
    FailureOr<SmallVector<TransferSegment>> segmentsOrFailure =
        computeTransferSegments(linearSize, elementBits, *subgroupSize,
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

    // Precompute laneOffset for each segment type.
    SmallVector<Value> segmentLaneOffsets;
    for (const TransferSegment &segment : segments) {
      Value laneOffset =
          arith::MulIOp::create(rewriter, loc, laneId,
                                arith::ConstantIndexOp::create(
                                    rewriter, loc, segment.elementsPerLane));
      segmentLaneOffsets.push_back(laneOffset);
    }

    emitTransfers(rewriter, loc, source, dest, destShape, numLinearDims,
                  elementType, indices, segments, segmentLaneOffsets);

    rewriter.eraseOp(dmaOp);
    return success();
  }

private:
  /// Emits GatherToLDS operations for a coalesced DMA transfer.
  ///
  /// The destination tensor is split into:
  ///   - Outer dimensions: iterated individually (one element at a time)
  ///   - Linear dimensions: the trailing `numLinearDims` dimensions are
  ///     linearized and distributed across lanes
  ///
  /// For each position in the outer dimensions, the linearized portion is
  /// transferred using multiple GatherToLDS operations. The lane offset
  /// (precomputed as laneId * elementsPerLane) is added to the linearized
  /// offset BEFORE delinearization. This ensures:
  ///   1. Correct distribution across all linearized dimensions
  ///   2. Natural overflow handling (delinearization performs the carry)
  ///   3. Proper per-element index lookup for gather operations
  ///
  /// Example: shape [16, 64] with 32 lanes, 4 elements/lane:
  ///   - Lane 16: linearOffset = 0 + 16*4 = 64
  ///   - delinearize(64, [16, 64]) → [1, 0] (row 1, col 0)
  ///   - Without pre-delinearization offset: would incorrectly access [0, 64]
  void emitTransfers(PatternRewriter &rewriter, Location loc, Value source,
                     Value dest, ArrayRef<int64_t> destShape,
                     int64_t numLinearDims, Type elementType,
                     OperandRange indices, ArrayRef<TransferSegment> segments,
                     ArrayRef<Value> segmentLaneOffsets) const {
    int64_t destRank = destShape.size();
    int64_t numOuterDims = destRank - numLinearDims;
    LDBG() << "Emitting transfers: " << numOuterDims << " outer dims, "
           << numLinearDims << " linear dims";

    // Build tile sizes: outer dims have size 1 (iterate individually),
    // linear dims have their full size (handled via linearization).
    SmallVector<int64_t> tileSizes(destRank, 1);
    for (int64_t i = numOuterDims; i < destRank; ++i) {
      tileSizes[i] = destShape[i];
    }

    // Build delinearization basis for the linear portion.
    SmallVector<int64_t> linearBasis;
    for (int64_t i = numOuterDims; i < destRank; ++i) {
      linearBasis.push_back(destShape[i]);
    }

    for (const SmallVector<int64_t> &outerOffsets :
         StaticTileOffsetRange(destShape, tileSizes)) {
      for (auto [segmentIdx, segment] : llvm::enumerate(segments)) {
        VectorType transferType =
            VectorType::get({segment.elementsPerLane}, elementType);
        Value laneOffset = segmentLaneOffsets[segmentIdx];

        for (int64_t transferIdx = 0; transferIdx < segment.numTransfers;
             ++transferIdx) {
          int64_t linearOffset =
              segment.startOffset + transferIdx * segment.elementsPerTransfer;

          // Build dimension offsets: outer dims use static offsets,
          // linear dims use delinearized offsets.
          SmallVector<Value> dimOffsets;

          // Add outer dimension offsets (constant values from iteration).
          for (int64_t i = 0; i < numOuterDims; ++i) {
            dimOffsets.push_back(
                arith::ConstantIndexOp::create(rewriter, loc, outerOffsets[i]));
          }

          // Add lane offset to linearized offset, then delinearize.
          // This distributes lane-parallel access correctly across dimensions.
          Value linearOffsetVal =
              arith::ConstantIndexOp::create(rewriter, loc, linearOffset);
          linearOffsetVal =
              arith::AddIOp::create(rewriter, loc, linearOffsetVal, laneOffset);

          SmallVector<OpFoldResult> basis =
              getAsIndexOpFoldResult(rewriter.getContext(), linearBasis);
          auto delinearize = affine::AffineDelinearizeIndexOp::create(
              rewriter, loc, linearOffsetVal, basis, /*hasOuterBound=*/true);
          for (Value v : delinearize.getResults()) {
            dimOffsets.push_back(v);
          }

          auto [srcIndices, dstIndices] =
              generateGatherIndices(rewriter, loc, dimOffsets, indices);

          amdgpu::GatherToLDSOp::create(rewriter, loc, source, srcIndices, dest,
                                        dstIndices,
                                        TypeAttr::get(transferType));
        }
      }
    }
  }

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
    if (DenseI64ArrayAttr dmaSizesAttr = target.getWgp().getDmaSizes()) {
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
