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
/// The gather_to_lds instruction requires:
///   - Source indices: per-lane divergent (each lane reads from different loc)
///   - Destination indices: subgroup-uniform (all lanes write to same LDS base)
///
/// Index computation rules for each dimension:
///
/// | Condition        | dstIdx       | srcIdx                              |
/// |------------------|--------------|-------------------------------------|
/// | Gather dimension | dstDimOffset | load(indices[dim][srcDimOffset])    |
/// | Non-gather       | dstDimOffset | srcDimOffset                        |
///
/// Where:
///   - srcDimOffset: position with lane offset (divergent per lane)
///   - dstDimOffset: position without lane offset (uniform across subgroup)
///   - indices[dim]: index memref mapping dest positions to source positions
static std::pair<SmallVector<Value>, SmallVector<Value>>
generateGatherIndices(PatternRewriter &rewriter, Location loc,
                      ValueRange srcDimOffsets, ValueRange dstDimOffsets,
                      OperandRange indices) {
  SmallVector<Value> srcIndices;
  SmallVector<Value> dstIndices;
  size_t numIndexDims = indices.size();

  for (auto [dim, srcDimOffset, dstDimOffset] :
       llvm::enumerate(srcDimOffsets, dstDimOffsets)) {
    // Destination uses uniform offset (no lane offset).
    dstIndices.push_back(dstDimOffset);

    // Build source index for this dimension (with lane offset).
    Value srcIdx;
    if (dim < numIndexDims) {
      // Gather dimension: load the source index from the index memref.
      // Use srcDimOffset (with lane offset) to look up the index.
      srcIdx = memref::LoadOp::create(rewriter, loc, indices[dim],
                                      ValueRange{srcDimOffset});
    } else {
      // Non-gather dimension: use srcDimOffset directly (with lane offset).
      srcIdx = srcDimOffset;
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
                  elementType, indices, segments, segmentLaneOffsets,
                  dmaOp.getInBounds());

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
  /// transferred using multiple GatherToLDS operations.
  ///
  /// The gather_to_lds instruction requires:
  ///   - Source indices: per-lane divergent (include lane offset)
  ///   - Destination indices: subgroup-uniform (exclude lane offset)
  ///
  /// We generate two delinearizations:
  ///   1. With lane offset: for source index computation (divergent)
  ///   2. Without lane offset: for destination index computation (uniform)
  ///
  /// Example: shape [16, 64] with 32 lanes, 4 elements/lane:
  ///   - Lane 16: srcLinearOffset = 0 + 16*4 = 64
  ///   - delinearize(64, [16, 64]) → [1, 0] (for source)
  ///   - delinearize(0, [16, 64])  → [0, 0] (for destination, uniform)
  void emitTransfers(PatternRewriter &rewriter, Location loc, Value source,
                     Value dest, ArrayRef<int64_t> destShape,
                     int64_t numLinearDims, Type elementType,
                     OperandRange indices, ArrayRef<TransferSegment> segments,
                     ArrayRef<Value> segmentLaneOffsets,
                     std::optional<ArrayAttr> inBoundsAttr) const {
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
    SmallVector<OpFoldResult> basis =
        getAsIndexOpFoldResult(rewriter.getContext(), linearBasis);

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

          // Build outer dimension offsets (constant values from iteration).
          // These are the same for both source and destination.
          SmallVector<Value> outerDimOffsets;
          for (int64_t i = 0; i < numOuterDims; ++i) {
            outerDimOffsets.push_back(
                arith::ConstantIndexOp::create(rewriter, loc, outerOffsets[i]));
          }

          Value linearOffsetVal =
              arith::ConstantIndexOp::create(rewriter, loc, linearOffset);

          // Source indices: add lane offset before delinearization (divergent).
          Value srcLinearOffset =
              arith::AddIOp::create(rewriter, loc, linearOffsetVal, laneOffset);
          auto srcDelinearize = affine::AffineDelinearizeIndexOp::create(
              rewriter, loc, srcLinearOffset, basis, /*hasOuterBound=*/true);

          SmallVector<Value> srcDimOffsets(outerDimOffsets);
          llvm::append_range(srcDimOffsets, srcDelinearize.getResults());

          // Destination indices: no lane offset (subgroup-uniform).
          auto dstDelinearize = affine::AffineDelinearizeIndexOp::create(
              rewriter, loc, linearOffsetVal, basis, /*hasOuterBound=*/true);

          SmallVector<Value> dstDimOffsets(outerDimOffsets);
          llvm::append_range(dstDimOffsets, dstDelinearize.getResults());

          auto [srcIndices, dstIndices] = generateGatherIndices(
              rewriter, loc, srcDimOffsets, dstDimOffsets, indices);

          // Raw buffer OOB clamping is 1D (linear): it returns 0 only when the
          // byte offset >= total buffer size. For non-outermost dimensions,
          // an OOB index wraps into the next row instead of returning 0.
          // Fix: when any non-outermost source index exceeds its dimension,
          // replace the outermost index with sourceShape[0] to force the
          // linearized offset past the buffer end → hardware returns 0.
          auto sourceType = cast<MemRefType>(source.getType());
          if (inBoundsAttr && hasAMDGPUFatRawBufferAddressSpace(sourceType)) {
            ArrayRef<int64_t> sourceShape = sourceType.getShape();
            Value anyNonOutermostOOB = arith::ConstantOp::create(
                rewriter, loc, rewriter.getBoolAttr(false));

            for (int64_t dim = 1; dim < sourceType.getRank(); ++dim) {
              if (dim >= static_cast<int64_t>(inBoundsAttr->size())) {
                break;
              }
              bool dimInBounds =
                  cast<BoolAttr>((*inBoundsAttr)[dim]).getValue();
              if (dimInBounds) {
                continue;
              }

              Value dimSize;
              if (ShapedType::isDynamic(sourceShape[dim])) {
                dimSize = memref::DimOp::create(rewriter, loc, source, dim);
              } else {
                dimSize = arith::ConstantIndexOp::create(rewriter, loc,
                                                         sourceShape[dim]);
              }

              Value isOOB = arith::CmpIOp::create(rewriter, loc,
                                                  arith::CmpIPredicate::uge,
                                                  srcIndices[dim], dimSize);

              anyNonOutermostOOB = arith::OrIOp::create(
                  rewriter, loc, anyNonOutermostOOB, isOOB);
            }

            Value oobOuterIdx;
            if (ShapedType::isDynamic(sourceShape[0])) {
              oobOuterIdx = memref::DimOp::create(rewriter, loc, source, 0);
            } else {
              oobOuterIdx =
                  arith::ConstantIndexOp::create(rewriter, loc, sourceShape[0]);
            }
            srcIndices[0] = arith::SelectOp::create(
                rewriter, loc, anyNonOutermostOOB, oobOuterIdx, srcIndices[0]);
          }

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
