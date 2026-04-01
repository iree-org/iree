// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <limits>
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Utils/Permutation.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-convert-to-coalesced-dma"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCONVERTTOCOALESCEDDMAPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

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

/// Create GPU thread mapping for lane mapping.
/// Returns a single-element array with gpu.lane_id<0>.
static SmallVector<Attribute> getThreadMapping(MLIRContext *ctx) {
  SmallVector<Attribute> mapping;
  // Since we only tile the innermost dimension, we only have one loop.
  // Map it to gpu.lane_id<0>.
  mapping.push_back(IREE::GPU::LaneIdAttr::get(ctx, 0));
  return mapping;
}

/// Check if a pad value matches what hardware OOB clamping returns (zero
/// bytes). For IEEE floats this is ±0.0. For types like f8E8M0FNU where the
/// all-zeros bit pattern (0x00) represents 2^(-127) rather than IEEE zero,
/// we check the bit pattern directly since gather_to_lds OOB clamping returns
/// zero bytes regardless of the element type's semantics.
static bool padValueMatchesHardwareOOBZero(Value padVal) {
  if (matchPattern(padVal, m_AnyZeroFloat()) ||
      matchPattern(padVal, m_Zero())) {
    return true;
  }
  APFloat floatVal(APFloat::IEEEdouble());
  if (matchPattern(padVal, m_ConstantFloat(&floatVal))) {
    return floatVal.bitcastToAPInt().isZero();
  }
  return false;
}

/// Trace through extract_slice operations to find an underlying tensor.pad.
/// Returns the PadOp if found, nullptr otherwise.
static tensor::PadOp traceToTensorPad(Value source) {
  while (auto extractSlice = source.getDefiningOp<tensor::ExtractSliceOp>()) {
    source = extractSlice.getSource();
  }
  return source.getDefiningOp<tensor::PadOp>();
}

/// Trace through swizzle promotion ops (swizzle_hint, expand_shape,
/// collapse_shape) to find the underlying value. These ops are inserted by
/// swizzle promotion and don't affect whether the tensor traces to empty.
static Value traceThroughSwizzleOps(Value value) {
  while (true) {
    if (auto expandOp = value.getDefiningOp<tensor::ExpandShapeOp>()) {
      value = expandOp.getSrc();
      continue;
    }
    if (auto collapseOp = value.getDefiningOp<tensor::CollapseShapeOp>()) {
      value = collapseOp.getSrc();
      continue;
    }
    if (auto swizzleOp = value.getDefiningOp<IREE::Codegen::SwizzleHintOp>()) {
      value = swizzleOp.getOperand();
      continue;
    }
    break;
  }
  return value;
}

/// Check if a value traces back to tensor.empty, possibly through swizzle
/// promotion ops (swizzle_hint, expand_shape, collapse_shape) and/or forall
/// block arguments.
static bool tracesToTensorEmpty(Value value) {
  // Trace through swizzle promotion ops first.
  value = traceThroughSwizzleOps(value);

  // Direct tensor.empty.
  if (value.getDefiningOp<tensor::EmptyOp>()) {
    return true;
  }

  // Check if value is an extract_slice from a forall block argument.
  auto extractSlice = value.getDefiningOp<tensor::ExtractSliceOp>();
  if (!extractSlice) {
    return false;
  }

  auto blockArg = dyn_cast<BlockArgument>(extractSlice.getSource());
  if (!blockArg) {
    return false;
  }

  auto forallOp = dyn_cast<scf::ForallOp>(blockArg.getOwner()->getParentOp());
  if (!forallOp) {
    return false;
  }

  // Find the corresponding shared_out init value.
  unsigned numIVs = forallOp.getInductionVars().size();
  unsigned argIndex = blockArg.getArgNumber();
  if (argIndex < numIVs) {
    return false;
  }

  unsigned sharedOutIndex = argIndex - numIVs;
  if (sharedOutIndex >= forallOp.getOutputs().size()) {
    return false;
  }

  // Trace through swizzle ops on the forall init value as well.
  Value initValue =
      traceThroughSwizzleOps(forallOp.getOutputs()[sharedOutIndex]);
  return initValue.getDefiningOp<tensor::EmptyOp>() != nullptr;
}

/// Check if the source of a copy traces to a fat_raw_buffer source.
/// Traces through extract_slice and pad ops to find the originating op.
/// Returns true if source is a block arg (opaque, allow DMA) or if it
/// traces to a LoadFromBufferOp with fat_raw_buffer address space.
/// Returns false if source traces to a LoadFromBufferOp without
/// fat_raw_buffer, or to any other concrete op (e.g. dispatch.tensor.load).
static bool sourceIsFromFatRawBuffer(Value source) {
  // Trace through extract_slice and pad ops.
  while (true) {
    if (auto extractSlice = source.getDefiningOp<tensor::ExtractSliceOp>()) {
      source = extractSlice.getSource();
      continue;
    }
    if (auto pad = source.getDefiningOp<tensor::PadOp>()) {
      source = pad.getSource();
      continue;
    }
    if (auto collapseOp = source.getDefiningOp<tensor::CollapseShapeOp>()) {
      source = collapseOp.getSrc();
      continue;
    }
    break;
  }

  // Block args are opaque; conservatively allow DMA.
  if (isa<BlockArgument>(source)) {
    return true;
  }

  // Check if source comes from a LoadFromBufferOp with fat_raw_buffer.
  auto loadOp = source.getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
  if (!loadOp) {
    return false;
  }

  auto memrefType = cast<MemRefType>(loadOp.getBuffer().getType());
  return hasAMDGPUFatRawBufferAddressSpace(memrefType);
}

/// Returns the minimum number of elements needed (after padding) for DMA
/// alignment, or std::nullopt if DMA is not supported for this element type.
static std::optional<int64_t>
getMinDMAAlignedElements(FunctionOpInterface funcOp, Type elementType) {
  std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
  if (!subgroupSize) {
    return std::nullopt;
  }

  int64_t elementBits = elementType.getIntOrFloatBitWidth();

  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target || !targetSupportsGlobalLoadDMA(target)) {
    return std::nullopt;
  }

  ArrayRef<int64_t> dmaSizes;
  if (auto dmaSizesAttr = target.getWgp().getDmaSizes()) {
    dmaSizes = dmaSizesAttr.asArrayRef();
  }

  int64_t minElementsPerTransfer = std::numeric_limits<int64_t>::max();
  for (int64_t dmaSize : dmaSizes) {
    if (dmaSize % elementBits != 0) {
      continue;
    }
    int64_t elementsPerLane = dmaSize / elementBits;
    int64_t elementsPerTransfer = *subgroupSize * elementsPerLane;
    minElementsPerTransfer =
        std::min(minElementsPerTransfer, elementsPerTransfer);
  }

  if (minElementsPerTransfer == std::numeric_limits<int64_t>::max()) {
    return std::nullopt;
  }
  return minElementsPerTransfer;
}

/// Returns the subgroup size if the available elements are aligned to DMA
/// transfer sizes, std::nullopt otherwise.
static std::optional<int64_t>
getDMAAlignedSubgroupSize(FunctionOpInterface funcOp, Type elementType,
                          int64_t availableElements) {
  std::optional<int64_t> minAligned =
      getMinDMAAlignedElements(funcOp, elementType);
  if (!minAligned || availableElements % *minAligned != 0) {
    return std::nullopt;
  }
  // getMinDMAAlignedElements already validated subgroupSize is present.
  return getSubgroupSize(funcOp);
}

/// Largest numWarps in [1, totalWarps] (by repeated halving) such that
/// `product(shape) / numWarps` satisfies the minimum DMA transfer alignment.
/// Conservative for shapes that don't divide evenly (the real greedy in
/// `computeSubgroupTileSizes` can sometimes pack more warps than this), but
/// always safe. Returns 0 only if even numWarps==1 fails — the pre-check in
/// `isCopyDMAConvertible` rejects such copies, so callers assert on 0.
static int64_t computeMaxFeasibleNumWarps(ArrayRef<int64_t> shape,
                                          int64_t totalWarps,
                                          int64_t minElementsPerTransfer) {
  int64_t totalElements = ShapedType::getNumElements(shape);
  for (int64_t n = std::max<int64_t>(totalWarps, 1); n >= 1; n /= 2) {
    int64_t perWarp = totalElements / n;
    if (perWarp >= minElementsPerTransfer &&
        perWarp % minElementsPerTransfer == 0) {
      return n;
    }
  }
  return 0;
}

/// Helper to compute thread number of threads based on translation_info.
/// Uses the subgroup_size from translation_info for thread-level tiling.
static SmallVector<OpFoldResult>
computeThreadNumThreadsImpl(OpBuilder &builder, Operation *op,
                            RankedTensorType outputType) {
  // Check that this operation has the use_global_load_dma config.
  auto dmaConfig = getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(op);
  if (!dmaConfig) {
    return {};
  }

  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  if (!funcOp) {
    return {};
  }

  int64_t rank = outputType.getRank();
  int64_t innermostDim = outputType.getShape()[rank - 1];
  if (ShapedType::isDynamic(innermostDim)) {
    return {};
  }

  // Determine how many elements are available for coalesced access.
  // For CopyOp with output tracing to tensor.empty(), we can linearize.
  ArrayRef<int64_t> shape = outputType.getShape();
  int64_t availableElements = innermostDim;
  if (auto copyOp = dyn_cast<linalg::CopyOp>(op)) {
    Value output = copyOp.getOutputs()[0];
    if (tracesToTensorEmpty(output) &&
        llvm::none_of(shape, ShapedType::isDynamic)) {
      availableElements = ShapedType::getNumElements(shape);
    }
  }

  auto subgroupSize = getDMAAlignedSubgroupSize(
      funcOp, outputType.getElementType(), availableElements);
  if (!subgroupSize) {
    return {};
  }

  return {builder.getIndexAttr(*subgroupSize)};
}

/// Check basic pad constraints for DMA: zero low padding and zero pad value.
/// TODO(#24156): Relax these checks by supporting individual cases.
static bool hasDMACompatiblePadding(tensor::PadOp pad) {
  for (OpFoldResult low : pad.getMixedLowPad()) {
    if (!isConstantIntValue(low, 0)) {
      return false;
    }
  }
  Value padVal = pad.getConstantPaddingValue();
  if (!padVal || !padValueMatchesHardwareOOBZero(padVal)) {
    return false;
  }
  return true;
}

/// Check if a tensor.pad source has DWORD-aligned innermost rows, accepting
/// sub-DWORD rows when the whole copy fits in a single DMA segment.
///
/// DWORD alignment matters because CDNA OOB clamping is per-DWORD: when
/// multiple DMA segments are issued, a partial-DWORD row causes overlap/
/// corruption between segments. For single-segment transfers (small copies
/// like scale operands), partial-DWORD OOB only affects trailing padding, so
/// the sub-DWORD row is safe.
static bool hasDWORDAlignedRows(tensor::PadOp pad, FunctionOpInterface funcOp) {
  auto sourceType = cast<RankedTensorType>(pad.getSource().getType());
  int64_t innermostDim = sourceType.getShape().back();
  if (ShapedType::isDynamic(innermostDim)) {
    return false;
  }
  Type elemType = sourceType.getElementType();
  int64_t rowBytes = innermostDim * (elemType.getIntOrFloatBitWidth() / 8);
  if (rowBytes % 4 == 0) {
    return true;
  }
  // Single-segment exemption: safe only when the innermost dimension has no
  // padding. When the inner dim is padded, output row width != source row
  // width, so linearized DMA reads wrong source elements for padded positions
  // instead of returning zeros.
  SmallVector<OpFoldResult> highPad = pad.getMixedHighPad();
  if (highPad.empty() || !isConstantIntValue(highPad.back(), 0)) {
    return false;
  }
  std::optional<int64_t> minAligned =
      getMinDMAAlignedElements(funcOp, elemType);
  if (!minAligned.has_value() || !sourceType.hasStaticShape()) {
    return false;
  }
  int64_t totalSrcElements = ShapedType::getNumElements(sourceType.getShape());
  return totalSrcElements <= *minAligned;
}

/// Check if a tensor.pad is valid for DMA conversion.
/// Requires: zero low padding, zero pad value, and DWORD-aligned source rows
/// (with the single-segment exemption documented on hasDWORDAlignedRows).
static bool isValidPadForDMA(tensor::PadOp pad, FunctionOpInterface funcOp) {
  return hasDMACompatiblePadding(pad) && hasDWORDAlignedRows(pad, funcOp);
}

/// Check if a linalg.copy is viable for DMA conversion based on alignment,
/// size and padding constraints. This does NOT modify the IR.
static bool isCopyDMAConvertible(linalg::CopyOp copyOp) {
  auto funcOp = copyOp->getParentOfType<FunctionOpInterface>();
  if (!funcOp) {
    return false;
  }

  auto outputType = cast<RankedTensorType>(copyOp.getOutputs()[0].getType());
  int64_t rank = outputType.getRank();
  ArrayRef<int64_t> shape = outputType.getShape();
  int64_t innermostDim = shape[rank - 1];
  if (ShapedType::isDynamic(innermostDim)) {
    return false;
  }

  // The pre-check runs before tiling but after promotion, so the output may
  // have swizzle promotion ops (swizzle_hint, expand_shape) between it and
  // tensor.empty. Use tracesToTensorEmpty to handle both cases.
  int64_t availableElements = innermostDim;
  Value output = copyOp.getOutputs()[0];
  if (tracesToTensorEmpty(output) &&
      llvm::none_of(shape, ShapedType::isDynamic)) {
    availableElements = ShapedType::getNumElements(shape);
  }

  tensor::PadOp pad = traceToTensorPad(copyOp.getInputs()[0]);
  if (pad && !isValidPadForDMA(pad, funcOp)) {
    return false;
  }

  return getDMAAlignedSubgroupSize(funcOp, outputType.getElementType(),
                                   availableElements)
      .has_value();
}

/// Check if the given forall op has warp mapping.
static bool hasWarpMapping(scf::ForallOp forallOp) {
  if (!forallOp) {
    return false;
  }

  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  if (!mapping.has_value()) {
    return false;
  }

  return llvm::all_of(mapping.value(), llvm::IsaPred<gpu::GPUWarpMappingAttr>);
}

template <typename OpTy>
static scf::ForallOp
tileToThreadLevel(OpTy op, PatternRewriter &rewriter,
                  ArrayRef<OpFoldResult> threadNumThreads) {
  if (threadNumThreads.empty()) {
    return nullptr;
  }

  // Get the rank of the operation.
  auto outputType = cast<RankedTensorType>(op.getDpsInits()[0].getType());
  int64_t rank = outputType.getRank();

  // threadNumThreads contains only the innermost dimension's num threads.
  // We need to create tile sizes for all dimensions, with 0 for dimensions
  // we don't want to tile.
  SmallVector<OpFoldResult> tileSizes;
  for (int64_t i = 0; i < rank; ++i) {
    if (i == rank - 1) {
      // Innermost dimension: tile with the given num threads.
      tileSizes.push_back(rewriter.getIndexAttr(1));
    } else {
      // Other dimensions: don't tile (size = 0).
      tileSizes.push_back(rewriter.getIndexAttr(0));
    }
  }

  // Configure tiling options using tile sizes.
  scf::SCFTilingOptions threadOptions;
  threadOptions.setTileSizeComputationFunction(
      [tileSizes](OpBuilder &b, Operation *op) { return tileSizes; });
  threadOptions.setNumThreadsComputationFunction(
      [threadNumThreads, rank](OpBuilder &b, Operation *op) {
        // Create numThreads array with zeros for all dims except innermost.
        SmallVector<OpFoldResult> fullNumThreads;
        for (int64_t i = 0; i < rank; ++i) {
          if (i == rank - 1) {
            fullNumThreads.push_back(threadNumThreads[0]);
          } else {
            fullNumThreads.push_back(b.getIndexAttr(0));
          }
        }
        return fullNumThreads;
      });
  threadOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

  // Set thread mapping for the single innermost dimension.
  threadOptions.setMapping(getThreadMapping(rewriter.getContext()));

  rewriter.setInsertionPoint(op);
  FailureOr<scf::SCFTilingResult> threadTilingResult = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(op.getOperation()), threadOptions);

  if (failed(threadTilingResult)) {
    return nullptr;
  }

  // Find the thread-level forall op.
  scf::ForallOp threadForallOp = nullptr;
  for (LoopLikeOpInterface loop : threadTilingResult->loops) {
    if (auto fop = dyn_cast<scf::ForallOp>(loop.getOperation())) {
      threadForallOp = fop;
      break;
    }
  }

  if (!threadForallOp) {
    return nullptr;
  }

  // Replace the original op with the tiled version.
  rewriter.replaceOp(op, threadTilingResult->replacements);

  return threadForallOp;
}

/// Create a sub-slice from the pre-pad source that corresponds to the tiled
/// view of the padded tensor. Sizes on padded dims are clamped to source
/// bounds; the DMA in_bounds attribute handles zero-fill for the remaining
/// padded area.
static Value createClampedSourceSliceFromPad(PatternRewriter &rewriter,
                                             Location loc, tensor::PadOp pad,
                                             scf::InParallelOp inParallelOp) {
  Value preSource = pad.getSource();

  // The per-tile view of the padded tensor is the (single) extract_slice user
  // of the pad, created either by `tileAtSubgroupLevel` (warp tile) or by
  // `tileToThreadLevel` in the `ConvertPadFusionCopyToCoalescedDMA` fallback
  // (thread tile). Either way, after tiling the copy's input always traces
  // through this slice, so it must exist.
  tensor::ExtractSliceOp tilingES;
  for (Operation *user : pad->getUsers()) {
    if (auto es = dyn_cast<tensor::ExtractSliceOp>(user)) {
      tilingES = es;
      break;
    }
  }
  assert(tilingES && "pad output must have an extract_slice user after tiling");

  // Build a sub-slice of the pre-pad source at this tile's offset, with sizes
  // clamped to source bounds on padded dims. Pad uses low=[0,...], so padded
  // and pre-pad coordinates align.
  rewriter.setInsertionPoint(inParallelOp);

  SmallVector<OpFoldResult> warpOffsets = tilingES.getMixedOffsets();
  SmallVector<OpFoldResult> warpSizes = tilingES.getMixedSizes();
  auto preSourceType = cast<RankedTensorType>(preSource.getType());
  int64_t rank = preSourceType.getRank();

  SmallVector<OpFoldResult> subOffsets, subSizes, subStrides;
  for (int64_t i = 0; i < rank; i++) {
    subStrides.push_back(rewriter.getIndexAttr(1));

    bool dimHasPadding = !isConstantIntValue(pad.getMixedHighPad()[i], 0);
    if (!dimHasPadding) {
      subOffsets.push_back(warpOffsets[i]);
      subSizes.push_back(warpSizes[i]);
      continue;
    }

    // Source may be smaller than the padded dimension. Clamp offset and size to
    // stay within source bounds.
    Value offsetVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, warpOffsets[i]);
    Value tileSizeVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, warpSizes[i]);

    int64_t staticDim = preSourceType.getShape()[i];
    Value sourceDimSize;
    if (ShapedType::isDynamic(staticDim)) {
      sourceDimSize = tensor::DimOp::create(rewriter, loc, preSource, i);
    } else {
      sourceDimSize = arith::ConstantIndexOp::create(rewriter, loc, staticDim);
    }

    Value clampedOffset =
        arith::MinSIOp::create(rewriter, loc, offsetVal, sourceDimSize);
    Value remaining =
        arith::SubIOp::create(rewriter, loc, sourceDimSize, clampedOffset);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value clampedRemaining =
        arith::MaxSIOp::create(rewriter, loc, remaining, zero);
    Value clampedSize =
        arith::MinSIOp::create(rewriter, loc, clampedRemaining, tileSizeVal);

    subOffsets.push_back(clampedOffset);
    subSizes.push_back(clampedSize);
  }

  auto sourceSlice = tensor::ExtractSliceOp::create(
      rewriter, loc, preSource, subOffsets, subSizes, subStrides);
  return sourceSlice.getResult();
}

/// Create a coalesced DMA operation in the in_parallel region.
/// Handles both copy and gather operations.
template <typename OpTy>
static LogicalResult createDMAInForall(scf::ForallOp threadForallOp,
                                       PatternRewriter &rewriter) {
  // Find the inner operation.
  OpTy innerOp = nullptr;
  threadForallOp->walk([&](OpTy foundOp) {
    innerOp = foundOp;
    return WalkResult::interrupt();
  });

  if (!innerOp) {
    return failure();
  }

  Block *forallBody = threadForallOp.getBody();
  Value sharedOut = forallBody->getArguments().back();
  size_t numIVs = forallBody->getNumArguments() - 1;
  Value laneId = forallBody->getArgument(numIVs - 1);

  auto inParallelOp = cast<scf::InParallelOp>(forallBody->getTerminator());
  Block &inParallelBlock = inParallelOp.getRegion().front();

  // Collect parallel_insert_slice ops to erase.
  SmallVector<tensor::ParallelInsertSliceOp> toErase;
  for (Operation &op : inParallelBlock) {
    if (auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op)) {
      toErase.push_back(insertOp);
    }
  }

  Location loc = innerOp.getLoc();
  Value source, indices;
  SmallVector<bool> inBoundsVec;

  // Extract source and indices based on op type.
  if constexpr (std::is_same_v<OpTy, linalg::CopyOp>) {
    Value input = innerOp.getInputs()[0];

    // After tiling, the input is typically:
    //   tensor.extract_slice %padded[...] [...] [1, 1]
    // We need to trace through extract_slice to find if source is tensor.pad.
    // Eligibility (including the DWORD / single-segment check) was already
    // vetted by isCopyDMAConvertible in the pre-check, so isValidPadForDMA
    // here is a consistency guard rather than a fresh decision.
    tensor::PadOp pad = traceToTensorPad(input);
    auto parentFuncOp =
        innerOp->template getParentOfType<FunctionOpInterface>();
    if (pad && isValidPadForDMA(pad, parentFuncOp)) {
      source =
          createClampedSourceSliceFromPad(rewriter, loc, pad, inParallelOp);

      // Compute in_bounds based on whether padding was added per dimension.
      for (auto [low, high] :
           llvm::zip(pad.getMixedLowPad(), pad.getMixedHighPad())) {
        bool isInBounds =
            isConstantIntValue(low, 0) && isConstantIntValue(high, 0);
        inBoundsVec.push_back(isInBounds);
      }
    }

    // Fallback: no tensor.pad fusion. The input is an extract_slice from
    // tiling; trace through it to get the actual source.
    if (!source) {
      if (auto extractSlice = input.getDefiningOp<tensor::ExtractSliceOp>()) {
        source = extractSlice.getSource();
      } else {
        return failure();
      }
    }
  }

  // Create the DMA op in the in_parallel region.
  rewriter.setInsertionPointToStart(&inParallelBlock);
  SmallVector<Value, 1> indicesOperands;
  if (indices) {
    indicesOperands.push_back(indices);
  }

  // Create in_bounds attribute if we fused a tensor.pad.
  ArrayAttr inBoundsAttr;
  if (!inBoundsVec.empty()) {
    inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsVec);
  }

  // When used in forall.in_parallel, the op doesn't return a result
  // as it performs an in-place update to the shared_outs tensor.
  IREE::GPU::CoalescedGatherDMAOp::create(rewriter, loc, Type(), source,
                                          indicesOperands, sharedOut, laneId,
                                          inBoundsAttr);

  // Erase the parallel_insert_slice ops and inner operation.
  for (tensor::ParallelInsertSliceOp &insertOp : toErase) {
    rewriter.eraseOp(insertOp);
  }
  rewriter.eraseOp(innerOp);

  return success();
}

/// Base class for converting operations to coalesced DMA operations.
template <typename OpTy>
struct ConvertToCoalescedDMABase : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto forallOp = op->template getParentOfType<scf::ForallOp>();
    if (!hasWarpMapping(forallOp)) {
      return failure();
    }

    SmallVector<OpFoldResult> threadNumThreads =
        computeThreadNumThreads(rewriter, op);
    if (threadNumThreads.empty()) {
      return failure();
    }

    scf::ForallOp threadForallOp =
        tileToThreadLevel(op, rewriter, threadNumThreads);
    if (!threadForallOp) {
      return failure();
    }

    // createDMAInForall must not fail after tileToThreadLevel, because
    // tileToThreadLevel already erased the original op via replaceOp.
    // Failing here would leave a dangling reference (use-after-free).
    // All eligibility checks must happen before this point (e.g., in
    // isCopyDMAConvertible / computeThreadNumThreads).
    [[maybe_unused]] LogicalResult result =
        createDMAInForall<OpTy>(threadForallOp, rewriter);
    assert(succeeded(result) &&
           "createDMAInForall must not fail after tileToThreadLevel erased "
           "the original op");
    return success();
  }

protected:
  /// Compute thread num threads for the operation.
  virtual SmallVector<OpFoldResult> computeThreadNumThreads(OpBuilder &builder,
                                                            OpTy op) const = 0;
};

struct ConvertCopyToCoalescedDMA : ConvertToCoalescedDMABase<linalg::CopyOp> {
  using ConvertToCoalescedDMABase::ConvertToCoalescedDMABase;

protected:
  SmallVector<OpFoldResult>
  computeThreadNumThreads(OpBuilder &builder,
                          linalg::CopyOp copyOp) const override {
    if (!sourceIsFromFatRawBuffer(copyOp.getInputs()[0])) {
      return {};
    }
    auto outputType = cast<RankedTensorType>(copyOp.getOutputs()[0].getType());
    return computeThreadNumThreadsImpl(builder, copyOp, outputType);
  }
};

/// Fallback pattern to convert tensor.pad fusion cases directly without
/// requiring warp-mapped forall parent. This handles edge cases where
/// subgroup tiling was unable to distribute the pad-fused copy across warps
/// (e.g., if computeSubgroupTileSizes fails). Copies that were already
/// successfully distributed into warp-mapped foralls are handled by
/// ConvertCopyToCoalescedDMA instead.
struct ConvertPadFusionCopyToCoalescedDMA : OpRewritePattern<linalg::CopyOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // Skip if already inside a warp-mapped forall — those are handled by
    // ConvertCopyToCoalescedDMA with proper source offset propagation.
    auto forallOp = copyOp->getParentOfType<scf::ForallOp>();
    if (hasWarpMapping(forallOp)) {
      return failure();
    }

    // Only match copies with use_global_load_dma config.
    auto config = getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(copyOp);
    if (!config) {
      return failure();
    }

    // Skip if source is not from fat_raw_buffer.
    if (!sourceIsFromFatRawBuffer(copyOp.getInputs()[0])) {
      return failure();
    }

    // Check if this is a tensor.pad fusion case.
    tensor::PadOp pad = traceToTensorPad(copyOp.getInputs()[0]);
    if (!pad) {
      return failure(); // Not a pad fusion case
    }

    // Check if padding exists (non-zero low/high pad).
    bool hasPadding = false;
    for (auto [low, high] :
         llvm::zip(pad.getMixedLowPad(), pad.getMixedHighPad())) {
      if (!isConstantIntValue(low, 0) || !isConstantIntValue(high, 0)) {
        hasPadding = true;
        break;
      }
    }
    if (!hasPadding) {
      return failure(); // No actual padding
    }

    // This is a tensor.pad fusion case. Convert directly to
    // coalesced_gather_dma without requiring warp-mapped forall.
    auto outputType = cast<RankedTensorType>(copyOp.getOutputs()[0].getType());
    SmallVector<OpFoldResult> threadNumThreads =
        computeThreadNumThreadsImpl(rewriter, copyOp, outputType);
    if (threadNumThreads.empty()) {
      return failure();
    }

    scf::ForallOp threadForallOp =
        tileToThreadLevel(copyOp, rewriter, threadNumThreads);
    if (!threadForallOp) {
      return failure();
    }

    [[maybe_unused]] LogicalResult result =
        createDMAInForall<linalg::CopyOp>(threadForallOp, rewriter);
    assert(succeeded(result) &&
           "createDMAInForall must not fail after tileToThreadLevel erased "
           "the original op");
    return success();
  }
};

struct ConvertGatherToCoalescedDMA
    : OpRewritePattern<IREE::LinalgExt::GatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::LinalgExt::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    // TODO: Add support for masked gather.
    if (gatherOp.getMask()) {
      return failure();
    }
    auto forallOp = gatherOp->getParentOfType<scf::ForallOp>();
    if (!hasWarpMapping(forallOp)) {
      return failure();
    }

    // For gather ops, tile only the innermost dimension to distribute across
    // threads.
    auto dmaConfig =
        getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(gatherOp);
    if (!dmaConfig) {
      return failure();
    }

    auto funcOp = gatherOp->getParentOfType<FunctionOpInterface>();
    if (!funcOp) {
      return failure();
    }

    // Validate DMA alignment and get subgroup size.
    auto outputType = cast<RankedTensorType>(gatherOp.getOutput().getType());
    int64_t rank = outputType.getRank();
    int64_t innermostDim = outputType.getShape()[rank - 1];
    if (ShapedType::isDynamic(innermostDim)) {
      return failure();
    }

    std::optional<int64_t> subgroupSize = getDMAAlignedSubgroupSize(
        funcOp, outputType.getElementType(), innermostDim);
    if (!subgroupSize) {
      return failure();
    }

    SmallVector<OpFoldResult> threadNumThreads;
    threadNumThreads.push_back(rewriter.getIndexAttr(*subgroupSize));

    scf::ForallOp threadForallOp =
        tileToThreadLevel(gatherOp, rewriter, threadNumThreads);
    if (!threadForallOp) {
      return failure();
    }

    // Create DMA ops directly without relying on the template version.
    // Find the tiled gather op.
    IREE::LinalgExt::GatherOp tiledGatherOp = nullptr;
    threadForallOp->walk([&](IREE::LinalgExt::GatherOp foundOp) {
      tiledGatherOp = foundOp;
      return WalkResult::interrupt();
    });

    assert(tiledGatherOp &&
           "tiled gather op must exist after tileToThreadLevel");

    Block *forallBody = threadForallOp.getBody();
    Value sharedOut = forallBody->getArguments().back();
    size_t numIVs = forallBody->getNumArguments() - 1;
    Value laneId = forallBody->getArgument(numIVs - 1);

    auto inParallelOp = cast<scf::InParallelOp>(forallBody->getTerminator());
    Block &inParallelBlock = inParallelOp.getRegion().front();

    Location loc = tiledGatherOp.getLoc();

    // Get source - need to find the source from before thread-level tiling.
    // The tiledGatherOp.getSource() is already sliced by thread-level tiling.
    // We need to trace back to get the original warp-level source.
    Value source = tiledGatherOp.getSource();

    // If source comes from an extract_slice, get its source (from warp-level).
    if (auto extractOp = source.getDefiningOp<tensor::ExtractSliceOp>()) {
      source = extractOp.getSource();
    }

    Value indices = tiledGatherOp.getIndices();

    // Create the DMA op with properly extracted indices (keeping tensor type).
    rewriter.setInsertionPoint(inParallelOp);
    SmallVector<Value> indicesVec;

    if (indices) {
      auto indicesType = cast<RankedTensorType>(indices.getType());

      if (indicesType.getRank() == 1) {
        // For 1D indices, use directly as tensor.
        indicesVec.push_back(indices);
      } else {
        int64_t batchSize = indicesType.getShape()[0];
        int64_t indexDepth = indicesType.getShape()[1];
        Type elementType = indicesType.getElementType();

        for (int64_t dim = 0; dim < indexDepth; ++dim) {
          OpFoldResult offsets[] = {rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(dim)};
          OpFoldResult sizes[] = {rewriter.getIndexAttr(batchSize),
                                  rewriter.getIndexAttr(1)};
          OpFoldResult strides[] = {rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1)};

          Value extractedSlice = tensor::ExtractSliceOp::create(
              rewriter, loc, indices, offsets, sizes, strides);

          // Collapse from [N, 1] to [N].
          ReassociationIndices reassociation[] = {{0, 1}};
          auto collapsedType = RankedTensorType::get({batchSize}, elementType);
          Value collapsedSlice = tensor::CollapseShapeOp::create(
              rewriter, loc, collapsedType, extractedSlice, reassociation);

          indicesVec.push_back(collapsedSlice);
        }
      }
    }

    // Create the DMA op.
    rewriter.setInsertionPointToStart(&inParallelBlock);

    IREE::GPU::CoalescedGatherDMAOp::create(rewriter, loc, Type(), source,
                                            indicesVec, sharedOut, laneId,
                                            /*in_bounds=*/nullptr);

    // Erase parallel_insert_slice ops and gather op.
    SmallVector<tensor::ParallelInsertSliceOp> toErase;
    for (Operation &op : inParallelBlock) {
      if (auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op)) {
        toErase.push_back(insertOp);
      }
    }
    for (tensor::ParallelInsertSliceOp insertOp : toErase) {
      rewriter.eraseOp(insertOp);
    }
    rewriter.eraseOp(tiledGatherOp);

    return success();
  }
};

/// Check if an im2col op is viable for conversion to gather + DMA.
/// Validates v1 constraints: identity perms, single K window,
/// channel-aligned k_off, DMA-aligned contiguous size, static shapes.
static bool isIm2colDMAConvertible(IREE::LinalgExt::Im2colOp im2colOp) {
  auto funcOp = im2colOp->getParentOfType<FunctionOpInterface>();
  if (!funcOp) {
    return false;
  }

  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target || !targetSupportsGlobalLoadDMA(target)) {
    return false;
  }

  // Note: sourceIsFromFatRawBuffer is not checked here or in
  // ConvertGatherToCoalescedDMA. At this pipeline stage (before
  // bufferization), the im2col input comes from dispatch.tensor.load,
  // not from LoadFromBufferOp/fat_raw_buffer. The fat_raw_buffer cast
  // happens during bufferization, and the coalesced_gather_dma lowering
  // handles the buffer type correctly at that point.

  // Padded im2col is not yet supported for DMA conversion.
  if (im2colOp.hasPadding()) {
    return false;
  }

  // v1: identity output_perm and input_k_perm only.
  if (!isIdentityPermutation(im2colOp.getOutputPerm()) ||
      !isIdentityPermutation(im2colOp.getInputKPerm())) {
    return false;
  }

  // getVectorizableDim enforces willBeContiguousSlice (single-window K_tile).
  OpBuilder b(im2colOp);
  Location loc = im2colOp.getLoc();
  std::optional<unsigned> vecDim = im2colOp.getVectorizableDim(b, loc);
  if (!vecDim.has_value()) {
    return false;
  }

  // v1: all output shapes must be static.
  auto outputType = cast<RankedTensorType>(im2colOp.getOutputType());
  if (!outputType.hasStaticShape()) {
    return false;
  }

  int64_t contiguousSize = outputType.getShape()[*vecDim];

  // v1: k_off must be channel-aligned (k_off % C == 0).
  auto inputType = cast<RankedTensorType>(im2colOp.getInputType());
  ArrayRef<int64_t> kPos = im2colOp.getKPos();
  int64_t cDim = kPos.back();
  int64_t C = inputType.getShape()[cDim];
  if (ShapedType::isDynamic(C)) {
    return false;
  }

  SmallVector<OpFoldResult> mixedOffsets = im2colOp.getMixedOffsets();
  int64_t numBatchDims = im2colOp.getBatchPos().size();
  int64_t numMDims = im2colOp.getNumMOutputDims();
  int64_t kCanonicalIdx = numBatchDims + numMDims;

  if (kCanonicalIdx < static_cast<int64_t>(mixedOffsets.size())) {
    OpFoldResult kOff = mixedOffsets[kCanonicalIdx];
    if (auto constVal = getConstantIntValue(kOff)) {
      if (*constVal % C != 0) {
        return false;
      }
    } else {
      // Dynamic k_off: if contiguousSize <= C, chooseDimToVectorize already
      // validated alignment. Otherwise reject.
      if (contiguousSize > C) {
        return false;
      }
    }
  }

  // DMA alignment check.
  return getDMAAlignedSubgroupSize(funcOp, outputType.getElementType(),
                                   contiguousSize)
      .has_value();
}

/// Build a 1D tensor<batch_size x index> where each element is the linearized
/// spatial offset in the collapsed source for that batch position.
///
/// For batch position i:
///   (b, m) = delinearize(i, [batch_tile, M_tile])
///   (oh, ow, ...) = delinearize(m_off + m, output_sizes_M)
///   (kh, kw, ...) = delinearize(k_off / C, window_sizes)
///   spatial[j] = m_coord[j] * stride[j] + window[j] * dilation[j]
///   n = batch_off + b
///   lin = n * dim[0] * dim[1] * ... + spatial[0] * dim[1] * ... + ...
static Value buildIm2colIndexTensor(PatternRewriter &rewriter, Location loc,
                                    IREE::LinalgExt::Im2colOp im2colOp,
                                    int64_t batchSize) {
  using namespace IREE::LinalgExt;

  auto inputType = cast<RankedTensorType>(im2colOp.getInputType());
  auto outputType = cast<RankedTensorType>(im2colOp.getOutputType());
  int64_t inputRank = inputType.getRank();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  ArrayRef<int64_t> strides = im2colOp.getStrides();
  ArrayRef<int64_t> dilations = im2colOp.getDilations();
  ArrayRef<int64_t> batchPos = im2colOp.getBatchPos();
  ArrayRef<int64_t> mPos = im2colOp.getMPos();
  ArrayRef<int64_t> kPos = im2colOp.getKPos();

  SmallVector<OpFoldResult> mixedOffsets = im2colOp.getMixedOffsets();
  SmallVector<SmallVector<OpFoldResult>> mixedOutputSizes =
      im2colOp.getMixedOutputSizes();

  int64_t numBatchDims = batchPos.size();
  int64_t numMDims = im2colOp.getNumMOutputDims();

  SmallVector<int64_t> batchOutputDims = im2colOp.getBatchOutputDims();
  SmallVector<int64_t> mOutputDims = im2colOp.getMOutputDims();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  int64_t batchTile = 1;
  for (int64_t d : batchOutputDims) {
    batchTile *= outputShape[d];
  }
  int64_t mTile = 1;
  for (int64_t d : mOutputDims) {
    mTile *= outputShape[d];
  }

  // Create tensor.empty for the index tensor.
  // Use index type so that after bufferization + DMA lowering, the loaded
  // index values are directly usable as gather_to_lds source indices.
  Type indexType = rewriter.getIndexType();
  Value emptyTensor = tensor::EmptyOp::create(
      rewriter, loc, ArrayRef<int64_t>{batchSize}, indexType);

  // Build linalg.generic with a single parallel iterator.
  AffineMap outputMap = rewriter.getMultiDimIdentityMap(1);
  SmallVector<utils::IteratorType> iterTypes = {utils::IteratorType::parallel};

  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, emptyTensor.getType(), /*inputs=*/ValueRange{},
      /*outputs=*/ValueRange{emptyTensor},
      /*indexingMaps=*/ArrayRef<AffineMap>{outputMap}, iterTypes,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        // Get the flat iteration index.
        Value idx = linalg::IndexOp::create(b, nestedLoc, 0);

        // Delinearize idx into (batchIdx, mIdx).
        SmallVector<OpFoldResult> batchMBasis = {b.getIndexAttr(batchTile),
                                                 b.getIndexAttr(mTile)};
        auto delinBM = affine::AffineDelinearizeIndexOp::create(
            b, nestedLoc, idx, batchMBasis, /*hasOuterBound=*/true);
        Value batchIdx = delinBM.getResult(0);
        Value mIdx = delinBM.getResult(1);

        // Compute batch offset: n = batch_off + batchIdx.
        // For each batch dim, delinearize the batch index using output_sizes.
        // With identity output_perm, canonical batch dims map directly.
        SmallVector<Value> batchCoords;
        if (numBatchDims == 1) {
          OpFoldResult batchOff = mixedOffsets[0];
          Value batchOffVal =
              getValueOrCreateConstantIndexOp(b, nestedLoc, batchOff);
          Value n = arith::AddIOp::create(b, nestedLoc, batchOffVal, batchIdx);
          batchCoords.push_back(n);
        } else {
          // Multiple batch dims: delinearize.
          SmallVector<OpFoldResult> batchBasis;
          for (int64_t i = 0; i < numBatchDims; ++i) {
            batchBasis.append(mixedOutputSizes[i].begin(),
                              mixedOutputSizes[i].end());
          }
          auto delinBatch = affine::AffineDelinearizeIndexOp::create(
              b, nestedLoc, batchIdx, batchBasis, /*hasOuterBound=*/true);
          for (int64_t i = 0; i < numBatchDims; ++i) {
            Value coord = delinBatch.getResult(i);
            OpFoldResult off = mixedOffsets[i];
            Value offVal = getValueOrCreateConstantIndexOp(b, nestedLoc, off);
            batchCoords.push_back(
                arith::AddIOp::create(b, nestedLoc, offVal, coord));
          }
        }

        // Delinearize M index using M output_sizes.
        // m_pos + m_off for each spatial dim.
        SmallVector<Value> mCoords;
        {
          // Collect all M output_sizes into a flat basis.
          SmallVector<OpFoldResult> mBasis;
          for (int64_t i = 0; i < numMDims; ++i) {
            int64_t canonIdx = numBatchDims + i;
            mBasis.append(mixedOutputSizes[canonIdx].begin(),
                          mixedOutputSizes[canonIdx].end());
          }
          // For each M output dim, add its offset then delinearize using
          // its output_sizes to get spatial coordinates.
          if (numMDims == 1) {
            int64_t canonIdx = numBatchDims;
            OpFoldResult mOff = mixedOffsets[canonIdx];
            Value mOffVal = getValueOrCreateConstantIndexOp(b, nestedLoc, mOff);
            Value mPos = arith::AddIOp::create(b, nestedLoc, mOffVal, mIdx);
            const SmallVector<OpFoldResult> &innerSizes =
                mixedOutputSizes[canonIdx];
            if (innerSizes.size() == 1) {
              mCoords.push_back(mPos);
            } else {
              auto delinM = affine::AffineDelinearizeIndexOp::create(
                  b, nestedLoc, mPos, innerSizes, /*hasOuterBound=*/true);
              for (unsigned j = 0; j < innerSizes.size(); ++j) {
                mCoords.push_back(delinM.getResult(j));
              }
            }
          } else {
            // Multiple M output dims. Delinearize mIdx into per-dim sizes.
            SmallVector<OpFoldResult> mDimSizes;
            for (int64_t d : mOutputDims) {
              mDimSizes.push_back(b.getIndexAttr(outputShape[d]));
            }
            auto delinMDims = affine::AffineDelinearizeIndexOp::create(
                b, nestedLoc, mIdx, mDimSizes, /*hasOuterBound=*/true);
            for (int64_t i = 0; i < numMDims; ++i) {
              int64_t canonIdx = numBatchDims + i;
              OpFoldResult mOff = mixedOffsets[canonIdx];
              Value mOffVal =
                  getValueOrCreateConstantIndexOp(b, nestedLoc, mOff);
              Value mDimIdx = delinMDims.getResult(i);
              Value mPosVal =
                  arith::AddIOp::create(b, nestedLoc, mOffVal, mDimIdx);
              const SmallVector<OpFoldResult> &innerSizes =
                  mixedOutputSizes[canonIdx];
              if (innerSizes.size() == 1) {
                mCoords.push_back(mPosVal);
              } else {
                auto delinM = affine::AffineDelinearizeIndexOp::create(
                    b, nestedLoc, mPosVal, innerSizes,
                    /*hasOuterBound=*/true);
                for (unsigned j = 0; j < innerSizes.size(); ++j) {
                  mCoords.push_back(delinM.getResult(j));
                }
              }
            }
          }
        }

        // Compute window offsets from k_off.
        // k_off / C gives the linearized window index, which we delinearize
        // using the kernel_size (window sizes for each spatial dim).
        SmallVector<Value> windowCoords;
        {
          int64_t kCanonIdx = numBatchDims + numMDims;
          OpFoldResult kOff = mixedOffsets[kCanonIdx];
          Value kOffVal = getValueOrCreateConstantIndexOp(b, nestedLoc, kOff);

          // Get C = innermost k_pos channel size.
          int64_t C = inputShape[kPos.back()];
          Value cVal = arith::ConstantIndexOp::create(b, nestedLoc, C);
          Value windowIdx = arith::DivUIOp::create(b, nestedLoc, kOffVal, cVal);

          // Delinearize window index using kernel_size.
          SmallVector<OpFoldResult> kernelSize = im2colOp.getMixedKernelSize();
          if (kernelSize.size() == 1) {
            windowCoords.push_back(windowIdx);
          } else {
            auto delinWin = affine::AffineDelinearizeIndexOp::create(
                b, nestedLoc, windowIdx, kernelSize,
                /*hasOuterBound=*/true);
            for (unsigned j = 0; j < kernelSize.size(); ++j) {
              windowCoords.push_back(delinWin.getResult(j));
            }
          }
        }

        // Compute spatial coordinates.
        // spatial[j] = mCoords[j] * strides[j] + windowCoords[j] *
        // dilations[j]
        SmallVector<Value> spatialCoords;
        AffineExpr d0, d1;
        bindDims(b.getContext(), d0, d1);
        for (unsigned j = 0; j < mPos.size(); ++j) {
          auto map =
              AffineMap::get(2, 0, {d0 * strides[j] + d1 * dilations[j]});
          Value spatial = affine::makeComposedAffineApply(
              b, nestedLoc, map, {mCoords[j], windowCoords[j]});
          spatialCoords.push_back(spatial);
        }

        // Build the full input coordinate vector, then linearize.
        // Input layout: dimensions at batchPos get batch coords,
        //               dimensions at mPos get spatial coords,
        //               dimensions at kPos are handled by the gather's
        //               contiguous slice (not part of the index).
        // We linearize all dims except the last (channel) dim.
        SmallVector<Value> inputCoords(inputRank);
        int batchCoordIdx = 0;
        int spatialCoordIdx = 0;
        SetVector<int64_t> batchPosSet(batchPos.begin(), batchPos.end());
        SetVector<int64_t> mPosSet(mPos.begin(), mPos.end());
        for (int64_t i = 0; i < inputRank; ++i) {
          if (batchPosSet.contains(i)) {
            inputCoords[i] = batchCoords[batchCoordIdx++];
          } else if (mPosSet.contains(i)) {
            inputCoords[i] = spatialCoords[spatialCoordIdx++];
          } else {
            // K (channel) dims — set to 0 for linearization; the gather
            // reads the contiguous slice along these dims.
            inputCoords[i] = arith::ConstantIndexOp::create(b, nestedLoc, 0);
          }
        }

        // Linearize all dims except the last (contiguous channel dim).
        // lin = coords[0] * (shape[1]*...*shape[R-2])
        //     + coords[1] * (shape[2]*...*shape[R-2])
        //     + ... + coords[R-2]
        SmallVector<Value> outerCoords(inputCoords.begin(),
                                       inputCoords.begin() + inputRank - 1);
        SmallVector<OpFoldResult> outerBasis;
        for (int64_t i = 0; i < inputRank - 1; ++i) {
          outerBasis.push_back(b.getIndexAttr(inputShape[i]));
        }

        Value linIdx = affine::AffineLinearizeIndexOp::create(
            b, nestedLoc, outerCoords, outerBasis, /*disjoint=*/false);

        linalg::YieldOp::create(b, nestedLoc, linIdx);
      });

  return genericOp.getResult(0);
}

/// Convert im2col to gather for DMA. Collapses the conv input, computes
/// a linearized index tensor, creates a gather with dimension_map=[0],
/// and reshapes the result back.
struct ConvertIm2colToGather : OpRewritePattern<IREE::LinalgExt::Im2colOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::Im2colOp im2colOp,
                                PatternRewriter &rewriter) const override {
    auto dmaConfig =
        getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(im2colOp);
    if (!dmaConfig) {
      return failure();
    }
    Location loc = im2colOp.getLoc();

    auto outputType = cast<RankedTensorType>(im2colOp.getOutputType());
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t outputRank = outputType.getRank();

    // batch_size = product of all dims except the last (K_tile).
    int64_t batchSize = ShapedType::getNumElements(outputShape.drop_back());

    // 1. Collapse source to 2D: [[0..rank-2], [rank-1]].
    Value input = im2colOp.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t inputRank = inputType.getRank();
    SmallVector<ReassociationIndices> srcReassoc = {
        llvm::to_vector(llvm::seq<int64_t>(0, inputRank - 1)), {inputRank - 1}};
    Value collapsed =
        tensor::CollapseShapeOp::create(rewriter, loc, input, srcReassoc);

    // 2. Compute index tensor.
    Value indices = buildIm2colIndexTensor(rewriter, loc, im2colOp, batchSize);

    // 3. Reshape im2col output to [batch_size, C_per_window].
    // Build reassociation: [[0..outputRank-2], [outputRank-1]].
    SmallVector<ReassociationIndices> outputReassoc = {
        llvm::to_vector(llvm::seq<int64_t>(0, outputRank - 1)),
        {outputRank - 1}};
    Value output = im2colOp.getOutput();
    Value reshapedOutput =
        tensor::CollapseShapeOp::create(rewriter, loc, output, outputReassoc);

    // 4. Create gather with dimension_map = [0].
    auto gatherOp = IREE::LinalgExt::GatherOp::create(
        rewriter, loc, reshapedOutput.getType(), collapsed, indices,
        reshapedOutput, rewriter.getDenseI64ArrayAttr({0}));
    setLoweringConfig(gatherOp, dmaConfig);

    // 5. Reshape gather result back to original output shape.
    Value result = tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, gatherOp.getResult(0), outputReassoc);

    rewriter.replaceOp(im2colOp, result);
    return success();
  }
};

struct GPUConvertToCoalescedDMAPass final
    : impl::GPUConvertToCoalescedDMAPassBase<GPUConvertToCoalescedDMAPass> {
  using GPUConvertToCoalescedDMAPassBase::GPUConvertToCoalescedDMAPassBase;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Pre-check: decide whether all linalg.copy ops should be DMA-converted.
    // Only activate when at least one copy already has use_global_load_dma
    // (indicating DMA intent from upstream config, e.g. --iree-llvmgpu-use-
    // direct-load). Collect all promoted copies (use_global_load_dma or
    // derived_thread_config). If ALL are DMA-convertible, upgrade them all to
    // use_global_load_dma. If ANY fails, downgrade them all to
    // derived_thread_config.
    // Note: GatherOps are excluded — they come from input IR (not from
    // GPUPromoteMatmulOperands) and are handled independently by
    // ConvertGatherToCoalescedDMA.
    SmallVector<linalg::CopyOp> promotedCopies;
    bool hasDMAIntent = false;
    funcOp->walk([&](linalg::CopyOp copyOp) {
      if (getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(copyOp)) {
        hasDMAIntent = true;
        promotedCopies.push_back(copyOp);
      } else if (getLoweringConfig<IREE::GPU::DerivedThreadConfigAttr>(
                     copyOp)) {
        promotedCopies.push_back(copyOp);
      }
    });

    if (hasDMAIntent) {
      bool allConvertible = llvm::all_of(promotedCopies, isCopyDMAConvertible);
      LLVM_DEBUG({
        if (!allConvertible) {
          llvm::dbgs() << "DMA pre-check: not all copies convertible, "
                       << "downgrading " << promotedCopies.size()
                       << " copies to derived_thread_config\n";
        }
      });
      for (linalg::CopyOp copyOp : promotedCopies) {
        if (allConvertible) {
          setLoweringConfig(copyOp,
                            IREE::GPU::UseGlobalLoadDMAAttr::get(context));
        } else {
          setLoweringConfig(copyOp,
                            IREE::GPU::DerivedThreadConfigAttr::get(context));
        }
      }
    }

    // Im2col pre-check: individually downgrade non-convertible im2cols.
    funcOp->walk([&](IREE::LinalgExt::Im2colOp im2colOp) {
      if (getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(im2colOp)) {
        if (!isIm2colDMAConvertible(im2colOp)) {
          setLoweringConfig(im2colOp,
                            IREE::GPU::DerivedThreadConfigAttr::get(context));
        }
      }
    });

    // Phase 0: convert im2col -> gather.
    // This produces new GatherOps that Phase 1 (subgroup tiling) will
    // pick up.
    {
      RewritePatternSet im2colPatterns(context);
      im2colPatterns.add<ConvertIm2colToGather>(context);
      if (failed(applyPatternsGreedily(funcOp, std::move(im2colPatterns)))) {
        return signalPassFailure();
      }
    }

    // Phase 1: subgroup tiling — also tiles new gather ops from Phase 0.
    if (failed(applySubgroupTiling(funcOp))) {
      return signalPassFailure();
    }

    // Phase 2: gather/copy -> DMA.
    RewritePatternSet patterns(context);
    patterns.add<ConvertGatherToCoalescedDMA>(context);
    patterns.add<ConvertCopyToCoalescedDMA>(context);
    patterns.add<ConvertPadFusionCopyToCoalescedDMA>(context);

    walkAndApplyPatterns(funcOp, std::move(patterns));
  }

private:
  /// Compute tile sizes for subgroup-level distribution.
  /// Returns {tileSizes, numTiledDims}.
  ///
  /// We keep the innermost dimension whole (not tiled) to ensure contiguous
  /// memory access patterns, and greedily redistribute warps to outer
  /// dimensions.
  std::pair<SmallVector<OpFoldResult>, int64_t>
  computeSubgroupTileSizes(IRRewriter &rewriter, ArrayRef<int64_t> shape,
                           ArrayRef<int64_t> numWarps) {
    SmallVector<OpFoldResult> tileSizes;
    int64_t numTiledDims = 0;
    int64_t rank = shape.size();

    // Calculate total number of warps available.
    // Note: numWarps may contain 0s for dimensions where wgSize < subgroupSize.
    // We treat 0 as 1 for the purpose of counting total warps.
    auto positiveWarps =
        llvm::make_filter_range(numWarps, [](int64_t n) { return n > 0; });
    int64_t totalWarps = llvm::product_of(positiveWarps);

    // Greedily distribute warps to outer dimensions, keeping innermost whole.
    // For 1D tensors, distribute across the single dimension (no inner/outer).
    int64_t remainingWarps = totalWarps;
    for (int64_t i = 0; i < rank; ++i) {
      bool isInnermostOfMultiDim = (i == rank - 1) && (rank > 1);
      if (isInnermostOfMultiDim) {
        // Keep innermost dimension whole (tile size = full dimension).
        tileSizes.push_back(rewriter.getIndexAttr(shape[i]));
        ++numTiledDims;
      } else if (remainingWarps > 1 && ShapedType::isStatic(shape[i])) {
        // Distribute remaining warps to this outer dimension.
        int64_t warpsForThisDim = std::min(remainingWarps, shape[i]);
        int64_t tileSize = llvm::divideCeil(shape[i], warpsForThisDim);
        tileSizes.push_back(rewriter.getIndexAttr(tileSize));
        // Update remaining parallelism for subsequent dimensions.
        remainingWarps = llvm::divideCeil(remainingWarps, warpsForThisDim);
        ++numTiledDims;
      } else {
        // No parallelism to distribute; skip tiling this dimension.
        tileSizes.push_back(rewriter.getIndexAttr(0));
      }
    }

    return {tileSizes, numTiledDims};
  }

  /// Tile operation at subgroup level using workgroup_size and subgroup_size
  /// from translation_info.
  template <typename OpTy>
  FailureOr<scf::SCFTilingResult> tileAtSubgroupLevel(IRRewriter &rewriter,
                                                      OpTy op) {
    MLIRContext *context = &getContext();
    auto dmaConfig = getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(op);
    if (!dmaConfig) {
      return failure();
    }

    // Get the function containing this operation.
    auto funcOp = op->template getParentOfType<FunctionOpInterface>();
    if (!funcOp) {
      return failure();
    }

    // Get workgroup size and subgroup size from translation_info.
    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
    if (!workgroupSize || !subgroupSize) {
      return failure();
    }

    // Calculate number of subgroups per dimension.
    // workgroupSize is [X, Y, Z], and we divide by subgroupSize to get warps.
    SmallVector<int64_t> numWarps;
    for (int64_t wgSize : *workgroupSize) {
      if (wgSize > 0 && *subgroupSize > 0) {
        numWarps.push_back(wgSize / *subgroupSize);
      } else {
        numWarps.push_back(1);
      }
    }

    // Get the output type to determine rank and shape.
    auto outputType = cast<RankedTensorType>(op.getDpsInits()[0].getType());
    int64_t rank = outputType.getRank();
    ArrayRef<int64_t> shape = outputType.getShape();

    Type elementType = outputType.getElementType();
    std::optional<int64_t> minAligned =
        getMinDMAAlignedElements(funcOp, elementType);
    if (!minAligned) {
      return failure();
    }

    // Determine how many elements are available for coalesced access.
    // For CopyOp with output tracing to tensor.empty() (possibly through
    // swizzle promotion ops), we can linearize all dimensions.
    int64_t innermostDim = shape[rank - 1];
    int64_t availableElements = innermostDim;
    bool linearizedAvailability = false;
    if (auto copyOp = dyn_cast<linalg::CopyOp>(op.getOperation())) {
      Value output = copyOp.getOutputs()[0];
      if (tracesToTensorEmpty(output) &&
          llvm::none_of(shape, ShapedType::isDynamic)) {
        availableElements = ShapedType::getNumElements(shape);
        linearizedAvailability = true;
      }
    }

    // If available elements are not aligned to transfer size, skip.
    if (availableElements % *minAligned != 0) {
      return failure();
    }

    // In the linearized-availability case the per-warp tile shrinks with more
    // warps and can fall below the minimum DMA transfer, leaving an orphan
    // `linalg.copy {use_global_load_dma}` inside a warp-mapped forall
    // (issue #24139). Halve `totalWarps` until each warp's tile is aligned.
    SmallVector<int64_t> effectiveNumWarps = numWarps;
    if (linearizedAvailability) {
      auto positiveWarps =
          llvm::make_filter_range(numWarps, [](int64_t n) { return n > 0; });
      int64_t totalWarps = llvm::product_of(positiveWarps);
      int64_t feasibleNumWarps =
          computeMaxFeasibleNumWarps(shape, totalWarps, *minAligned);
      // The whole-tile alignment check in isCopyDMAConvertible is exactly the
      // numWarps==1 feasibility check, so reaching here with feasible==0
      // means the pre-check is out of sync with this distribution policy.
      assert(feasibleNumWarps >= 1 &&
             "DMA pre-check should have rejected this copy: even a single "
             "warp cannot satisfy the minimum DMA transfer alignment");
      effectiveNumWarps = {feasibleNumWarps};
    }

    SmallVector<OpFoldResult> tileSizes;
    int64_t numTiledDims = 0;

    // Distribute across subgroups (warps) for both pad fusion and non-pad
    // cases.
    std::tie(tileSizes, numTiledDims) =
        computeSubgroupTileSizes(rewriter, shape, effectiveNumWarps);

    if (numTiledDims == 0) {
      return failure();
    }

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    // Only create mapping for the dimensions that are actually tiled.
    tilingOptions.setMapping(getWarpMapping(context, numTiledDims));

    rewriter.setInsertionPoint(op);
    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(op.getOperation()), tilingOptions);

    return tilingResult;
  }

  LogicalResult applySubgroupTiling(FunctionOpInterface funcOp) {
    // Check if the target supports global load DMA (gfx950+).
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!targetSupportsGlobalLoadDMA(target)) {
      return success();
    }

    MLIRContext *context = &getContext();
    SmallVector<Operation *> opsToTile;

    // Collect all ops with iree_gpu.use_global_load_dma lowering config.
    // Skip ops that are already inside a warp-mapped forall.
    funcOp->walk([&](Operation *op) {
      if (auto copyOp = dyn_cast<linalg::CopyOp>(op)) {
        auto config = getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(op);
        if (!config || !sourceIsFromFatRawBuffer(copyOp.getInputs()[0])) {
          return;
        }
        auto parentForall = op->getParentOfType<scf::ForallOp>();
        if (!hasWarpMapping(parentForall)) {
          opsToTile.push_back(op);
        }
      } else if (isa<IREE::LinalgExt::GatherOp>(op)) {
        auto config = getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(op);
        if (config) {
          auto parentForall = op->getParentOfType<scf::ForallOp>();
          if (!hasWarpMapping(parentForall)) {
            opsToTile.push_back(op);
          }
        }
      }
    });

    // Apply subgroup-level tiling to each op. If tiling fails (e.g., dynamic
    // shapes, alignment mismatch), the op is left untiled and handled by the
    // fallback pattern ConvertPadFusionCopyToCoalescedDMA in Phase 2.
    IRRewriter rewriter(context);
    for (Operation *op : opsToTile) {
      FailureOr<scf::SCFTilingResult> tilingResult =
          TypeSwitch<Operation *, FailureOr<scf::SCFTilingResult>>(op)
              .Case([&](linalg::CopyOp copyOp) {
                return tileAtSubgroupLevel(rewriter, copyOp);
              })
              .Case([&](IREE::LinalgExt::GatherOp gatherOp) {
                return tileAtSubgroupLevel(rewriter, gatherOp);
              })
              .Default(failure());

      if (failed(tilingResult)) {
        continue;
      }

      // Replace the original op with the tiled version.
      rewriter.replaceOp(op, tilingResult->replacements);
    }

    return success();
  }
};

} // namespace

} // namespace mlir::iree_compiler
