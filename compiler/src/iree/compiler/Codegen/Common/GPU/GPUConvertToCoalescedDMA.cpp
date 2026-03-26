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
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
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

/// Check if the target architecture supports global load DMA.
/// Returns true only for CDNA4+ (gfx950+) architectures.
static bool targetSupportsGlobalLoadDMA(IREE::GPU::TargetAttr target) {
  if (!target) {
    return false;
  }
  FailureOr<amdgpu::Chipset> chipset = amdgpu::Chipset::parse(target.getArch());
  if (failed(chipset)) {
    return false;
  }
  // CDNA4 is gfx950+ (major=9, minor>=5). Other major versions (RDNA, etc.)
  // do not support global load DMA.
  return chipset->majorVersion == 9 && chipset->minorVersion >= 5;
}

/// Returns the subgroup size if the available elements are aligned to DMA
/// transfer sizes, std::nullopt otherwise.
static std::optional<int64_t>
getDMAAlignedSubgroupSize(FunctionOpInterface funcOp, Type elementType,
                          int64_t availableElements) {
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

  if (minElementsPerTransfer == std::numeric_limits<int64_t>::max() ||
      availableElements % minElementsPerTransfer != 0) {
    return std::nullopt;
  }

  return subgroupSize;
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

/// Check if a linalg.copy is viable for DMA conversion based on alignment and
/// size constraints. This does NOT modify the IR.
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
    if (tensor::PadOp pad = traceToTensorPad(input)) {
      // Verify pad constraints: low padding must be all zeros, pad value must
      // be 0.
      // TODO(#23365): Support non-zero pad values (e.g., -inf, 1) by emitting
      // a select on the loaded values from LDS to replace OOB zeros with the
      // desired padding element.
      bool validPad = true;
      for (OpFoldResult low : pad.getMixedLowPad()) {
        if (!isConstantIntValue(low, 0)) {
          validPad = false;
          break;
        }
      }
      Value padVal = pad.getConstantPaddingValue();
      if (!padVal || !(matchPattern(padVal, m_AnyZeroFloat()) ||
                       matchPattern(padVal, m_Zero()))) {
        validPad = false;
      }

      if (validPad) {
        // Use pad.getSource() directly as the DMA source.
        // This is the tensor.extract_slice result (e.g., tensor<?x64xf32>).
        source = pad.getSource();

        // Check if source tensor's innermost row size is DWORD (4-byte)
        // aligned. On AMD CDNA, per-component range checking is performed for
        // each DWORD. If a DWORD is partially out-of-bounds, the entire DWORD
        // returns zero, causing incorrect results. Additionally, partial OOB
        // triggers the slow path with multi-cycling and instruction issue
        // penalties.
        auto sourceType = cast<RankedTensorType>(source.getType());
        int64_t innermostDim = sourceType.getShape().back();
        if (!ShapedType::isDynamic(innermostDim)) {
          Type elemType = sourceType.getElementType();
          int64_t elemBytes = elemType.getIntOrFloatBitWidth() / 8;
          int64_t rowBytes = innermostDim * elemBytes;
          if (rowBytes % 4 != 0) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Skipping DMA: row size " << rowBytes
                       << " bytes not DWORD-aligned (slow path)\n");
            return failure();
          }
        }

        // Compute in_bounds based on whether padding was added per dimension.
        for (auto [low, high] :
             llvm::zip(pad.getMixedLowPad(), pad.getMixedHighPad())) {
          bool isInBounds =
              isConstantIntValue(low, 0) && isConstantIntValue(high, 0);
          inBoundsVec.push_back(isInBounds);
        }
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
  } else if constexpr (std::is_same_v<OpTy, IREE::LinalgExt::GatherOp>) {
    source = innerOp.getSource();
    indices = innerOp.getIndices();

    // Convert indices tensor to vector for DMA if present.
    if (indices) {
      rewriter.setInsertionPoint(inParallelOp);
      auto indicesType = cast<RankedTensorType>(indices.getType());
      Type elementType = indicesType.getElementType();

      // First, read the indices tensor as a vector with the original element
      // type.
      auto vectorTypeOriginal =
          VectorType::get(indicesType.getShape(), elementType);

      int64_t rank = indicesType.getRank();
      SmallVector<Value> readIndices(rank);
      for (int64_t i = 0; i < rank; ++i) {
        readIndices[i] = arith::ConstantIndexOp::create(rewriter, loc, 0);
      }

      // Create padding value - use i32 for index type.
      Type paddingType = elementType;
      if (elementType.isIndex()) {
        paddingType = rewriter.getI32Type();
      }
      TypedAttr zeroPadAttr = rewriter.getIntegerAttr(paddingType, 0);
      Value zeroPad = arith::ConstantOp::create(rewriter, loc, zeroPadAttr);

      Value indicesVec = vector::TransferReadOp::create(
          rewriter, loc, vectorTypeOriginal, indices, readIndices, zeroPad);

      // Convert to i32 type if needed.
      Type i32Type = rewriter.getI32Type();
      if (elementType != i32Type) {
        VectorType i32VectorType =
            VectorType::get(indicesType.getShape(), i32Type);
        indices = arith::IndexCastOp::create(rewriter, loc, i32VectorType,
                                             indicesVec);
      } else {
        indices = indicesVec;
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

    return createDMAInForall<OpTy>(threadForallOp, rewriter);
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

/// Pattern to convert tensor.pad fusion cases directly without requiring
/// warp-mapped forall parent.
struct ConvertPadFusionCopyToCoalescedDMA : OpRewritePattern<linalg::CopyOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
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

    return createDMAInForall<linalg::CopyOp>(threadForallOp, rewriter);
  }
};

struct ConvertGatherToCoalescedDMA
    : OpRewritePattern<IREE::LinalgExt::GatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::LinalgExt::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
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

    // Get the function containing this operation.
    auto funcOp = gatherOp->getParentOfType<FunctionOpInterface>();
    if (!funcOp) {
      return failure();
    }

    // Get subgroup size from translation_info.
    std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
    if (!subgroupSize) {
      return failure();
    }

    // Validate that innermost dimension is large enough for coalesced DMA.
    auto outputType = cast<RankedTensorType>(gatherOp.getOutput().getType());
    int64_t rank = outputType.getRank();
    int64_t innermostDim = outputType.getShape()[rank - 1];
    if (ShapedType::isDynamic(innermostDim)) {
      return failure();
    }

    Type elementType = outputType.getElementType();
    int64_t elementBits = elementType.getIntOrFloatBitWidth();

    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!target || !targetSupportsGlobalLoadDMA(target)) {
      return failure();
    }

    ArrayRef<int64_t> dmaSizes;
    if (DenseI64ArrayAttr dmaSizesAttr = target.getWgp().getDmaSizes()) {
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

    if (minElementsPerTransfer == std::numeric_limits<int64_t>::max() ||
        innermostDim % minElementsPerTransfer != 0) {
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

    if (!tiledGatherOp) {
      return failure();
    }

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

  // Note: we do NOT check sourceIsFromFatRawBuffer here because at this
  // pipeline stage (before bufferization), the im2col input comes from
  // dispatch.tensor.load, not from LoadFromBufferOp/fat_raw_buffer.
  // The fat_raw_buffer cast happens during bufferization. The gather DMA
  // conversion (ConvertGatherToCoalescedDMA) checks the source later when
  // it matters.

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

    // Skip coalesced DMA if the innermost dimension is smaller than the minimum
    // transfer size. The minimum transfer size is subgroupSize *
    // minElementsPerLane, where minElementsPerLane is determined by the
    // smallest DMA size and element type.
    int64_t innermostDim = shape[rank - 1];
    if (ShapedType::isDynamic(innermostDim)) {
      return failure();
    }

    // Get the element type bit width.
    Type elementType = outputType.getElementType();
    int64_t elementBits = elementType.getIntOrFloatBitWidth();

    // Get DMA sizes from target to compute minimum transfer size.
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!target) {
      return failure();
    }

    ArrayRef<int64_t> dmaSizes;
    if (DenseI64ArrayAttr dmaSizesAttr = target.getWgp().getDmaSizes()) {
      dmaSizes = dmaSizesAttr.asArrayRef();
    }

    // Find minimum elements per transfer across all DMA sizes.
    // We need innermostDim >= subgroupSize * minElementsPerLane.
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

    // Determine how many elements are available for coalesced access.
    // For CopyOp with output tracing to tensor.empty() (possibly through
    // swizzle promotion ops), we can linearize all dimensions.
    int64_t availableElements = innermostDim;
    if (auto copyOp = dyn_cast<linalg::CopyOp>(op.getOperation())) {
      Value output = copyOp.getOutputs()[0];
      if (tracesToTensorEmpty(output) &&
          llvm::none_of(shape, ShapedType::isDynamic)) {
        availableElements = ShapedType::getNumElements(shape);
      }
    }

    // If no valid DMA size found or available elements are not aligned to
    // transfer size, skip.
    if (minElementsPerTransfer == std::numeric_limits<int64_t>::max() ||
        availableElements % minElementsPerTransfer != 0) {
      return failure();
    }

    // Check if this is a tensor.pad fusion case.
    bool isPadFusion = false;
    if (auto copyOp = dyn_cast<linalg::CopyOp>(op.getOperation())) {
      if (tensor::PadOp pad = traceToTensorPad(copyOp.getInputs()[0])) {
        // Check if padding exists (non-zero low/high pad).
        for (auto [low, high] :
             llvm::zip(pad.getMixedLowPad(), pad.getMixedHighPad())) {
          if (!isConstantIntValue(low, 0) || !isConstantIntValue(high, 0)) {
            isPadFusion = true;
            break;
          }
        }
      }
    }

    SmallVector<OpFoldResult> tileSizes;
    int64_t numTiledDims = 0;

    if (isPadFusion) {
      // TODO(#23365): Tile to subgroups for pad fusion by propagating source
      // offsets through tiling. Currently, after subgroup tiling each warp's
      // DMA gets the full pre-pad source but a sub-tiled init, and the DMA
      // lowering has no way to offset into the source. This requires adding
      // source offset support to CoalescedGatherDMAOp. For now, create a
      // single-iteration wrapper forall so the DMA sees the full buffer.
      // Bail out if any dimension is dynamic since we need static tile sizes.
      if (llvm::any_of(shape, ShapedType::isDynamic)) {
        return failure();
      }
      for (int64_t i = 0; i < rank; ++i) {
        tileSizes.push_back(rewriter.getIndexAttr(shape[i]));
        ++numTiledDims;
      }
    } else {
      // Compute tile sizes for subgroup-level distribution.
      std::tie(tileSizes, numTiledDims) =
          computeSubgroupTileSizes(rewriter, shape, numWarps);
    }

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

    // Apply subgroup-level tiling to each op.
    // For tensor.pad fusion cases, tileAtSubgroupLevel creates a
    // single-iteration wrapper forall to maintain the expected structure while
    // allowing the DMA to operate on the full buffer.
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
