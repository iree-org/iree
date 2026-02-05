// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <limits>
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
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

/// Trace through extract_slice operations to find an underlying tensor.pad.
/// Returns the PadOp if found, nullptr otherwise.
static tensor::PadOp traceToTensorPad(Value source) {
  while (auto extractSlice = source.getDefiningOp<tensor::ExtractSliceOp>()) {
    source = extractSlice.getSource();
  }
  return source.getDefiningOp<tensor::PadOp>();
}

/// Check if a value traces back to tensor.empty (possibly through forall args).
static bool tracesToTensorEmpty(Value value) {
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

  Value initValue = forallOp.getOutputs()[sharedOutIndex];
  return initValue.getDefiningOp<tensor::EmptyOp>() != nullptr;
}

/// Helper to compute thread number of threads based on translation_info.
/// Uses the subgroup_size from translation_info for thread-level tiling.
static SmallVector<OpFoldResult>
computeThreadNumThreadsImpl(OpBuilder &builder, Operation *op,
                            RankedTensorType outputType) {
  // Get the function containing this operation.
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  if (!funcOp) {
    return {};
  }

  // Get subgroup size from translation_info.
  std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
  if (!subgroupSize) {
    return {};
  }

  // Skip coalesced DMA if the innermost dimension is smaller than the minimum
  // transfer size. The minimum transfer size is subgroupSize *
  // minElementsPerLane.
  int64_t rank = outputType.getRank();
  int64_t innermostDim = outputType.getShape()[rank - 1];
  if (ShapedType::isDynamic(innermostDim)) {
    return {};
  }

  // Get element type bit width.
  Type elementType = outputType.getElementType();
  int64_t elementBits = elementType.getIntOrFloatBitWidth();

  // Get DMA sizes from target to compute minimum transfer size.
  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target) {
    return {};
  }

  ArrayRef<int64_t> dmaSizes;
  if (auto dmaSizesAttr = target.getWgp().getDmaSizes()) {
    dmaSizes = dmaSizesAttr.asArrayRef();
  }

  // Find minimum elements per transfer across all DMA sizes.
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

  // If no valid DMA size found or elements not aligned to transfer size, skip.
  if (minElementsPerTransfer == std::numeric_limits<int64_t>::max() ||
      availableElements % minElementsPerTransfer != 0) {
    return {};
  }

  return {builder.getIndexAttr(*subgroupSize)};
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
  auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
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
struct ConvertToCoalescedDMABase : public OpRewritePattern<OpTy> {
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

struct ConvertCopyToCoalescedDMA
    : public ConvertToCoalescedDMABase<linalg::CopyOp> {
  using ConvertToCoalescedDMABase::ConvertToCoalescedDMABase;

protected:
  SmallVector<OpFoldResult>
  computeThreadNumThreads(OpBuilder &builder,
                          linalg::CopyOp copyOp) const override {
    auto outputType = cast<RankedTensorType>(copyOp.getOutputs()[0].getType());
    return computeThreadNumThreadsImpl(builder, copyOp, outputType);
  }
};

/// Pattern to convert tensor.pad fusion cases directly without requiring
/// warp-mapped forall parent.
struct ConvertPadFusionCopyToCoalescedDMA
    : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // Only match copies with use_global_load_dma config.
    auto config = getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(copyOp);
    if (!config) {
      return failure();
    }

    // Check if this is a tensor.pad fusion case.
    auto pad = traceToTensorPad(copyOp.getInputs()[0]);
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
    : public OpRewritePattern<IREE::LinalgExt::GatherOp> {
  using OpRewritePattern<IREE::LinalgExt::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    auto forallOp = gatherOp->getParentOfType<scf::ForallOp>();
    if (!hasWarpMapping(forallOp)) {
      return failure();
    }

    // Get the function containing this operation.
    auto funcOp = gatherOp->getParentOfType<FunctionOpInterface>();
    if (!funcOp) {
      return failure();
    }

    // Check target supports global load DMA.
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!target || !IREE::GPU::targetSupportsGlobalLoadDMA(target)) {
      return failure();
    }

    // Get subgroup size from translation_info.
    std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
    if (!subgroupSize) {
      return failure();
    }

    // Validate that innermost dimension is large enough for coalesced DMA.
    auto outputType =
        cast<RankedTensorType>(gatherOp.getOutputs()[0].getType());
    int64_t rank = outputType.getRank();
    int64_t innermostDim = outputType.getShape()[rank - 1];
    if (ShapedType::isDynamic(innermostDim)) {
      return failure();
    }

    Type elementType = outputType.getElementType();
    int64_t elementBits = elementType.getIntOrFloatBitWidth();

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

struct GPUConvertToCoalescedDMAPass final
    : impl::GPUConvertToCoalescedDMAPassBase<GPUConvertToCoalescedDMAPass> {
  using GPUConvertToCoalescedDMAPassBase::GPUConvertToCoalescedDMAPassBase;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Preprocessing: apply subgroup-level tiling.
    if (failed(applySubgroupTiling(funcOp))) {
      return signalPassFailure();
    }

    // Only tile and convert ops within forall ops with warp mapping.
    // Also handle tensor.pad fusion cases that don't have warp mapping.
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
    auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
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
    // For CopyOp with tensor.empty() output, we can linearize all dimensions.
    // Otherwise, we can only use the innermost dimension.
    int64_t availableElements = innermostDim;
    if (auto copyOp = dyn_cast<linalg::CopyOp>(op.getOperation())) {
      Value output = copyOp.getOutputs()[0];
      if (output.getDefiningOp<tensor::EmptyOp>()) {
        // Can linearize all dimensions - compute total static elements.
        if (llvm::none_of(shape, ShapedType::isDynamic)) {
          availableElements = ShapedType::getNumElements(shape);
        }
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
      // For tensor.pad fusion, create a single-iteration wrapper forall
      // by setting tile sizes to the full shape. This allows the DMA to
      // operate on the full buffer while satisfying the warp-mapped parent
      // requirement.
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
    MLIRContext *context = &getContext();

    // Check if target supports global load DMA.
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!target || !IREE::GPU::targetSupportsGlobalLoadDMA(target)) {
      return success();
    }

    SmallVector<Operation *> opsToTile;

    // Collect copy/gather ops that are eligible for coalesced DMA.
    // Skip ops that are already inside a warp-mapped forall.
    funcOp->walk([&](Operation *op) {
      if (isa<linalg::CopyOp, IREE::LinalgExt::GatherOp>(op)) {
        auto parentForall = op->getParentOfType<scf::ForallOp>();
        if (!hasWarpMapping(parentForall)) {
          opsToTile.push_back(op);
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
