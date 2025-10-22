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
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/APInt.h"
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

/// Create GPU thread mapping attributes for the given rank.
static SmallVector<Attribute> getThreadMapping(MLIRContext *ctx, int64_t rank) {
  SmallVector<Attribute> mapping;
  for (int64_t i = 0; i < rank; ++i) {
    mapping.push_back(IREE::GPU::LaneIdAttr::get(ctx, rank - 1 - i));
  }
  return mapping;
}

static std::optional<int64_t> getElementsPerThread(linalg::CopyOp copyOp) {
  // Get the GPU target attributes.
  IREE::GPU::TargetAttr targetAttr = getGPUTargetAttr(copyOp);
  if (!targetAttr) {
    return std::nullopt;
  }

  IREE::GPU::TargetWgpAttr wgp = targetAttr.getWgp();
  if (!wgp) {
    return std::nullopt;
  }

  DenseI64ArrayAttr dmaSizesAttr = wgp.getDmaSizes();
  if (!dmaSizesAttr || dmaSizesAttr.empty()) {
    return std::nullopt;
  }

  auto outputType = cast<RankedTensorType>(copyOp.getOutputs()[0].getType());
  Type elementType = outputType.getElementType();

  unsigned elementBitWidth = elementType.getIntOrFloatBitWidth();
  if (elementBitWidth == 0) {
    return std::nullopt;
  }

  int64_t maxDmaSize = *std::max_element(dmaSizesAttr.asArrayRef().begin(),
                                         dmaSizesAttr.asArrayRef().end());
  int64_t elementsPerThread = maxDmaSize / elementBitWidth;
  if (elementsPerThread <= 0) {
    return std::nullopt;
  }

  return elementsPerThread;
}

/// Helper to compute thread number of threads based on operation type.
/// For copy ops: uses dma_sizes to determine how many elements per thread.
/// For gather ops: currently limits to 1 element per thread (32-bit).
template <typename OpTy>
static SmallVector<OpFoldResult>
computeThreadNumThreadsImpl(OpBuilder &builder, OpTy op,
                            RankedTensorType outputType) {
  int64_t rank = outputType.getRank();
  int64_t innermostStep;

  if constexpr (std::is_same_v<OpTy, linalg::CopyOp>) {
    // For copy: use dma_sizes from target attributes.
    IREE::GPU::TargetAttr targetAttr = getGPUTargetAttr(op);
    if (!targetAttr) {
      return {};
    }

    IREE::GPU::TargetWgpAttr wgp = targetAttr.getWgp();
    if (!wgp) {
      return {};
    }

    DenseI64ArrayAttr dmaSizesAttr = wgp.getDmaSizes();
    if (!dmaSizesAttr || dmaSizesAttr.empty()) {
      return {};
    }

    Type elementType = outputType.getElementType();
    unsigned elementBitWidth = elementType.getIntOrFloatBitWidth();
    if (elementBitWidth == 0) {
      return {};
    }

    int64_t maxDmaSize = *std::max_element(dmaSizesAttr.asArrayRef().begin(),
                                           dmaSizesAttr.asArrayRef().end());

    // Compute the step for the innermost dimension:
    // step = (trailing_dim_size * element_bit_width) / max_dma_size.
    int64_t trailingDimSize = outputType.getDimSize(rank - 1);
    int64_t trailingDimByteSize = trailingDimSize * elementBitWidth;
    innermostStep = trailingDimByteSize / maxDmaSize;

    if (innermostStep <= 0) {
      return {};
    }
  } else if constexpr (std::is_same_v<OpTy, IREE::LinalgExt::GatherOp>) {
    // For gather: use the size of the innermost dimension.
    int64_t innermostDimSize = outputType.getDimSize(rank - 1);
    if (innermostDimSize == ShapedType::kDynamic || innermostDimSize <= 0) {
      return {};
    }
    innermostStep = innermostDimSize;
  }

  // Compute number of threads:
  // For 2D tensors, create 2D loops: [1, innermostStep].
  SmallVector<OpFoldResult> numThreads;
  for (int64_t i = 0; i < rank; ++i) {
    if (i == rank - 1) {
      numThreads.push_back(builder.getIndexAttr(innermostStep));
    } else {
      // Create 2D loops by using 1 for non-innermost dimensions.
      numThreads.push_back(builder.getIndexAttr(1));
    }
  }
  return numThreads;
}

/// Check if the given forall op has warp mapping.
static bool hasWarpMapping(scf::ForallOp forallOp) {
  if (!forallOp) {
    return false;
  }

  auto mapping = forallOp.getMapping();
  if (!mapping.has_value()) {
    return false;
  }

  return llvm::any_of(mapping.value(), [](Attribute attr) {
    return isa<gpu::GPUWarpMappingAttr>(attr);
  });
}

/// Helper to count non-zero thread counts for mapping.
static int64_t
countNonZeroThreads(const SmallVector<OpFoldResult> &threadNumThreads) {
  int64_t numLoops = 0;
  for (const auto &numThread : threadNumThreads) {
    if (auto intAttr = dyn_cast<IntegerAttr>(numThread.dyn_cast<Attribute>())) {
      if (intAttr.getInt() != 0) {
        numLoops++;
      }
    } else {
      numLoops++;
    }
  }
  return numLoops;
}

template <typename OpTy>
static scf::ForallOp
tileToThreadLevel(OpTy op, PatternRewriter &rewriter,
                  const SmallVector<OpFoldResult> &threadNumThreads) {
  if (threadNumThreads.empty()) {
    return nullptr;
  }

  // Configure tiling options using numThreads.
  scf::SCFTilingOptions threadOptions;
  threadOptions.setNumThreadsComputationFunction(
      [threadNumThreads](OpBuilder &b, Operation *op) {
        return threadNumThreads;
      });
  threadOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

  // Set thread mapping if needed.
  int64_t numLoops = countNonZeroThreads(threadNumThreads);
  if (numLoops > 0) {
    threadOptions.setMapping(getThreadMapping(rewriter.getContext(), numLoops));
  }

  rewriter.setInsertionPoint(op);
  FailureOr<scf::SCFTilingResult> threadTilingResult = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(op.getOperation()), threadOptions);

  if (failed(threadTilingResult)) {
    return nullptr;
  }

  // Find the thread-level forall op.
  scf::ForallOp threadForallOp = nullptr;
  for (auto loop : threadTilingResult->loops) {
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
  for (auto &op : inParallelBlock) {
    if (auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op)) {
      toErase.push_back(insertOp);
    }
  }

  Location loc = innerOp.getLoc();
  Value source, indices;

  // Extract source and indices based on op type.
  if constexpr (std::is_same_v<OpTy, linalg::CopyOp>) {
    source = innerOp.getInputs()[0];
  } else if constexpr (std::is_same_v<OpTy, IREE::LinalgExt::GatherOp>) {
    source = innerOp.getSource();
    indices = innerOp.getIndices();

    // Convert indices tensor to vector for DMA if present.
    if (indices) {
      rewriter.setInsertionPoint(inParallelOp);
      auto indicesType = cast<RankedTensorType>(indices.getType());
      VectorType vectorType =
          VectorType::get(indicesType.getShape(), rewriter.getIndexType());
      Value zeroPad = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      SmallVector<Value> readIndices(indicesType.getRank());
      for (int64_t i = 0; i < indicesType.getRank(); ++i) {
        readIndices[i] = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      }

      indices = rewriter.create<vector::TransferReadOp>(
          loc, vectorType, indices, readIndices, zeroPad);
    }
  }

  // Create the DMA op in the in_parallel region.
  rewriter.setInsertionPointToStart(&inParallelBlock);
  SmallVector<Value> operands;
  operands.push_back(source);
  if (indices) {
    operands.push_back(indices);
  }
  operands.push_back(sharedOut);
  operands.push_back(laneId);
  IREE::GPU::CoalescedGatherDMAOp::create(rewriter, loc, sharedOut.getType(),
                                          operands);

  // Erase the parallel_insert_slice ops and inner operation.
  for (auto insertOp : toErase) {
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
  using ConvertToCoalescedDMABase<linalg::CopyOp>::ConvertToCoalescedDMABase;

protected:
  SmallVector<OpFoldResult>
  computeThreadNumThreads(OpBuilder &builder,
                          linalg::CopyOp copyOp) const override {
    auto outputType = cast<RankedTensorType>(copyOp.getOutputs()[0].getType());
    return computeThreadNumThreadsImpl(builder, copyOp, outputType);
  }
};

struct ConvertGatherToCoalescedDMA
    : public ConvertToCoalescedDMABase<IREE::LinalgExt::GatherOp> {
  using ConvertToCoalescedDMABase<
      IREE::LinalgExt::GatherOp>::ConvertToCoalescedDMABase;

protected:
  SmallVector<OpFoldResult>
  computeThreadNumThreads(OpBuilder &builder,
                          IREE::LinalgExt::GatherOp gatherOp) const override {
    auto outputType =
        cast<RankedTensorType>(gatherOp.getOutputs()[0].getType());
    return computeThreadNumThreadsImpl(builder, gatherOp, outputType);
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
    RewritePatternSet patterns(context);
    patterns.add<ConvertGatherToCoalescedDMA>(context);
    patterns.add<ConvertCopyToCoalescedDMA>(context);

    walkAndApplyPatterns(funcOp, std::move(patterns));
  }

private:
  /// Tile operation at subgroup level using GPU lowering config.
  template <typename OpTy>
  FailureOr<scf::SCFTilingResult> tileAtSubgroupLevel(IRRewriter &rewriter,
                                                      OpTy op) {
    MLIRContext *context = &getContext();
    auto dmaConfig = getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(op);
    if (!dmaConfig)
      return failure();

    SmallVector<OpFoldResult> tileSizes = dmaConfig.getTilingLevelSizes(
        rewriter, llvm::to_underlying(IREE::GPU::TilingLevel::Subgroup), op);

    if (tileSizes.empty())
      return failure();

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

    auto rank = cast<RankedTensorType>(op.getOutputs()[0].getType()).getRank();
    tilingOptions.setMapping(getWarpMapping(context, rank));

    rewriter.setInsertionPoint(op);
    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(op.getOperation()), tilingOptions);

    return tilingResult;
  }

  LogicalResult applySubgroupTiling(FunctionOpInterface funcOp) {
    MLIRContext *context = &getContext();
    SmallVector<Operation *> opsToTile;

    // Collect all ops with iree_gpu.use_global_load_dma lowering config.
    funcOp->walk([&](Operation *op) {
      if (isa<linalg::CopyOp, IREE::LinalgExt::GatherOp>(op)) {
        auto config = getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(op);
        if (config) {
          opsToTile.push_back(op);
        }
      }
    });

    // Apply subgroup-level tiling to each op.
    IRRewriter rewriter(context);
    for (auto *op : opsToTile) {
      FailureOr<scf::SCFTilingResult> tilingResult =
          TypeSwitch<Operation *, FailureOr<scf::SCFTilingResult>>(op)
              .Case<linalg::CopyOp>([&](auto copyOp) {
                return tileAtSubgroupLevel(rewriter, copyOp);
              })
              .Case<IREE::LinalgExt::GatherOp>([&](auto gatherOp) {
                return tileAtSubgroupLevel(rewriter, gatherOp);
              })
              .Default([](Operation *) { return failure(); });

      if (failed(tilingResult))
        continue;

      // Replace the original op with the tiled version.
      rewriter.replaceOp(op, tilingResult->replacements);
    }

    return success();
  }
};

} // namespace

} // namespace mlir::iree_compiler
