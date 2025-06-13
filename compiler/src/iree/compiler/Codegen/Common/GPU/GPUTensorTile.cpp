// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-tensor-tile"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUTENSORTILEPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

class TileConsumerAndFuseInputProducer final
    : public OpInterfaceRewritePattern<TilingInterface> {
public:
  TileConsumerAndFuseInputProducer(MLIRContext *context,
                                   LinalgTransformationFilter filter,
                                   bool fuseInputProducer, bool coalesceLoops,
                                   PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
        filter(std::move(filter)), fuseInputProducer(fuseInputProducer),
        coalesceLoops(coalesceLoops) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();

    // Make sure we have a PartitionableLoopInterface op here and query the tile
    // sizes from the partitionable loops.
    auto plOp = dyn_cast<PartitionableLoopsInterface>(*op);
    if (!plOp) {
      return rewriter.notifyMatchFailure(
          op, "Op does not implement PartitionableLoopsInterface");
    }
    auto partitionedLoops = plOp.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<int64_t> tileSizes = getTileSizes(op, 0);
    if (tileSizes.empty()) {
      return rewriter.notifyMatchFailure(
          op, "Op does not have configuration to get tile_sizes from");
    }
    // Mask out non reduction dimensions.
    for (unsigned depth : partitionedLoops) {
      if (depth < tileSizes.size())
        tileSizes[depth] = 0;
    }

    // Make sure we have a tile size for each dimension.
    // TODO: This is currently needed for LLVMGPU, where we propagate the
    // lowering configuration to all linalg ops. Some linalg ops may not have
    // the same rank, e.g., the configuration for a matmul attached to a
    // producer linalg.fill op. It implicitly assumes that the leading
    // dimensions of different linalg ops match, which is the current status;
    // but may not hold true in the long term.
    tileSizes.resize(op.getLoopIteratorTypes().size(), 0);

    if (llvm::all_of(tileSizes, [](int64_t s) { return s == 0; })) {
      return rewriter.notifyMatchFailure(op, "No dimensions are tiled");
    }

    // Tile the current op and fuse its immediate input operands.
    SmallVector<OpFoldResult> tileSizesOfr =
        getAsIndexOpFoldResult(rewriter.getContext(), tileSizes);
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileConsumerAndFuseInputProducer(rewriter, op, tileSizesOfr);
    if (failed(tilingResult)) {
      return rewriter.notifyMatchFailure(op, "failed to tile consumer");
    }

    // Replace the tiled op with replacements.
    rewriter.replaceOp(op, tilingResult->replacements);
    filter.replaceLinalgTransformationFilter(rewriter,
                                             tilingResult->tiledOps.front());

    if (coalesceLoops && tilingResult->loops.size() > 1) {
      SmallVector<scf::ForOp> loops = llvm::map_to_vector(
          tilingResult->loops, [](LoopLikeOpInterface loop) {
            return cast<scf::ForOp>(loop.getOperation());
          });
      if (failed(mlir::coalesceLoops(rewriter, loops))) {
        return failure();
      }
    }

    return success();
  }

private:
  FailureOr<scf::SCFTilingResult>
  tileConsumerAndFuseInputProducer(PatternRewriter &rewriter,
                                   TilingInterface consumer,
                                   ArrayRef<OpFoldResult> tileSizes) const {
    // First tile the current op as the consumer op.
    auto tilingOptions = scf::SCFTilingOptions().setTileSizes(tileSizes);
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileUsingSCF(rewriter, consumer, tilingOptions);
    if (failed(tilingResult)) {
      return rewriter.notifyMatchFailure(consumer, "failed to tile consumer");
    }

    if (!fuseInputProducer)
      return tilingResult;
    // If there are no generated loops generated, fusion is immaterial.
    if (tilingResult->loops.empty())
      return tilingResult;

    // Collect immediate input operands that are fusable into the tiled loop.
    // We have tensor extract slice ops taking slices of the untiled op.
    //
    // Note that this excludes init operands for correctness. Input operands are
    // fine to fuse, at the cost of recomputation though.
    SmallVector<tensor::ExtractSliceOp> candidates;
    assert(tilingResult->tiledOps.size() == 1);
    Operation *tiledOp = tilingResult->tiledOps.front();
    auto dsOp = dyn_cast<DestinationStyleOpInterface>(tiledOp);
    if (!dsOp)
      return tilingResult;
    for (OpOperand *operand : dsOp.getDpsInputOperands()) {
      auto sliceOp = operand->get().getDefiningOp<tensor::ExtractSliceOp>();
      if (!sliceOp)
        continue;
      auto tilingOp = sliceOp.getSource().getDefiningOp<TilingInterface>();
      if (!tilingOp)
        continue;
      if (isa<tensor::PadOp>(sliceOp.getSource().getDefiningOp())) {
        continue;
      }
      // Restrict to fully parallel ops for now for simplicity.
      auto isParallel = [](utils::IteratorType it) {
        return linalg::isParallelIterator(it);
      };
      if (llvm::all_of(tilingOp.getLoopIteratorTypes(), isParallel)) {
        candidates.push_back(sliceOp);
      }
    }

    // Fuse the candidate immeidate operands into the tiled loop.
    OpBuilder::InsertionGuard guard(rewriter);
    while (!candidates.empty()) {
      tensor::ExtractSliceOp sliceOp = candidates.back();
      candidates.pop_back();
      std::optional<scf::SCFFuseProducerOfSliceResult> result =
          scf::tileAndFuseProducerOfSlice(rewriter, sliceOp,
                                          tilingResult->loops);
      if (result) {
        // Mark the fused input producer for distribution when writing to shared
        // memory. We cannot use the current matmul op's tiling scheme here
        // given dimensions are different.
        LinalgTransformationFilter f(
            ArrayRef<StringAttr>(),
            rewriter.getStringAttr(getCopyToWorkgroupMemoryMarker()));
        f.replaceLinalgTransformationFilter(
            rewriter, result->tiledAndFusedProducer.getDefiningOp());
      }
    }
    return tilingResult;
  }

  LinalgTransformationFilter filter;
  bool fuseInputProducer;
  bool coalesceLoops;
};

/// Patterns for workgroup level tiling. Workgroup tiling is done at the flow
/// level but we may have extra tiling for the reduction dimension. Therefore we
/// tile again without distributing.
static void populateTilingPatterns(RewritePatternSet &patterns,
                                   bool fuseInputProducer, bool coalesceLoops) {
  MLIRContext *context = patterns.getContext();

  LinalgTransformationFilter filter(
      ArrayRef<StringAttr>{
          StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getWorkgroupKTiledMarker()));
  filter.setMatchByDefault();

  patterns.add<TileConsumerAndFuseInputProducer>(
      context, filter, fuseInputProducer, coalesceLoops);
}

} // namespace

LogicalResult tileReductionToSerialLoops(mlir::FunctionOpInterface funcOp,
                                         bool fuseInputProducer,
                                         bool coalesceLoops) {
  {
    // Tile again at the workgroup level since redution dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    RewritePatternSet wgTilingPatterns(funcOp.getContext());
    populateTilingPatterns(wgTilingPatterns, fuseInputProducer, coalesceLoops);
    if (failed(applyPatternsGreedily(funcOp, std::move(wgTilingPatterns)))) {
      return failure();
    }
  }

  {
    RewritePatternSet wgTilingCanonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(funcOp.getContext());
    populateAffineMinSCFCanonicalizationPattern(
        wgTilingCanonicalizationPatterns);
    scf::populateSCFForLoopCanonicalizationPatterns(
        wgTilingCanonicalizationPatterns);
    if (failed(applyPatternsGreedily(
            funcOp, std::move(wgTilingCanonicalizationPatterns)))) {
      return failure();
    }
    return success();
  }
}

namespace {
/// Tile parallel dimensions according to the attribute tile sizes attached to
/// each op.
static LogicalResult tileParallelDims(mlir::FunctionOpInterface funcOp,
                                      SmallVectorImpl<int64_t> &workgroupSize,
                                      bool distributeToWarp) {
  std::array<int64_t, 3> elementPerWorkgroup = {
      distributeToWarp ? workgroupSize[0] / kWarpSize : workgroupSize[0],
      workgroupSize[1], workgroupSize[2]};
  SmallVector<TilingInterface> computeOps;
  funcOp.walk([&](TilingInterface op) { computeOps.push_back(op); });

  auto marker =
      StringAttr::get(funcOp.getContext(), getCopyToWorkgroupMemoryMarker());

  for (TilingInterface tilingOp : computeOps) {
    auto attr = tilingOp->getAttr(LinalgTransforms::kLinalgTransformMarker);
    if (attr == marker)
      continue;

    size_t numLoops = 0;
    for (auto type : tilingOp.getLoopIteratorTypes()) {
      if (type == utils::IteratorType::parallel)
        numLoops++;
    }
    IRRewriter rewriter(tilingOp->getContext());
    rewriter.setInsertionPoint(tilingOp);
    auto interfaceOp =
        cast<PartitionableLoopsInterface>(*tilingOp.getOperation());
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    // If there are no dimensions to tile skip the transformation.
    if (partitionedLoops.empty())
      continue;
    SmallVector<OpFoldResult> numThreads(numLoops, rewriter.getIndexAttr(0));
    int64_t id = 0, threadDim = 0;
    SmallVector<Attribute> idDims;
    auto getThreadMapping = [&](int64_t dim) {
      return mlir::gpu::GPUThreadMappingAttr::get(
          tilingOp->getContext(), dim == 0   ? mlir::gpu::MappingId::DimX
                                  : dim == 1 ? mlir::gpu::MappingId::DimY
                                             : mlir::gpu::MappingId::DimZ);
    };
    for (unsigned loop : llvm::reverse(partitionedLoops)) {
      int64_t num = elementPerWorkgroup[id++];
      if (num > 1) {
        numThreads[loop] = rewriter.getIndexAttr(num);
        idDims.push_back(getThreadMapping(threadDim++));
      }
    }
    std::reverse(idDims.begin(), idDims.end());
    scf::SCFTilingOptions options;
    options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    options.setMapping(idDims);
    options.setNumThreads(numThreads);
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingOp, options);
    if (failed(tilingResult)) {
      return tilingOp->emitOpError("failed to tile to scf.forall");
    }
    rewriter.replaceOp(tilingOp, tilingResult->replacements);
  }
  return success();
}

// Tile convolution output window dimension by 1 to prepare downsizing.
static LogicalResult tileAndUnrollConv(mlir::FunctionOpInterface funcOp) {
  SmallVector<linalg::ConvolutionOpInterface, 1> convOps;
  funcOp.walk([&convOps](linalg::ConvolutionOpInterface convOp) {
    convOps.push_back(convOp);
  });
  for (linalg::ConvolutionOpInterface convOp : convOps) {
    auto consumerOp = cast<linalg::LinalgOp>(*convOp);
    IRRewriter rewriter(funcOp.getContext());
    SmallVector<OpFoldResult> tileSizes = getAsIndexOpFoldResult(
        funcOp.getContext(), getTileSizes(consumerOp, 1));
    if (tileSizes.empty())
      return success();

    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, cast<TilingInterface>(consumerOp.getOperation()),
            scf::SCFTileAndFuseOptions().setTilingOptions(
                scf::SCFTilingOptions().setTileSizes(tileSizes)));

    if (failed(tileAndFuseResult)) {
      consumerOp.emitOpError("failed tiling and fusing producers");
      return failure();
    }

    SmallVector<Value> replacements;
    replacements.resize(consumerOp->getNumResults());
    for (const auto &[index, result] :
         llvm::enumerate(consumerOp->getResults())) {
      replacements[index] = tileAndFuseResult->replacements.lookup(result);
    }
    consumerOp->replaceAllUsesWith(replacements);

    // Fully unroll the generated loop. This allows us to remove the loop
    // for parallel output window dimension, so it helps future vector
    // transformations.
    ArrayRef<LoopLikeOpInterface> loops = tileAndFuseResult.value().loops;
    if (!loops.empty()) {
      assert(loops.size() == 1);
      scf::ForOp loopOp = cast<scf::ForOp>(loops.front());
      IntegerAttr ub;
      if (!matchPattern(loopOp.getUpperBound(), m_Constant(&ub))) {
        loopOp.emitOpError("upper bound should be a constant");
        return failure();
      }
      if (failed(mlir::loopUnrollByFactor(loopOp, ub.getInt()))) {
        loopOp.emitOpError("failed unrolling by factor 1");
        return failure();
      }
    }
  }
  return success();
}

struct GPUTensorTilePass final
    : impl::GPUTensorTilePassBase<GPUTensorTilePass> {
  using GPUTensorTilePassBase::GPUTensorTilePassBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    if (!workgroupSize) {
      return;
    }
    if (failed(tileParallelDims(funcOp, workgroupSize.value(),
                                distributeToSubgroup))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "// --- After second level of tiling:\n";
      funcOp.dump();
    });

    // Tile to serial loops to the wg tile size to handle reductions and other
    // dimension that have not been distributed.
    if (failed(tileReductionToSerialLoops(funcOp, /*fuseInputProducer=*/false,
                                          /*coalesceLoops=*/false)))
      return signalPassFailure();

    LLVM_DEBUG({
      llvm::dbgs() << "// --- After tile reductions:\n";
      funcOp.dump();
    });

    if (failed(tileAndUnrollConv(funcOp))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "// --- After conv unrolling:\n";
      funcOp.dump();
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
