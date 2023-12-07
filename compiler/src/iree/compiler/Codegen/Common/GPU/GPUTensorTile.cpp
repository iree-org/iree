// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;

#define DEBUG_TYPE "iree-codegen-gpu-tensor-tile"

namespace mlir::iree_compiler {

class TileConsumerAndFuseInputProducer final
    : public OpInterfaceRewritePattern<TilingInterface> {
public:
  TileConsumerAndFuseInputProducer(
      MLIRContext *context, IREE::LinalgExt::LinalgTransformationFilter filter,
      bool fuseInputProducer, PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
        filter(std::move(filter)), fuseInputProducer(fuseInputProducer) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();

    // Make sure we have a PartitionableLoopInterface op here and query the tile
    // sizes from the partitionable loops.
    auto plOp = dyn_cast<PartitionableLoopsInterface>(*op);
    if (!plOp)
      return failure();
    auto partitionedLoops = plOp.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<int64_t> tileSizes = getTileSizes(op, 0);
    if (tileSizes.empty())
      return failure();
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
    tileSizes.resize(op.getLoopIteratorTypes().size());

    if (llvm::all_of(tileSizes, [](int64_t s) { return s == 0; })) {
      return failure();
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
        tileUsingSCFForOp(rewriter, consumer, tilingOptions);
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
      auto linalgOp = sliceOp.getSource().getDefiningOp<linalg::LinalgOp>();
      if (!linalgOp)
        continue;
      // Restrict to fully parallel linalg ops for now for simplicity.
      auto isParallel = [](utils::IteratorType it) {
        return linalg::isParallelIterator(it);
      };
      if (llvm::all_of(linalgOp.getIteratorTypesArray(), isParallel)) {
        candidates.push_back(sliceOp);
      }
    }

    // Fuse the candidate immeidate operands into the tiled loop.
    OpBuilder::InsertionGuard guard(rewriter);
    auto forLoops =
        llvm::to_vector(llvm::map_range(tilingResult->loops, [](Operation *op) {
          return cast<scf::ForOp>(op);
        }));
    while (!candidates.empty()) {
      tensor::ExtractSliceOp sliceOp = candidates.back();
      candidates.pop_back();
      std::optional<scf::SCFFuseProducerOfSliceResult> result =
          tileAndFuseProducerOfSlice(rewriter, sliceOp, forLoops);
      if (result) {
        // Mark the fused input producer for distribution when writing to shared
        // memory. We cannot use the current matmul op's tiling scheme here
        // given dimensions are different.
        IREE::LinalgExt::LinalgTransformationFilter f(
            ArrayRef<StringAttr>(),
            rewriter.getStringAttr(getCopyToWorkgroupMemoryMarker()));
        f.replaceLinalgTransformationFilter(
            rewriter, result->tiledAndFusedProducer.getDefiningOp());
      }
    }
    tilingResult->loops = llvm::to_vector(
        llvm::map_range(forLoops, [](auto op) -> Operation * { return op; }));
    return tilingResult;
  }

  IREE::LinalgExt::LinalgTransformationFilter filter;
  bool fuseInputProducer;
};

/// Patterns for workgroup level tiling. Workgroup tiling is done at the flow
/// level but we may have extra tiling for the reduction dimension. Therefore we
/// tile again without distributing.
static void populateTilingPatterns(RewritePatternSet &patterns,
                                   bool fuseInputProducer) {
  MLIRContext *context = patterns.getContext();

  IREE::LinalgExt::LinalgTransformationFilter filter(
      ArrayRef<StringAttr>{
          StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getWorkgroupKTiledMarker()));
  filter.setMatchByDefault();

  patterns.add<TileConsumerAndFuseInputProducer>(context, filter,
                                                 fuseInputProducer);
}

LogicalResult tileReductionToSerialLoops(func::FuncOp funcOp,
                                         bool fuseInputProducer) {
  {
    // Tile again at the workgroup level since redution dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    RewritePatternSet wgTilingPatterns(funcOp.getContext());
    populateTilingPatterns(wgTilingPatterns, fuseInputProducer);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(wgTilingPatterns)))) {
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
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(wgTilingCanonicalizationPatterns)))) {
      return failure();
    }
    return success();
  }
}

/// Tile parallel dimensions according to the attribute tile sizes attached to
/// each op.
static LogicalResult tileParallelDims(func::FuncOp funcOp,
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
    auto attr = tilingOp->getAttr(
        IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker);
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
    ArrayAttr mapping = rewriter.getArrayAttr(idDims);
    auto tilingResult =
        linalg::tileToForallOp(rewriter, tilingOp, numThreads, mapping);
    rewriter.replaceOp(tilingOp, tilingResult->tileOp->getResults());
  }
  return success();
}

// Tile convolution output window dimension by 1 to prepare downsizing.
static LogicalResult tileAndUnrollConv(func::FuncOp funcOp) {
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
        scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
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
    ArrayRef<Operation *> loops = tileAndFuseResult.value().loops;
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

namespace {
struct GPUTensorTilePass : public GPUTensorTileBase<GPUTensorTilePass> {
private:
  // Distribute the workloads to warp if true otherwise distribute to threads.
  bool distributeToWarp = false;

public:
  GPUTensorTilePass(bool distributeToWarp)
      : distributeToWarp(distributeToWarp) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!isEntryPoint(funcOp))
      return;

    auto workgroupSize = llvm::map_to_vector(
        getEntryPoint(funcOp)->getWorkgroupSize().value(),
        [&](Attribute attr) { return llvm::cast<IntegerAttr>(attr).getInt(); });
    if (failed(tileParallelDims(funcOp, workgroupSize, distributeToWarp))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "// --- After second level of tiling:\n";
      funcOp.dump();
    });

    // Tile to serial loops to the wg tile size to handle reductions and other
    // dimension that have not been distributed.
    if (failed(tileReductionToSerialLoops(funcOp)))
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

std::unique_ptr<OperationPass<func::FuncOp>>
createGPUTensorTile(bool distributeToWarp) {
  return std::make_unique<GPUTensorTilePass>(distributeToWarp);
}

} // namespace mlir::iree_compiler
