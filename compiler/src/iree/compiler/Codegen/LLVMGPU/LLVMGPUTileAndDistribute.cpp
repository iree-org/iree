// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-tile-and-distribute"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUTILEANDDISTRIBUTEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

/// Tiles to workgroup level. Workgroup tiling is done at the flow level but we
/// may have extra tiling for the reduction dimension. Therefore we tile again
/// without distributing.
static LogicalResult tileReductionLoops(mlir::FunctionOpInterface funcOp) {
  auto tileSizesFn = [](OpBuilder &builder,
                        Operation *op) -> SmallVector<OpFoldResult> {
    auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(op->getContext(), getTileSizes(op, 0));
    auto zeroAttr = builder.getIndexAttr(0);
    for (unsigned depth : partitionedLoops) {
      if (depth < tileSizes.size()) {
        tileSizes[depth] = zeroAttr;
      }
    }

    int numLoops = cast<TilingInterface>(op).getLoopIteratorTypes().size();
    tileSizes.resize(numLoops, zeroAttr);
    return tileSizes;
  };

  auto tilingOptions =
      scf::SCFTilingOptions().setTileSizeComputationFunction(tileSizesFn);

  MLIRContext *context = funcOp.getContext();
  LinalgTransformationFilter filter(
      ArrayRef<StringAttr>{
          StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getWorkgroupKTiledMarker()));
  filter.setMatchByDefault();

  return tileLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

static LogicalResult tileToSerialLoops(mlir::FunctionOpInterface funcOp) {
  {
    // Tile again at the workgroup level since redution dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    if (failed(tileReductionLoops(funcOp))) {
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

/// Return the tile size associated to one thread or warp based on the number of
/// element in the group.
static SmallVector<Value>
calculateDistributedTileSize(ArrayRef<int64_t> numElements, OpBuilder &builder,
                             Operation *operation) {
  SmallVector<int64_t> blockTileSize = getTileSizes(operation, 0);
  SmallVector<Value> tileSizesVal;
  // Use partitionedLoop to know what loop needs to be distributed.
  auto interfaceOp = cast<PartitionableLoopsInterface>(operation);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    return tileSizesVal;
  }
  auto zero = arith::ConstantIndexOp::create(builder, operation->getLoc(), 0);
  tileSizesVal.resize(
      cast<TilingInterface>(operation).getLoopIteratorTypes().size(), zero);

  // partitionedLoops contains the dimensions we want to distribute.
  // We are distributing them in order onto the different workgroup
  // dimensions.
  SmallVector<int64_t> distributedDim(numElements);
  distributedDim.resize(partitionedLoops.size());
  unsigned idIdx = 0;
  std::reverse(distributedDim.begin(), distributedDim.end());
  for (unsigned depth : partitionedLoops) {
    if (depth >= blockTileSize.size())
      continue;
    tileSizesVal[depth] = arith::ConstantIndexOp::create(
        builder, operation->getLoc(),
        llvm::divideCeil(blockTileSize[depth], distributedDim[idIdx++]));
    if (idIdx == kNumMaxParallelDims)
      break;
  }
  return tileSizesVal;
}

/// Tiles to warp.
static LogicalResult tileToWarp(mlir::FunctionOpInterface funcOp,
                                SmallVectorImpl<int64_t> &workgroupSize) {
  std::array<int64_t, 3> warpPerWorkgroup = {
      workgroupSize[0] / kWarpSize, workgroupSize[1], workgroupSize[2]};

  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [warpPerWorkgroup](OpBuilder &builder, Operation *operation) {
        return calculateDistributedTileSize(warpPerWorkgroup, builder,
                                            operation);
      };
  auto getWarpProcInfoFn = [warpPerWorkgroup](
                               OpBuilder &builder, Location loc,
                               ArrayRef<Range> parallelLoopRanges) {
    return getSubgroupIdsAndCounts(builder, loc, /*warpSize=*/32u,
                                   parallelLoopRanges.size(), warpPerWorkgroup);
  };
  linalg::LinalgLoopDistributionOptions warpDistributionOptions;
  warpDistributionOptions.procInfo = getWarpProcInfoFn;

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getInnerTileSizeFn)
                           .setDistributionOptions(warpDistributionOptions);
  MLIRContext *context = funcOp.getContext();
  LinalgTransformationFilter filter(
      {StringAttr::get(context, getWorkgroupKTiledMarker()),
       StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  filter.setMatchByDefault();
  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

/// Patterns for thread level tiling.
static LogicalResult tileToInvocation(mlir::FunctionOpInterface funcOp,
                                      SmallVectorImpl<int64_t> &workgroupSize) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [&](OpBuilder &builder, Operation *operation) {
        return calculateDistributedTileSize(workgroupSize, builder, operation);
      };
  auto getThreadProcInfoFn =
      [&workgroupSize](OpBuilder &builder, Location loc,
                       ArrayRef<Range> parallelLoopRanges) {
        return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                        workgroupSize);
      };
  linalg::LinalgLoopDistributionOptions invocationDistributionOptions;
  invocationDistributionOptions.procInfo = getThreadProcInfoFn;

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(invocationDistributionOptions);

  MLIRContext *context = funcOp.getContext();
  LinalgTransformationFilter f(
      {StringAttr::get(context, getWorkgroupKTiledMarker()),
       StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  f.addFilter([](Operation *op) {
     // FFT doesn't support second level of tiling yet.
     return success(!isa<IREE::LinalgExt::FftOp>(op));
   }).setMatchByDefault();

  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, f);
}

namespace {
class LLVMGPUTileAndDistributePass final
    : public impl::LLVMGPUTileAndDistributePassBase<
          LLVMGPUTileAndDistributePass> {
private:
  // Distribute the workloads to warp if true otherwise distribute to threads.
  bool distributeToWarp = false;

public:
  using impl::LLVMGPUTileAndDistributePassBase<
      LLVMGPUTileAndDistributePass>::LLVMGPUTileAndDistributePassBase;
  LLVMGPUTileAndDistributePass(bool distributeToWarp)
      : distributeToWarp(distributeToWarp) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    // Promote C matrix and propagate the potential  fill producer into the temp
    // allocation. This needs to be done before reduction tiling.
    {
      RewritePatternSet promotionPatterns(&getContext());
      populateContractPromotionPatterns(promotionPatterns, {2});
      if (failed(applyPatternsGreedily(funcOp, std::move(promotionPatterns)))) {
        return signalPassFailure();
      }
      propagateSharedMemoryCopy(funcOp);
    }

    // Tile again at the workgroup level since reduction dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    if (failed(tileToSerialLoops(funcOp))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After tile reductions:";
      funcOp.dump();
    });

    std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
        getWorkgroupSize(funcOp);
    if (!maybeWorkgroupSize) {
      funcOp.emitOpError("expected workgroup size to be set on the lowering "
                         "config for the function");
      return signalPassFailure();
    }

    SmallVector<int64_t> workgroupSize = maybeWorkgroupSize.value();
    int64_t flatWorkgroupSize =
        workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
    // Only promote to workgroup size if there are multiple warps.
    if (flatWorkgroupSize > kWarpSize) {
      RewritePatternSet promotionPatterns(&getContext());

      populateContractPromotionPatterns(promotionPatterns, {0, 1});

      if (failed(applyPatternsGreedily(funcOp, std::move(promotionPatterns)))) {
        return signalPassFailure();
      }
      // Insert barriers before and after copies to workgroup memory.
      insertBarriersAroundSharedMemoryCopy(funcOp);
    }

    {
      RewritePatternSet promotionCanonicalization =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      if (failed(applyPatternsGreedily(funcOp,
                                       std::move(promotionCanonicalization)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After promotion:";
      funcOp.dump();
    });

    if (distributeToWarp) {
      // Apply last level of tiling and distribute to warps.
      if (failed(tileToWarp(funcOp, workgroupSize))) {
        return signalPassFailure();
      }

    } else {
      // Apply last level of tiling and distribute to threads.
      if (failed(tileToInvocation(funcOp, workgroupSize))) {
        return signalPassFailure();
      }
    }
    {
      // Apply canonicalization patterns.
      RewritePatternSet threadTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinSCFCanonicalizationPattern(
          threadTilingCanonicalizationPatterns);
      if (failed(applyPatternsGreedily(
              funcOp, std::move(threadTilingCanonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After tile and distribute to threads:";
      funcOp.dump();
    });
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUTileAndDistributePass(bool distributeToWarp) {
  return std::make_unique<LLVMGPUTileAndDistributePass>(distributeToWarp);
}

} // namespace mlir::iree_compiler
