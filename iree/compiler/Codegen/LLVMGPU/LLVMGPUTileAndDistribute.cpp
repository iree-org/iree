// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvmgpu-tile-and-distribute"

namespace mlir {
namespace iree_compiler {

/// Patterns for workgroup level tiling. Workgroup tiling is done at the flow
/// level but we may have extra tiling for the reduction dimension. Therefore we
/// tile again without distributing.
static void populateTilingReductionPatterns(
    OwningRewritePatternList &patterns) {
  auto tileSizesFn = [&](OpBuilder &builder,
                         Operation *op) -> SmallVector<Value, 4> {
    SmallVector<unsigned> partitionedLoops = getPartitionedLoops(op);
    SmallVector<Value, 4> tileSizes = getTileSizes(builder, op, 0);
    auto zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    for (unsigned depth : partitionedLoops) {
      if (depth < tileSizes.size()) {
        tileSizes[depth] = zero;
      }
    }
    return tileSizes;
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(tileSizesFn);
  MLIRContext *context = patterns.getContext();
  patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
                  linalg::LinalgTilingPattern<linalg::GenericOp>>(
      context, tilingOptions,
      linalg::LinalgTransformationFilter(
          ArrayRef<Identifier>{},
          Identifier::get(getWorkgroupKTiledMarker(), context)));
}

/// Patterns for warp level tiling.
static void populateTilingToWarpPatterns(
    OwningRewritePatternList &patterns, SmallVectorImpl<int64_t> &workgroupSize,
    SmallVectorImpl<int64_t> &workloadPerWorkgroup) {
  std::array<int64_t, 3> warpPerWorkgroup = {
      workgroupSize[0] / kWarpSize, workgroupSize[1], workgroupSize[2]};

  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [&workloadPerWorkgroup, warpPerWorkgroup](OpBuilder &builder,
                                                Operation *operation) {
        SmallVector<Value, 4> tileSizesVal;
        SmallVector<int64_t, 4> tileSizes;
        for (auto workload : llvm::enumerate(workloadPerWorkgroup)) {
          tileSizes.push_back(workload.value() /
                              warpPerWorkgroup[workload.index()]);
        }
        std::reverse(tileSizes.begin(), tileSizes.end());
        if (tileSizes.empty()) return SmallVector<Value, 4>();
        SmallVector<unsigned> partitionedLoops = getPartitionedLoops(operation);
        unsigned maxDepth = partitionedLoops.back() + 1;
        auto zero =
            builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
        tileSizesVal.resize(maxDepth, zero);
        size_t tileSizeIdx = 0;
        for (unsigned depth : partitionedLoops) {
          tileSizesVal[depth] = builder.create<arith::ConstantIndexOp>(
              operation->getLoc(), tileSizes[tileSizeIdx++]);
          if (tileSizeIdx == tileSizes.size()) break;
        }
        return tileSizesVal;
      };
  auto getWarpProcInfoFn = [warpPerWorkgroup](
                               OpBuilder &builder, Location loc,
                               ArrayRef<Range> parallelLoopRanges) {
    return getSubgroupIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                   warpPerWorkgroup);
  };
  linalg::LinalgLoopDistributionOptions warpDistributionOptions;
  warpDistributionOptions.procInfo = getWarpProcInfoFn;
  warpDistributionOptions.distributionMethod = {
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getInnerTileSizeFn)
                           .setDistributionOptions(warpDistributionOptions);
  MLIRContext *context = patterns.getContext();
  patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  linalg::LinalgTilingPattern<linalg::FillOp>,
                  linalg::LinalgTilingPattern<linalg::CopyOp>,
                  linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
                  linalg::LinalgTilingPattern<linalg::GenericOp>>(
      context, tilingOptions,
      linalg::LinalgTransformationFilter(
          {Identifier::get(getWorkgroupKTiledMarker(), context),
           Identifier::get(getWorkgroupMemoryMarker(), context)},
          Identifier::get(getVectorizeMarker(), context))
          .setMatchByDefault());
}

/// Patterns for thread level tiling.
static void populateTilingToInvocationPatterns(
    OwningRewritePatternList &patterns, SmallVectorImpl<int64_t> &workgroupSize,
    SmallVectorImpl<int64_t> &workloadPerWorkgroup) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [&](OpBuilder &builder, Operation *operation) {
        SmallVector<Value, 4> tileSizesVal;
        SmallVector<int64_t, 4> tileSizes;
        for (auto workload : llvm::enumerate(workloadPerWorkgroup)) {
          tileSizes.push_back(workload.value() /
                              workgroupSize[workload.index()]);
        }
        std::reverse(tileSizes.begin(), tileSizes.end());
        if (tileSizes.empty()) return SmallVector<Value, 4>();
        SmallVector<unsigned> partitionedLoops = getPartitionedLoops(operation);
        unsigned maxDepth = partitionedLoops.back() + 1;
        auto zero =
            builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
        tileSizesVal.resize(maxDepth, zero);
        size_t tileSizeIdx = 0;
        for (unsigned depth : partitionedLoops) {
          tileSizesVal[depth] = builder.create<arith::ConstantIndexOp>(
              operation->getLoc(), tileSizes[tileSizeIdx++]);
          if (tileSizeIdx == tileSizes.size()) break;
        }
        return tileSizesVal;
      };

  auto getThreadProcInfoFn = [&workgroupSize](
                                 OpBuilder &builder, Location loc,
                                 ArrayRef<Range> parallelLoopRanges) {
    return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                    workgroupSize);
  };
  linalg::LinalgLoopDistributionOptions invocationDistributionOptions;
  invocationDistributionOptions.procInfo = getThreadProcInfoFn;
  invocationDistributionOptions.distributionMethod = {
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(invocationDistributionOptions);

  MLIRContext *context = patterns.getContext();
  patterns
      .insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
              linalg::LinalgTilingPattern<linalg::FillOp>,
              linalg::LinalgTilingPattern<linalg::CopyOp>,
              linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
              linalg::LinalgTilingPattern<linalg::GenericOp>,
              linalg::LinalgTilingPattern<linalg::Conv2DNhwcHwcfOp>,
              linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwcHwcOp>,
              linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwcHwcmOp>,
              linalg::LinalgTilingPattern<linalg::PoolingNhwcMaxOp>,
              linalg::LinalgTilingPattern<linalg::PoolingNhwcMinOp>,
              linalg::LinalgTilingPattern<linalg::PoolingNhwcSumOp>,
              IREE::LinalgExt::TiledOpInterfaceTilingPattern>(
          context, tilingOptions,
          linalg::LinalgTransformationFilter(
              {Identifier::get(getWorkgroupKTiledMarker(), context),
               Identifier::get(getWorkgroupMemoryMarker(), context)},
              Identifier::get(getVectorizeMarker(), context))
              .addFilter([](Operation *op) {
                // FFT doesn't support second level of tiling yet.
                return success(!isa<IREE::LinalgExt::FftOp>(op));
              })
              .setMatchByDefault());
}

static LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst) {
  auto copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  return success();
}

static Optional<Value> allocateWorkgroupMemory(
    OpBuilder &b, memref::SubViewOp subview,
    ArrayRef<Value> boundingSubViewSize, DataLayout &layout) {
  // In CUDA workgroup memory is represented by a global variable. Create a
  // global variable and a memref.GetGlobalOp at the beginning of the function
  // to get the memref.
  OpBuilder::InsertionGuard guard(b);
  FuncOp funcOp = subview->getParentOfType<FuncOp>();
  if (!funcOp) {
    subview.emitError("expected op to be within std.func");
    return llvm::None;
  }
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(moduleOp);

  // The bounding subview size is expected to be constant. This specified the
  // shape of the allocation.
  SmallVector<int64_t, 2> shape = llvm::to_vector<2>(
      llvm::map_range(boundingSubViewSize, [](Value v) -> int64_t {
        APInt value;
        if (matchPattern(v, m_ConstantInt(&value))) return value.getSExtValue();
        return -1;
      }));
  if (llvm::any_of(shape, [](int64_t v) { return v == -1; })) return {};
  MemRefType allocType =
      MemRefType::get(shape, subview.getType().getElementType(), {},
                      gpu::GPUDialect::getWorkgroupAddressSpace());
  b.setInsertionPoint(&moduleOp.front());
  auto global = b.create<memref::GlobalOp>(
      funcOp.getLoc(), "__shared_memory__",
      /*sym_visibility=*/b.getStringAttr("private"),
      /*type=*/allocType,
      /*initial_value=*/ElementsAttr(),
      /*constant=*/false, /*alignment=*/IntegerAttr());
  symbolTable.insert(global);

  b.setInsertionPointToStart(&(*funcOp.getBody().begin()));
  Value buffer = b.create<memref::GetGlobalOp>(funcOp.getLoc(), global.type(),
                                               global.getName());
  return buffer;
}

static LogicalResult deallocateWorkgroupMemory(OpBuilder &b, Value buffer) {
  // Nothing to do.
  return success();
}

static void populatePromotionPatterns(MLIRContext *context,
                                      OwningRewritePatternList &patterns) {
  patterns.insert<linalg::LinalgPromotionPattern<linalg::MatmulOp>,
                  linalg::LinalgPromotionPattern<linalg::BatchMatmulOp>>(
      context,
      linalg::LinalgPromotionOptions()
          .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                        deallocateWorkgroupMemory)
          .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory)
          .setOperandsToPromote({0, 1})
          .setUseFullTileBuffers({false, false}),
      linalg::LinalgTransformationFilter(
          {Identifier::get(getWorkgroupKTiledMarker(), context)},
          Identifier::get(getWorkgroupMemoryMarker(), context)));
}

namespace {
struct LLVMGPUTileAndDistributePass
    : public LLVMGPUTileAndDistributeBase<LLVMGPUTileAndDistributePass> {
 private:
  // Distribute the workloads to warp if true otherwise distribute to threads.
  bool distributeToWarp = false;

 public:
  LLVMGPUTileAndDistributePass(bool distributeToWarp)
      : distributeToWarp(distributeToWarp) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();
    if (!isEntryPoint(funcOp)) return;
    {
      // Tile again at the workgroup level since redution dimension were
      // ignored. Dimensions already tiled will be ignore since we tile to the
      // same size.
      OwningRewritePatternList wgTilingPatterns(context);
      populateTilingReductionPatterns(wgTilingPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(wgTilingPatterns));
    }

    {
      RewritePatternSet wgTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinSCFCanonicalizationPattern(
          wgTilingCanonicalizationPatterns);
      (void)applyPatternsAndFoldGreedily(
          funcOp, std::move(wgTilingCanonicalizationPatterns));
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After tile reductions:";
      funcOp.dump();
    });

    auto workgroupSize = llvm::to_vector<4>(llvm::map_range(
        getEntryPoint(funcOp).workgroup_size().getValue(),
        [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
    auto workloadPerWorkgroup =
        getTranslationInfo(getEntryPoint(funcOp)).getWorkloadPerWorkgroupVals();

    int64_t flatWorkgroupSize =
        workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
    // Only promote to workgroup size if there are multiple warps.
    if (flatWorkgroupSize > kWarpSize) {
      OwningRewritePatternList promotionPatterns(&getContext());
      populatePromotionPatterns(context, promotionPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(promotionPatterns));
      // Insert barriers before and after copies to workgroup memory and skip
      // insert barriers between back to back copy to workgroup memory.
      OpBuilder builder(&getContext());
      funcOp.walk([&builder](linalg::CopyOp copyOp) {
        if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
          Operation *prevOp = copyOp->getPrevNode();
          if (!prevOp || !hasMarker(prevOp, getCopyToWorkgroupMemoryMarker())) {
            builder.setInsertionPoint(copyOp);
            builder.create<gpu::BarrierOp>(copyOp.getLoc());
          }
          Operation *nextOp = copyOp->getNextNode();
          if (!nextOp || !hasMarker(nextOp, getCopyToWorkgroupMemoryMarker())) {
            builder.setInsertionPointAfter(copyOp);
            builder.create<gpu::BarrierOp>(copyOp.getLoc());
          }
        }
      });
    }

    {
      RewritePatternSet promotionCanonicalization =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(promotionCanonicalization));
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After promotion:";
      funcOp.dump();
    });

    if (distributeToWarp) {
      // Apply last level of tiling and distribute to warps.
      OwningRewritePatternList warpLevelTilingPatterns(context);
      populateTilingToWarpPatterns(warpLevelTilingPatterns, workgroupSize,
                                   workloadPerWorkgroup);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(warpLevelTilingPatterns));

    } else {
      // Apply last level of tiling and distribute to threads.
      OwningRewritePatternList threadLevelTilingPatterns(context);
      populateTilingToInvocationPatterns(threadLevelTilingPatterns,
                                         workgroupSize, workloadPerWorkgroup);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(threadLevelTilingPatterns));
    }
    {
      // Apply canonicalization patterns.
      RewritePatternSet threadTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinSCFCanonicalizationPattern(
          threadTilingCanonicalizationPatterns);
      (void)applyPatternsAndFoldGreedily(
          funcOp, std::move(threadTilingCanonicalizationPatterns));
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After tile and distribute to threads:";
      funcOp.dump();
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUTileAndDistribute(
    bool distributeToWarp) {
  return std::make_unique<LLVMGPUTileAndDistributePass>(distributeToWarp);
}

}  // namespace iree_compiler
}  // namespace mlir
