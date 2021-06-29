// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static constexpr int32_t kNumGPUDims = 3;

static SmallVector<linalg::ProcInfo, 2> getGPUThreadIdsAndCounts(
    OpBuilder &builder, Location loc, unsigned numDims,
    ArrayRef<int64_t> workgroupSize) {
  assert(numDims <= kNumGPUDims);
  SmallVector<linalg::ProcInfo, 2> procInfo(numDims);
  std::array<StringRef, kNumGPUDims> dimAttr{"x", "y", "z"};
  Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    StringAttr attr = builder.getStringAttr(dimAttr[i]);
    procInfo[numDims - 1 - i] = {
        builder.create<gpu::ThreadIdOp>(loc, indexType, attr),
        builder.create<ConstantOp>(loc,
                                   builder.getIndexAttr(workgroupSize[i]))};
  }
  return procInfo;
}

/// Patterns for workgroup level tiling. Workgroup tiling is done at the flow
/// level but we may have extra tiling for the reduction dimension. Therefore we
/// tile again without distributing.
static void populateTilingReductionPatterns(MLIRContext *context,
                                            OwningRewritePatternList &patterns,
                                            ArrayRef<int64_t> tileSizes) {
  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizes(tileSizes);

  patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  linalg::LinalgTilingPattern<linalg::BatchMatmulOp>>(
      context, tilingOptions,
      linalg::LinalgTransformationFilter(
          {Identifier::get(getWorkgroupMarker(), context)},
          Identifier::get(getWorkgroupKTiledMarker(), context)));
}

/// Patterns for thread level tiling.
static void populateTilingToInvocationPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    ArrayRef<int64_t> workgroupSize) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [](OpBuilder &builder, Operation *operation) {
        SmallVector<Value, 4> tileSizesVal;
        SmallVector<int64_t, 4> tileSizes = getTileSizes(operation, 2);
        if (tileSizes.empty()) return SmallVector<Value, 4>();
        tileSizesVal.reserve(tileSizes.size());
        for (auto val : llvm::enumerate(tileSizes)) {
          // Only tile the last 3 dimensions. Use tile size of 0 for any higher
          // dimension as we only support distributing on 3 dimensions.
          int64_t t =
              (tileSizes.size() - val.index()) <= kNumGPUDims ? val.value() : 0;
          tileSizesVal.push_back(
              builder.create<ConstantIndexOp>(operation->getLoc(), t));
        }
        return tileSizesVal;
      };

  auto getThreadProcInfoFn = [workgroupSize](
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

  patterns.insert<
      linalg::LinalgTilingPattern<linalg::MatmulOp>,
      linalg::LinalgTilingPattern<linalg::FillOp>,
      linalg::LinalgTilingPattern<linalg::CopyOp>,
      linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
      linalg::LinalgTilingPattern<linalg::GenericOp>,
      linalg::LinalgTilingPattern<linalg::ConvInputNHWCFilterHWCFOp>,
      linalg::LinalgTilingPattern<linalg::DepthwiseConvInputNHWCFilterHWCOp>,
      linalg::LinalgTilingPattern<linalg::ConvInputNHWCFilterHWCFOp>,
      linalg::LinalgTilingPattern<linalg::DepthwiseConvInputNHWCFilterHWCFOp>,
      linalg::LinalgTilingPattern<linalg::DepthwiseConvInputNHWCFilterHWCOp>>(
      context, tilingOptions,
      linalg::LinalgTransformationFilter(
          {Identifier::get(getWorkgroupMarker(), context),
           Identifier::get(getWorkgroupKTiledMarker(), context),
           Identifier::get(getWorkgroupMemoryMarker(), context)},
          Identifier::get(getVectorizeMarker(), context)));
}

/// Patterns for copy to shared memory mapping. Copy to shared memory are not
/// part of the launch config but needs to be distributed on the workgroup
/// picked by the root op.
static void populateTilingCopyToWorkgroupMemPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    ArrayRef<int64_t> workgroupSize) {
  // Tile and distribute copy to workgroup memory.
  linalg::TileSizeComputationFunction wgCopyTileSizeFn =
      [](OpBuilder &builder, Operation *operation) {
        const int64_t copyTileSize = 4;
        // We tile to 4 as we want each thread to load 4 element in a cyclic
        // distribution.
        SmallVector<Value, 4> tileSizesVal;
        unsigned rank =
            cast<linalg::CopyOp>(operation).getOutputBufferTypes()[0].getRank();
        for (unsigned i = 0; i < rank - 1; i++) {
          int64_t t = (rank - i) <= kNumGPUDims ? 1 : 0;
          tileSizesVal.push_back(
              builder.create<ConstantIndexOp>(operation->getLoc(), t));
        }
        tileSizesVal.push_back(
            builder.create<ConstantIndexOp>(operation->getLoc(), copyTileSize));
        return tileSizesVal;
      };
  auto getCopyThreadProcInfoFn = [workgroupSize](
                                     OpBuilder &builder, Location loc,
                                     ArrayRef<Range> parallelLoopRanges) {
    SmallVector<std::array<int64_t, 3>, 2> staticRanges;
    bool hasDynamicRange = false;
    // If the ranges are not constant fall back to naive disribution.
    for (auto range : parallelLoopRanges) {
      auto cstOffset = range.offset.getDefiningOp<ConstantIndexOp>();
      auto cstSize = range.size.getDefiningOp<ConstantIndexOp>();
      auto cstStride = range.stride.getDefiningOp<ConstantIndexOp>();
      if (!cstOffset || !cstSize || !cstStride) {
        hasDynamicRange = true;
        break;
      }
      staticRanges.push_back(
          {cstOffset.getValue(), cstSize.getValue(), cstStride.getValue()});
    }
    // Only support static dimension with 1D workgroups for now. Fall back to
    // the naive distribution for other cases.
    if (hasDynamicRange || workgroupSize[1] != 1 || workgroupSize[2] != 1)
      return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                      workgroupSize);
    Value serializedId =
        builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(), "x");
    int64_t numIds = workgroupSize[0];
    int numDims = parallelLoopRanges.size();
    SmallVector<linalg::ProcInfo, 2> procInfo(numDims);
    assert(numDims <= kNumGPUDims);
    // Distribute the available Ids on the loop dimensions.
    for (int i = numDims - 1; i >= 0; i--) {
      std::array<int64_t, 3> &range = staticRanges[i];
      Value id = serializedId;
      int64_t interval = (range[1] - range[0]) / range[2];
      Value intervalValue = builder.create<ConstantIndexOp>(loc, interval);
      int64_t count = 0;
      if (numIds <= 1) {
        count = 1;
        id = builder.create<ConstantIndexOp>(loc, 0);
      } else if (numIds > interval) {
        AffineExpr d0 = getAffineDimExpr(0, builder.getContext());
        AffineExpr s0 = getAffineSymbolExpr(0, builder.getContext());
        if (i > 0)
          id = makeComposedAffineApply(builder, loc, d0 % s0,
                                       {id, intervalValue});
        count = interval;
      } else {
        count = numIds;
      }
      numIds = numIds / interval;
      AffineExpr d0 = getAffineDimExpr(0, builder.getContext());
      AffineExpr s0 = getAffineSymbolExpr(0, builder.getContext());
      serializedId = makeComposedAffineApply(builder, loc, d0.floorDiv(s0),
                                             {serializedId, intervalValue});
      procInfo[i] = {id, builder.create<ConstantIndexOp>(loc, count)};
    }
    return procInfo;
  };
  linalg::LinalgLoopDistributionOptions copyInvocationDistributionOptions;
  copyInvocationDistributionOptions.procInfo = getCopyThreadProcInfoFn;
  copyInvocationDistributionOptions.distributionMethod = {
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(wgCopyTileSizeFn)
          .setDistributionOptions(copyInvocationDistributionOptions);
  patterns.insert<linalg::LinalgTilingPattern<linalg::CopyOp>>(
      context, tilingOptions,
      linalg::LinalgTransformationFilter(
          {Identifier::get(getCopyToWorkgroupMemoryMarker(), context)},
          Identifier::get(getVectorizeMarker(), context)));
}

static LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst) {
  // TODO(thomasraoux): Improve barrier placement.
  b.create<gpu::BarrierOp>(src.getLoc());
  auto copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  b.create<gpu::BarrierOp>(src.getLoc());
  return success();
}

static Optional<Value> allocateWorkgroupMemory(
    OpBuilder &b, memref::SubViewOp subview,
    ArrayRef<Value> boundingSubViewSize, DataLayout &layout) {
  // In CUDA workgroup memory is represented by a global variable. Create a
  // global variable and a memref.GetGlobalOp at the beginning of the funtion to
  // get the memref.
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
  Type allocType =
      MemRefType::get(shape, subview.getType().getElementType(), {},
                      gpu::GPUDialect::getWorkgroupAddressSpace());
  b.setInsertionPoint(&moduleOp.front());
  auto global =
      b.create<memref::GlobalOp>(funcOp.getLoc(), "__shared_memory__",
                                 /*sym_visibility=*/b.getStringAttr("private"),
                                 /*type=*/allocType,
                                 /*initial_value=*/ElementsAttr(),
                                 /*constant=*/false);
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
  patterns.insert<linalg::LinalgPromotionPattern<linalg::MatmulOp>>(
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

static constexpr unsigned kWorkgroupDimCount = 3;

namespace {

/// Replaces hal.interface.workgroup.size op with the constant value chosen
/// from tiling scheme.
class ConcretizeWorkgroupSizeOp final
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
 public:
  ConcretizeWorkgroupSizeOp(MLIRContext *context, ArrayRef<int64_t> tileSize)
      : OpRewritePattern(context, /*benefit=*/1), tileSize(tileSize) {}

  LogicalResult matchAndRewrite(IREE::HAL::InterfaceWorkgroupSizeOp op,
                                PatternRewriter &rewriter) const override {
    unsigned dimIndex = op.dimension().getZExtValue();

    if (dimIndex < kWorkgroupDimCount && tileSize[dimIndex] != 0) {
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, rewriter.getIndexAttr(tileSize[dimIndex]));
      return success();
    }

    return failure();
  }

 private:
  ArrayRef<int64_t> tileSize;
};

struct LLVMGPUTileAndDistributePass
    : public LLVMGPUTileAndDistributeBase<LLVMGPUTileAndDistributePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp module = variantOp.getInnerModule();

    MLIRContext *context = module->getContext();
    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      if (!isEntryPoint(funcOp)) continue;

      SmallVector<linalg::LinalgOp, 4> linalgOps;
      SmallVector<Operation *, 4> tiledLoops;

      if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) {
        return signalPassFailure();
      }
      if (linalgOps.empty() || !hasLoweringConfig(*linalgOps.begin())) return;
      std::array<int64_t, 3> workgroupSize;
      for (auto it : llvm::enumerate(funcOp->getAttr("llvmgpu_workgroup_size")
                                         .cast<DenseIntElementsAttr>()
                                         .getIntValues())) {
        workgroupSize[it.index()] = it.value().getZExtValue();
      }
      IREE::HAL::LoweringConfig config = getLoweringConfig(*linalgOps.begin());
      SmallVector<int64_t, 4> wgTileSize = getTileSizes(config, 0);
      // If there is no tile size, skip tiling.
      if (wgTileSize.empty()) return;
      unsigned numOuterParallelLoops = 0;
      unsigned numContractionLoops = 0;
      for (linalg::LinalgOp linalgOp : linalgOps) {
        numOuterParallelLoops =
            std::max(getNumOuterParallelLoops(linalgOp), numOuterParallelLoops);
        numContractionLoops =
            std::max(linalgOp.getNumReductionLoops(), numContractionLoops);
      }
      size_t numTilableDims =
          std::min(kWorkgroupDimCount, numOuterParallelLoops);
      wgTileSize.resize(numTilableDims);
      std::reverse(wgTileSize.begin(), wgTileSize.end());
      {
        // Replace the opaque tile size for workgroup level tiling and update
        // the number of workgroups based on the tile size.
        OwningRewritePatternList patterns(context);
        patterns.insert<ConcretizeWorkgroupSizeOp>(context, wgTileSize);

        (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
        if (failed(materializeStaticLaunchInformation(funcOp, wgTileSize))) {
          funcOp.emitOpError("failed to materialize static launch information");
          return signalPassFailure();
        }
      }

      if (numContractionLoops > 0) {
        // Tile again at the workgroup level since redution dimension were
        // ignored. Dimensions already tiled will be ignore since we tile to the
        // same size.
        OwningRewritePatternList wgTilingPatterns(context);
        populateTilingReductionPatterns(context, wgTilingPatterns,
                                        getTileSizes(config, 0));
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

      {
        OwningRewritePatternList promotionPatterns(&getContext());
        populatePromotionPatterns(context, promotionPatterns);
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(promotionPatterns));
      }

      {
        RewritePatternSet promotionCanonicalization =
            linalg::getLinalgTilingCanonicalizationPatterns(context);
        (void)applyPatternsAndFoldGreedily(
            funcOp, std::move(promotionCanonicalization));
      }

      {
        // Apply last level of tiling and distribute to threads.
        OwningRewritePatternList threadLevelTilingPatterns(context);
        populateTilingToInvocationPatterns(context, threadLevelTilingPatterns,
                                           workgroupSize);
        populateTilingCopyToWorkgroupMemPatterns(
            context, threadLevelTilingPatterns, workgroupSize);
        (void)applyPatternsAndFoldGreedily(
            funcOp, std::move(threadLevelTilingPatterns));
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
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMGPUTileAndDistributeToThreads() {
  return std::make_unique<LLVMGPUTileAndDistributePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
