// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
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
    MLIRContext *context, OwningRewritePatternList &patterns) {
  auto tileSizesFn = [&](OpBuilder &builder,
                         Operation *op) -> SmallVector<Value, 4> {
    SmallVector<unsigned> partitionedLoops = getPartitionedLoops(op);
    SmallVector<int64_t, 4> tileSizes = getTileSizes(op, 0);
    Location loc = op->getLoc();
    auto tileSizesVal =
        llvm::to_vector<4>(llvm::map_range(tileSizes, [&](int64_t v) -> Value {
          return builder.create<arith::ConstantIndexOp>(loc, v);
        }));
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    for (unsigned depth : partitionedLoops) {
      if (depth < tileSizesVal.size()) {
        tileSizesVal[depth] = zero;
      }
    }
    return tileSizesVal;
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(tileSizesFn);

  patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
                  linalg::LinalgTilingPattern<linalg::GenericOp>>(
      context, tilingOptions,
      linalg::LinalgTransformationFilter(
          {Identifier::get(getWorkgroupMarker(), context)},
          Identifier::get(getWorkgroupKTiledMarker(), context)));
}

/// Patterns for thread level tiling.
static void populateTilingToInvocationPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    SmallVector<int64_t, 4> &workgroupSize,
    SmallVector<int64_t, 4> &workloadPerWorkgroup) {
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
        llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                                     partitionedLoops.end());
        tileSizesVal.reserve(tileSizes.size());
        for (auto val : llvm::enumerate(tileSizes)) {
          int64_t useTileSize =
              partitionedLoopsSet.count(val.index()) ? val.value() : 0;
          tileSizesVal.push_back(builder.create<arith::ConstantIndexOp>(
              operation->getLoc(), useTileSize));
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

  patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  linalg::LinalgTilingPattern<linalg::FillOp>,
                  linalg::LinalgTilingPattern<linalg::CopyOp>,
                  linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
                  linalg::LinalgTilingPattern<linalg::GenericOp>,
                  linalg::LinalgTilingPattern<linalg::Conv2DNhwcHwcfOp>,
                  linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwOp>,
                  linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwcOp>,
                  linalg_ext::TiledOpInterfaceTilingPattern>(
      context, tilingOptions,
      linalg::LinalgTransformationFilter(
          {Identifier::get(getWorkgroupMarker(), context),
           Identifier::get(getWorkgroupKTiledMarker(), context),
           Identifier::get(getWorkgroupMemoryMarker(), context)},
          Identifier::get(getVectorizeMarker(), context))
          .addFilter([](Operation *op) {
            // FFT doesn't support second level of tiling yet.
            return success(!isa<linalg_ext::FftOp>(op));
          }));
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
      populateTilingReductionPatterns(context, wgTilingPatterns);
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
    auto workloadPerWorkgroup = llvm::to_vector<4>(llvm::map_range(
        getTranslationInfo(getEntryPoint(funcOp))
            .workloadPerWorkgroup()
            .getValue(),
        [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));

    int64_t flatWorkgroupSize =
        workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
    // Only promote to workgroup size if there are multiple warps.
    if (flatWorkgroupSize > 32) {
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

    {
      // Apply last level of tiling and distribute to threads.
      OwningRewritePatternList threadLevelTilingPatterns(context);
      populateTilingToInvocationPatterns(context, threadLevelTilingPatterns,
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

std::unique_ptr<OperationPass<FuncOp>>
createLLVMGPUTileAndDistributeToThreads() {
  return std::make_unique<LLVMGPUTileAndDistributePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
