// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToNVVM/KernelConfig.h"
#include "iree/compiler/Conversion/LinalgToNVVM/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static SmallVector<linalg::ProcInfo, 2> getGPUThreadIdsAndCounts(
    OpBuilder &builder, Location loc, unsigned numDims) {
  static constexpr int32_t kNumGPUDims = 3;
  SmallVector<linalg::ProcInfo, 2> procInfo(numDims);
  std::array<StringRef, kNumGPUDims> dimAttr{"x", "y", "z"};
  Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    StringAttr attr =
        builder.getStringAttr(dimAttr[std::min<unsigned>(i, kNumGPUDims)]);
    procInfo[numDims - 1 - i] = {
        builder.create<gpu::ThreadIdOp>(loc, indexType, attr),
        builder.create<gpu::BlockDimOp>(loc, indexType, attr)};
  }
  return procInfo;
}

/// Patterns for thread level tiling.
static void populateTilingToInvocationPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [](OpBuilder &builder, Operation *operation) {
        ArrayRef<int64_t> tileSizes = {4};
        if (tileSizes.empty()) return SmallVector<Value, 4>();
        SmallVector<Value, 4> tileSizesVal;
        tileSizesVal.reserve(tileSizes.size());
        for (auto val : tileSizes) {
          tileSizesVal.push_back(
              builder.create<ConstantIndexOp>(operation->getLoc(), val));
        }
        return tileSizesVal;
      };

  auto getThreadProcInfoFn = [](OpBuilder &builder, Location loc,
                                ArrayRef<Range> parallelLoopRanges) {
    return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size());
  };
  linalg::LinalgLoopDistributionOptions invocationDistributionOptions = {
      getThreadProcInfoFn,
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(invocationDistributionOptions);

  patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  linalg::LinalgTilingPattern<linalg::FillOp>,
                  linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
                  linalg::LinalgTilingPattern<linalg::GenericOp>>(
      context, tilingOptions,
      linalg::LinalgTransformationFilter(
          {Identifier::get(getWorkgroupMarker(), context)},
          Identifier::get(getVectorizeMarker(), context)));
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

struct TileAndDistributeToThreads
    : public PassWrapper<TileAndDistributeToThreads,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    IREE::HAL::ExecutableTargetOp targetOp = getOperation();
    ModuleOp module = targetOp.getInnerModule();

    MLIRContext *context = module->getContext();
    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      if (!isEntryPoint(funcOp)) continue;

      SmallVector<linalg::LinalgOp, 4> linalgOps;
      SmallVector<Operation *, 4> tiledLoops;

      if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) {
        return signalPassFailure();
      }
      linalg::Aliases aliases;
      linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
      auto config = getCUDALaunchConfig(context, dependenceGraph, linalgOps);
      if (!config) return signalPassFailure();

      // Attach the workgroup size as an attribute. This will be used when
      // creating the flatbuffer.
      funcOp->setAttr("cuda_workgroup_size",
                      DenseElementsAttr::get<int64_t>(
                          VectorType::get(3, IntegerType::get(context, 64)),
                          config->getWorkgroupSize()));

      Operation *rootOp = config->getRootOperation(llvm::to_vector<4>(
          llvm::map_range(linalgOps, [](linalg::LinalgOp op) {
            return op.getOperation();
          })));
      SmallVector<int64_t, 4> wgTileSize =
          llvm::to_vector<4>(config->getTileSizes(rootOp, 0));
      // If there is no tile size, skip tiling.
      if (wgTileSize.empty()) return;
      std::reverse(wgTileSize.begin(), wgTileSize.end());
      size_t numTilableDims =
          std::min(kWorkgroupDimCount,
                   getNumOuterParallelLoops(cast<linalg::LinalgOp>(rootOp)));
      wgTileSize.resize(numTilableDims);
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

      {
        // Apply last level of tiling and distribute to threads.
        OwningRewritePatternList threadLevelTilingPatterns(context);
        populateTilingToInvocationPatterns(context, threadLevelTilingPatterns);
        (void)applyPatternsAndFoldGreedily(
            funcOp, std::move(threadLevelTilingPatterns));
        applyCanonicalizationPatternsForTiling(context, funcOp);
      }
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createTileAndDistributeToThreads() {
  return std::make_unique<TileAndDistributeToThreads>();
}

static PassRegistration<TileAndDistributeToThreads> pass(
    "iree-codegen-cuda-tile-and-distribute",
    "Pass to tile and distribute linalg ops within a workgroup.");

}  // namespace iree_compiler
}  // namespace mlir
