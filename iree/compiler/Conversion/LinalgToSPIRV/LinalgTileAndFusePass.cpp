// Copyright 2020 Google LLC
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

//===- LinalgTilingOnBuffers.cpp - Tile and fuse Linalg on Buffers --------===//
//
// Implements a pass to tile and fuse linalg operations on buffers.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Attributes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MarkerUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MemorySpace.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

#define DEBUG_TYPE "iree-linalg-tile-and-fuse"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns true if the linalg op has padding attribute, and that it has
/// non-zero entries.
template <typename OpTy>
static bool hasPadding(OpTy op) {
  Optional<DenseIntElementsAttr> padding = op.padding();
  if (!padding) return false;
  return llvm::any_of(padding.getValue(),
                      [](APInt v) -> bool { return !v.isNullValue(); });
}

//===----------------------------------------------------------------------===//
// Pass and patterns
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
struct LinalgTileAndFusePass
    : public PassWrapper<LinalgTileAndFusePass, OperationPass<ModuleOp>> {
  LinalgTileAndFusePass() = default;
  LinalgTileAndFusePass(const SPIRVCodegenOptions &passedOptions) {
    options = passedOptions;
  }
  LinalgTileAndFusePass(const LinalgTileAndFusePass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, linalg::LinalgDialect,
                    scf::SCFDialect, ShapeDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;

 private:
  SPIRVCodegenOptions options;

  // TODO: Find a common place to put these options. They are defined three
  // times, once here, once for the pass pipeline and once for the binary.
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes", llvm::cl::desc("Set tile sizes to use"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};

  ListOption<int64_t> workgroupSize{
      *this, "workgroup-size",
      llvm::cl::desc(
          "Number of workgroups to dispatch for the SPIR-V module; at most "
          "three integers standarding for the x, y, and z dimension; "
          "additional arguments will be ignored (used only for testing)"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};

  Option<bool> useWorkgroupMemory{
      *this, "use-workgroup-memory",
      llvm::cl::desc(
          "Enable use of workgroup memory in SPIR-V code generation pipeline"),
      llvm::cl::init(false)};

  Option<bool> useVectorization{
      *this, "use-vectorization",
      llvm::cl::desc(
          "Enable use of vectorization in SPIR-V code generation pipeline"),
      llvm::cl::init(false)};
};

//===----------------------------------------------------------------------===//
// Patterns to tile computation to map to workgroups.
//===----------------------------------------------------------------------===//

/// Distribution options for linalg.matmul when targeting workgroups.
static linalg::LinalgLoopDistributionOptions matmulDistributionOptions = {
    [](OpBuilder &builder, Location loc,
       ArrayRef<SubViewOp::Range> parallelLoopRanges) {
      return getGPUProcessorIdsAndCounts<gpu::BlockIdOp, gpu::GridDimOp>(
          builder, loc, parallelLoopRanges.size());
    },
    {linalg::DistributionMethod::CyclicNumProcsEqNumIters,
     linalg::DistributionMethod::CyclicNumProcsEqNumIters,
     linalg::DistributionMethod::CyclicNumProcsEqNumIters}};

/// Pattern for tiling operations. Updates the workgroup size in the surrounding
/// function operation if tiling succeeds.
template <typename MatmulOp>
struct TileMatmulPattern : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileMatmulPattern(MLIRContext *context, linalg::LinalgTilingOptions options,
                    const LaunchConfig &launchConfig,
                    PatternBenefit benefit = 1)
      : Base(MatmulOp::getOperationName(), context,
             options.setDistributionOptions(matmulDistributionOptions),
             linalg::LinalgMarker(
                 ArrayRef<Identifier>(),
                 Identifier::get(getWorkgroupNumItemsGENumItersMarker(),
                                 context)),
             benefit),
        launchConfig(launchConfig) {}

  virtual LogicalResult matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
    // Find the parent FuncOp before tiling. If tiling succeeds, the op will be
    // erased.
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    if (!funcOp || failed(Base::matchAndRewrite(op, rewriter)) ||
        failed(updateWorkGroupSize(funcOp, launchConfig.getWorkgroupSize())) ||
        (funcOp.getAttr(getNumWorkgroupsFnAttrName()) &&
         failed(createNumWorkgroupsFromResultShape(
             rewriter, cast<linalg::LinalgOp>(op), funcOp,
             launchConfig.getTileSizes(op, 0))))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  const LaunchConfig &launchConfig;
};

/// Pattern to tile linalg.matmul for subgroups.
struct TileMatmulSubgroupPattern
    : public linalg::LinalgTilingPattern<linalg::MatmulOp> {
  using Base = linalg::LinalgTilingPattern<linalg::MatmulOp>;
  TileMatmulSubgroupPattern(MLIRContext *context,
                            linalg::LinalgTilingOptions options,
                            PatternBenefit benefit = 1)
      : Base(context, options,
             linalg::LinalgMarker(
                 Identifier::get(getWorkgroupNumItemsGENumItersMarker(),
                                 context),
                 Identifier::get(getVectorizeMarker(), context)),
             benefit) {}
};

/// Distribution options for targeting workgroups for convolution/pooling
/// operations.
static linalg::LinalgLoopDistributionOptions convPoolDistributionOptions = {
    [](OpBuilder &builder, Location loc,
       ArrayRef<SubViewOp::Range> parallelLoopRanges) {
      return getGPUProcessorIdsAndCounts<gpu::BlockIdOp, gpu::GridDimOp>(
          builder, loc, parallelLoopRanges.size());
    },
    {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
     linalg::DistributionMethod::Cyclic}};

/// Pattern for tiling convolution and pooling operations.
template <typename OpTy>
struct TileConvPoolPattern : public linalg::LinalgTilingPattern<OpTy> {
  using Base = linalg::LinalgTilingPattern<OpTy>;
  TileConvPoolPattern(MLIRContext *context, linalg::LinalgTilingOptions options,
                      const LaunchConfig &launchConfig,
                      PatternBenefit benefit = 1)
      : Base(context,
             options.setDistributionOptions(convPoolDistributionOptions),
             linalg::LinalgMarker(
                 ArrayRef<Identifier>(),
                 Identifier::get(getWorkgroupMarker(), context)),
             benefit),
        launchConfig(launchConfig) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (hasPadding(cast<OpTy>(op))) return failure();
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    if (!funcOp || failed(Base::matchAndRewrite(op, rewriter)) ||
        failed(updateWorkGroupSize(funcOp, launchConfig.getWorkgroupSize())))
      return failure();
    funcOp.removeAttr(getNumWorkgroupsFnAttrName());
    return success();
  }

  const LaunchConfig &launchConfig;
};

/// Populate patterns for first-level tiling.
static void populateTilingToWorkgroupPatterns(
    MLIRContext *context, const LaunchConfig &launchConfig,
    OwningRewritePatternList &patterns) {
  // Function to compute first level tiling values.
  std::function<SmallVector<Value, 4>(OpBuilder &, Operation *)>
      getOuterTileSizeFn =
          [&launchConfig](OpBuilder &builder,
                          Operation *operation) -> SmallVector<Value, 4> {
    ArrayRef<int64_t> tileSizes = launchConfig.getTileSizes(operation, 0);
    if (tileSizes.empty()) return {};
    SmallVector<Value, 4> tileSizesVal;
    tileSizesVal.reserve(tileSizes.size());
    for (auto val : tileSizes) {
      tileSizesVal.push_back(
          builder.create<ConstantIndexOp>(operation->getLoc(), val));
    }
    return tileSizesVal;
  };
  patterns.insert<TileConvPoolPattern<linalg::ConvOp>,
                  TileMatmulPattern<linalg::MatmulOp>,
                  TileMatmulPattern<linalg::BatchMatmulOp>,
                  TileConvPoolPattern<linalg::PoolingMaxOp>,
                  TileConvPoolPattern<linalg::PoolingMinOp>,
                  TileConvPoolPattern<linalg::PoolingSumOp>>(
      context,
      linalg::LinalgTilingOptions()
          .setTileSizeComputationFunction(getOuterTileSizeFn)
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
      launchConfig);
}

//===----------------------------------------------------------------------===//
// Patterns to promote subviews to workgroup memory
//===----------------------------------------------------------------------===//

/// Pattern to promote matmul operands to workgroup memory.
struct PromoteMatmulSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::MatmulOp> {
  PromoteMatmulSubviewsPattern(
      MLIRContext *context, linalg::LinalgPromotionOptions options,
      linalg::LinalgMarker marker = linalg::LinalgMarker(),
      PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::MatmulOp>(
            context,
            options.setOperandsToPromote({0, 1}).setUseFullTileBuffers(
                {false, false}),
            linalg::LinalgMarker(
                Identifier::get(getWorkgroupNumItemsGENumItersMarker(),
                                context),
                Identifier::get(getWorkgroupMemoryNumItemsGENumItersMarker(),
                                context)),
            benefit) {}
};

/// Patterns to promote convolution operands to workgroup memory.
// TODO(ravishankarm): This pattern is only promoting the image subview to
// workgroup memory. In reality we should also be able to promote the filter
// subview to workgroup memory as well. Since none of the loops used to access
// the filter are tiled, this would mean the entire filter is moved to workgroup
// memory. Two reasons this is not done right now:
// 1) Linalg when tiling doesnt create a subview for the filter (since none of
//    its dimensions are tiled. This needs to be relaxed (maybe by using an
//    option.
// 2) Maybe there are better alternatives for handling filter (using different
//    StorageClasses, since for inference workloads these are model
//    constants. This is TBD.
struct PromoteConvolutionSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::ConvOp> {
  PromoteConvolutionSubviewsPattern(
      MLIRContext *context, linalg::LinalgPromotionOptions options,
      linalg::LinalgMarker marker = linalg::LinalgMarker(),
      PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::ConvOp>(
            context,
            options.setOperandsToPromote({1}).setUseFullTileBuffers(
                {false, false}),
            linalg::LinalgMarker(
                Identifier::get(getWorkgroupMarker(), context),
                Identifier::get(getWorkgroupMemoryMarker(), context)),
            benefit) {}
};
}  // namespace

static void populatePromotionPatterns(MLIRContext *context,
                                      OwningRewritePatternList &patterns) {
  patterns
      .insert<PromoteMatmulSubviewsPattern, PromoteConvolutionSubviewsPattern>(
          context,
          linalg::LinalgPromotionOptions()
              .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                            deallocateWorkgroupMemory)
              .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory));
}

//===----------------------------------------------------------------------===//
// Patterns and methods for subgroup tiling.
//===----------------------------------------------------------------------===//

/// Computes the Value for subgroupID along each dimension given number of
/// subgroups `numSubGroups` along each dimension (x-first, y-second, z-third).
static SmallVector<linalg::ProcInfo, 2> getSubgroupIdsAndCounts(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> numSubgroups) {
  Type indexType = builder.getIndexType();
  Value subgroupId = builder.create<gpu::SubgroupIdOp>(loc, indexType);
  SmallVector<linalg::ProcInfo, 2> procInfo(numSubgroups.size());

  // subgroupID
  //   = id.z * nsubgroups.y * nsubgroups.x + id.y * nsubgroups.x + id.x
  using edsc::op::operator%;
  for (size_t i = 0, e = numSubgroups.size(); i != e; ++i) {
    Value nprocs = builder.create<ConstantIndexOp>(loc, numSubgroups[i]);
    Value procId = subgroupId % nprocs;
    procInfo[e - i - 1] = linalg::ProcInfo{procId, nprocs};
    subgroupId = builder.create<SignedDivIOp>(loc, subgroupId, nprocs);
  }
  return procInfo;
}

/// Patterns for second level tiling to target subgroups.
static void populateTilingToSubgroupPatterns(
    MLIRContext *context, const LaunchConfig &launchConfig,
    OwningRewritePatternList &patterns) {
  std::function<SmallVector<Value, 4>(OpBuilder &, Operation *)>
      getInnerTileSizeFn =
          [&launchConfig](OpBuilder &builder,
                          Operation *operation) -> SmallVector<Value, 4> {
    ArrayRef<int64_t> tileSizes = launchConfig.getTileSizes(operation, 1);
    if (tileSizes.empty()) return {};
    SmallVector<Value, 4> tileSizesVal;
    tileSizesVal.reserve(tileSizes.size());
    for (auto val : tileSizes) {
      tileSizesVal.push_back(
          builder.create<ConstantIndexOp>(operation->getLoc(), val));
    }
    return tileSizesVal;
  };

  auto getSubgroupProcInfoFn =
      [&launchConfig](OpBuilder &builder, Location loc,
                      ArrayRef<SubViewOp::Range> parallelLoopRanges) {
        ArrayRef<int64_t> numSubgroups =
            launchConfig.getNumSubgroups().take_front(
                parallelLoopRanges.size());
        return getSubgroupIdsAndCounts(builder, loc, numSubgroups);
      };
  linalg::LinalgLoopDistributionOptions subgroupDistributionOptions = {
      getSubgroupProcInfoFn,
      {linalg::DistributionMethod::CyclicNumProcsEqNumIters,
       linalg::DistributionMethod::CyclicNumProcsEqNumIters}};
  patterns.insert<TileMatmulSubgroupPattern>(
      context, linalg::LinalgTilingOptions()
                   .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
                   .setTileSizeComputationFunction(getInnerTileSizeFn)
                   .setDistributionOptions(subgroupDistributionOptions));
}

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(MLIRContext *context,
                                          const LaunchConfig &launchConfig,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<linalg::LinalgVectorizationPattern<linalg::MatmulOp>>(
      context,
      linalg::LinalgMarker(Identifier::get(getVectorizeMarker(), context)));
}

/// Apply canonicalizations related to tiling to make promotion/vectorization
/// easier.
static void applyCanonicalizationPatterns(MLIRContext *context, Operation *op) {
  OwningRewritePatternList canonicalizationPatterns;
  canonicalizationPatterns.insert<AffineMinCanonicalizationPattern>(context);
  AffineApplyOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  AffineMinOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  SubViewOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  applyPatternsAndFoldGreedily(op, canonicalizationPatterns);
}

void LinalgTileAndFusePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  // Override options with command line values.
  if (!tileSizes.empty())
    options.tileSizes.assign(tileSizes.begin(), tileSizes.end());
  if (!workgroupSize.empty())
    options.workgroupSize.assign(workgroupSize.begin(), workgroupSize.end());
  if (useWorkgroupMemory) options.useWorkgroupMemory = true;
  if (useVectorization) options.useVectorization = true;

  LLVM_DEBUG(
      llvm::dbgs() << "--- IREE Linalg tile and fuse configuration ---\n";);
  for (FuncOp funcOp : module.getOps<FuncOp>()) {
    if (!isEntryPoint(funcOp)) continue;

    Region &body = funcOp.getBody();
    if (!llvm::hasSingleElement(body.getBlocks())) {
      funcOp.emitError("unhandled dispatch function with multiple blocks");
      return signalPassFailure();
    }
    Block &block = body.front();
    auto linalgOps = block.getOps<linalg::LinalgOp>();
    if (linalgOps.empty()) continue;

    LaunchConfig launchConfig;
    SmallVector<linalg::LinalgOp, 4> linalgOpsVec(linalgOps.begin(),
                                                  linalgOps.end());
    if (failed(launchConfig.init(options, linalgOpsVec))) {
      funcOp.emitError("unable to find launch configuration");
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "@func " << funcOp.getName() << ": # workgroup sizes: [";
      interleaveComma(launchConfig.getWorkgroupSize(), llvm::dbgs());
      llvm::dbgs() << "]\n";
      for (auto op : linalgOps) {
        llvm::dbgs() << "\t" << op.getOperation()->getName() << " : ";
        TileSizesListType const &tileSizes = launchConfig.getTileSizes(op);
        llvm::dbgs() << "{";
        std::string sep = "";
        for (auto &level : enumerate(tileSizes)) {
          llvm::dbgs() << sep << level.index() << " : [";
          sep = ", ";
          interleaveComma(level.value(), llvm::dbgs());
          llvm::dbgs() << "]";
        }
        llvm::dbgs() << "}\n";
      }
    });

    OwningRewritePatternList firstLevelTilingPatterns;
    populateTilingToWorkgroupPatterns(context, launchConfig,
                                      firstLevelTilingPatterns);
    applyPatternsAndFoldGreedily(funcOp, firstLevelTilingPatterns);
    applyCanonicalizationPatterns(context, funcOp);

    if (options.useWorkgroupMemory) {
      // The promotion patterns are put separate from the tiling patterns to
      // make sure that the allocated scratchspace memory is constant sizes
      // which requires some folding to trigger.
      OwningRewritePatternList promotionPatterns;
      populatePromotionPatterns(context, promotionPatterns);
      applyPatternsAndFoldGreedily(funcOp, promotionPatterns);
      applyCanonicalizationPatterns(context, funcOp);
    }

    if (options.useVectorization) {
      OwningRewritePatternList secondLevelTilingPatterns;
      populateTilingToSubgroupPatterns(context, launchConfig,
                                       secondLevelTilingPatterns);
      applyPatternsAndFoldGreedily(funcOp, secondLevelTilingPatterns);
      applyCanonicalizationPatterns(context, funcOp);

      OwningRewritePatternList vectorizationPatterns;
      populateVectorizationPatterns(context, launchConfig,
                                    vectorizationPatterns);
      applyPatternsAndFoldGreedily(funcOp, vectorizationPatterns);
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createLinalgTileAndFusePass(
    const SPIRVCodegenOptions &options) {
  return std::make_unique<LinalgTileAndFusePass>(options);
}

static PassRegistration<LinalgTileAndFusePass> pass(
    "iree-codegen-linalg-tile-and-fuse",
    "Tile and fuse Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndFusePass>(); });

}  // namespace iree_compiler
}  // namespace mlir
