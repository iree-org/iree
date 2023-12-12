#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/LLVMCPU/TargetMLTransformInfo.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/PluginAPI/Client.h"

#include "llvm/Support/Debug.h"

namespace mlir::iree_compiler {

namespace {

using IREE::Codegen::DispatchLoweringPassPipeline;

static SizesAndScalableFlags
getNonScalableTileSizes(ArrayRef<int64_t> tileSizes) {
  return {SmallVector<int64_t>(tileSizes),
          SmallVector<bool>(tileSizes.size(), false)};
}

template <typename T>
static SmallVector<T> copySuffixAndPrependArray(unsigned targetLength,
                                                ArrayRef<T> suffix,
                                                T paddingValue) {
  SmallVector<T> result(targetLength, paddingValue);
  for (unsigned i = 1; i <= std::min(result.size(), suffix.size()); ++i) {
    result[result.size() - i] = suffix[suffix.size() - i];
  }
  return result;
}

static int64_t getValueRangeMinVectorSize(func::FuncOp funcOp,
                                          ValueRange values) {
  std::optional<int64_t> minVectorSize;
  for (auto valueType : values.getTypes()) {
    auto shapedType = cast<ShapedType>(valueType);
    int64_t vectorSize = getVectorSize(funcOp, shapedType);
    if (minVectorSize) {
      minVectorSize = std::min(*minVectorSize, vectorSize);
    } else {
      minVectorSize = vectorSize;
    }
  }
  return *minVectorSize;
}

static int64_t getMinVectorSize(func::FuncOp funcOp,
                                linalg::ContractionOpInterface op) {
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  return getValueRangeMinVectorSize(
      funcOp, {op.lhs(), op.rhs(), linalgOp.getDpsInitOperand(0)->get()});
}

class X86TransposeLikeOpPattern
    : public OpTileSizeSelectionPattern<linalg::GenericOp> {
public:
  FailureOr<TileSizeAndPipelineConfig>
  matchAndConfig(func::FuncOp funcOp, linalg::GenericOp rootOp) const override {
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!hasAVX2Feature(targetAttr) || !isSupportedTransposeOp(rootOp) ||
        !rootOp.hasPureTensorSemantics()) {
      return failure();
    }

    auto targetMLTransInfo =
        TargetMLTransformInfo::getTargetMLTransformInfo(targetAttr);
    unsigned numLoops = rootOp.getNumLoops();
    auto linalgOpInfo = LinalgOpInfo(rootOp);

    DistributionHeuristicConfig distConfig;
    distConfig.minTileSizes = getMinTilingSizesForEachDim(
        funcOp, rootOp, linalgOpInfo, targetMLTransInfo);
    if (llvm::all_of(distConfig.minTileSizes,
                     [](int64_t vs) { return vs == 1; })) {
      // Nothing to vectorize just lower to loops.
      return failure();
    }

    if (llvm::count_if(distConfig.minTileSizes,
                       [](int64_t tileSize) { return tileSize > 1; }) != 2) {
      // Transpose patterns are not applicable if vectorizing more or less than
      // two dims.
      return failure();
    }

    // Make sure that the original tile sizes are multiple of the tile sizes
    // to be used for the transpose op (i.e., 8x8).
    // TODO(diegocaballero): Enable 4x8 tile sizes if we find it useful.
    if (llvm::any_of(distConfig.minTileSizes, [](int64_t tileSize) {
          return tileSize > 1 && (tileSize % 8) != 0;
        })) {
      return failure();
    }

    // Replace dims to be vectorized with the new 8x8 tile sizes.
    std::replace_if(
        distConfig.minTileSizes.begin(), distConfig.minTileSizes.end(),
        [](int64_t tileSize) { return tileSize > 1; }, 8);

    SmallVector<int64_t> distTileSizes =
        getDefaultDistributedLevelTileSizes(rootOp, distConfig);

    // Set the vector level tile sizes.
    SmallVector<int64_t> vecTileSizes;
    setX86VectorTileSizes(rootOp, numLoops, distTileSizes,
                          distConfig.minTileSizes, distConfig.maxTileSizes,
                          VectorPreProcStrategy::Masking, vecTileSizes);

    TileSizeConfig tileSizeConfig(getNonScalableTileSizes(distTileSizes),
                                  getNonScalableTileSizes(vecTileSizes));
    // return setTileSizeConfigAndPipeline(
    //     funcOp, cast<TilingInterface>(rootOp.getOperation()), tileSizeConfig,
    //     DispatchLoweringPassPipeline::CPUDoubleTilingExpert);
    return TileSizeAndPipelineConfig{
        tileSizeConfig, DispatchLoweringPassPipeline::CPUDoubleTilingExpert};
  }

private:
  /// Returns true if the operation is a GenericOp implementing a supported
  /// transposition.
  static bool isSupportedTransposeOp(linalg::GenericOp genericOp) {
    // Check that the op has at least 2 dimensions.
    if (genericOp.getNumLoops() < 2) {
      return false;
    }

    // Check that the op has only one input and one output.
    // TODO(diegocaballero): Generalize to multiple inputs.
    if ((genericOp.getNumDpsInputs() != 1) ||
        (genericOp.getNumDpsInits() != 1)) {
      return false;
    }

    // Check that all the iterators are parallel.
    if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
      return false;
    }

    // Check that the two indexing maps are a permutation of each other.
    auto indexingMaps = genericOp.getIndexingMapsArray();
    return !indexingMaps[0].isEmpty() && !indexingMaps[1].isEmpty() &&
           ((indexingMaps[0].isIdentity() && !indexingMaps[1].isIdentity() &&
             indexingMaps[1].isPermutation()) ||
            (!indexingMaps[0].isIdentity() && indexingMaps[0].isPermutation() &&
             indexingMaps[1].isIdentity()));
  }
};

class X86ContractionOpPattern : public ContractionOpTileSizeSelectionPattern {
public:
  FailureOr<TileSizeAndPipelineConfig>
  matchAndConfig(func::FuncOp funcOp,
                 linalg::ContractionOpInterface rootOp) const override {
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!isX86(targetAttr)) {
      return failure();
    }
    if (!isInnermostReduction(rootOp)) {
      return failure();
    }

    auto linalgOp = cast<linalg::LinalgOp>(rootOp.getOperation());

    int64_t vectorSize = getMinVectorSize(funcOp, rootOp);
    SmallVector<int64_t> vecTileSizes = getVecTileSizes(linalgOp);
    SmallVector<int64_t> cacheTileSizes = getCacheTileSizes(linalgOp);
    SmallVector<int64_t> distTileSizes =
        getDistTileSizes(linalgOp, vecTileSizes, vectorSize);

    // TODO: We set cache tile sizes to the distribution sizes for now (no-op)
    // to make sure there are no performance changes. This will let us change
    // the distribution sizes while still preserving the cache behavior of the
    // original sizes. When we set proper sizes, we should call again
    // `getMatmulCacheTileSizesForShape(cacheTileSizes, distTileSizes);` here as
    // the `getDefaultDistributedLevelTileSizes` above may return sizes that are
    // smaller than `minTileSizes`, so we have to adjust the cache sizes again.
    cacheTileSizes = distTileSizes;

    TileSizeConfig tileSizeConfig(getNonScalableTileSizes(distTileSizes),
                                  getNonScalableTileSizes(cacheTileSizes),
                                  getNonScalableTileSizes(vecTileSizes));
    return TileSizeAndPipelineConfig{
        tileSizeConfig,
        DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert};
  }

private:
  const SmallVector<int64_t> kDefaultVecTileSizes = {8, 32, 16};
  const SmallVector<int64_t> kCacheMaxTileSizes = {8, 128, 16};
  const int64_t kDistMaxTileSize = 128;

  SmallVector<int64_t> getVecTileSizes(linalg::LinalgOp op) const {
    unsigned numLoops = op.getNumLoops();
    SmallVector<int64_t> staticShape = op.getStaticLoopRanges();
    SmallVector<int64_t> tileSizes = kDefaultVecTileSizes;
    // In default we only do masking/peeling on M and K dims.
    if (numLoops >= 3 && !ShapedType::isDynamic(staticShape[numLoops - 3])) {
      tileSizes[0] = std::min(tileSizes[0], staticShape[numLoops - 3]);
    }
    return copySuffixAndPrependArray<int64_t>(numLoops, tileSizes, 1);
  }

  // Compute cache-level tile sizes. Cache a dimension only if there are enough
  // iterations.
  SmallVector<int64_t> getCacheTileSizes(linalg::LinalgOp op) const {
    auto lhsShapedType = llvm::cast<ShapedType>(
        cast<linalg::ContractionOpInterface>(op.getOperation())
            .lhs()
            .getType());
    auto resShapedType =
        cast<ShapedType>(op.getDpsInitOperand(0)->get().getType());
    bool isQuantized =
        lhsShapedType.getElementType() != resShapedType.getElementType();
    unsigned numLoops = op.getNumLoops();
    // Cache-level tiling is only supported for 2-D non-quantized matmuls.
    if (numLoops < 3 || isQuantized) {
      return SmallVector<int64_t>(numLoops, 0);
    }
    SmallVector<int64_t> maxTileSizes(numLoops - 3, 1);
    maxTileSizes.append(kCacheMaxTileSizes);
    return getMatmulCacheTileSizesForShape(maxTileSizes,
                                           op.getStaticLoopRanges());
  }

  SmallVector<int64_t> getDistTileSizes(linalg::LinalgOp op,
                                        SmallVector<int64_t> vecTileSizes,
                                        int64_t vectorSize) const {
    DistributionHeuristicConfig distConfig;
    unsigned numLoops = op.getNumLoops();
    distConfig.maxTileSizes.resize(numLoops, kDistMaxTileSize);
    // It's inspired from https://github.com/iree-org/iree-llvm-sandbox repo.
    // Sandbox has [[288, 128, 512], [12, 32, 1]] setup. We scale 288 to 192
    // because 288/12*8=192
    if (numLoops == 3) {
      distConfig.maxTileSizes[0] = 192;
      distConfig.maxTileSizes[1] = 128;
    }
    // TODO: This should be cacheTileSizes?
    distConfig.minTileSizes = vecTileSizes;
    distConfig.allowIncompleteTile = true;
    distConfig.vectorSizeHints.resize(numLoops, vectorSize);
    if (isa<linalg::BatchMatmulOp>(op.getOperation())) {
      distConfig.maxTileSizes[0] = 1;
      distConfig.vectorSizeHints[0] = 1;
    }
    return getDefaultDistributedLevelTileSizes(op, distConfig);
  }
};

class AArch64ContractionOpPattern
    : public ContractionOpTileSizeSelectionPattern {
public:
  FailureOr<TileSizeAndPipelineConfig>
  matchAndConfig(func::FuncOp funcOp,
                 linalg::ContractionOpInterface rootOp) const override {
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!isAArch64(targetAttr)) {
      return failure();
    }
    if (!isInnermostReduction(rootOp)) {
      return failure();
    }

    auto linalgOp = cast<linalg::LinalgOp>(rootOp.getOperation());

    int64_t vectorSize = getMinVectorSize(funcOp, rootOp);
    SizesAndScalableFlags vecTileSizes = getVecTileSizes(linalgOp, targetAttr);
    SizesAndScalableFlags distTileSizes =
        getDistTileSizes(linalgOp, vecTileSizes, vectorSize);

    return TileSizeAndPipelineConfig{
        {distTileSizes, vecTileSizes},
        hasAnySVEFeature(targetAttr)
            ? DispatchLoweringPassPipeline::CPUDoubleTilingExpert
            : DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert};
  }

private:
  const SmallVector<int64_t> kDefaultVecTileSizes = {8, 16, 1};
  const int64_t kDistMaxTileSize = 128;

  SizesAndScalableFlags
  getVecTileSizes(linalg::LinalgOp op,
                  IREE::HAL::ExecutableTargetAttr targetAttr) const {
    SmallVector<int64_t> tileSizes;
    SmallVector<bool> scalableFlags;
    if (hasSMEFeature(targetAttr)) {
      // Note: This may not pick any sizes (which will fallback to the default
      // SVE) sizes below.
      getMatmulAArch64SMEVectorSizes(op, tileSizes, scalableFlags);
    }

    unsigned numLoops = op.getNumLoops();
    if (tileSizes.empty()) {
      // Use default hard-coded tile sizes if we couldn't compute anything
      // better.
      tileSizes = kDefaultVecTileSizes;

      SmallVector<int64_t> staticShape = op.getStaticLoopRanges();
      // In default we only do masking/peeling on M and K dims.
      if (numLoops >= 3 && !ShapedType::isDynamic(staticShape[numLoops - 3])) {
        tileSizes[0] = std::min(tileSizes[0], staticShape[numLoops - 3]);
      }

      // Specialisation for SVE.
      if (hasAnySVEFeature(targetAttr)) {
        // Mark middle dimensions as scalable, so sizes are (8, [16], 1).
        scalableFlags = {false, true, false};
      } else {
        scalableFlags = {false, false, false};
      }
    }

    tileSizes = copySuffixAndPrependArray<int64_t>(numLoops, tileSizes, 1);
    scalableFlags =
        copySuffixAndPrependArray<bool>(numLoops, scalableFlags, false);
    return {tileSizes, scalableFlags};
  }

  SizesAndScalableFlags getDistTileSizes(linalg::LinalgOp op,
                                         SizesAndScalableFlags vecTileSizes,
                                         int64_t vectorSize) const {
    DistributionHeuristicConfig distConfig;
    unsigned numLoops = op.getNumLoops();
    distConfig.maxTileSizes.resize(numLoops, kDistMaxTileSize);
    std::tie(distConfig.minTileSizes, std::ignore) = vecTileSizes;
    distConfig.allowIncompleteTile = true;
    distConfig.vectorSizeHints.resize(numLoops, vectorSize);
    if (isa<linalg::BatchMatmulOp>(op.getOperation())) {
      distConfig.maxTileSizes[0] = 1;
      distConfig.vectorSizeHints[0] = 1;
    }
    SmallVector<int64_t> tileSizes =
        getDefaultDistributedLevelTileSizes(op, distConfig);
    SmallVector<bool> scalableFlags(numLoops, false);
    return {tileSizes, scalableFlags};
  }
};

class PatternRegister : public TileSizeSelectionPatternRegister {
public:
  void populatePatterns(SmallVector<std::unique_ptr<TileSizeSelectionPattern>>
                            &patterns) const override {
    patterns.emplace_back(std::make_unique<X86TransposeLikeOpPattern>());
    patterns.emplace_back(std::make_unique<X86ContractionOpPattern>());
    patterns.emplace_back(std::make_unique<AArch64ContractionOpPattern>());
    return;
  }
};

class LLVMCPUTargetConfig
    : public PluginSession<LLVMCPUTargetConfig, EmptyPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {

  void populateTileSizeSelectionPatterns(
      TileSizeSelectionPatternList &list) override {
    list.registers.emplace_back(std::make_shared<PatternRegister>());
  }
};

} // namespace

} // namespace mlir::iree_compiler

extern "C" bool
iree_register_compiler_plugin_lowering_config_llvmcpu_target_config(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::LLVMCPUTargetConfig>(
      "lowering_config_llvmcpu_target_config");
  return true;
}
