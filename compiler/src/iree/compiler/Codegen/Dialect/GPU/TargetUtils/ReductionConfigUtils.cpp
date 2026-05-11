// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/SliceUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-gpu-reduction-config-utils"

// If the size of the reduction dimension is not a dispatch compile-time
// constant, choose a default size that the config should optimize for.
constexpr unsigned kVectorDistributeReductionSizeToTargetIfDynamic = (1 << 31);

namespace mlir::iree_compiler::IREE::GPU {

namespace {

bool isROCmBackend(IREE::GPU::TargetAttr target) {
  return target.getArch().starts_with("gfx");
}

static bool isMatmulLike(linalg::LinalgOp linalgOp) {
  return linalg::isaContractionOpInterface(linalgOp) &&
         linalgOp.getNumParallelLoops() >= 1;
};

static SmallVector<unsigned> getParallelDims(Operation *op) {
  SmallVector<unsigned> result;
  for (auto [idx, type] :
       llvm::enumerate(cast<TilingInterface>(op).getLoopIteratorTypes())) {
    if (type == utils::IteratorType::parallel) {
      result.push_back(idx);
    }
  }
  return result;
}

static SmallVector<unsigned> getReductionDims(Operation *op) {
  SmallVector<unsigned> result;
  for (auto [idx, type] :
       llvm::enumerate(cast<TilingInterface>(op).getLoopIteratorTypes())) {
    if (type == utils::IteratorType::reduction) {
      result.push_back(idx);
    }
  }
  return result;
}

static unsigned getNumLoops(Operation *op) {
  return cast<TilingInterface>(op).getLoopIteratorTypes().size();
}

static bool hasReductionIterator(Operation *op) {
  auto tilingOp = dyn_cast<TilingInterface>(op);
  if (!tilingOp) {
    return false;
  }
  return llvm::any_of(tilingOp.getLoopIteratorTypes(), [](auto type) {
    return type == utils::IteratorType::reduction;
  });
}

struct TensorSizeEstimate {
  int64_t elementBitwidth;
  int64_t staticSize;
  int64_t numDynamicDims;
  Value value; // This field is to simplify debugging / printing, not for
               // calculation.
};

/// Calculates the estimated tensor value size. Looks through known bit
/// extension ops.
static FailureOr<TensorSizeEstimate> calculateTensorSize(Value val) {
  if (auto genericOp = val.getDefiningOp<linalg::GenericOp>()) {
    if (IREE::LinalgExt::isBitExtendOp(genericOp) &&
        genericOp.getNumDpsInputs() > 0) {
      val = genericOp.getDpsInputs()[0];
    }
  }
  auto tensorType = dyn_cast<RankedTensorType>(val.getType());
  if (!tensorType) {
    return failure();
  }

  auto elemType = getElementTypeOrSelf(tensorType);
  if (!elemType.isIntOrIndexOrFloat()) {
    return failure();
  }

  if (isa<IndexType>(elemType)) {
    // On LLVMGPU, we can conservatively assume 64-bit index types.
    elemType = IntegerType::get(val.getContext(), 64);
  }

  TensorSizeEstimate result = {};
  result.elementBitwidth = elemType.getIntOrFloatBitWidth();
  result.numDynamicDims = tensorType.getNumDynamicDims();
  result.staticSize = 1;
  result.value = val;
  for (int64_t dim :
       llvm::make_filter_range(tensorType.getShape(), ShapedType::isStatic)) {
    result.staticSize *= dim;
  }

  return result;
}

/// Returns the bitwidth of the largest operand or init. We estimate this based
/// on the known static size and the number of dynamic dimensions. This is meant
/// to cater towards the memory bandwidth required to load the largest of the
/// inputs.
static FailureOr<int64_t> getBitWidth(Operation *op) {
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  SmallVector<FailureOr<TensorSizeEstimate>> inputSizes =
      llvm::filter_to_vector(
          llvm::map_range(
              llvm::concat<Value>(dpsOp.getDpsInits(), dpsOp.getDpsInputs()),
              calculateTensorSize),
          succeeded);
  if (inputSizes.empty()) {
    return failure();
  }

  TensorSizeEstimate largestInput = *inputSizes.front();
  for (FailureOr<TensorSizeEstimate> candidateSize :
       llvm::drop_begin(inputSizes)) {
    if (std::tie(candidateSize->numDynamicDims, candidateSize->staticSize,
                 candidateSize->elementBitwidth) >
        std::tie(largestInput.numDynamicDims, largestInput.staticSize,
                 largestInput.elementBitwidth)) {
      largestInput = *candidateSize;
    }
  }

  LDBG() << "Largest input: " << largestInput.value;
  LDBG() << "Largest input bitwidth: " << largestInput.elementBitwidth;
  return largestInput.elementBitwidth;
}

/// Check if the reduction op has a single combiner operation.
static LogicalResult checkSingleCombiner(linalg::LinalgOp op) {
  bool foundSingleReductionOutput = false;
  for (auto [index, initOpOperand] : llvm::enumerate(op.getDpsInitsMutable())) {
    // Only single combiner operations are supported for now.
    SmallVector<Operation *> combinerOps;
    if (matchReduction(op.getRegionOutputArgs(), index, combinerOps) &&
        combinerOps.size() == 1) {
      if (foundSingleReductionOutput) {
        return failure();
      }
      foundSingleReductionOutput = true;
      continue;
    }
    if (!op.getMatchingIndexingMap(&initOpOperand).isIdentity()) {
      return failure();
    }
  }
  if (!foundSingleReductionOutput) {
    return failure();
  }

  return success();
}

/// Storage for tracking tile size configuration. Workgroup tile sizes are
/// tracked separate. "tileSizes" materializes as serial level for parallel
/// iterators and reduction level for reduction iterators.
struct TileSizesConfig {
  SmallVector<int64_t> tileSizes;
  SmallVector<int64_t> threadTileSizes;
  SmallVector<int64_t> threadCounts;
  SmallVector<int64_t> subgroupCounts;
  bool isReduction = false;

  LoweringConfigAttr
  buildConfig(MLIRContext *context,
              std::optional<ArrayRef<int64_t>> workgroupTileSizes) const {
    Builder b(context);
    auto mapping = llvm::to_vector(llvm::seq<int64_t>(0, tileSizes.size()));

    ArrayAttr subgroupBasisAttr = b.getArrayAttr(
        {b.getI64ArrayAttr(subgroupCounts), b.getI64ArrayAttr(mapping)});
    ArrayAttr laneBasisAttr = b.getArrayAttr(
        {b.getI64ArrayAttr(threadCounts), b.getI64ArrayAttr(mapping)});

    SmallVector<NamedAttribute> configAttrs;
    if (workgroupTileSizes) {
      configAttrs.push_back(
          b.getNamedAttr("workgroup", b.getI64ArrayAttr(*workgroupTileSizes)));
    }
    configAttrs.push_back(
        b.getNamedAttr(isReduction ? "partial_reduction" : "serial",
                       b.getI64ArrayAttr(tileSizes)));
    configAttrs.push_back(
        b.getNamedAttr("thread", b.getI64ArrayAttr(threadTileSizes)));
    configAttrs.push_back(b.getNamedAttr("lane_basis", laneBasisAttr));
    configAttrs.push_back(b.getNamedAttr("subgroup_basis", subgroupBasisAttr));

    auto configDict = b.getDictionaryAttr(configAttrs);
    return IREE::GPU::LoweringConfigAttr::get(context, configDict);
  }
};

/// `allowMaskedTail` says the lowering can mask off positions where the
/// per-thread tile overshoots the iteration dim, so the per-op tile-size
/// computation does not need to halve threadLoads or shrink to a divisor
/// via GCD. ArgCompare passes false; ordinary linalg reductions pass true.
static FailureOr<TileSizesConfig>
getVectorDistributeReductionConfig(Operation *op, IREE::GPU::TargetAttr target,
                                   SmallVector<int64_t> &localWgpTiles,
                                   int64_t workgroupSize, int64_t subgroupSize,
                                   int64_t threadLoads, bool allowMaskedTail) {
  SmallVector<unsigned> parallelDims = getParallelDims(op);
  SmallVector<unsigned> reductionDims = getReductionDims(op);

  SmallVector<int64_t> bounds =
      cast<IndexingMapOpInterface>(op).getStaticLoopRanges();
  unsigned numLoops = getNumLoops(op);

  SmallVector<int64_t> threadTileSizes(numLoops, 0);
  SmallVector<int64_t> threadCounts(numLoops, 1);
  SmallVector<int64_t> subgroupCounts(numLoops, 1);

  // Set the configuration for the operation with no reduction dims.
  // The workgroup tile sizes are set by the reduction operation.
  if (reductionDims.empty()) {
    SmallVector<int64_t> serialTileSizes(numLoops, 1);

    // For dimensions that are being tiled on workgroups, do not tile them on
    // the serial level.
    for (auto [serialTile, wgpTile] :
         llvm::zip_equal(serialTileSizes, localWgpTiles)) {
      if (wgpTile != 0) {
        serialTile = 0;
      }
    }

    // Find the innermost dim not tiled on workgroups - this is the dim
    // corresponding to a reduction group globally.
    unsigned distributeDim = parallelDims.back();
    for (int i = numLoops - 1; i >= 0; i--) {
      if (localWgpTiles[i] == 0) {
        distributeDim = i;
        break;
      }
    }

    int64_t parallelSize = bounds[distributeDim];
    if (ShapedType::isDynamic(parallelSize)) {
      parallelSize = kVectorDistributeReductionSizeToTargetIfDynamic;
    }
    // Adjust threadLoads for this op's distribution dim. With masking, an
    // overshooting tile is fine - the lowering masks off the tail. Without
    // masking, we have to halve threadLoads until it cleanly tiles.
    if (!allowMaskedTail) {
      while (threadLoads > 1 && parallelSize % threadLoads != 0) {
        threadLoads /= 2;
      }
    }

    int64_t lastDimReductionTileSize = workgroupSize * threadLoads;
    // Setting subgroupBasis to minimum i.e., 1 and threadBasis
    // to maximum i.e., subgroupSize.
    int64_t subgroupBasis = 1;
    int64_t threadBasis = subgroupSize;

    // Without masking, shrink the tile to a divisor of parallelSize so the
    // per-thread tile actually tiles cleanly. With masking, keep the full
    // tile (workgroupSize * threadLoads) so every thread participates - the
    // lowering masks any threads whose positions overshoot parallelSize.
    if (!allowMaskedTail) {
      lastDimReductionTileSize =
          llvm::APIntOps::GreatestCommonDivisor(
              {64, static_cast<uint64_t>(parallelSize)},
              {64, static_cast<uint64_t>(lastDimReductionTileSize)})
              .getZExtValue();
    }

    bool allDimsCovered =
        llvm::none_of(localWgpTiles, [](int64_t t) { return t == 0; });
    if (!allDimsCovered) {
      int subgroupStride = threadBasis * threadLoads;
      while (lastDimReductionTileSize % subgroupStride != 0) {
        threadBasis >>= 1;
        subgroupStride = threadBasis * threadLoads;
      }
      int subgroup = lastDimReductionTileSize / subgroupStride;
      subgroupBasis = (subgroup == 0) ? 1 : subgroup;
    }

    // Since all the dimensions are contained within the shared parallel
    // dimension, set the tile sizes to 1.
    if (allDimsCovered) {
      lastDimReductionTileSize = 1;
      threadLoads = 1;
      threadBasis = 1;
      subgroupBasis = 1;
    }

    // Note: with allowMaskedTail, lastDimReductionTileSize can exceed
    // bounds[distributeDim] (e.g. 128 > 3 for a small channel dim). The
    // LLVMGPUVectorDistribute pass masks the per-lane reads that fall past
    // the actual bound.
    serialTileSizes[distributeDim] = lastDimReductionTileSize;
    threadTileSizes[distributeDim] = threadLoads;
    threadCounts[distributeDim] = threadBasis;
    subgroupCounts[distributeDim] = subgroupBasis;

    return TileSizesConfig{serialTileSizes, threadTileSizes, threadCounts,
                           subgroupCounts, /*isReduction=*/false};
  }

  // Setting the config for operation with atleast one reduction dimension.
  SmallVector<int64_t> partialReductionTileSizes(numLoops, 0);
  int64_t lastReductionDim = reductionDims.back();

  // TODO: This is enabled for matvec on ROCm for now. We should
  // validate this strategy and extend to more linalg generics and to CUDA.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (linalgOp && isROCmBackend(target) && ShapedType::isStaticShape(bounds) &&
      isMatmulLike(linalgOp)) {
    int64_t parallelIdx = *llvm::find_if(
        parallelDims, [&](int64_t currIdx) { return bounds[currIdx] != 1; });
    int64_t parallelBound = bounds[parallelIdx];
    int64_t numParallelReductions = 1;
    const int64_t maxParallelFactor = workgroupSize / 4;
    for (int64_t parallelFactor = 2; (parallelFactor < maxParallelFactor) &&
                                     (parallelBound % parallelFactor == 0) &&
                                     (parallelBound >= parallelFactor);
         parallelFactor *= 2) {
      numParallelReductions = parallelFactor;
    }
    localWgpTiles[parallelIdx] = numParallelReductions;
  }

  for (int64_t dim : reductionDims) {
    threadTileSizes[dim] = 1;
    partialReductionTileSizes[dim] = 1;
  }

  int64_t lastReductionDimSize = bounds[reductionDims.back()];

  if (ShapedType::isDynamic(lastReductionDimSize)) {
    lastReductionDimSize = kVectorDistributeReductionSizeToTargetIfDynamic;
  }
  // Adjust threadLoads for this op's reduction dim. See the parallel-only
  // branch above for the masking rationale.
  if (!allowMaskedTail) {
    while (threadLoads > 1 && lastReductionDimSize % threadLoads != 0) {
      threadLoads /= 2;
    }
  }

  int64_t partialReductionSize = workgroupSize * threadLoads;
  // Without masking, shrink to a divisor of lastReductionDimSize so the
  // tile lines up exactly with the data and no tail iterations exist.
  // With masking, keep the full `workgroup * threadLoads` tile: this
  // preserves the invariant that `lane_basis = subgroupSize` and
  // `subgroup_basis = workgroupSize / subgroupSize` divide cleanly.
  // Tail positions beyond `lastReductionDimSize` are padded with the
  // reduction's combiner identity by the materialize-vector-masking
  // infrastructure (PRs #23679, #24044, #24187).
  if (!allowMaskedTail) {
    partialReductionSize =
        llvm::APIntOps::GreatestCommonDivisor(
            {64, static_cast<uint64_t>(partialReductionSize)},
            {64, static_cast<uint64_t>(lastReductionDimSize)})
            .getZExtValue();
  }

  int64_t threadBasis = subgroupSize;
  int subgroupStride = threadBasis * threadLoads;
  while (partialReductionSize % subgroupStride != 0) {
    threadBasis >>= 1;
    subgroupStride = threadBasis * threadLoads;
  }
  int subgroup = partialReductionSize / subgroupStride;
  int64_t subgroupBasis = (subgroup == 0) ? 1 : subgroup;

  // Note: with allowMaskedTail, partialReductionSize can exceed
  // lastReductionDimSize (e.g. 512 > 257). The LLVMGPUVectorDistribute pass
  // masks the per-lane reads that fall past lastReductionDimSize.
  partialReductionTileSizes[lastReductionDim] = partialReductionSize;
  threadTileSizes[lastReductionDim] = threadLoads;
  threadCounts[lastReductionDim] = threadBasis;
  subgroupCounts[lastReductionDim] = subgroupBasis;

  return TileSizesConfig{partialReductionTileSizes, threadTileSizes,
                         threadCounts, subgroupCounts,
                         /*isReduction=*/true};
}

/// Populate lowering configurations based on the following assumptions:
///
/// 1. All parallel dimensions in the root operation are parallel across the
///   entire dispatch. Any other parallel dimensions are sequential
///   dimensions. If there was a case where a parallel dimension on the root
///   op would be sequential, i.e. it was sequential in some other operation,
///   then the root operation shouldn't have been the root operation at all.
///
/// 2. All inputs/outputs of the dispatch have a "good" memory layout that has
///   sequential dimensions as the innermost dimensions. This is a "good" memory
///   layout because distributing operations on workgroups does not cause
///   strided accesses per-workgroup. NOTE: This assumption is true when we can
///   get the "good" layout by fusing the memory layout transformations into the
///   previous dispatch, but is not true at boundaries. Generally the boundaries
///   are okay to be less optimized, but when used as a single operator compiler
///   (e.g. Fusilli) this is suboptimal and needs to be improved.
///   TODO: Remove this assumption.
static LogicalResult populateConfigInfo(
    Operation *root, const llvm::SetVector<Operation *> &computeOps,
    IREE::GPU::TargetAttr target, int64_t workgroupSize, int64_t subgroupSize,
    int64_t threadLoads, bool allowMaskedTail) {
  if (computeOps.empty()) {
    return failure();
  }

  // We define shared parallel dimensions as the parallel dimensions of the root
  // op. These dimensions are parallel across the entire dispatch (See
  // Assumption 1).
  //
  // Build a tracker to track shared dimensions across the compute ops.
  SmallVector<Operation *> ops(computeOps.begin(), computeOps.end());
  IterationDimTracker tracker(ops);
  // Tracker for minimum tile size required for shared parallel dimensions.
  DenseMap<int64_t, int64_t> wgTilePerDim;
  unsigned rootNumLoops = getNumLoops(root);
  auto rootIterTypes = cast<TilingInterface>(root).getLoopIteratorTypes();
  for (auto dim : llvm::seq<int64_t>(0, rootNumLoops)) {
    if (rootIterTypes[dim] == utils::IteratorType::parallel) {
      int64_t globalDim = tracker.getGlobalDimIdx(root, dim);
      wgTilePerDim[globalDim] = 1;
    }
  }

  // An operation needs a lowering config in 2 cases:
  //   - It has some dims which are not part of shared workgroup dims (we only
  //     distribute threads on non shared dims).
  //   - It cannot infer its config from its consumer compute ops.
  auto needsLoweringConfig = [&](Operation *computeOp) -> bool {
    // Track which dims we have enough information about.
    unsigned opNumLoops = getNumLoops(computeOp);
    llvm::SmallBitVector coveredDims(opNumLoops);

    // We have config information about shared parallel dims. We don't
    // distribute on them.
    //
    // Note: This is only true because of Assumption 2, when that assumption
    // is dropped, the root op needs to set some thread config on the root op
    // for shared parallel dims. Only the root op needs to do this.
    for (unsigned dim = 0; dim < opNumLoops; dim++) {
      int64_t gd = tracker.getGlobalDimIdx(computeOp, dim);
      if (wgTilePerDim.contains(gd)) {
        coveredDims.set(dim);
      }
    }

    // Since compute ops are iterated in reverse topological order, the
    // consumer compute ops already have config information. We can assume
    // that they can be used to infer the config of the current op.
    //
    // Cover dimensions that can be inferred by consumer compute ops.
    auto dpsOp = cast<DestinationStyleOpInterface>(computeOp);
    auto indexingMapOp = cast<IndexingMapOpInterface>(computeOp);
    for (OpOperand &output : dpsOp.getDpsInitsMutable()) {
      OpResult result = dpsOp.getTiedOpResult(&output);
      bool hasComputeOpUser =
          llvm::any_of(result.getUsers(), [&](Operation *user) {
            return computeOps.contains(user);
          });
      if (!hasComputeOpUser) {
        continue;
      }

      AffineMap outputMap = indexingMapOp.getMatchingIndexingMap(&output);
      for (unsigned i = 0; i < outputMap.getNumResults(); ++i) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(outputMap.getResult(i))) {
          coveredDims.set(dimExpr.getPosition());
        }
      }
    }

    // If any dimension is not covered, then we need a lowering config for
    // this op.
    return !coveredDims.all();
  };

  // Compute tile sizes for ops that need a lowering config.
  SmallVector<std::pair<Operation *, TileSizesConfig>, 4> tileSizeConfigs;
  for (Operation *computeOp : computeOps) {
    if (!needsLoweringConfig(computeOp)) {
      continue;
    }

    // Build localWgpTiles from the tracker.
    unsigned opNumLoops = getNumLoops(computeOp);
    SmallVector<int64_t> localWgpTiles(opNumLoops, 0);
    for (unsigned iterDim = 0; iterDim < opNumLoops; iterDim++) {
      int64_t gd = tracker.getGlobalDimIdx(computeOp, iterDim);
      if (wgTilePerDim.contains(gd)) {
        localWgpTiles[iterDim] = wgTilePerDim[gd];
      }
    }

    auto config = getVectorDistributeReductionConfig(
        computeOp, target, localWgpTiles, workgroupSize, subgroupSize,
        threadLoads, allowMaskedTail);
    if (failed(config)) {
      return failure();
    }

    // Propagate any WG tile updates (e.g., from local split-k) back to
    // wgTilePerDim.
    for (auto [dim, tile] : llvm::enumerate(localWgpTiles)) {
      int64_t gd = tracker.getGlobalDimIdx(computeOp, dim);
      if (wgTilePerDim.contains(gd) && tile > wgTilePerDim[gd]) {
        wgTilePerDim[gd] = std::lcm(wgTilePerDim[gd], tile);
      }
    }
    tileSizeConfigs.push_back({computeOp, *config});
  }

  // Build and set lowering configurations. Only the root operation in the
  // dispatch gets the workgroup tile sizes.
  assert(tileSizeConfigs.size() >= 1 &&
         "There should be at least one op that needs lowering config");

  MLIRContext *context = root->getContext();
  SmallVector<int64_t> workgroupTileSizes(rootNumLoops, 0);
  for (auto dim : llvm::seq<int64_t>(0, rootNumLoops)) {
    int64_t gd = tracker.getGlobalDimIdx(root, dim);
    if (wgTilePerDim.contains(gd)) {
      workgroupTileSizes[dim] = wgTilePerDim[gd];
    }
  }

  for (auto &[computeOp, config] : tileSizeConfigs) {
    if (computeOp == root) {
      setLoweringConfig(computeOp,
                        config.buildConfig(context, workgroupTileSizes));
      continue;
    }
    setLoweringConfig(computeOp, config.buildConfig(context, std::nullopt));
  }
  return success();
}

/// Check if the dispatch has a single store operation.
/// If the dispatch meets the criterion, it returns the set of
/// compute ops.
template <typename... StoreOpTy>
static FailureOr<SetVector<Operation *>>
checkDispatchForVectorDistribution(Operation *parentOp) {
  SmallVector<Operation *> storeOps;

  parentOp->walk([&](Operation *op) {
    if (isa<StoreOpTy...>(op)) {
      storeOps.push_back(op);
    }
  });

  if (storeOps.empty()) {
    return failure();
  }

  BackwardSliceOptions sliceOptions;
  sliceOptions.inclusive = false;
  sliceOptions.omitBlockArguments = true;
  sliceOptions.omitUsesFromAbove = false;
  SetVector<Operation *> slice;

  for (Operation *op : storeOps) {
    [[maybe_unused]] LogicalResult result =
        getBackwardSlice(op, &slice, sliceOptions);
    assert(result.succeeded());
  }

  SetVector<Operation *> computeOps;
  // Check if the op contains an scf.forall. This could be generalized, but
  // for now only check for split-reduction generated scf.forall.
  std::optional<scf::ForallOp> forallOp;
  bool foundComputeOp = false;
  for (Operation *op : slice) {
    if (isa<scf::ForallOp>(op)) {
      if (forallOp) {
        return failure();
      }
      forallOp = cast<scf::ForallOp>(op);
      continue;
    }
    if (isa<IndexingMapOpInterface>(op)) {
      foundComputeOp = true;
    }
  }
  if (forallOp) {
    std::optional<ArrayAttr> mapping = forallOp->getMapping();
    if (!mapping) {
      return failure();
    }
    if (failed(IREE::LinalgExt::SplitReductionMappingAttr::verifyAttrList(
            forallOp->getContext(), forallOp->getLoc(), mapping->getValue(),
            /*emitDiagnosticErrors =*/false))) {
      return failure();
    }
    if (foundComputeOp) {
      return failure();
    }
    return checkDispatchForVectorDistribution<tensor::ParallelInsertSliceOp>(
        forallOp.value());
  }

  bool containsValidReductionOp = true;
  for (Operation *op : llvm::reverse(slice)) {
    if (isa<linalg::FillOp>(op)) {
      continue;
    }
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    bool isIndexingMapOp = isa<IndexingMapOpInterface>(op) &&
                           isa<TilingInterface>(op) &&
                           isa<DestinationStyleOpInterface>(op);
    if (!linalgOp && !isIndexingMapOp) {
      continue;
    }
    // Pooling operations are currently not supported properly. The pipeline
    // can support them, it's just not tested properly.
    // TODO: Check if we can support pooling operations.
    if (linalgOp && linalg::isaConvolutionOpInterface(linalgOp)) {
      return failure();
    }
    if (hasReductionIterator(op)) {
      if (linalgOp && failed(checkSingleCombiner(linalgOp))) {
        containsValidReductionOp = false;
        break;
      }
    }
    computeOps.insert(op);
  }

  // Return failure if the dispatch contains no reduction op.
  if (!containsValidReductionOp) {
    return failure();
  }

  // Get the reduction dimensions.
  auto getOpReductionDims = [](Operation *op) -> SetVector<int64_t> {
    SetVector<int64_t> dims;
    for (auto [idx, type] :
         llvm::enumerate(cast<TilingInterface>(op).getLoopIteratorTypes())) {
      if (type == utils::IteratorType::reduction) {
        dims.insert(idx);
      }
    }
    return dims;
  };

  for (Operation *computeOp : computeOps) {
    auto dpsOp = cast<DestinationStyleOpInterface>(computeOp);
    auto indexingMapOp = cast<IndexingMapOpInterface>(computeOp);
    SmallVector<AffineMap> maps = indexingMapOp.getIndexingMapsArray();
    for (OpOperand *operand : dpsOp.getDpsInputOperands()) {
      // Skip operands without indexing maps (e.g., ArgCompareOp's index_base).
      if (operand->getOperandNumber() >= maps.size()) {
        continue;
      }
      AffineMap indexingMap = maps[operand->getOperandNumber()];

      auto opResult = dyn_cast<OpResult>(operand->get());
      if (!opResult) {
        continue;
      }

      Operation *producer = opResult.getOwner();
      if (!computeOps.contains(producer)) {
        continue;
      }

      // Check whether the op operand is not reduced and producer of that
      // operand is not a reduction op.
      auto reductionDims = getOpReductionDims(computeOp);
      bool isOperandReduced = llvm::any_of(
          llvm::seq<int>(indexingMap.getNumResults()), [&](int val) {
            return reductionDims.contains(indexingMap.getDimPosition(val));
          });
      if (isOperandReduced && hasReductionIterator(producer)) {
        return failure();
      }
    }
  }
  return computeOps;
}

/// Pick the subgroup size for an ArgCompare reduction. ArgCompare's
/// lowering does not yet support tail masking, so the reduction dim must
/// divide cleanly across the subgroup. Returns `std::nullopt` when no
/// available subgroup-size choice divides a static `reductionSize`.
static std::optional<int64_t>
pickArgCompareSubgroupSize(IREE::GPU::TargetWgpAttr wgp, int64_t reductionSize,
                           bool hasDynamicReductionDim) {
  ArrayRef<int> choices = wgp.getSubgroupSizeChoices().asArrayRef();
  const auto *it = llvm::find_if(choices, [&](int s) {
    return reductionSize % s == 0 || hasDynamicReductionDim;
  });
  if (it == choices.end()) {
    return std::nullopt;
  }
  return *it;
}

/// Pick `threadLoads` (number of elements each thread loads from the
/// reduction dim). ArgCompare halves until perfect divisibility; linalg
/// only halves until each workgroup is guaranteed to have at least one
/// full subgroup of work, since masking handles non-divisible tails.
///
/// Preconditions: `initialThreadLoads >= 1` and `subgroupSize >= 1`.
/// Postcondition: returned value is `>= 1`. Both branches use a
/// `threadLoads > 1` guard, so the loops always terminate regardless of
/// the divisibility relationship between `reductionSize` and
/// `subgroupSize`.
static unsigned pickThreadLoads(unsigned initialThreadLoads,
                                int64_t reductionSize, int64_t subgroupSize,
                                bool hasDynamicReductionDim,
                                bool isArgCompare) {
  unsigned threadLoads = initialThreadLoads;
  if (hasDynamicReductionDim) {
    return threadLoads;
  }
  if (isArgCompare) {
    while (threadLoads > 1 &&
           (reductionSize / threadLoads) % subgroupSize != 0) {
      threadLoads /= 2;
    }
    return threadLoads;
  }
  while (threadLoads > 1 && static_cast<int64_t>(llvm::divideCeil(
                                reductionSize, threadLoads)) < subgroupSize) {
    threadLoads /= 2;
  }
  return threadLoads;
}

} // namespace

/// The `setReductionConfig` attaches `lowering_config` to
/// multiple operations within a dispatch containing at least a single reduction
/// operation. It's divided into two parts:
/// 1. `checkDispatchForVectorDistribution` checks that the dispatch is
/// compatible with the vector distribution pipeline. If it's compatible, then
/// it returns a set of linalg operations to which `lowering_config` might be
/// attached.
/// 2. `populateConfigInfo` determines to which linalg operations it might
/// attach `lowering_config`. Currently, it attaches `lowering_config` to
/// reduction operations and parallel operations that have new dimensions or
/// non-identity output indexing maps (e.g., transposed outputs).
///   a. `getVectorDistributeReductionConfig` determines the `lowering_config`
///   for the reduction as well as parallel operations with new dimensions or
///   non-identity outputs.

/// The workgroup, subgroup, and threadTileSizes are determined by the
/// `setReductionConfig` operation, which are global
/// information that is used by `populateConfigInfo` while determining the
/// `lowering_config`.

/// TODO (pashu123):
/// The threadTileSizes should be determined per operation rather than passed as
/// global information. This is due to the current limitation of the vector
/// distribution pipeline, which demands that the `vector.transfer_read` with
/// multiple users have the same layout.
/// TODO (pashu123):
/// The workgroup and subgroup sizes are determined by the single operation
/// within the dispatch. Extend it to analyze the dispatch and determine the
/// workgroup and subgroup sizes.
LogicalResult setReductionConfig(IREE::GPU::TargetAttr target,
                                 mlir::FunctionOpInterface entryPoint,
                                 Operation *op) {
  MLIRContext *context = op->getContext();

  if (!isa<linalg::LinalgOp>(op) && !isa<IREE::LinalgExt::ArgCompareOp>(op)) {
    return failure();
  }

  if (!hasReductionIterator(op)) {
    return failure();
  }

  // Use strict alignment (no inner-dim fallback for non-divisible cases,
  // strict divisibility-based subgroup selection, and strict threadLoads
  // halving) when:
  //   - The op is ArgCompare (its lowering does not yet support tail
  //     masking).
  //   - The op is a matmul-like contraction (uses the file-level
  //     `isMatmulLike` helper). Matmul / batch-matmul / matvec have their
  //     own dedicated selectors (`setMatmulLoweringConfig`,
  //     `setContractConfig`) that run before us and use MMA instructions.
  //     If those selectors decline a particular shape, we want the
  //     dispatch to keep falling through to TileAndFuse, not get captured
  //     here and routed through the generic VectorDistribute reduction
  //     path which would lose MMA on real workloads.
  const bool isArgCompare = isa<IREE::LinalgExt::ArgCompareOp>(op);
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  const bool useStrictAlignment =
      isArgCompare || (linalgOp && isMatmulLike(linalgOp));

  FailureOr<SetVector<Operation *>> computeOps =
      checkDispatchForVectorDistribution<IREE::TensorExt::DispatchTensorStoreOp,
                                         IREE::Codegen::StoreToBufferOp>(
          entryPoint);

  if (failed(computeOps)) {
    return failure();
  }

  // The VectorDistribute lowering pipeline cannot emit `vector.store` /
  // `vector.maskedstore` to sub-byte (< 8 bit) memrefs - the EmulateBitWidth
  // pass marks those ops illegal. Bail so the dispatch falls through to a
  // pipeline that handles sub-byte stores.
  for (Operation *computeOp : *computeOps) {
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(computeOp);
    if (!dpsOp) {
      continue;
    }
    for (OpOperand &init : dpsOp.getDpsInitsMutable()) {
      Type elemType = getElementTypeOrSelf(init.get().getType());
      if (elemType.isIntOrFloat() && elemType.getIntOrFloatBitWidth() < 8) {
        return failure();
      }
    }
  }

  SmallVector<unsigned> parallelDims = getParallelDims(op);
  SmallVector<unsigned> reductionDims = getReductionDims(op);

  SmallVector<int64_t> bounds =
      cast<IndexingMapOpInterface>(op).getStaticLoopRanges();
  IREE::GPU::TargetWgpAttr wgp = target.getWgp();

  // Reduce-all-dims operations with a non-aligned inner reduction dim
  // (i.e. no parallel iterators *and* the inner dim is not a multiple of
  // the preferred subgroup size) hit a VectorDistribute lowering gap.
  // After vectorization, the reduction shows up as a vector op of the
  // full inner dim size - e.g., torch.aten.max over `tensor<1x100xf32>`
  // First consider the inner reduction dimension. Fall back to the product
  // of all reduction dims when:
  //   - For ArgCompare (no masking support): the inner dim is not a multiple
  //     of the preferred subgroup size.
  //   - For linalg reductions: the inner dim is smaller than one subgroup,
  //     so a single subgroup couldn't cover even one row (e.g. scaled matmul
  //     with block size 32 on gfx950).
  int64_t reductionSize = bounds[reductionDims.back()];
  const int64_t preferredSubgroupSize = target.getPreferredSubgroupSize();
  const bool needsInnerDimFallback =
      ShapedType::isStatic(reductionSize) &&
      (useStrictAlignment ? (reductionSize % preferredSubgroupSize != 0)
                          : (reductionSize < preferredSubgroupSize));
  if (needsInnerDimFallback) {
    reductionSize = 1;
    for (unsigned dim : reductionDims) {
      if (ShapedType::isDynamic(bounds[dim])) {
        reductionSize = ShapedType::kDynamic;
        break;
      }
      reductionSize *= bounds[dim];
    }
  }

  // Reduce-all-dims (no parallel iterator) with a non-aligned inner
  // dim, e.g. `torch.aten.max` over `tensor<1x100xf32>` -> scalar. The
  // dispatch contains fused ops (e.g. from torch.aten.index) producing
  // vector<NxT> patterns that the distribute pass cannot lower
  // alongside the reduction's distribution layout. Bail so the dispatch
  // falls through. ArgCompare is unaffected (strict path above). Tiny
  // reductions with `reductionSize <= subgroupSize` are handled via the
  // tile-clamp in `getVectorDistributeReductionConfig` below.
  if (!useStrictAlignment && parallelDims.empty()) {
    int64_t innerRedDim = bounds[reductionDims.back()];
    if (ShapedType::isStatic(innerRedDim) &&
        innerRedDim % preferredSubgroupSize != 0) {
      return failure();
    }
  }

  if (ShapedType::isDynamic(reductionSize)) {
    reductionSize = kVectorDistributeReductionSizeToTargetIfDynamic;
  }

  bool hasDynamicReductionDim = false;
  for (unsigned dim : reductionDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      hasDynamicReductionDim = true;
    }
  }

  std::optional<int64_t> pickedSubgroupSize =
      useStrictAlignment
          ? pickArgCompareSubgroupSize(wgp, reductionSize,
                                       hasDynamicReductionDim)
          : std::optional<int64_t>{target.getPreferredSubgroupSize()};
  if (!pickedSubgroupSize) {
    return failure();
  }
  int64_t subgroupSize = *pickedSubgroupSize;

  FailureOr<int64_t> bitWidth = getBitWidth(op);
  if (failed(bitWidth)) {
    return failure();
  }

  // Reduction distribution only supports 4/8/16/32 bit types now.
  if (!llvm::is_contained({4, 8, 16, 32}, *bitWidth)) {
    return failure();
  }

  const std::optional<int64_t> maxLoadBits = wgp.getMaxLoadInstructionBits();
  const unsigned largestLoadSizeInBits = maxLoadBits.value_or(128);

  unsigned threadLoads =
      pickThreadLoads(largestLoadSizeInBits / *bitWidth, reductionSize,
                      subgroupSize, hasDynamicReductionDim, useStrictAlignment);

  std::optional<int64_t> parallelSize = 1;
  for (int64_t dim : parallelDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      parallelSize = std::nullopt;
      break;
    }
    *parallelSize *= bounds[dim];
  }

  // Deduce the workgroup size we should use for reduction. Currently a
  // workgroup processes all elements in reduction dimensions. Need to make sure
  // the workgroup size we use can divide the total reduction size, and it's
  // also within hardware limitations.
  ArrayRef<int32_t> maxWgSizes = wgp.getMaxWorkgroupSizes();
  int32_t maxWorkgroupSize = *llvm::max_element(maxWgSizes);
  IREE::GPU::TargetChipAttr chip = target.getChip();
  const int numWGPs = chip ? chip.getWgpCount() : 512;
  const int numSIMDs = wgp.getSimdsPerWgp().value_or(4);

  // If there is more than enough work to saturate all WGPs, use single subgroup
  // per workgroup.
  // TODO: Similarly decide on the local split k factor based on total number of
  // SIMDs.
  if (parallelSize && *parallelSize > numWGPs * numSIMDs) {
    maxWorkgroupSize = target.getPreferredSubgroupSize();
  }

  int64_t workgroupSize = llvm::divideCeil(reductionSize, threadLoads);
  // Round up to the next multiple of subgroupSize so every workgroup is a
  // whole number of subgroups. Tail threads will mask their loads when
  // reductionSize is not a multiple of (workgroupSize * threadLoads).
  workgroupSize = llvm::divideCeil(workgroupSize, subgroupSize) * subgroupSize;
  if (workgroupSize > maxWorkgroupSize) {
    workgroupSize = llvm::APIntOps::GreatestCommonDivisor(
                        {64, static_cast<uint64_t>(workgroupSize)},
                        {64, static_cast<uint64_t>(maxWorkgroupSize)})
                        .getZExtValue();
  }

  // Total parallel size that can fill the GPU with enough workgroups.
  // TODO: query from the target device; roughly 2x hardware compute unit.
  const int parallelThreshold = 256;
  // How many 128-bit vectors each thread should at least read.
  const int targetVectorCount = 8;
  // For non-aligned linalg reductions (where reductionSize is not a multiple
  // of workgroupSize * threadLoads), the integer division below truncates
  // toward zero, making this loop slightly more eager to halve workgroupSize
  // than for aligned shapes. The (workgroupSize / 2) % subgroupSize guard
  // still preserves the structural invariant that each workgroup is a whole
  // number of subgroups, so the resulting config is always valid; it may
  // just have fewer "vectors per thread" than the heuristic targets.
  while (parallelSize && *parallelSize > parallelThreshold &&
         (workgroupSize / 2) % subgroupSize == 0 &&
         reductionSize / (workgroupSize * threadLoads) < targetVectorCount) {
    // Use less subgroups per workgroup in order to host more workgroups per
    // hardware compute unit.
    workgroupSize /= 2;
    *parallelSize /= 2;
  }

  // Multi-reduction fusions (e.g. mean + variance for layer/group-norm)
  // share the input via a private alloca. When the inner reduction dim
  // is smaller than the workgroup size, `threadLoads` collapses to 1
  // and the workgroup-level `vector.transfer_read` ends up
  // `vector<1xN>` with N being the data shape (not the per-thread
  // tile). Such a vector has no valid lane layout under
  // `lane_basis = subgroupSize`, and the LLVMGPUVectorDistribute pass
  // rejects with `'func.func' op failed to distribute`. Bail to
  // TileAndFuse for these specific cases. Single-reduction overshoots
  // (canonical masking case, e.g. `tensor<2x508xf16>`) are unaffected
  // because they do not need the shared-input alloca pattern, and
  // larger multi-reductions where each thread reads a contiguous
  // `threadLoads`-tile are also unaffected because their per-thread
  // vector matches the lane layout.
  if (!useStrictAlignment) {
    int64_t innerRedDim = bounds[reductionDims.back()];
    if (ShapedType::isStatic(innerRedDim) && innerRedDim < workgroupSize) {
      int reductionOpCount = 0;
      for (Operation *computeOp : *computeOps) {
        if (auto reductionLinalg = dyn_cast<linalg::LinalgOp>(computeOp)) {
          if (reductionLinalg.getNumReductionLoops() > 0) {
            ++reductionOpCount;
          }
        }
      }
      if (reductionOpCount > 1) {
        return failure();
      }
    }
  }

  if (failed(populateConfigInfo(op, *computeOps, target, workgroupSize,
                                subgroupSize, threadLoads,
                                /*allowMaskedTail=*/!useStrictAlignment))) {
    return failure();
  }

  OpBuilder b(context);
  auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(context);
  SmallVector<NamedAttribute, 1> pipelineAttrs = {NamedAttribute(
      IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName(), pipelineOptions)};
  auto pipelineConfig = b.getDictionaryAttr(pipelineAttrs);

  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      context, PipelineAttr::get(context, LoweringPipeline::VectorDistribute),
      SymbolRefAttr(), {workgroupSize, 1, 1}, subgroupSize, pipelineConfig);

  if (shouldSetTunerAttributes()) {
    setRootOpInfo(op);
  }
  return setTranslationInfo(entryPoint, translationInfo);
}

} // namespace mlir::iree_compiler::IREE::GPU
