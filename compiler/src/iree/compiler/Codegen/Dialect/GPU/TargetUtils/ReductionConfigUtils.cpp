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
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

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

static bool hasReductionIterator(linalg::LinalgOp op) {
  return llvm::any_of(op.getIteratorTypesArray(), linalg::isReductionIterator);
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
static FailureOr<int64_t> getBitWidth(linalg::LinalgOp op) {
  SmallVector<FailureOr<TensorSizeEstimate>> inputSizes =
      llvm::filter_to_vector(
          llvm::map_range(
              llvm::concat<Value>(op.getDpsInits(), op.getDpsInputs()),
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

/// Returns the dimension size that threadLoads must divide for the given op.
static int64_t getThreadLoadsConstraint(linalg::LinalgOp op) {
  SmallVector<unsigned> parallelDims, reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);
  SmallVector<int64_t> bounds = op.getStaticLoopRanges();

  if (reductionDims.empty()) {
    // Parallel-only op: threadLoads must divide last parallel dim
    return bounds[parallelDims.back()];
  }
  // Reduction op: threadLoads must divide last reduction dim
  return bounds[reductionDims.back()];
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
    SmallVector<int64_t> mapping(tileSizes.size());
    std::iota(mapping.begin(), mapping.end(), 0);

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

static FailureOr<TileSizesConfig> getVectorDistributeReductionConfig(
    linalg::LinalgOp op, IREE::GPU::TargetAttr target,
    llvm::SmallDenseMap<unsigned, unsigned> &sharedWgpTiles,
    int64_t workgroupSize, int64_t subgroupSize, int64_t threadLoads) {
  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();

  SmallVector<int64_t> threadTileSizes(op.getNumLoops(), 0);
  SmallVector<int64_t> threadCounts(op.getNumLoops(), 1);
  SmallVector<int64_t> subgroupCounts(op.getNumLoops(), 1);

  // Set the configuration for the operation with no reduction dims.
  // The workgroup tile sizes are set by the reduction operation.
  if (reductionDims.empty()) {
    SmallVector<int64_t> serialTileSizes(op.getNumLoops(), 1);

    // For the shared wgp dimension, set the serial tile sizes to be zero.
    for (const auto &[dim, tile_size] : sharedWgpTiles) {
      serialTileSizes[dim] = 0;
    }

    int64_t parallelSize = bounds[parallelDims.back()];
    if (ShapedType::isDynamic(parallelSize)) {
      parallelSize = kVectorDistributeReductionSizeToTargetIfDynamic;
    }
    if (parallelSize % threadLoads != 0) {
      LDBG() << "Failed to get vector distribute config for op ("
             << "parallelSize=" << parallelSize
             << " is not divisible by threadLoads=" << threadLoads << "):\n"
             << op << "\n";
      return failure();
    }

    int64_t lastDimReductionTileSize = workgroupSize * threadLoads;
    // Setting subgroupBasis to minimum i.e., 1 and threadBasis
    // to maximum i.e., subgroupSize.
    int64_t subgroupBasis = 1;
    int64_t threadBasis = subgroupSize;

    lastDimReductionTileSize =
        llvm::APIntOps::GreatestCommonDivisor(
            {64, static_cast<uint64_t>(parallelSize)},
            {64, static_cast<uint64_t>(lastDimReductionTileSize)})
            .getZExtValue();

    if (!(sharedWgpTiles.size() == op.getNumLoops())) {
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
    if (sharedWgpTiles.size() == op.getNumLoops()) {
      lastDimReductionTileSize = 1;
      threadLoads = 1;
      threadBasis = 1;
      subgroupBasis = 1;
    }

    serialTileSizes[parallelDims.back()] = lastDimReductionTileSize;
    threadTileSizes[parallelDims.back()] = threadLoads;
    threadCounts[parallelDims.back()] = threadBasis;
    subgroupCounts[parallelDims.back()] = subgroupBasis;

    return TileSizesConfig{serialTileSizes, threadTileSizes, threadCounts,
                           subgroupCounts, /*isReduction=*/false};
  }

  // Setting the config for operation with atleast one reduction dimension.
  SmallVector<int64_t> partialReductionTileSizes(op.getNumLoops(), 0);
  int64_t lastReductionDim = reductionDims.back();

  // TODO: This is enabled for matvec on ROCm for now. We should
  // validate this strategy and extend to more linalg generics and to CUDA.
  if (isROCmBackend(target) && ShapedType::isStaticShape(bounds) &&
      isMatmulLike(op)) {
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
    sharedWgpTiles[parallelIdx] = numParallelReductions;
  }

  for (int64_t dim : reductionDims) {
    threadTileSizes[dim] = 1;
    partialReductionTileSizes[dim] = 1;
  }

  int64_t lastReductionDimSize = bounds[reductionDims.back()];

  if (ShapedType::isDynamic(lastReductionDimSize)) {
    lastReductionDimSize = kVectorDistributeReductionSizeToTargetIfDynamic;
  }
  if (lastReductionDimSize % threadLoads != 0) {
    LDBG() << "Failed to get vector distribute config for op ("
           << "lastReductionDimSize=" << lastReductionDimSize
           << " is not divisible by threadLoads=" << threadLoads << "):\n"
           << op << "\n";
    return failure();
  }

  int64_t partialReductionSize = workgroupSize * threadLoads;
  partialReductionSize = llvm::APIntOps::GreatestCommonDivisor(
                             {64, static_cast<uint64_t>(partialReductionSize)},
                             {64, static_cast<uint64_t>(lastReductionDimSize)})
                             .getZExtValue();

  int64_t threadBasis = subgroupSize;
  int subgroupStride = threadBasis * threadLoads;
  while (partialReductionSize % subgroupStride != 0) {
    threadBasis >>= 1;
    subgroupStride = threadBasis * threadLoads;
  }
  int subgroup = partialReductionSize / subgroupStride;
  int64_t subgroupBasis = (subgroup == 0) ? 1 : subgroup;

  partialReductionTileSizes[lastReductionDim] = partialReductionSize;
  threadTileSizes[lastReductionDim] = threadLoads;
  threadCounts[lastReductionDim] = threadBasis;
  subgroupCounts[lastReductionDim] = subgroupBasis;

  return TileSizesConfig{partialReductionTileSizes, threadTileSizes,
                         threadCounts, subgroupCounts,
                         /*isReduction=*/true};
}

static LogicalResult
populateConfigInfo(const llvm::SetVector<linalg::LinalgOp> &computeOps,
                   IREE::GPU::TargetAttr target, int64_t workgroupSize,
                   int64_t subgroupSize, int64_t threadLoads) {
  if (computeOps.empty()) {
    return failure();
  }

  SmallVector<unsigned> sharedParallelDims;
  linalg::LinalgOp op = computeOps.front();
  op.getParallelDims(sharedParallelDims);
  llvm::SmallDenseSet<unsigned> sharedParallelSet(sharedParallelDims.begin(),
                                                  sharedParallelDims.end());
  for (linalg::LinalgOp linalgOp : computeOps) {
    SmallVector<unsigned> currParallelDims;
    linalgOp.getParallelDims(currParallelDims);
    llvm::SmallDenseSet<unsigned> currParallelSet(currParallelDims.begin(),
                                                  currParallelDims.end());
    llvm::set_intersect(sharedParallelSet, currParallelSet);
  }
  llvm::SmallDenseMap<unsigned, unsigned> sharedWgpTiles;

  // Initialize the tile sizes of the shared workgroup dims to be 1.
  for (auto i : sharedParallelSet) {
    sharedWgpTiles[i] = 1;
  }

  // An operation needs a lowering config if it cannot infer its configuration
  // from its outputs. For each output that feeds into a computeOp, the
  // consumer's config determines the tiling for the dimensions referenced by
  // the output indexing map. If all iteration dimensions are covered this way,
  // no config is needed.
  auto needsLoweringConfig = [&](linalg::LinalgOp linalgOp) -> bool {
    llvm::SmallDenseSet<unsigned> coveredDims;
    for (OpOperand &output : linalgOp.getDpsInitsMutable()) {
      OpResult result = linalgOp.getTiedOpResult(&output);
      bool hasComputeOpUser =
          llvm::any_of(result.getUsers(), [&](Operation *user) {
            auto linalgUser = dyn_cast<linalg::LinalgOp>(user);
            return linalgUser && computeOps.contains(linalgUser);
          });
      if (!hasComputeOpUser) {
        continue;
      }

      AffineMap outputMap = linalgOp.getMatchingIndexingMap(&output);
      for (unsigned i = 0; i < outputMap.getNumResults(); ++i) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(outputMap.getResult(i))) {
          coveredDims.insert(dimExpr.getPosition());
        }
      }
    }

    for (unsigned i = 0; i < linalgOp.getNumLoops(); ++i) {
      if (!coveredDims.contains(i)) {
        return true;
      }
    }
    return false;
  };

  // Compute tile sizes for ops that need a lowering config.
  SmallVector<std::pair<linalg::LinalgOp, TileSizesConfig>, 4> tileSizeConfigs;
  for (linalg::LinalgOp linalgOp : computeOps) {
    if (!needsLoweringConfig(linalgOp)) {
      continue;
    }

    auto config = getVectorDistributeReductionConfig(
        linalgOp, target, sharedWgpTiles, workgroupSize, subgroupSize,
        threadLoads);
    if (failed(config)) {
      return failure();
    }
    tileSizeConfigs.push_back({linalgOp, *config});
  }

  // Build and set lowering configurations. Only the first op in
  // tileSizeConfigs (the last compute op in topological order) gets the
  // workgroup tile sizes.
  if (tileSizeConfigs.empty()) {
    return success();
  }

  MLIRContext *context = tileSizeConfigs.front().first->getContext();
  auto &[firstOp, firstConfig] = tileSizeConfigs.front();
  SmallVector<int64_t> workgroupTileSizes(firstOp.getNumLoops(), 0);
  for (const auto &[dim, tileSize] : sharedWgpTiles) {
    workgroupTileSizes[dim] = tileSize;
  }
  setLoweringConfig(firstOp,
                    firstConfig.buildConfig(context, workgroupTileSizes));

  for (auto &[linalgOp, config] : llvm::drop_begin(tileSizeConfigs)) {
    setLoweringConfig(linalgOp, config.buildConfig(context, std::nullopt));
  }
  return success();
}

/// Check if the dispatch has a single store operation.
/// If the dispatch meets the criterion, it returns the set of
/// compute ops.
template <typename... StoreOpTy>
static FailureOr<SetVector<linalg::LinalgOp>>
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

  SetVector<linalg::LinalgOp> computeOps;
  // Check if the op contains an scf.forall. This could be generalized, but for
  // now only check for split-reduction generated scf.forall.
  std::optional<scf::ForallOp> forallOp;
  bool foundLinalgOp = false;
  for (Operation *op : slice) {
    if (isa<scf::ForallOp>(op)) {
      if (forallOp) {
        return failure();
      }
      forallOp = cast<scf::ForallOp>(op);
      continue;
    }
    if (isa<linalg::LinalgOp>(op)) {
      foundLinalgOp = true;
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
    if (foundLinalgOp) {
      return failure();
    }
    return checkDispatchForVectorDistribution<tensor::ParallelInsertSliceOp>(
        forallOp.value());
  }

  bool containsValidReductionOp = true;
  for (Operation *op : llvm::reverse(slice)) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (isa<linalg::FillOp>(op)) {
        continue;
      }
      // Pooling operations are currently not supported properly. The pipeline
      // can support them, it's just not tested properly.
      // TODO: Check if we can support pooling operations.
      if (linalg::isaConvolutionOpInterface(linalgOp)) {
        return failure();
      }
      if (hasReductionIterator(linalgOp) &&
          failed(checkSingleCombiner(linalgOp))) {
        containsValidReductionOp = false;
        break;
      }
      computeOps.insert(linalgOp);
    }
  }

  // Return failure if the dispatch contains no reduction op.
  if (!containsValidReductionOp) {
    return failure();
  }

  // Get the reduction dimensions.
  auto getReductionDims = [](linalg::LinalgOp &linalgOp) -> SetVector<int64_t> {
    SetVector<int64_t> reductionDims;
    for (auto [idx, iterator] :
         llvm::enumerate(linalgOp.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(iterator)) {
        reductionDims.insert(idx);
      }
    }
    return reductionDims;
  };

  for (linalg::LinalgOp linalgOp : computeOps) {
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      int64_t operandIdx = linalgOp.getIndexingMapIndex(operand);
      AffineMap indexingMap = linalgOp.getIndexingMapsArray()[operandIdx];

      auto opResult = dyn_cast<OpResult>(operand->get());
      if (!opResult) {
        continue;
      }

      auto producerOp = dyn_cast<linalg::LinalgOp>(opResult.getOwner());
      if (!producerOp || !computeOps.contains(producerOp)) {
        continue;
      }

      // Check whether the op operand is not reduced and producer of that
      // operand is not a reduction op.
      auto reductionDims = getReductionDims(linalgOp);
      bool isOperandReduced = llvm::any_of(
          llvm::seq<int>(indexingMap.getNumResults()), [&](int val) {
            return reductionDims.contains(indexingMap.getDimPosition(val));
          });
      if (isOperandReduced && hasReductionIterator(producerOp)) {
        return failure();
      }
    }
  }
  return computeOps;
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
                                 linalg::LinalgOp op) {
  MLIRContext *context = op.getContext();
  OpBuilder b(context);

  if (!hasReductionIterator(op)) {
    return failure();
  }

  FailureOr<SetVector<linalg::LinalgOp>> computeOps =
      checkDispatchForVectorDistribution<IREE::TensorExt::DispatchTensorStoreOp,
                                         IREE::Codegen::StoreToBufferOp>(
          entryPoint);

  if (failed(computeOps)) {
    return failure();
  }

  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  IREE::GPU::TargetWgpAttr wgp = target.getWgp();
  // First consider the inner reduction dimension. If this is a multiple of a
  // subgroup size choice, use this as the reduction dimension, and choose
  // subgroup, thread loads etc based on it. Otherwise, consider the entire
  // reduction dimension.  This happens for example in case of multiple
  // reductions in scaled matmul with the last dimension being the block size
  // (32 for gfx950).
  int64_t reductionSize = bounds[reductionDims.back()];
  if (ShapedType::isStatic(reductionSize) &&
      reductionSize % target.getPreferredSubgroupSize() != 0) {
    // Consider the entire reduction dimension.
    reductionSize = 1;
    for (unsigned dim : reductionDims) {
      if (ShapedType::isDynamic(bounds[dim])) {
        reductionSize = ShapedType::kDynamic;
        break;
      }
      reductionSize *= bounds[dim];
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

  int64_t subgroupSize = 0;
  for (int s : wgp.getSubgroupSizeChoices().asArrayRef()) {
    if (reductionSize % s == 0 || hasDynamicReductionDim) {
      subgroupSize = s;
      break;
    }
  }

  if (subgroupSize == 0) {
    return failure();
  }

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

  unsigned threadLoads = largestLoadSizeInBits / *bitWidth;
  if (!hasDynamicReductionDim) {
    while ((reductionSize / threadLoads) % subgroupSize != 0) {
      threadLoads /= 2;
    }
  }

  // Adjust threadLoads to satisfy constraints from all compute ops in the
  // dispatch.
  for (linalg::LinalgOp linalgOp : *computeOps) {
    int64_t constraint = getThreadLoadsConstraint(linalgOp);
    if (ShapedType::isStatic(constraint)) {
      while (threadLoads > 1 && constraint % threadLoads != 0) {
        threadLoads /= 2;
      }
    }
    if (threadLoads <= 1) {
      break;
    }
  }

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

  int64_t workgroupSize = reductionSize / threadLoads;
  if (workgroupSize > maxWorkgroupSize) {
    workgroupSize = llvm::APIntOps::GreatestCommonDivisor(
                        {64, static_cast<uint64_t>(workgroupSize)},
                        {64, static_cast<uint64_t>(maxWorkgroupSize)})
                        .getZExtValue();
  }

  // Total parallel size that can fill the GPU with enough workgorups.
  // TODO: query from the target device; roughly 2x hardware compute unit.
  const int parallelThreshold = 256;
  // How many 128-bit vectors each thread should at least read.
  const int targetVectorCount = 8;
  while (parallelSize && *parallelSize > parallelThreshold &&
         (workgroupSize / 2) % subgroupSize == 0 &&
         reductionSize / (workgroupSize * threadLoads) < targetVectorCount) {
    // Use less subgroups per workgroup..
    workgroupSize /= 2;
    // in order to host more workgroups per hardware compute unit.
    *parallelSize /= 2;
  }

  if (failed(populateConfigInfo(*computeOps, target, workgroupSize,
                                subgroupSize, threadLoads))) {
    return failure();
  }

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
