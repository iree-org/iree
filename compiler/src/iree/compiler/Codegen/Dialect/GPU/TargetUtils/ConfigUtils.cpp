// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-gpu-config-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::IREE::GPU {

constexpr int64_t kCacheLineSizeBits = 128 * 8;

LogicalResult
setDataTiledMultiMmaLoweringConfig(IREE::GPU::TargetAttr target,
                                   mlir::FunctionOpInterface entryPoint,
                                   Operation *op) {
  auto multiMmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op);
  if (!multiMmaOp) {
    return failure();
  }
  auto dataTiledMmaAttr = dyn_cast<DataTiledMMAAttr>(multiMmaOp.getKind());
  if (!dataTiledMmaAttr) {
    return failure();
  }

  LDBG("MultiMMA TileAndFuse Config");

  // Compute workgroup size, which is given by the subgroup size times the
  // number of subgroups. The number of subgroups is found by the product of
  // subgroup unrolling factors, since the non-unrolled inner kernel takes a
  // single subgroup.
  const int64_t targetSubgroupSize = dataTiledMmaAttr.getSubgroupSize();
  int64_t flatWorkgroupSize = targetSubgroupSize *
                              dataTiledMmaAttr.getUnrollMToSubgroups() *
                              dataTiledMmaAttr.getUnrollNToSubgroups();
  std::array<int64_t, 3> workgroupSize{flatWorkgroupSize, 1, 1};

  // Set all workgroup and reduction tile sizes to 1, since the data tiled
  // kernel has the scope of an entire workgroup, and the reduction tiling is
  // already baked into the "opaque" data tiled inner layout of the multi_mma.
  SmallVector<AffineMap> indexingMaps = multiMmaOp.getIndexingMapsArray();
  mlir::linalg::ContractionDimensions contractionDims =
      mlir::linalg::inferContractionDims(indexingMaps).value();

  int64_t iterationRank = indexingMaps.front().getNumDims();
  SmallVector<int64_t> workgroupTileSizes(iterationRank, 1);
  SmallVector<int64_t> reductionTileSizes(iterationRank, 0);
  for (int64_t kDim : contractionDims.k) {
    workgroupTileSizes[kDim] = 0;
    reductionTileSizes[kDim] = 1;
  }

  // Set tile sizes.
  MLIRContext *context = multiMmaOp.getContext();
  SmallVector<NamedAttribute, 1> attrs;
  Builder b(context);
  attrs.emplace_back(b.getStringAttr("workgroup"),
                     b.getI64ArrayAttr(workgroupTileSizes));
  attrs.emplace_back(b.getStringAttr("reduction"),
                     b.getI64ArrayAttr(reductionTileSizes));
  // Promote operands to use shared memory for LHS and RHS.
  GPU::LoweringConfigAttr::setPromotedOperandList(context, attrs, {0, 1});
  auto configDict = b.getDictionaryAttr(attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  // Don't add any special padding or prefetching, since the data-tiled layout
  // is already what we want.
  SmallVector<NamedAttribute, 1> pipelineAttrs;
  auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
      context, /*prefetchSharedMemory=*/false,
      /*no_reduce_shared_memory_bank_conflicts=*/true,
      /*reorder_workgroups_strategy=*/std::nullopt);
  pipelineAttrs.emplace_back(
      b.getStringAttr(IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName()),
      pipelineOptions);
  auto pipelineConfig = b.getDictionaryAttr(pipelineAttrs);

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, targetSubgroupSize, pipelineConfig);
}

LogicalResult setMatmulLoweringConfig(IREE::GPU::TargetAttr target,
                                      mlir::FunctionOpInterface entryPoint,
                                      Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) {
    return failure();
  }

  if (target.getWgp().getMma().empty())
    return failure();

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  mlir::linalg::ContractionDimensions contractionDims =
      mlir::linalg::inferContractionDims(linalgOp).value();

  if (contractionDims.k.empty() || contractionDims.m.empty() ||
      contractionDims.n.empty()) {
    return failure();
  }

  // For now we are not being smart and trying to reshape dimensions to allow
  // for better usage of intrinsics, and instead are tiling all dimensions
  // except the inner most m, n, and k dimensions to 1.
  int64_t mDim = contractionDims.m.back();
  int64_t nDim = contractionDims.n.back();
  int64_t kDim = contractionDims.k.back();

  // Dynamic dims are expected to be taken care of earlier in the pipeline.
  if (ShapedType::isDynamic(bounds[mDim]) ||
      ShapedType::isDynamic(bounds[nDim]) ||
      ShapedType::isDynamic(bounds[kDim])) {
    return failure();
  }

  Value lhs = linalgOp.getDpsInputOperand(0)->get();
  Value rhs = linalgOp.getDpsInputOperand(1)->get();
  Value init = linalgOp.getDpsInitOperand(0)->get();

  Type lhsElemType = getElementTypeOrSelf(lhs);
  Type rhsElemType = getElementTypeOrSelf(rhs);
  Type initElemType = getElementTypeOrSelf(init);

  GPUMatmulShapeType problem{bounds[mDim], bounds[nDim], bounds[kDim],
                             lhsElemType,  rhsElemType,  initElemType};

  SmallVector<GPUMatmulShapeType> intrinsics;
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    if (mma.getSubgroupSize() != targetSubgroupSize)
      continue;
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType);
  }
  if (intrinsics.empty())
    return failure();

  GPUMMAHeuristicSeeds seeds;
  int64_t inBitWidth = lhsElemType.getIntOrFloatBitWidth();

  // Note that the following heuristic seeds are just placeholder values.
  // We need to clean it up and make it adjusting to different targets.
  // See https://github.com/iree-org/iree/issues/16341 for details.
  if (problem.mSize * problem.nSize <= 512 * 512) {
    // For matmuls with small M*N size, we want to distribute M*N onto more
    // workgroups to fill the GPU. Use a smaller bestMNTileCountPerSubgroup
    // and a larger bestKTileCountPerSubgroup.
    seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
             /*bestMNTileCountPerSubgroup=*/4,
             /*bestKTileCountPerSubgroup=*/8,
             /*bestKElementCountPerSubgroup*/ kCacheLineSizeBits / inBitWidth};
  } else {
    seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
             /*bestMNTileCountPerSubgroup=*/16,
             /*bestKTileCountPerSubgroup=*/4,
             /*bestKElementCountPerSubgroup*/ kCacheLineSizeBits / 2 /
                 inBitWidth};
  }

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  LDBG("Matmul TileAndFuse Config");

  // Infer if lhs or rhs is transposed to help generate better schedule.
  // TODO: Drop this. This is only a consideration for other pipelines.
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  bool transposedLhs =
      kDim !=
      llvm::cast<AffineDimExpr>(maps[0].getResults().back()).getPosition();
  bool transposedRhs =
      nDim !=
      llvm::cast<AffineDimExpr>(maps[1].getResults().back()).getPosition();

  // First try to find a schedule with an exactly matching intrinsic.
  std::optional<GPUMMASchedule> schedule =
      deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                        targetSubgroupSize, transposedLhs, transposedRhs);
  if (!schedule) {
    // Then try again by allowing upcasting accumulator.
    schedule = deduceMMASchedule(
        problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize,
        transposedLhs, transposedRhs, /*canUpcastAcc=*/true);
  }

  if (!schedule) {
    LDBG("Failed to deduce TileAndFuse MMA schedule");
    return failure();
  }

  LDBG("Target Subgroup size: " << targetSubgroupSize);
  LDBG("Schedule: sizes [" << schedule->mSize << ", " << schedule->nSize << ", "
                           << schedule->kSize << "]");
  LDBG("Schedule: tile counts [" << schedule->mTileCount << ", "
                                 << schedule->nTileCount << ", "
                                 << schedule->kTileCount << "]");
  LDBG("Schedule: warp counts [" << schedule->mWarpCount << ", "
                                 << schedule->nWarpCount << "]");

  std::array<int64_t, 3> workgroupSize{
      schedule->nWarpCount * targetSubgroupSize, schedule->mWarpCount, 1};

  SmallVector<int64_t> workgroupTileSizes(linalgOp.getNumLoops(), 0);
  SmallVector<int64_t> reductionTileSizes(linalgOp.getNumLoops(), 0);
  SmallVector<int64_t> subgroupTileSizes(linalgOp.getNumLoops(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : contractionDims.batch) {
    workgroupTileSizes[batch] = 1;
  }

  // Tile all m, n, and k dimensions to 1 except the innermost. Unit dims
  // from this tiling are folded before vectorization.
  for (int64_t m : llvm::drop_end(contractionDims.m)) {
    workgroupTileSizes[m] = 1;
  }
  for (int64_t n : llvm::drop_end(contractionDims.n)) {
    workgroupTileSizes[n] = 1;
  }
  for (int64_t k : llvm::drop_end(contractionDims.k)) {
    reductionTileSizes[k] = 1;
  }

  // Compute the M/N dimension tile size by multiplying subgroup information.
  workgroupTileSizes[mDim] =
      schedule->mWarpCount * schedule->mTileCount * schedule->mSize;
  workgroupTileSizes[nDim] =
      schedule->nWarpCount * schedule->nTileCount * schedule->nSize;

  // Specify the subgroup tile sizes from the mma schedule. This is applied
  subgroupTileSizes[mDim] = schedule->mTileCount;
  subgroupTileSizes[nDim] = schedule->nTileCount;

  // Similarly the reduction tile size is just the post-packing tile count.
  reductionTileSizes[kDim] = schedule->kTileCount;

  IREE::GPU::MmaInterfaceAttr mmaKind =
      target.getWgp().getMma()[schedule->index];

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  MLIRContext *context = linalgOp.getContext();
  SmallVector<NamedAttribute, 1> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getI64ArrayAttr(workgroupTileSizes));
  attrs.emplace_back(StringAttr::get(context, "reduction"),
                     b.getI64ArrayAttr(reductionTileSizes));
  attrs.emplace_back(StringAttr::get(context, "subgroup"),
                     b.getI64ArrayAttr(subgroupTileSizes));
  attrs.emplace_back(StringAttr::get(context, "mma_kind"), mmaKind);
  GPU::LoweringConfigAttr::setPromotedOperandList(context, attrs, {0, 1});
  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  SmallVector<NamedAttribute, 1> pipelineAttrs;
  auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
      context, /*prefetchSharedMemory=*/true,
      /*no_reduce_shared_memory_bank_conflicts=*/false,
      /*reorder_workgroups_strategy=*/std::nullopt);
  pipelineAttrs.emplace_back(
      StringAttr::get(context,
                      IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName()),
      pipelineOptions);
  auto pipelineConfig = DictionaryAttr::get(context, pipelineAttrs);

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, targetSubgroupSize, pipelineConfig);
}

/// Helper to identify contraction like operations for operand promotiong.
static bool isNonMatvecContraction(linalg::LinalgOp linalgOp) {
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return false;
  }

  if (contractionDims->k.size() < 1 || contractionDims->m.size() < 1 ||
      contractionDims->n.size() < 1) {
    return false;
  }

  auto getElementCount = [&](ArrayRef<unsigned> dims) {
    int64_t acc = 1;
    for (auto mDim : dims) {
      int64_t size = bounds[mDim];
      if (ShapedType::isDynamic(size)) {
        return size;
      }
      acc *= size;
    }
    return acc;
  };
  return getElementCount(contractionDims->m) != 1 &&
         getElementCount(contractionDims->n) != 1;
}

LogicalResult setTileAndFuseLoweringConfig(IREE::GPU::TargetAttr target,
                                           mlir::FunctionOpInterface entryPoint,
                                           Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // Bail out on multi result cases as consumer fusion currently does not
  // support multi result ops.
  if (!linalgOp || linalgOp.getNumDpsInits() != 1) {
    return failure();
  }

  // This pipeline requires tensor semantics. Also fail for gather semantics
  // for now to simplify tile + fuse.
  if (!linalgOp.hasPureTensorSemantics() || linalgOp.hasIndexSemantics()) {
    return failure();
  }

  SmallVector<unsigned int> partitionableLoops;
  linalgOp.getParallelDims(partitionableLoops);

  // Bail out if op is not tilable.
  if (partitionableLoops.empty()) {
    return failure();
  }

  const int subgroupSize = target.getPreferredSubgroupSize();
  const unsigned loopDepth = linalgOp.getNumLoops();

  // Configurations we need to decide.
  int64_t flatWorkgroupSize = 1;
  SmallVector<int64_t> workgroupTileSizes(loopDepth, 0);
  SmallVector<int64_t> threadTileSizes(loopDepth, 0);

  // Common case for all linalg ops.

  // The core idea is to distribute the partitioned loops to the workgroup
  // dimensions. The goal is to fill up the GPU as much as possible, which means
  // 1) distributing to as many threads as possible, and 2) avoid assigning too
  // many threads to handle out-of-bound elements (thus idle).

  auto elementHasPowerOfTwoBitwidth = [](Value operand) {
    Type elementType = getElementTypeOrSelf(operand.getType());
    return isa<IntegerType, FloatType>(elementType) &&
           llvm::isPowerOf2_64(IREE::Util::getTypeBitWidth(elementType));
  };

  // Whether we can try to use the vectorization pipeline.
  SmallVector<int64_t> loopBounds = linalgOp.getStaticLoopRanges();
  bool projPerm =
      llvm::all_of(linalgOp.getIndexingMapsArray(),
                   [](AffineMap map) { return map.isProjectedPermutation(); });
  bool powTwo =
      llvm::all_of(linalgOp->getOperands(), elementHasPowerOfTwoBitwidth);
  bool staticShape = llvm::none_of(loopBounds, ShapedType::isDynamic);

  // Require all affine maps to be projected permutation so that we can
  // generate vector transfer ops.
  bool vectorizable = projPerm && powTwo && staticShape;

  const unsigned minBitwidth = getMinElementBitwidth(linalgOp);
  // Make sure we use a tile size that results in some integral number of bytes.
  const unsigned scaleToByte =
      std::max(8 / minBitwidth, static_cast<unsigned>(1));

  // Distribute workload to the given `numThreads` by allowing a potental loss.
  auto distributeToThreads = [&](int64_t numThreads,
                                 std::optional<int64_t> lossFactor =
                                     std::nullopt) {
    LDBG("Loss factor: " << lossFactor << "\n");
    // Initialize the configuration.
    flatWorkgroupSize = 1;
    // Initialize tiling along all partitioned loops with size 1.
    for (int64_t loopIndex : partitionableLoops) {
      workgroupTileSizes[loopIndex] = threadTileSizes[loopIndex] = 1;
    }
    // Override the innermost dimension to distribute to threads in a subgroup.
    workgroupTileSizes[partitionableLoops.back()] = subgroupSize;

    // If there are more than 3 parallel dim try to tile the extra higher level
    // dimensions to 1 for extra dimensions.
    if (isa<linalg::GenericOp>(linalgOp.getOperation())) {
      for (auto [i, tileSize] : llvm::enumerate(workgroupTileSizes)) {
        if (tileSize != 0)
          break;
        if (loopBounds[i] != 1)
          tileSize = 1;
      }
    }
    // Scan from the innermost shape dimension and try to deduce the
    // configuration for the corresponding GPU workgroup dimension.
    int64_t wgDim = 0;
    for (auto shapeDim : llvm::reverse(partitionableLoops)) {
      int64_t loopBound = loopBounds[shapeDim];
      // Skip dynamic dimensions.
      if (ShapedType::isDynamic(loopBound))
        continue;

      // Try to find some power of two that can devide the current shape dim
      // size. This vector keeps the candidate tile sizes.
      SmallVector<int64_t, 8> candidates;

      // For the inner most workgroup dim, try to see if we can have 4
      // elements per thread. This enables vectorization.
      if (vectorizable && wgDim == 0 && !lossFactor) {
        candidates.push_back(4 * numThreads);
      }
      // Try all power of two numbers up to the subgroup size.
      for (unsigned i = numThreads; i >= 1; i >>= 1) {
        candidates.push_back(i);
      }
      LLVM_DEBUG({
        llvm::dbgs() << "Base candidate tile sizes: [";
        llvm::interleaveComma(candidates, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      int64_t candidateWorkgroupSize = 1;
      for (int64_t candidate : candidates) {
        int64_t scaledTileSize = candidate * scaleToByte;
        if (loopBound % scaledTileSize != 0) {
          if (!lossFactor)
            continue;
          // Skip this candidate if it causes many threads to be idle.
          int64_t idleThreads = candidate - (loopBound % scaledTileSize);
          if (idleThreads > candidate / *lossFactor)
            continue;
        }
        // If the workload is too small and we cannot distribute to more than 2
        // workgroups, try a smaller tile size to increase parallelism.
        if (partitionableLoops.size() == 1 && candidate > subgroupSize &&
            llvm::divideCeil(loopBound, scaledTileSize) <= 2) {
          continue;
        }

        // Found a suitable candidate. Try to let each thread handle 4
        // elements if this is the workgroup x dimension.
        // TODO: Try to take into account element type bit width to get
        // 4xdword reads instead of 4x{elements}.
        workgroupTileSizes[shapeDim] = scaledTileSize;
        LLVM_DEBUG(llvm::dbgs()
                   << "Chosen workgroup tile size: " << scaledTileSize << "\n");
        if (vectorizable && wgDim == 0 && !lossFactor && candidate % 4 == 0) {
          // Use size-1 vectors to increase parallelism if larger ones causes
          // idle threads in the subgroup.
          bool hasIdleThreads =
              partitionableLoops.size() == 1 && candidate <= subgroupSize;
          int vectorSize = hasIdleThreads ? 1 : 4;
          LLVM_DEBUG(llvm::dbgs() << "Use vector size: " << vectorSize << "\n");
          threadTileSizes[shapeDim] = vectorSize * scaleToByte;
          candidateWorkgroupSize = candidate / vectorSize;
          assert(numThreads % (candidate / vectorSize) == 0);
          numThreads /= candidate / vectorSize;
        } else {
          if (wgDim == 0)
            vectorizable = false;
          threadTileSizes[shapeDim] = scaleToByte;
          candidateWorkgroupSize = candidate;
          assert(numThreads % candidate == 0);
          numThreads /= candidate;
        }
        assert(numThreads >= 1);
        break;
      }

      flatWorkgroupSize *= candidateWorkgroupSize;

      // Stop if we have distributed all threads.
      if (numThreads == 1)
        break;
      wgDim++;
    }
    return numThreads;
  };

  // First try to see if we can use up all threads without any loss.
  if (distributeToThreads(subgroupSize) != 1) {
    // Otherwise, allow larger and larger loss factor.

    // Threads for distribution. Use 32 at least.
    int64_t numThreads = std::max(subgroupSize, 32);
    // We can tolerate (1 / lossFactor) of threads in the workgroup to be idle.
    int64_t lossFactor = 32;

    for (; lossFactor >= 1; lossFactor >>= 1) {
      if (distributeToThreads(numThreads, lossFactor) == 1)
        break;
    }
  }

  // TODO(qedawkins): Currently scf.forall resolution only supports static
  // trip counts, meaning the workgroup tile size must perfectly divide the
  // loop bound (and thread tile size must perfectly divide the workgroup tile)
  // so that the trip count won't be static. Remove this check once proper
  // dynamic trip count resolution support is added.
  for (auto [loopId, threadTile] : llvm::enumerate(threadTileSizes)) {
    if (threadTile == 0) {
      continue;
    }
    int64_t bound = loopBounds[loopId];
    int64_t wkgpTile = workgroupTileSizes[loopId];
    if (bound % wkgpTile != 0 || wkgpTile % threadTile != 0) {
      return failure();
    }
  }

  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(threadTileSizes);

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  MLIRContext *context = linalgOp.getContext();
  SmallVector<NamedAttribute, 1> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getI64ArrayAttr(workgroupTileSizes));

  attrs.emplace_back(StringAttr::get(context, "thread"),
                     b.getI64ArrayAttr(threadTileSizes));

  if (isNonMatvecContraction(linalgOp)) {
    GPU::LoweringConfigAttr::setPromotedOperandList(context, attrs, {0, 1});
  }

  // Heuristic value chosen to limit maximum vector sizes when tiling below.
  const unsigned maxVectorSize = 32;

  // Try to tile all reductions by some small factor, preferrably 4, when
  // possible. This gives us a chance to perform vector4 load if an input has
  // its innnermost dimension being reduction. It also avoids generating too
  // many instructions when unrolling vector later. We limit the expected
  // vector size by estimating it from the size of the iteration space tile and
  // limit it to a reasonable value. We process the loops from inner most to
  // outer most to try to align loads along inner dimensions.
  int64_t vectorSize = 1;
  int64_t numLoops = linalgOp.getNumLoops();
  SmallVector<utils::IteratorType> iterTypes = linalgOp.getIteratorTypesArray();
  SmallVector<int64_t> loopTileSizes(numLoops, 0);
  for (auto [reverseIdx, iter] : llvm::enumerate(llvm::reverse(iterTypes))) {
    unsigned i = numLoops - reverseIdx - 1;
    if (linalg::isReductionIterator(iter) || i >= workgroupTileSizes.size() ||
        workgroupTileSizes[i] == 0) {
      int64_t tileSize = getReductionTilingFactor(loopBounds[i]);
      if (vectorSize * tileSize > maxVectorSize) {
        tileSize = 1;
      }
      vectorSize *= tileSize;
      loopTileSizes[i] = tileSize;
    }
  }
  if (llvm::any_of(loopTileSizes, [](int64_t s) { return s != 0; })) {
    attrs.emplace_back(StringAttr::get(context, "reduction"),
                       b.getI64ArrayAttr(loopTileSizes));
  }

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  LDBG("Selected tile and fuse lowering config: " << loweringConfig << "\n");

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      {flatWorkgroupSize, 1, 1}, subgroupSize, DictionaryAttr());
}

//===----------------------------------------------------------------------===//
// Lowering Config Attributes
//===----------------------------------------------------------------------===//

GPUPipelineOptions
getPipelineOptions(FunctionOpInterface funcOp,
                   IREE::Codegen::TranslationInfoAttr translationInfo) {
  GPUPipelineOptions pipelineOptions = {};
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);

  if (DictionaryAttr config = translationInfo.getConfiguration()) {
    std::optional<NamedAttribute> maybePipelineOptionsAttr =
        config.getNamed(GPUPipelineOptionsAttr::getDictKeyName());
    if (!maybePipelineOptionsAttr.has_value()) {
      return pipelineOptions;
    }
    auto pipelineOptionsAttr =
        cast<GPUPipelineOptionsAttr>(maybePipelineOptionsAttr->getValue());
    BoolAttr prefetchSharedMemory =
        pipelineOptionsAttr.getPrefetchSharedMemory();
    if (prefetchSharedMemory) {
      pipelineOptions.prefetchSharedMemory = prefetchSharedMemory.getValue();
    }
    BoolAttr noReduceBankConflicts =
        pipelineOptionsAttr.getNoReduceSharedMemoryBankConflicts();
    if (noReduceBankConflicts) {
      pipelineOptions.enableReduceSharedMemoryBankConflicts =
          !noReduceBankConflicts.getValue();
    }
    ReorderWorkgroupsStrategyAttr reorderWorkgroupsStrategy =
        pipelineOptionsAttr.getReorderWorkgroupsStrategy();
    if (reorderWorkgroupsStrategy) {
      pipelineOptions.reorderStrategy = reorderWorkgroupsStrategy.getValue();
    }
  }

  pipelineOptions.enableUkernels = targetAttr && hasUkernel(targetAttr);

  LLVM_DEBUG(llvm::dbgs() << "GPU Pipeline Options: " << pipelineOptions
                          << "\n");
  return pipelineOptions;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const GPUPipelineOptions &options) {
  StringRef reorderStr = "<not set>";
  if (options.reorderStrategy) {
    switch (options.reorderStrategy.value()) {
    case ReorderWorkgroupsStrategy::Transpose:
      reorderStr = "transpose";
      break;
    case ReorderWorkgroupsStrategy::Swizzle:
      reorderStr = "swizzle";
      break;
    case ReorderWorkgroupsStrategy::None:
      reorderStr = "none";
      break;
    default:
      assert(false && "Unhandled reorder option");
    }
  }

  return os << "{" << "enableReduceSharedMemoryBankConflicts = "
            << options.enableReduceSharedMemoryBankConflicts << ", "
            << ", prefetchSharedMemory = " << options.prefetchSharedMemory
            << ", reorderWorkgroupsStrategy = " << reorderStr
            << ", enableUkernels = " << options.enableUkernels << "}";
}

} // namespace mlir::iree_compiler::IREE::GPU
