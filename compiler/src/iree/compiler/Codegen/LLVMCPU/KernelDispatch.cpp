// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"

#include <numeric>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

/// NOTE: None of these flags are supported in any form long term. This are
/// temporary hooks added for development purposes. They could be
/// changed/modified at any time.
/// TODO: Find a way to plumb this through to not rely on these flags.

static llvm::cl::opt<int> clNativeVectorSizeInBytes(
    "iree-codegen-llvm-vector-size-in-bytes",
    llvm::cl::desc("native vector size to use on the hardware"),
    llvm::cl::init(16));

static llvm::cl::opt<int> clNumberOfRuntimeThreads(
    "iree-codegen-llvm-number-of-threads",
    llvm::cl::desc("number of threads that are used at runtime"),
    llvm::cl::init(8));

static llvm::cl::list<int> mmt4dWorkgroupTileSizes(
    "iree-codegen-llvm-mmt4d-workgroup-tile-sizes",
    llvm::cl::desc("linalg.mmt4d workgroup tile size"), llvm::cl::ZeroOrMore);

static llvm::cl::list<int> mmt4dL1TileSizes(
    "iree-codegen-llvm-mmt4d-l1-tile-size",
    llvm::cl::desc("linalg.mmt4d L1 tile size"), llvm::cl::ZeroOrMore);

static llvm::cl::list<int> mmt4dVectorSizes(
    "iree-codegen-llvm-mmt4d-vector-size",
    llvm::cl::desc("linalg.mmt4d vector tile size"), llvm::cl::ZeroOrMore);

static llvm::cl::opt<int> defaultWorkgroupTileSize(
    "iree-codegen-llvm-generic-ops-workgroup-size",
    llvm::cl::desc(
        "linalg.generic and linalg.indexed_generic workgroup tile size"),
    llvm::cl::init(64));

static llvm::cl::opt<bool> useLinalgTransformInterp(
    "iree-codegen-use-linalg-transform-interp",
    llvm::cl::desc(
        "experimental path to use the linalg transform dialect interpreter"),
    llvm::cl::init(false));

// TODO(hanchung): Remove the flag. This is the flag for fastly falling back to
// the previous snapshot.
static llvm::cl::opt<bool> disableMatmulPadPipeline(
    "iree-codegen-disable-matmul-pad-pipeline",
    llvm::cl::desc("disable padding options in Matmul codegen"),
    llvm::cl::init(false));

using IREE::Codegen::DispatchLoweringPassPipeline;

/// Looks for the `native_vector_size` attribute in the hal.executable.variant
/// op.
static Optional<int64_t> getNativeVectorSizeInBytes(func::FuncOp entryPointFn) {
  auto variantOp =
      entryPointFn->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!variantOp) return llvm::None;
  IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.target();
  if (!targetAttr) return llvm::None;
  auto config = targetAttr.getConfiguration();
  if (!config) return llvm::None;
  auto nativeVectorSizeAttr = config.getAs<IntegerAttr>("native_vector_size");
  if (!nativeVectorSizeAttr) return llvm::None;
  int64_t nativeVectorSizeVal = nativeVectorSizeAttr.getInt();
  if (!nativeVectorSizeVal) return llvm::None;
  return nativeVectorSizeVal;
}

/// For a given `shapedType` or (`byteWidth` of element type) return the number
/// of elements that correspond to the native vector size. Returns 1 as the
/// fallback.
static int64_t getVectorSize(func::FuncOp entryPointFn, unsigned byteWidth) {
  if (Optional<int64_t> nativeVectorSize =
          getNativeVectorSizeInBytes(entryPointFn)) {
    return nativeVectorSize.getValue() / byteWidth;
  }
  return clNativeVectorSizeInBytes / byteWidth;
}
static int64_t getVectorSize(func::FuncOp entryPointFn, ShapedType shapedType) {
  Type elementType = shapedType.getElementType();
  if (!elementType.isIntOrFloat()) return 1;
  unsigned byteWidth = IREE::Util::getRoundedElementByteWidth(elementType);
  return getVectorSize(entryPointFn, byteWidth);
}

/// Returns minimum tiling sizes for each dimension. One dimension is possible
/// to access at different element types. It determines the tiling sizes by
/// looking into all the operands.
static SmallVector<int64_t> getMinTilingSizesForEachDim(
    func::FuncOp entryPointFn, linalg::LinalgOp op) {
  unsigned numLoops = op.getNumLoops();
  SmallVector<int64_t> minTileSizes(numLoops, 1);
  auto inputOutputOpOperands = op.getInputAndOutputOperands();
  for (auto map : llvm::enumerate(op.getIndexingMaps())) {
    // Check the fastest varying dimension of the operand. Set the vector size
    // of the corresponding loop to the vector size.
    if (map.value().getNumResults() == 0) continue;
    auto fastestVaryingDimExpr =
        map.value().getResults().back().dyn_cast<AffineDimExpr>();
    if (!fastestVaryingDimExpr) continue;
    unsigned fastestVaryingDim = fastestVaryingDimExpr.getPosition();

    // If the indexing map has result it has to be a shaped type.
    auto operandType =
        inputOutputOpOperands[map.index()]->get().getType().cast<ShapedType>();
    minTileSizes[fastestVaryingDim] =
        std::max<int64_t>(minTileSizes[fastestVaryingDim],
                          getVectorSize(entryPointFn, operandType));
  }
  return minTileSizes;
}

/// Returns the type length in bytes. Looks through all the interface binding
/// ops to see the ABI types and guess-timates the type size to use. This is
/// used to convert the vector size in bytes to vector size in number of
/// elements.
static unsigned getReferenceTypeLengthInBytes(func::FuncOp entryPointFn) {
  unsigned referenceTypeLengthInBytes = 4;
  entryPointFn.walk([&](IREE::HAL::InterfaceBindingSubspanOp subSpanOp) {
    Type type = subSpanOp.getResult().getType();
    Type elementType = TypeSwitch<Type, Type>(type)
                           .Case<ShapedType, IREE::Flow::DispatchTensorType>(
                               [&](auto shapedType) -> Type {
                                 // Ignore operands that are 0D tensors. These
                                 // are not vector-loadable, so using these to
                                 // get vector length would be a pessimization.
                                 if (!shapedType.getRank()) return nullptr;
                                 return shapedType.getElementType();
                               })
                           .Default([&](Type t) -> Type { return nullptr; });
    if (!elementType || !elementType.isIntOrFloat()) return;
    unsigned typeWidthInBytes =
        IREE::Util::getRoundedElementByteWidth(elementType);
    referenceTypeLengthInBytes =
        std::min<unsigned>(referenceTypeLengthInBytes, typeWidthInBytes);
  });
  return referenceTypeLengthInBytes;
}

/// Returns the default tile sizes to use for the loops that are distributed at
/// Flow level.
static SmallVector<int64_t> getDefaultDistributedLoopTileSizes(
    ArrayRef<int64_t> lbs, ArrayRef<int64_t> ubs,
    ArrayRef<int64_t> minTileSizes, ArrayRef<int64_t> maxTileSizes,
    ArrayRef<int64_t> vectorSizeHints) {
  assert(lbs.size() == ubs.size() && lbs.size() == minTileSizes.size() &&
         lbs.size() == maxTileSizes.size() &&
         "expected all vectors to be of equal size");
  size_t numDims = lbs.size();
  SmallVector<int64_t> distributedTileSizes(numDims, 1);
  SmallVector<int64_t> numWorkgroupsPerDim(numDims, 1);
  SmallVector<int64_t> workload(numDims, 1);
  for (auto i : llvm::seq<size_t>(0, numDims)) {
    if (maxTileSizes[i] == 0 || ShapedType::isDynamic(lbs[i]) ||
        ShapedType::isDynamic(ubs[i])) {
      distributedTileSizes[i] = maxTileSizes[i];
      workload[i] = ShapedType::kDynamicSize;
      continue;
    }

    assert(lbs[i] <= ubs[i]);
    workload[i] = ubs[i] - lbs[i];
    int64_t candidateTileSize = 1;
    int64_t targetSize = std::min(workload[i] / 2, maxTileSizes[i]);
    int64_t vectorSize = vectorSizeHints[i];
    if (vectorSize > 1) {
      // Pick the factor of dim which is closest to the target tile size and
      // is a multiplier of vector size.
      for (int64_t k = vectorSize; k <= targetSize; k += vectorSize) {
        if (workload[i] % k == 0 && k >= minTileSizes[i]) {
          candidateTileSize = k;
        }
      }
    }
    // Fallback to power of 2 if there's no hint or can't find the ideal size.
    if (vectorSize <= 1 || candidateTileSize == 1) {
      candidateTileSize =
          std::max<int64_t>(llvm::PowerOf2Floor(targetSize), minTileSizes[i]);
    }

    // Limit the workload per workgroup to the default being the max to keep the
    // work per invocation reasonable.
    distributedTileSizes[i] =
        std::min<int64_t>(candidateTileSize, maxTileSizes[i]);
    numWorkgroupsPerDim[i] =
        llvm::divideCeil(workload[i], distributedTileSizes[i]);
  }

  // Reduce the number of workgroups in cases where we are dividing the work too
  // much. Over-provision the number of workgroups to twice the number of
  // threads.
  int64_t numWorkgroupsLimit = 2 * clNumberOfRuntimeThreads;
  int64_t numWorkgroups =
      std::accumulate(numWorkgroupsPerDim.begin(), numWorkgroupsPerDim.end(),
                      1LL, std::multiplies<int64_t>{});
  unsigned currDim = numDims;
  while (numWorkgroups > numWorkgroupsLimit && currDim > 0) {
    unsigned index = currDim - 1;
    int64_t currSize = distributedTileSizes[index];
    if (currSize >= maxTileSizes[index] ||
        workload[index] == ShapedType::kDynamicSize ||
        currSize >= workload[index]) {
      currDim--;
      continue;
    }
    int64_t newSize = std::min<int64_t>(currSize * 2, maxTileSizes[index]);
    int64_t vectorSize = vectorSizeHints[index];
    // Chech if it's the ideal size with vector size hint. And skip if the new
    // size will break the ideal size.
    if (vectorSize > 1 &&
        (currSize % vectorSize == 0 && workload[index] % currSize == 0) &&
        (newSize % vectorSize != 0 || workload[index] % newSize != 0)) {
      currDim--;
      continue;
    }

    distributedTileSizes[index] = newSize;
    int64_t nwg =
        llvm::divideCeil(workload[index], distributedTileSizes[index]);
    if (nwg < numWorkgroupsPerDim[index]) {
      numWorkgroups /= numWorkgroupsPerDim[index];
      numWorkgroups *= nwg;
    } else {
      currDim--;
    }
  }
  return distributedTileSizes;
}

/// Adjusts the workload per workgroup to be a multiple of vector size to ensure
/// that the op vectorizes.
static int64_t getMaxTileSize(int64_t lb, int64_t ub, int64_t maxSize,
                              int64_t vectorSizeVal) {
  if (ub == ShapedType::kDynamicSize || lb == ShapedType::kDynamicSize) {
    return maxSize;
  }
  int64_t dim = ub - lb;
  if (dim < vectorSizeVal) return dim;
  for (int64_t i = std::min(maxSize, dim); i > 0; --i) {
    if (dim % i == 0 && i % vectorSizeVal == 0) {
      return i;
    }
  }
  // If it can't be a multiple of vectorSizeVal, let's choose a factor of dim
  // sizes heuristically.
  int64_t start = std::min(maxSize, dim);
  start = std::min(start, vectorSizeVal * 2);
  for (int64_t i = start; i > 0; --i) {
    if (dim % i == 0) {
      return i;
    }
  }
  return 1;
}

/// Returns the tile size to use for the Flow level.
///
/// The vectorSizeHints can be empty or as many as the number of loops. When not
/// empty, each hint should be 1 or the vector size. On the dimensions where the
/// hints != 1, it will try to find the tile sizes which are multipliers of the
/// hints.
static SmallVector<int64_t> getDefaultDistributedLevelTileSizes(
    ArrayRef<unsigned> partitionableLoops, ArrayRef<int64_t> lbs,
    ArrayRef<int64_t> ubs, ArrayRef<int64_t> minTileSizes,
    ArrayRef<int64_t> maxTileSizes, ArrayRef<int64_t> vectorSizeHints = {}) {
  int64_t numLoops = lbs.size();
  assert(numLoops == minTileSizes.size() && maxTileSizes.size() == numLoops &&
         "expected as many min/max tile sizes as number of loops");
  assert(
      vectorSizeHints.empty() ||
      vectorSizeHints.size() == numLoops &&
          "vector size hints should be empty or equal to the number of loops");

  // Only set values when the loop is partitionable.
  SmallVector<int64_t> adjustedMinTileSizes(numLoops, 0);
  SmallVector<int64_t> adjustedMaxTileSizes(numLoops, 0);
  SmallVector<int64_t> adjustedVectorSizeHints(numLoops, 1);
  for (auto i : partitionableLoops) {
    adjustedMinTileSizes[i] = minTileSizes[i];
    adjustedMaxTileSizes[i] = maxTileSizes[i];
    if (!vectorSizeHints.empty()) {
      adjustedVectorSizeHints[i] = vectorSizeHints[i];
    }
  }

  SmallVector<int64_t> distributedTileSizes =
      getDefaultDistributedLoopTileSizes(lbs, ubs, adjustedMinTileSizes,
                                         adjustedMaxTileSizes,
                                         adjustedVectorSizeHints);
  // Final fix up of the tile sizes to make sure that they divide the problem
  // size to make it vectorizable.
  for (auto i : llvm::seq<unsigned>(0, distributedTileSizes.size())) {
    if (!distributedTileSizes[i]) continue;
    distributedTileSizes[i] = getMaxTileSize(
        lbs[i], ubs[i], distributedTileSizes[i], minTileSizes[i]);
  }
  return distributedTileSizes;
}
static SmallVector<int64_t> getDefaultDistributedLevelTileSizes(
    linalg::LinalgOp linalgOp, ArrayRef<int64_t> minTileSizes,
    ArrayRef<int64_t> maxTileSizes, ArrayRef<int64_t> vectorSizeHints = {}) {
  OpBuilder builder(linalgOp.getContext());
  builder.setInsertionPoint(linalgOp);
  SmallVector<int64_t> lbs(linalgOp.getNumLoops(), 0);
  SmallVector<int64_t> ubs = linalgOp.getStaticLoopRanges();
  auto loops =
      cast<IREE::Flow::PartitionableLoopsInterface>(linalgOp.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  return getDefaultDistributedLevelTileSizes(loops, lbs, ubs, minTileSizes,
                                             maxTileSizes, vectorSizeHints);
}

/// Splits the tile sizes in `parallelSizes` into `reductionSizes` for the
/// reduction loops.
static void splitParallelAndReductionTiles(
    linalg::LinalgOp op, SmallVectorImpl<int64_t> &parallelSizes,
    SmallVectorImpl<int64_t> &reductionSizes) {
  reductionSizes.assign(parallelSizes.begin(), parallelSizes.end());
  for (auto iteratorType : llvm::enumerate(op.iterator_types())) {
    if (iteratorType.value().cast<StringAttr>().getValue() ==
        getParallelIteratorTypeName()) {
      reductionSizes[iteratorType.index()] = 0;
    } else {
      parallelSizes[iteratorType.index()] = 0;
    }
  }
}

static void setAlwaysVectorizeSizes(linalg::LinalgOp op,
                                    SmallVectorImpl<int64_t> &parallelSizes,
                                    SmallVectorImpl<int64_t> &reductionSizes) {
  SmallVector<int64_t, 4> staticLoopRanges = op.getStaticLoopRanges();
  for (auto en :
       llvm::enumerate(llvm::zip(staticLoopRanges, op.iterator_types()))) {
    auto size = std::get<0>(en.value());
    if (!ShapedType::isDynamic(size)) continue;
    auto iterType = std::get<1>(en.value()).cast<StringAttr>().getValue();
    if (iterType == getParallelIteratorTypeName()) {
      parallelSizes[en.index()] = 1;
    } else {
      reductionSizes[en.index()] = 1;
    }
  }
}

/// Sets the default configuration to use for an operation that implements the
/// `PartitionableLoopsInterface`, given the `lbs` and `ubs` of all the loops.
static LogicalResult setDefaultRootConfig(
    func::FuncOp entryPointFn,
    IREE::Flow::PartitionableLoopsInterface partitionableLoopsInterfaceOp,
    ArrayRef<int64_t> lbs, ArrayRef<int64_t> ubs) {
  if (getLoweringConfig(partitionableLoopsInterfaceOp)) return success();

  SmallVector<unsigned> partitionableLoops =
      partitionableLoopsInterfaceOp.getPartitionableLoops(kNumMaxParallelDims);

  SmallVector<int64_t> minTileSizes(lbs.size(), 1);
  SmallVector<int64_t> maxTileSizes(lbs.size(), 1);
  if (!partitionableLoops.empty()) {
    // TODO: Here the min tile size is just looking at the type of the data in
    // the entry point function, and using a vector size that depends on just
    // that. For `LinalgOp`s we can use the indexing map, find the loops that
    // are fastest varying and set those to have a min tile size of vector
    // length. A version of this is done for generic ops. Generalize that and
    // use it for `LinalgOp`s.
    unsigned typeWidthInBytes = getReferenceTypeLengthInBytes(entryPointFn);
    minTileSizes[partitionableLoops.back()] =
        getVectorSize(entryPointFn, typeWidthInBytes);
    for (auto partitionableLoopId : partitionableLoops) {
      maxTileSizes[partitionableLoopId] = defaultWorkgroupTileSize;
    }
  }

  SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
      partitionableLoops, lbs, ubs, minTileSizes, maxTileSizes);
  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(flowTileSizes));
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, partitionableLoopsInterfaceOp, tileSizes,
      DispatchLoweringPassPipeline::CPUDefault);
}

static LogicalResult setMatmulPadRootConfig(
    func::FuncOp entryPointFn, linalg::ContractionOpInterface op,
    ArrayRef<int64_t> flowTileSizes, ArrayRef<int64_t> target2ndTileSizes,
    int vectorSize) {
  assert(target2ndTileSizes.size() == 3 &&
         "the current configuration is driven by matmul which has exactly "
         "three loops");
  // The tiling for parallel dims and reduction dims should be separated.
  SmallVector<int64_t> parallelTileSizes;
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  int64_t nLoops = linalgOp.getNumLoops();
  if (nLoops >= 3) {
    parallelTileSizes.append(nLoops - 3, 1);
    parallelTileSizes.push_back(target2ndTileSizes[0]);
  }
  if (nLoops >= 2) {
    parallelTileSizes.push_back(target2ndTileSizes[1]);
  }
  parallelTileSizes.push_back(0);

  // TODO(hanchung): Make logic more heuristic. Padding hurts performance a lot
  // if the dim size is small (e.g., K=24).
  auto lhsShapedType = op.lhs().getType().cast<ShapedType>();
  int64_t K = lhsShapedType.getShape().back();
  SmallVector<int64_t> reductionTileSizes;
  reductionTileSizes.append(nLoops - 1, 0);
  reductionTileSizes.push_back(
      getMaxTileSize(0, K, target2ndTileSizes[2], vectorSize));

  TileSizesListType tileSizes;
  tileSizes.emplace_back(flowTileSizes.begin(), flowTileSizes.end());
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes,
      DispatchLoweringPassPipeline::CPUDoubleTilingPadExpert);
}

static LogicalResult setMatmulNoPadRootConfig(
    func::FuncOp entryPointFn, linalg::ContractionOpInterface op,
    ArrayRef<int64_t> flowTileSizes, ArrayRef<int64_t> target2ndTileSizes,
    int vectorSize) {
  assert(target2ndTileSizes.size() == 3 &&
         "the current configuration is driven by matmul which has exactly "
         "three loops");
  // The tiling for parallel dims and reduction dims should be separated.
  SmallVector<int64_t> parallelTileSizes;
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  int64_t nLoops = linalgOp.getNumLoops();
  if (nLoops >= 3) {
    parallelTileSizes.append(nLoops - 3, 1);
    parallelTileSizes.push_back(getMaxTileSize(
        0, flowTileSizes[nLoops - 3], target2ndTileSizes[0], vectorSize));
  }
  if (nLoops >= 2) {
    parallelTileSizes.push_back(getMaxTileSize(
        0, flowTileSizes[nLoops - 2], target2ndTileSizes[1], vectorSize));
  }
  parallelTileSizes.push_back(0);

  auto lhsShapedType = op.lhs().getType().cast<ShapedType>();
  int64_t K = lhsShapedType.getShape().back();
  SmallVector<int64_t> reductionTileSizes;
  reductionTileSizes.append(nLoops - 1, 0);
  reductionTileSizes.push_back(
      getMaxTileSize(0, K, target2ndTileSizes[2], vectorSize));

  setAlwaysVectorizeSizes(linalgOp, parallelTileSizes, reductionTileSizes);

  TileSizesListType tileSizes;
  tileSizes.emplace_back(flowTileSizes.begin(), flowTileSizes.end());
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes,
      DispatchLoweringPassPipeline::CPUDoubleTilingExpert);
}

static LogicalResult setARMRootConfig(func::FuncOp entryPointFn,
                                      linalg::ContractionOpInterface op,
                                      ArrayRef<int64_t> flowTileSizes,
                                      int vectorSize) {
  // Hardcoded tile sizes, where v is the native vector size.
  // L1 tile sizes are {1, ..., 5v, v, 16v}.
  // Vector tile sizes are {1, ..., v, v, v}
  SmallVector<int64_t> l1TileSizes, vectorTileSizes;
  int64_t nLoops = cast<linalg::LinalgOp>(op.getOperation()).getNumLoops();
  if (nLoops >= 3) {
    l1TileSizes.append(nLoops - 3, 1);
    l1TileSizes.push_back(getMaxTileSize(0, flowTileSizes[nLoops - 3],
                                         5 * vectorSize, vectorSize));
    vectorTileSizes.append(nLoops - 3, 1);
    vectorTileSizes.push_back(vectorSize);
  }
  if (nLoops >= 2) {
    l1TileSizes.push_back(
        getMaxTileSize(0, flowTileSizes[nLoops - 2], vectorSize, vectorSize));
    vectorTileSizes.push_back(vectorSize);
  }

  // L1/vector tile size for k dimensions.
  auto lhsShapedType = op.lhs().getType().cast<ShapedType>();
  int64_t K = lhsShapedType.getShape().back();
  l1TileSizes.push_back(getMaxTileSize(0, K, 16 * vectorSize, vectorSize));
  vectorTileSizes.push_back(vectorSize);
  TileSizesListType tileSizes;
  tileSizes.emplace_back(flowTileSizes.begin(), flowTileSizes.end());
  tileSizes.push_back(l1TileSizes);
  tileSizes.push_back(vectorTileSizes);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes,
      DispatchLoweringPassPipeline::CPUTileFuseAndVectorize);
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, linalg::ContractionOpInterface contractionOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  auto linalgOp = cast<linalg::LinalgOp>(contractionOp.getOperation());
  unsigned numLoops = linalgOp.getNumLoops();
  {
    SmallVector<unsigned> dims;
    linalgOp.getReductionDims(dims);
    if (dims.size() != 1 || dims[0] != numLoops - 1) {
      return contractionOp.emitOpError(
          "expected to have exactly one reduction dim, and it is the innermost "
          "dim");
    }
  }

  // Consider all element types and use the smallest vector size. The tiling
  // sizes are chosen based on the vector size.
  auto lhsShapedType = contractionOp.lhs().getType().cast<ShapedType>();
  auto rhsShapedType = contractionOp.rhs().getType().cast<ShapedType>();
  auto resShapedType =
      linalgOp.getOutputOperand(0)->get().getType().cast<ShapedType>();
  int64_t vectorSize = getVectorSize(entryPointFn, lhsShapedType);
  vectorSize = std::min(vectorSize, getVectorSize(entryPointFn, rhsShapedType));
  vectorSize = std::min(vectorSize, getVectorSize(entryPointFn, resShapedType));

  // Use the default distribution for the matmul loops.
  int64_t defaultMaxSize = defaultWorkgroupTileSize;
  if (isX86(entryPointFn) || isRISCV(entryPointFn)) {
    defaultMaxSize = 128;
  }
  SmallVector<int64_t> minTileSizes =
      getMinTilingSizesForEachDim(entryPointFn, linalgOp);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultMaxSize);
  if (numLoops > 3) {
    minTileSizes[0] = 1;
    maxTileSizes[0] = 1;
  }
  SmallVector<int64_t> flowTileSizes =
      getDefaultDistributedLevelTileSizes(linalgOp, minTileSizes, maxTileSizes);

  // TODO(dcaballe): Find better configurations for RISC-V backends.
  if (isX86(entryPointFn) || isRISCV(entryPointFn)) {
    SmallVector<int64_t> tileSizes = {8, 32, 16};
    // There are hard-coded configurations in DoubleTilingPadExpert, so it only
    // works for linalg.matmul cases. We can relax it once we have better
    // scheduling, e.g., transform dialect.
    if (disableMatmulPadPipeline || numLoops != 3) {
      return setMatmulNoPadRootConfig(entryPointFn, contractionOp,
                                      flowTileSizes, tileSizes, vectorSize);
    }
    return setMatmulPadRootConfig(entryPointFn, contractionOp, flowTileSizes,
                                  tileSizes, vectorSize);
  }

  // Fall back to ARM configurations.
  bool isQuantized =
      lhsShapedType.getElementType() != resShapedType.getElementType();
  if (isQuantized) {
    SmallVector<int64_t> tileSizes = {4, 16, 4};
    return setMatmulPadRootConfig(entryPointFn, contractionOp, flowTileSizes,
                                  tileSizes, vectorSize);
  } else {
    return setARMRootConfig(entryPointFn, contractionOp, flowTileSizes,
                            vectorSize);
  }
}

/// Sets the lowering configuration for dispatch region for linalg.mmt4d root
/// op
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, linalg::Mmt4DOp mmt4dOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  // TODO(ataei): These are hand tuned for some performance benchmarks for
  // now, we want to adapt the same strategy as matmul that dynamically sets
  // tile size.
  auto getWorkgroupTileSizes = [&]() -> SmallVector<int64_t> {
    if (!mmt4dWorkgroupTileSizes.empty()) {
      return SmallVector<int64_t>(mmt4dWorkgroupTileSizes.begin(),
                                  mmt4dWorkgroupTileSizes.end());
    }
    return {48, 32};
  };

  auto getL1TileSizes = [&]() -> SmallVector<int64_t> {
    auto lhsShape = mmt4dOp.inputs()[0].getType().cast<ShapedType>().getShape();
    auto rhsShape = mmt4dOp.inputs()[1].getType().cast<ShapedType>().getShape();
    int M0 = lhsShape[2];
    int N0 = rhsShape[2];
    int K0 = lhsShape[3];
    if (!mmt4dL1TileSizes.empty()) {
      return SmallVector<int64_t>(mmt4dL1TileSizes.begin(),
                                  mmt4dL1TileSizes.end());
    }
    return {1, 1, 1, M0, N0, K0};
  };

  auto getVectorSizes = [&]() -> SmallVector<int64_t> {
    auto lhsShape = mmt4dOp.inputs()[0].getType().cast<ShapedType>().getShape();
    auto rhsShape = mmt4dOp.inputs()[1].getType().cast<ShapedType>().getShape();
    int M0 = lhsShape[2];
    int N0 = rhsShape[2];
    int K0 = lhsShape[3];
    if (!mmt4dVectorSizes.empty()) {
      return SmallVector<int64_t>(mmt4dVectorSizes.begin(),
                                  mmt4dVectorSizes.end());
    }
    return {1, 1, 1, M0, N0, K0};
  };

  SmallVector<int64_t> nativeVectorSize = getVectorSizes();

  TileSizesListType tileSizes = {getWorkgroupTileSizes(), getL1TileSizes(),
                                 nativeVectorSize};

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, mmt4dOp, tileSizes,
      DispatchLoweringPassPipeline::CPUTileFuseAndVectorize);
}

/// Sets the lowering configuration for dispatch region for linalg_ext.fft
/// root op.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, IREE::LinalgExt::FftOp fftOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  unsigned numLoops = fftOp.getLoopIteratorTypes().size();
  auto partitionedLoops = fftOp.getPartitionableLoops(kNumMaxParallelDims);
  SmallVector<int64_t> workgroupTileSizes(numLoops, defaultWorkgroupTileSize);
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto dim : llvm::seq<int64_t>(0, workgroupTileSizes.size())) {
    if (!partitionedLoopsSet.count(dim)) {
      workgroupTileSizes[dim] = 0;
    }
  }

  auto rank = fftOp.getOperandRank();
  if (workgroupTileSizes.size() >= rank && workgroupTileSizes[rank - 1] != 0) {
    APInt value;
    if (matchPattern(fftOp.getStage(), m_ConstantInt(&value))) {
      workgroupTileSizes[rank - 1] = 1ll << value.getSExtValue();
      workgroupTileSizes[rank - 1] =
          std::max(workgroupTileSizes[rank - 1],
                   static_cast<int64_t>(defaultWorkgroupTileSize));
    } else {
      return fftOp.emitOpError("non-constant stage might not work for fft op");
    }
  }
  TileSizesListType tileSizes = {workgroupTileSizes};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, fftOp, tileSizes, DispatchLoweringPassPipeline::CPUDefault);
}

static void setX86WorkgroupTileSizes(
    linalg::GenericOp genericOp, unsigned numLoops,
    ArrayRef<int64_t> flowTileSizes, ArrayRef<int64_t> minTileSizes,
    ArrayRef<int64_t> maxTileSizes,
    SmallVectorImpl<int64_t> &workgroupTileSizes) {
  workgroupTileSizes.append(numLoops, 0);
  SmallVector<int64_t, 4> staticLoopRanges = genericOp.getStaticLoopRanges();
  for (auto loopNum : llvm::seq<unsigned>(0, numLoops)) {
    if (flowTileSizes[loopNum]) {
      workgroupTileSizes[loopNum] =
          getMaxTileSize(0, flowTileSizes[loopNum], minTileSizes[loopNum],
                         minTileSizes[loopNum]);
    } else {
      // If the flow level tile size is zero, and static loop range is 0 as
      // well, set the tile sizes here to zero as well.
      workgroupTileSizes[loopNum] =
          staticLoopRanges[loopNum] == 1 ? 0 : minTileSizes[loopNum];
    }
  }
}

/// Returns true if the operation is a GenericOp implementing a supported
/// transposition.
static bool isSupportedTransposeOp(linalg::GenericOp genericOp) {
  // Check that the op has at least 2 dimensions.
  if (genericOp.getNumLoops() < 2) {
    return false;
  }

  // Check that the op has only one input and one output.
  // TODO(diegocaballero): Generalize to multiple inputs.
  if ((genericOp.getNumInputs() != 1) || (genericOp.getNumOutputs() != 1)) {
    return false;
  }

  // Check that all the iterators are parallel.
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }

  // Check that the two indexing maps are a permutation of each other.
  auto indexing_maps = genericOp.getIndexingMaps();
  return !indexing_maps[0].isEmpty() && !indexing_maps[1].isEmpty() &&
         ((indexing_maps[0].isIdentity() && !indexing_maps[1].isIdentity() &&
           indexing_maps[1].isPermutation()) ||
          (!indexing_maps[0].isIdentity() && indexing_maps[0].isPermutation() &&
           indexing_maps[1].isIdentity()));
}

/// Sets the default lowering configuration for a generic op to use
/// CPUDoubleTilingExpert pipeline.
static LogicalResult setDefaultGenericOpRootConfig(
    func::FuncOp entryPointFn, linalg::GenericOp genericOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  if (getLoweringConfig(genericOp)) {
    return success();
  }

  // If there are no loops, there is nothing to do.
  unsigned numLoops = genericOp.getNumLoops();
  if (numLoops == 0) return success();

  SmallVector<int64_t> minTileSizes =
      getMinTilingSizesForEachDim(entryPointFn, genericOp);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultWorkgroupTileSize);
  if (llvm::all_of(minTileSizes, [](int64_t vs) { return vs == 1; })) {
    // Nothing to vectorize just lower to loops.
    return success();
  }

  // Set the flow level tiling to the default.
  SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
      genericOp, minTileSizes, maxTileSizes);

  // Set the next level tile sizes.
  SmallVector<int64_t> parallelTileSizes;
  SmallVector<int64_t> reductionTileSizes;
  setX86WorkgroupTileSizes(genericOp, numLoops, flowTileSizes, minTileSizes,
                           maxTileSizes, parallelTileSizes);
  splitParallelAndReductionTiles(genericOp, parallelTileSizes,
                                 reductionTileSizes);
  setAlwaysVectorizeSizes(genericOp, parallelTileSizes, reductionTileSizes);

  TileSizesListType tileSizes;
  tileSizes.push_back(flowTileSizes);
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);

  // For non-tensor based ops use the Buffer ops pipeline.
  auto passPipeline =
      genericOp.hasTensorSemantics()
          ? DispatchLoweringPassPipeline::CPUDoubleTilingExpert
          : DispatchLoweringPassPipeline::CPUBufferOpsTileAndVectorize;
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, genericOp,
                                               tileSizes, passPipeline);
}

/// Sets the lowering configuration for a generic op implementing a
/// transposition to use CPUDoubleTilingExpert pipeline.
static LogicalResult setTransposeLikeOpRootConfig(
    func::FuncOp entryPointFn, linalg::GenericOp genericOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  if (getLoweringConfig(genericOp)) {
    return success();
  }

  if (!hasAVX2Features(genericOp) || !isSupportedTransposeOp(genericOp)) {
    return success();
  }

  unsigned numLoops = genericOp.getNumLoops();
  SmallVector<int64_t> minTileSizes =
      getMinTilingSizesForEachDim(entryPointFn, genericOp);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultWorkgroupTileSize);
  if (llvm::all_of(minTileSizes, [](int64_t vs) { return vs == 1; })) {
    // Nothing to vectorize just lower to loops.
    return success();
  }

  if (llvm::count_if(minTileSizes,
                     [](int64_t tileSize) { return tileSize > 1; }) != 2) {
    // Transpose patterns are not applicable if vectorizing more or less than
    // two dims.
    return success();
  }

  // Make sure that the original tile sizes are multiple of the tile sizes
  // to be used for the transpose op (i.e., 8x8).
  // TODO(diegocaballero): Enable 4x8 tile sizes if we find it useful.
  if (llvm::any_of(minTileSizes, [](int64_t tileSize) {
        return tileSize > 1 && (tileSize % 8) != 0;
      })) {
    return success();
  }

  // Replace dims to be vectorized with the new 8x8 tile sizes.
  std::replace_if(
      minTileSizes.begin(), minTileSizes.end(),
      [](int64_t tileSize) { return tileSize > 1; }, 8);

  // Set the flow level tiling to the default.
  SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
      genericOp, minTileSizes, maxTileSizes);

  // Set the next level tile sizes.
  SmallVector<int64_t> parallelTileSizes;
  setX86WorkgroupTileSizes(genericOp, numLoops, flowTileSizes, minTileSizes,
                           maxTileSizes, parallelTileSizes);

  TileSizesListType tileSizes;
  tileSizes.push_back(flowTileSizes);
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(/*reduction tile sizes=*/{});

  // For non-tensor based ops use the Buffer ops pipeline.
  auto passPipeline =
      genericOp.hasTensorSemantics()
          ? DispatchLoweringPassPipeline::CPUDoubleTilingExpert
          : DispatchLoweringPassPipeline::CPUBufferOpsTileAndVectorize;
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, genericOp,
                                               tileSizes, passPipeline);
}

/// Sets the lowering configuration for a generic op to use
/// CPUDoubleTilingExpert pipeline.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, linalg::GenericOp genericOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  if (failed(
          setTransposeLikeOpRootConfig(entryPointFn, genericOp, tiledLoops)) ||
      failed(
          setDefaultGenericOpRootConfig(entryPointFn, genericOp, tiledLoops))) {
    return failure();
  }
  return success();
}

/// Sets the lowering configuration for linalg.conv_2d_nhwc_hwcf and
/// linalg.depthwise_conv_2d_nhwc_hwc operations.
static LogicalResult setConvRootConfig(
    func::FuncOp entryPointFn, linalg::LinalgOp convOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops,
    ArrayRef<int64_t> targetTileSizes, int64_t vectorSize) {
  if (!isa<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(
          convOp.getOperation())) {
    return failure();
  }

  // Use the default distribution for the conv loops.
  unsigned numLoops = convOp.getNumLoops();
  SmallVector<int64_t> minTileSizes(numLoops, 1);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultWorkgroupTileSize);
  SmallVector<int64_t> vectorSizeHints(numLoops, 1);

  // Give the vector size hint on OC.
  vectorSizeHints[3] = vectorSize;

  // Set the flow level tiling to the default.
  SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
      convOp, minTileSizes, maxTileSizes, vectorSizeHints);

  // Shapes of N, OH, OW, OC, KH, KW, (IC)
  SmallVector<int64_t, 4> shapes = convOp.getStaticLoopRanges();
  SmallVector<int64_t> parallelTileSizes(targetTileSizes.begin(),
                                         targetTileSizes.end());
  for (auto i : llvm::seq<unsigned>(0, parallelTileSizes.size())) {
    auto tileSize = flowTileSizes[i] ? flowTileSizes[i] : shapes[i];
    // If the tile size is intended to be 1, do not adjust it to `vectorSize`.
    // The ops will be decomposed to lower-rank named ops.
    if (parallelTileSizes[i] != 1) {
      parallelTileSizes[i] =
          getMaxTileSize(0, tileSize, parallelTileSizes[i], vectorSize);
    }
  }
  SmallVector<int64_t> reductionTileSizes;
  splitParallelAndReductionTiles(convOp, parallelTileSizes, reductionTileSizes);
  setAlwaysVectorizeSizes(convOp, parallelTileSizes, reductionTileSizes);

  TileSizesListType tileSizes;
  tileSizes.push_back(flowTileSizes);
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, convOp, tileSizes,
      DispatchLoweringPassPipeline::CPUConvTileAndDecomposeExpert);
}

static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, linalg::Conv2DNhwcHwcfOp convOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  int64_t vectorSize =
      getVectorSize(entryPointFn, convOp.getResult(0).getType());
  SmallVector<int64_t> targetTileSizes = {1, 1, 8, vectorSize * 2, 1, 1, 8};
  return setConvRootConfig(entryPointFn, convOp, tiledLoops, targetTileSizes,
                           vectorSize);
}

/// Sets the lowering configuration for linalg.depthwise_conv_2d_nhwc_hwc
/// operations.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, linalg::DepthwiseConv2DNhwcHwcOp convOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  int64_t vectorSize =
      getVectorSize(entryPointFn, convOp.getResult(0).getType());
  SmallVector<int64_t> targetTileSizes = {1, 1, 8, vectorSize * 2, 1, 3};
  return setConvRootConfig(entryPointFn, convOp, tiledLoops, targetTileSizes,
                           vectorSize);
}

/// Set default configuration for Linalg ops.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, linalg::LinalgOp linalgOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  if (getLoweringConfig(linalgOp)) return success();

  auto partitionableLoopOp =
      cast<IREE::Flow::PartitionableLoopsInterface>(linalgOp.getOperation());
  SmallVector<int64_t> lbs(linalgOp.getNumLoops(), 0);
  SmallVector<int64_t> ubs = linalgOp.getStaticLoopRanges();
  return setDefaultRootConfig(entryPointFn, partitionableLoopOp, lbs, ubs);
}

/// Set the default configuration for operations that implement the
/// `TiledOpInterface`.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn,
    IREE::LinalgExt::TiledOpInterface tiledOpInterfaceOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  if (getLoweringConfig(tiledOpInterfaceOp)) return success();

  auto partitionableLoopOp = cast<IREE::Flow::PartitionableLoopsInterface>(
      tiledOpInterfaceOp.getOperation());

  // TODO(hanchung): Implement getStaticLoopRanges method for TiledOpInterface.
  OpBuilder builder(tiledOpInterfaceOp.getContext());
  builder.setInsertionPoint(tiledOpInterfaceOp);
  SmallVector<Range> iterationDomain =
      tiledOpInterfaceOp.getIterationDomain(builder);
  auto getStaticValue = [](Value v) -> int64_t {
    IntegerAttr attr;
    if (!matchPattern(v, m_Constant(&attr))) return ShapedType::kDynamicSize;
    return attr.getInt();
  };
  auto lbs = llvm::to_vector(llvm::map_range(
      iterationDomain, [&](Range r) { return getStaticValue(r.offset); }));
  auto ubs = llvm::to_vector(llvm::map_range(
      iterationDomain, [&](Range r) { return getStaticValue(r.size); }));
  return setDefaultRootConfig(entryPointFn, partitionableLoopOp, lbs, ubs);
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult setRootConfigImpl(
    func::FuncOp entryPointFn, Operation *op,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  // Do not overwrite default configuration.
  if (getLoweringConfig(op)) return success();

  // Redirect to individual operations.
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<IREE::LinalgExt::FftOp, linalg::GenericOp, linalg::Mmt4DOp,
              linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(
            [&](auto op) {
              return setRootConfig(entryPointFn, op, tiledLoops);
            })
        .Case<linalg::ContractionOpInterface>([&](auto op) {
          return setRootConfig(entryPointFn, op, tiledLoops);
        })
        .Case<linalg::LinalgOp, IREE::LinalgExt::TiledOpInterface>(
            [&](auto op) {
              return setRootConfig(entryPointFn, op, tiledLoops);
            })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Redirects to methods that set the configuration based on operation type for
/// VMVX backend.
static LogicalResult setVMVXRootConfigImpl(
    func::FuncOp entryPointFn, Operation *op,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  if (getLoweringConfig(op)) return success();

  // Redirect to individual operations.
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<linalg::LinalgOp, IREE::LinalgExt::TiledOpInterface>(
            [&](auto op) {
              return setRootConfig(entryPointFn, op, tiledLoops);
            })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Find the root operation for the dispatch region.
static FailureOr<Operation *> getRootOperation(
    ArrayRef<Operation *> computeOps) {
  Operation *rootOperation = nullptr;
  auto updateRootOperation = [&](Operation *op) -> LogicalResult {
    if (rootOperation) {
      return op->emitOpError(
          "unhandled multiple root operations in dispatch region");
    }
    rootOperation = op;
    return success();
  };
  for (auto op : computeOps) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Do not not treat linalg ops that are all parallel as root operations in
      // this sweep.
      if (linalgOp.getNumLoops() == linalgOp.getNumParallelLoops()) continue;

      // All other linalg ops are root ops.
      if (failed(updateRootOperation(op))) return failure();
      continue;
    }

    if (auto tiledOpInterfaceOp =
            dyn_cast<IREE::LinalgExt::TiledOpInterface>(op)) {
      // TODO(ravishankarm): For now
      // `tensor.extract_slice`/`tensor.insert_slice` implement the
      // `tiledInterfaceOp`. With tile + distribute moved out of Flow
      // dialect, this doesnt work anymore. Remove this when the external
      // model implementation of
      // `tensor.extract_slice`/`tensor.insert_slice` are dropped.
      if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op)) continue;

      // All other operations that implement this interface are root ops.
      if (failed(updateRootOperation(op))) return failure();
      continue;
    }
  }
  if (rootOperation) return rootOperation;

  // If no root operation is found yet. Look for linalg generic ops.
  for (auto op : llvm::reverse(computeOps)) {
    if (isa<linalg::LinalgOp>(op)) {
      if (failed(updateRootOperation(op))) return failure();
    }
  }
  return rootOperation;
}

/// Finds the root operation in the given list of Linalg operations and sets
/// its configuration. Returns error for multiple root operations.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, ArrayRef<Operation *> computeOps,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp)) {
    return failure();
  }
  Operation *rootOperation = rootOp.getValue();

  if (rootOperation) {
    if (isVMVXBackend(entryPointFn)) {
      if (failed(
              setVMVXRootConfigImpl(entryPointFn, rootOperation, tiledLoops))) {
        return failure();
      }
    } else {
      if (failed(setRootConfigImpl(entryPointFn, rootOperation, tiledLoops))) {
        return failure();
      }
    }
  }

  if (!getTranslationInfo(entryPointFn)) {
    // Fall back, just set the translation to CPUDefault.
    setTranslationInfo(entryPointFn, DispatchLoweringPassPipeline::CPUDefault,
                       /*workloadPerWorkgroup=*/ArrayRef<int64_t>{},
                       /*workgroupSize=*/ArrayRef<int64_t>{});
  }

  return success();
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult setTranslationInfoAndRootConfig(
    func::FuncOp entryPointFn, ArrayRef<Operation *> computeOps,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  // First check if the operations have a preset pipeline.
  for (auto computeOp : computeOps) {
    if (IREE::Codegen::CompilationInfoAttr compilationInfo =
            getCompilationInfo(computeOp)) {
      // If the function already has a translation, error out.
      if (auto translationInfo = getTranslationInfo(entryPointFn)) {
        return computeOp->emitOpError(
            "multiple ops within dispatch trying to set the translation "
            "info");
      }

      SmallVector<int64_t> workgroupSize =
          compilationInfo.getWorkgroupSizeVals();
      setTranslationInfo(entryPointFn, compilationInfo.getTranslationInfo(),
                         workgroupSize);
      setLoweringConfig(computeOp, compilationInfo.getLoweringConfig());
      eraseCompilationInfo(computeOp);
    }
  }

  // Next set the configuration of the operations.
  return setRootConfig(entryPointFn, computeOps, tiledLoops);
}

LogicalResult initCPULaunchConfig(ModuleOp moduleOp) {
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps =
      getAllEntryPoints(moduleOp);
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto entryPointOp = entryPointOps.lookup(funcOp.getName());
    if (!entryPointOp) continue;
    if (getTranslationInfo(entryPointOp)) continue;

    // If using sandbox passes, currently set the workload_per_wg to be
    // empty for single-threaded execution.
    if (useLinalgTransformInterp) {
      auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
          moduleOp.getContext(), IREE::Codegen::DispatchLoweringPassPipeline::
                                     LinalgTransformInterpCodegen);
      setTranslationInfo(funcOp, translationInfo);
      continue;
    }

    SmallVector<Operation *> computeOps;
    SmallVector<LoopTilingAndDistributionInfo> tiledLoops;

    // If there are no linalg ops, not using Linalg based lowering.
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      return failure();
    }

    if (failed(
            setTranslationInfoAndRootConfig(funcOp, computeOps, tiledLoops))) {
      return failure();
    }
  }

  // The root confguration setting introduces `tensor.dim` operations. Resolve
  // those away.
  RewritePatternSet patterns(moduleOp.getContext());
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

}  // namespace iree_compiler
}  // namespace mlir
