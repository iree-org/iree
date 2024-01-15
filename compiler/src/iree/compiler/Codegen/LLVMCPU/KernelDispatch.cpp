// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/LLVMCPU/TargetMLTransformInfo.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/TransformStrategies/CPU/Common.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "kernel-dispatch"
#define KD_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir::iree_compiler {

/// NOTE: None of these flags are supported in any form long term. This are
/// temporary hooks added for development purposes. They could be
/// changed/modified at any time.
/// TODO: Find a way to plumb this through to not rely on these flags.

static llvm::cl::opt<int> clNumberOfRuntimeThreads(
    "iree-llvmcpu-number-of-threads",
    llvm::cl::desc("number of threads that are used at runtime if codegen "
                   "thread distribution is enabled"),
    llvm::cl::init(8));

static llvm::cl::opt<bool> clDisableDistribution(
    "iree-llvmcpu-disable-distribution",
    llvm::cl::desc("disable thread distribution in codegen"),
    llvm::cl::init(false));

static llvm::cl::opt<int>
    clDefaultDistTileSize("iree-llvmcpu-distribution-size",
                          llvm::cl::desc("default distribution tile size"),
                          llvm::cl::init(64));

static llvm::cl::opt<int> clNarrowMatmulTileBytes(
    "iree-llvmcpu-narrow-matmul-tile-bytes",
    llvm::cl::desc(
        "target distribution tile size for wide matrix operand of narrow "
        "matmuls, expressed in bytes. Currently only used in data-tiled "
        "matmuls (mmt4d). Since this is only used for narrow matmuls, which "
        "traverse their wide matrix operand once, there is no reuse here and "
        "this doesn't have to be sized to fit in some CPU cache. This is more "
        "about distributing work to threads."),
    llvm::cl::init(64 * 1024));

static llvm::cl::opt<int> clGeneralMatmulTileBytes(
    "iree-llvmcpu-general-matmul-tile-bytes",
    llvm::cl::desc("target distribution tile size for matrix operands of "
                   "general matmuls, expressed in bytes. Currently only used "
                   "in data-tiled matmuls (mmt4d)."),
    llvm::cl::init(64 * 1024));

static llvm::cl::opt<bool> clDisableVectorPeeling(
    "iree-llvmcpu-disable-vector-peeling",
    llvm::cl::desc("Disable peeling as a pre-processing step for "
                   "vectorization (only relevant when using compiler "
                   "heuristics to select the strategy)."),
    llvm::cl::init(false));

// Non-static options are used in other places.
llvm::cl::opt<bool> clEnableTransformDialectJit(
    "iree-llvmcpu-enable-transform-dialect-jit",
    llvm::cl::desc("enable the usage of the transform dialect JIT"),
    llvm::cl::init(false));

using IREE::Codegen::DispatchLoweringPassPipeline;

// Encodes the pre-processing strategy to be applied on a Linalg operation
// before vectorization.
enum class VectorPreProcStrategy {
  // Peel iterations from the vector dimensions so that they become multiple of
  // the vector length.
  Peeling,
  // Compute vector dimensions assuming vector masking support. Vector sizes may
  // be rounded up to the nearest power of two and out-of-bounds elements would
  // be masked-out.
  Masking,
  // Do not apply any vectorization pre-processing transformation.
  None,
  // A hint for the compiler to use its heuristics to determine an
  // actual pre-processing strategy.
  Heuristics
};

static llvm::cl::opt<VectorPreProcStrategy> clPProcStrategy(
    "iree-codegen-llvmcpu-vector-pproc-strategy",
    llvm::cl::desc("Set the strategy for pre-processing Linalg operation "
                   "before vectorization:"),
    llvm::cl::values(
        clEnumValN(VectorPreProcStrategy::Peeling, "peel",
                   "Peel iterations from the vector dimensions so that they "
                   "become multiple of the vector length"),
        clEnumValN(
            VectorPreProcStrategy::Masking, "mask",
            " Compute vector dimensions assuming vector masking support. "
            "Vector sizes may be rounded up to the nearest power of two "
            "and out-of-bounds elements would be masked-out."),
        clEnumValN(
            VectorPreProcStrategy::None, "none",
            "Do not apply any vectorization pre-processing transformation."),
        clEnumValN(VectorPreProcStrategy::Heuristics, "heuristics",
                   "To be determined by IREE's heuristics (default).")),
    llvm::cl::init(VectorPreProcStrategy::Heuristics));

// TODO(dcaballe): Move operator<< to DebugUtils.h.
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const VectorPreProcStrategy &strategy) {
  switch (strategy) {
  case VectorPreProcStrategy::Peeling:
    os << "Peeling";
    break;
  case VectorPreProcStrategy::Masking:
    os << "Masking";
    break;
  case VectorPreProcStrategy::None:
    os << "None";
    break;
  case VectorPreProcStrategy::Heuristics:
    os << "Heuristics";
    break;
  }
  return os;
}

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::SmallVectorImpl<T> &vector) {
  for (T element : vector) {
    os << element << " ";
  }

  return os;
}

static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const mlir::iree_compiler::TileSizesListType &tileSizeList) {
  os << "[";
  for (auto &tuple : tileSizeList) {
    os << "[" << tuple << "]";
  }
  os << "]";

  return os;
}

/// Splits the iteration ranges from `op` and returns the `lbs` and the `ubs` as
/// separate lists.
static void getRangeBounds(TilingInterface op, SmallVectorImpl<int64_t> &lb,
                           SmallVectorImpl<int64_t> &ub) {
  OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);
  SmallVector<Range> loopRange = op.getIterationDomain(builder);
  auto getStaticValue = [](OpFoldResult ofr) -> int64_t {
    std::optional<int64_t> intVal = getConstantIntValue(ofr);
    if (!intVal)
      return ShapedType::kDynamic;
    return intVal.value();
  };
  lb = llvm::map_to_vector(loopRange,
                           [&](Range r) { return getStaticValue(r.offset); });
  ub = llvm::map_to_vector(loopRange,
                           [&](Range r) { return getStaticValue(r.size); });
}

/// Returns true if all the input and output tensor operands of 'op' are fully
/// dynamic.
static bool isFullyDynamicOp(linalg::LinalgOp op) {
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  return llvm::all_of(loopRanges,
                      [](int64_t size) { return ShapedType::isDynamic(size); });
}

/// Returns the vectorization pre-processing strategy (peeling, masking) for the
/// given LinalgOp. It is based on either:
///   * user-specified value, or
///   * heuristics (e.g. the op traits and the target architecture).
static VectorPreProcStrategy
getVectorPreProcStrategy(linalg::LinalgOp linalgOp) {
  // If set, use the strategy selected by a user.
  if (clPProcStrategy != VectorPreProcStrategy::Heuristics) {
    return clPProcStrategy;
  }

  // Select a strategy based on heuristics.
  if (linalgOp.hasBufferSemantics()) {
    return VectorPreProcStrategy::None;
  }

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(linalgOp);
  bool isLinalgGeneric = isa<linalg::GenericOp>(linalgOp.getOperation());

  // Default X86 specific strategy.
  if (isX86(targetAttr)) {
    if (isLinalgGeneric) {
      return VectorPreProcStrategy::Masking;
    }

    if (!clDisableVectorPeeling) {
      return VectorPreProcStrategy::Peeling;
    }
  }

  // Default RISC-V specific strategies.
  if (isRISCV(targetAttr)) {
    if (isLinalgGeneric) {
      return VectorPreProcStrategy::Masking;
    }

    if (!clDisableVectorPeeling) {
      return VectorPreProcStrategy::Peeling;
    }
  }

  // Default AArch64 specific strategies.
  if (isAArch64(targetAttr)) {
    if (hasAnySVEFeature(targetAttr)) {
      return VectorPreProcStrategy::Masking;
    }

    if (!clDisableVectorPeeling) {
      return VectorPreProcStrategy::Peeling;
    }
  }

  return VectorPreProcStrategy::None;
}

/// Looks for the `native_vector_size` attribute in the hal.executable.target
/// looked up from this op.
static int64_t getNativeVectorSizeInBytes(func::FuncOp entryPointFn) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  auto nativeVectorSizeAttr =
      getConfigIntegerAttr(targetAttr, "native_vector_size");
  if (nativeVectorSizeAttr) {
    int64_t nativeVectorSizeVal = nativeVectorSizeAttr->getInt();
    if (nativeVectorSizeVal) {
      return nativeVectorSizeVal;
    }
  }

  assert(0 && "Missing 'native_vector_size' attribute");
  return 0;
}

/// For a given `shapedType` or (`byteWidth` of element type) return the number
/// of elements that correspond to the native vector size. Returns 1 as the
/// fallback.
static int64_t getVectorSize(func::FuncOp entryPointFn, unsigned byteWidth) {
  return getNativeVectorSizeInBytes(entryPointFn) / byteWidth;
}
static int64_t getVectorSize(func::FuncOp entryPointFn, ShapedType shapedType) {
  Type elementType = shapedType.getElementType();
  if (!elementType.isIntOrFloat())
    return 1;
  unsigned byteWidth = IREE::Util::getRoundedElementByteWidth(elementType);
  return getVectorSize(entryPointFn, byteWidth);
}

/// Returns minimum tiling sizes for each dimension. One dimension is possible
/// to access at different element types. It determines the tiling sizes by
/// looking into all the operands.
// TODO(diegocaballero): Refactor this logic to a method that computes the final
// tile sizes for vectorization/unrolling in one shot.
static SmallVector<int64_t>
getMinTilingSizesForEachDim(func::FuncOp entryPointFn, linalg::LinalgOp op,
                            const LinalgOpInfo &linalgOpInfo,
                            const TargetMLTransformInfo &targetMLTransInfo) {
  unsigned numLoops = op.getNumLoops();
  SmallVector<int64_t> minTileSizes(numLoops, 1);
  auto inputOutputOpOperands = op->getOpOperands();

  for (auto [index, map] : llvm::enumerate(op.getIndexingMapsArray())) {
    // Check the fastest varying dimension of the operand. Set the vector size
    // of the corresponding loop to the vector size.
    if (map.getNumResults() == 0)
      continue;
    auto fastestVaryingDimExpr =
        dyn_cast<AffineDimExpr>(map.getResults().back());
    if (!fastestVaryingDimExpr)
      continue;
    unsigned fastestVaryingDim = fastestVaryingDimExpr.getPosition();

    // If the indexing map has result it has to be a shaped type.
    auto operandType =
        llvm::cast<ShapedType>(inputOutputOpOperands[index].get().getType());
    int64_t tileSize = getVectorSize(entryPointFn, operandType);

    minTileSizes[fastestVaryingDim] =
        std::max<int64_t>(minTileSizes[fastestVaryingDim], tileSize);
  }

  // Limit unroll factor. For now, we assume the rightmost non-one tiled
  // dimension is for vectorization and any other non-one dimension is for
  // unrolling.
  auto limitUnrollFactor = [&](int64_t maxUnrollFactor) {
    int vecDim;
    for (vecDim = minTileSizes.size() - 1; vecDim >= 0; --vecDim) {
      if (minTileSizes[vecDim] > 1) {
        break;
      }
    }
    for (int unrollDim = vecDim - 1; unrollDim >= 0; --unrollDim) {
      minTileSizes[unrollDim] =
          std::min<int64_t>(minTileSizes[unrollDim], maxUnrollFactor);
    }
  };

  if (linalgOpInfo.isTranspose()) {
    // Limit unrolling on transpose operations.
    // TODO(dcaballe): Consider input and output transposes.
    limitUnrollFactor(targetMLTransInfo.defaultMaxTransposeUnrollFactor);
  } else {
    // Limit unrolling to the default target maximum.
    limitUnrollFactor(targetMLTransInfo.defaultMaxUnrollFactor);
  }

  return minTileSizes;
}

// Reduces the number of workgroups in cases where we are dividing the work too
// much. Over-provision the number of workgroups to twice the number of
// threads.
static void reduceDistributionWorkgroups(
    ArrayRef<int64_t> workload, SmallVectorImpl<int64_t> &distributedTileSizes,
    std::optional<ArrayRef<int64_t>> maxTileSizes = std::nullopt,
    std::optional<ArrayRef<int64_t>> vectorSizeHints = std::nullopt) {
  assert(workload.size() == distributedTileSizes.size());
  SmallVector<int64_t> numWorkgroupsPerDim(workload.size(), 1);
  for (auto [idx, value] : llvm::enumerate(workload)) {
    if (distributedTileSizes[idx] == 0 || ShapedType::isDynamic(value)) {
      continue;
    }
    numWorkgroupsPerDim[idx] =
        llvm::divideCeil(value, distributedTileSizes[idx]);
  }

  int64_t numWorkgroupsLimit = 2 * clNumberOfRuntimeThreads;
  int64_t numWorkgroups =
      std::accumulate(numWorkgroupsPerDim.begin(), numWorkgroupsPerDim.end(),
                      1LL, std::multiplies<int64_t>{});
  unsigned currDim = workload.size();
  while (numWorkgroups > numWorkgroupsLimit && currDim > 0) {
    unsigned index = currDim - 1;
    int64_t currSize = distributedTileSizes[index];
    if (ShapedType::isDynamic(workload[index]) || currSize == 0 ||
        (maxTileSizes && currSize >= maxTileSizes.value()[index]) ||
        currSize >= workload[index]) {
      currDim--;
      continue;
    }

    int64_t newSize = std::min<int64_t>(currSize * 2, workload[index]);
    int64_t vectorSize = vectorSizeHints ? vectorSizeHints.value()[index] : 0;

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
      numWorkgroupsPerDim[index] = nwg;
      numWorkgroups *= nwg;
    } else {
      currDim--;
    }
  }

  // Final fixup for dividing workload evenly.
  for (auto i : llvm::seq<unsigned>(0, distributedTileSizes.size())) {
    if (distributedTileSizes[i] == 0 || ShapedType::isDynamic(workload[i])) {
      continue;
    }

    int64_t nwg = llvm::divideCeil(workload[i], distributedTileSizes[i]);
    int64_t newSize = llvm::divideCeil(workload[i], nwg);

    // Chech if it's the ideal size with vector size hint. And skip if the new
    // size will break the ideal size.
    int64_t vectorSize = vectorSizeHints ? vectorSizeHints.value()[i] : 0;
    if (vectorSize > 1 &&
        (newSize % vectorSize != 0 || workload[i] % newSize != 0)) {
      continue;
    }

    distributedTileSizes[i] = newSize;
  }
}

/// Returns the default tile sizes to use for the loops that are distributed.
static SmallVector<int64_t>
getDefaultDistributionTileSizes(ArrayRef<int64_t> lbs, ArrayRef<int64_t> ubs,
                                ArrayRef<int64_t> minTileSizes,
                                ArrayRef<int64_t> maxTileSizes,
                                ArrayRef<int64_t> vectorSizeHints) {
  assert(lbs.size() == ubs.size() && lbs.size() == minTileSizes.size() &&
         lbs.size() == maxTileSizes.size() &&
         "expected all vectors to be of equal size");

  size_t numDims = lbs.size();
  // Set all the distribution tile sizes to zero if thread distribution is
  // disabled.
  if (clDisableDistribution) {
    return SmallVector<int64_t>(numDims, 0);
  }

  SmallVector<int64_t> distributedTileSizes(numDims, 1);
  SmallVector<int64_t> workload(numDims, 1);
  for (auto i : llvm::seq<size_t>(0, numDims)) {
    if (maxTileSizes[i] == 0 || ShapedType::isDynamic(lbs[i]) ||
        ShapedType::isDynamic(ubs[i])) {
      distributedTileSizes[i] = maxTileSizes[i];
      workload[i] = ShapedType::kDynamic;
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
      candidateTileSize = std::max<int64_t>(
          llvm::bit_floor<uint64_t>(targetSize), minTileSizes[i]);
    }

    // Limit the workload per workgroup to the default being the max to keep the
    // work per invocation reasonable.
    distributedTileSizes[i] =
        std::min<int64_t>(candidateTileSize, maxTileSizes[i]);
  }

  reduceDistributionWorkgroups(workload, distributedTileSizes, maxTileSizes,
                               vectorSizeHints);

  return distributedTileSizes;
}

/// Returns the nearest power of two of `size` if `predicate` is true.
/// Otherwise, returns `size`.
static int64_t roundUpToPow2(int64_t size, bool predicate) {
  if (!predicate) {
    return size;
  }
  assert(size > 0 && "Negative size");
  return llvm::PowerOf2Ceil(size);
}

/// Computes the maximum tile size that can be used to distribute a dimension
/// based on its number of iterations and the native vector size used of the
/// target. The resulting tile size will be a multiple of the provided vector
/// size, except when `allowIncompleteTile` is set to true.
static int64_t getMaxDistributionTileSize(int64_t lb, int64_t ub,
                                          int64_t maxSize, int64_t vectorSize,
                                          bool allowIncompleteTile = false) {
  if (ShapedType::isDynamic(ub) || ShapedType::isDynamic(lb)) {
    return maxSize;
  }
  int64_t numIters = ub - lb;
  if (numIters <= maxSize && numIters < vectorSize) {
    return numIters;
  }

  int64_t scaledUB = std::min(maxSize, numIters) / vectorSize * vectorSize;
  for (int64_t i = scaledUB; i > 0; i -= vectorSize) {
    if (numIters % i == 0) {
      return i;
    }
  }
  if (allowIncompleteTile) {
    // Set bound to half to avoid too many workgroup.
    int64_t start = std::min(maxSize, numIters);
    int64_t end = start / 2;
    for (int64_t i = start; i >= end; --i) {
      if (numIters % i == 0) {
        return i;
      }
    }
    return maxSize;
  }
  // If it can't be a multiple of `vectorSize`, let's choose a factor of
  // `numIters` sizes heuristically.
  int64_t start = std::min(maxSize, numIters);
  for (int64_t i = start; i > 0; --i) {
    if (numIters % i == 0) {
      return i;
    }
  }
  return 1;
}

/// Computes the maximum tile size that can be used to vectorize (or unroll) a
/// dimension based on its number of elements and the native vector size of
/// the target. If `enforcePowerOfTwo` is set to true, the resulting tile size
/// will be a power of two.
static int64_t getMaxVectorTileSize(int64_t numElem, int64_t tileSize,
                                    int64_t vectorSize,
                                    bool enforcePowerOfTwo = false) {
  if (ShapedType::isDynamic(numElem)) {
    return roundUpToPow2(tileSize, enforcePowerOfTwo);
  }
  if (numElem <= tileSize && numElem < vectorSize) {
    return roundUpToPow2(numElem, enforcePowerOfTwo);
  }

  // Return the largest suitable power of two if power of two is enforced.
  if (enforcePowerOfTwo) {
    return roundUpToPow2(std::min(tileSize, numElem), enforcePowerOfTwo);
  }

  // Try to find a tile size that is multiple of the vector size.
  int64_t scaledUB = std::min(tileSize, numElem) / vectorSize * vectorSize;
  for (int64_t i = scaledUB; i > 0; i -= vectorSize) {
    if (numElem % i == 0) {
      return i;
    }
  }

  // If it can't be a multiple of `vectorSize`, let's choose a factor of
  // `numElem` sizes heuristically.
  int64_t start = std::min(tileSize, numElem);
  for (int64_t i = start; i > 0; --i) {
    if (numElem % i == 0) {
      return i;
    }
  }

  return 1;
}

/// Struct that holds factors for heuristic distribution tile sizes selection.
/// The `minTileSizes`, `maxTileSizes` and `vectorSizeHints` can be empty or
/// as many as the number of loops.
struct DistributionHeuristicConfig {
  // TODO(hanchung): Remove `allowIncompleteTile` option after codegen can
  // vectorize all the shapes. Allowing incomplete tile is critical for odd
  // shapes (e.g., some dim sizes could be prime number).
  bool allowIncompleteTile = false;

  SmallVector<int64_t> minTileSizes;
  SmallVector<int64_t> maxTileSizes;

  // On the dimensions where the hints != 1, it will try to find the tile sizes
  // which are multipliers of the hints.
  SmallVector<int64_t> vectorSizeHints;
};

/// Returns the tile size to use for distribution. The `op` needs to be a
/// TilingInterface op.
static SmallVector<int64_t>
getDefaultDistributedLevelTileSizes(Operation *op,
                                    const DistributionHeuristicConfig &config) {
  SmallVector<int64_t> lbs, ubs;
  getRangeBounds(cast<TilingInterface>(op), lbs, ubs);
  int64_t numLoops = lbs.size();

  assert(
      (config.minTileSizes.empty() || config.minTileSizes.size() == numLoops) &&
      "min tile sizes should be empty or equal to the number of loops");
  assert(
      (config.maxTileSizes.empty() || config.maxTileSizes.size() == numLoops) &&
      "max tile sizes should be empty or equal to the number of loops");
  assert((config.vectorSizeHints.empty() ||
          config.vectorSizeHints.size() == numLoops) &&
         "vector size hints should be empty or equal to the number of loops");

  // Only set values when the loop is partitionable.
  SmallVector<int64_t> adjustedMinTileSizes(numLoops, 0);
  SmallVector<int64_t> adjustedMaxTileSizes(numLoops, 0);
  SmallVector<int64_t> adjustedVectorSizeHints(numLoops, 1);
  SmallVector<unsigned> partitionableLoops =
      cast<PartitionableLoopsInterface>(op).getPartitionableLoops(
          kNumMaxParallelDims);
  for (auto i : partitionableLoops) {
    adjustedMinTileSizes[i] =
        config.minTileSizes.empty() ? 1 : config.minTileSizes[i];
    adjustedMaxTileSizes[i] = config.maxTileSizes.empty()
                                  ? clDefaultDistTileSize
                                  : config.maxTileSizes[i];
    adjustedVectorSizeHints[i] =
        config.vectorSizeHints.empty() ? 1 : config.vectorSizeHints[i];
  }

  LLVM_DEBUG(KD_DBGS() << "Adjusted min tile sizes: " << adjustedMinTileSizes
                       << "\n");
  LLVM_DEBUG(KD_DBGS() << "Adjusted max tile sizes: " << adjustedMaxTileSizes
                       << "\n");
  LLVM_DEBUG(KD_DBGS() << "Adjusted vector size hints: "
                       << adjustedVectorSizeHints << "\n");

  SmallVector<int64_t> distributedTileSizes = getDefaultDistributionTileSizes(
      lbs, ubs, adjustedMinTileSizes, adjustedMaxTileSizes,
      adjustedVectorSizeHints);

  LLVM_DEBUG(KD_DBGS() << "Distributed tile sizes before fixups: "
                       << distributedTileSizes << "\n");

  // Final fix up of the tile sizes to make sure that they divide the problem
  // size to make it vectorizable.
  for (auto i : llvm::seq<unsigned>(0, distributedTileSizes.size())) {
    if (!distributedTileSizes[i])
      continue;
    distributedTileSizes[i] = getMaxDistributionTileSize(
        lbs[i], ubs[i], distributedTileSizes[i], adjustedMinTileSizes[i],
        config.allowIncompleteTile);
  }
  LLVM_DEBUG(KD_DBGS() << "Distributed tile sizes after fixups: "
                       << distributedTileSizes << "\n");
  return distributedTileSizes;
}

/// Splits the tile sizes in `parallelSizes` into `reductionSizes` for the
/// reduction loops.
static void splitParallelAndReductionTiles(
    linalg::LinalgOp op, SmallVectorImpl<int64_t> &parallelSizes,
    SmallVectorImpl<int64_t> &reductionSizes,
    SmallVectorImpl<bool> *parallelScalableFlags = nullptr,
    SmallVectorImpl<bool> *reductionScalableFlags = nullptr) {
  reductionSizes.assign(parallelSizes.begin(), parallelSizes.end());
  if (reductionScalableFlags) {
    assert(parallelScalableFlags && "expected parallel scalable flags!");
    reductionScalableFlags->assign(parallelScalableFlags->begin(),
                                   parallelScalableFlags->end());
  }
  for (auto [index, iteratorType] :
       llvm::enumerate(op.getIteratorTypesArray())) {
    if (iteratorType == utils::IteratorType::parallel) {
      reductionSizes[index] = 0;
      if (reductionScalableFlags)
        (*reductionScalableFlags)[index] = false;
    } else {
      parallelSizes[index] = 0;
      if (parallelScalableFlags)
        (*parallelScalableFlags)[index] = false;
    }
  }
}

static void setAlwaysVectorizeSizes(linalg::LinalgOp op,
                                    SmallVectorImpl<int64_t> &parallelSizes,
                                    SmallVectorImpl<int64_t> &reductionSizes) {
  SmallVector<int64_t> staticLoopRanges = op.getStaticLoopRanges();
  for (auto [index, valuePair] : llvm::enumerate(
           llvm::zip_equal(staticLoopRanges, op.getIteratorTypesArray()))) {
    auto [size, iterType] = valuePair;
    if (!ShapedType::isDynamic(size))
      continue;
    if (iterType == utils::IteratorType::parallel) {
      parallelSizes[index] = 1;
    } else {
      reductionSizes[index] = 1;
    }
  }

  LLVM_DEBUG(KD_DBGS() << "Set always-vectorize parallel sizes: "
                       << parallelSizes << "\n");
  LLVM_DEBUG(KD_DBGS() << "Set always-vectorize reduction sizes: "
                       << reductionSizes << "\n");
}

static void
setVectorSizesForDynamicShapes(linalg::LinalgOp op,
                               VectorPreProcStrategy vecPreProcStrategy,
                               SmallVectorImpl<int64_t> &parallelSizes,
                               SmallVectorImpl<int64_t> &reductionSizes) {
  // Masking doesn't need any dim set to 1.
  if (vecPreProcStrategy == VectorPreProcStrategy::Masking) {
    return;
  }

  SmallVector<int64_t> origParallelSizes(parallelSizes.begin(),
                                         parallelSizes.end());
  SmallVector<int64_t> origReductionSizes(reductionSizes.begin(),
                                          reductionSizes.end());
  setAlwaysVectorizeSizes(op, parallelSizes, reductionSizes);

  if (llvm::all_of(parallelSizes, [](int64_t size) { return size <= 1; })) {
    // Make sure we vectorize at least the first innermost parallel dim with a
    // vector size greater than one.
    for (int i = origParallelSizes.size() - 1; i >= 0; --i) {
      if (origParallelSizes[i] > 1) {
        parallelSizes[i] = origParallelSizes[i];
        break;
      }
    }
  } else if (llvm::all_of(reductionSizes,
                          [](int64_t size) { return size <= 1; })) {
    // Make sure we vectorize at least the first innermost reduction dim with a
    // vector size greater than one.
    for (int i = origReductionSizes.size() - 1; i >= 0; --i) {
      if (origReductionSizes[i] > 1) {
        reductionSizes[i] = origReductionSizes[i];
        break;
      }
    }
  }

  LLVM_DEBUG(KD_DBGS() << "Parallel sizes for dynamic sizes: " << parallelSizes
                       << "\n");
  LLVM_DEBUG(KD_DBGS() << "Reduction sizes for dynamic sizes: "
                       << reductionSizes << "\n");
  return;
}

/// Returns the default cache-level tile sizes for a matmul op and a specific
/// target. There shouldn't be proper heuristics here, just fixed values.
static SmallVector<int64_t> getDefaultMatmulCacheSizes(linalg::LinalgOp op,
                                                       bool isQuantized) {
  unsigned numLoops = op.getNumLoops();
  SmallVector<int64_t> noCacheLevelTiling(numLoops, 0);

  // Cache-level tiling is only supported for 2-D matmuls.
  if (numLoops < 3) {
    return noCacheLevelTiling;
  }

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (isX86(targetAttr)) {
    if (isQuantized) {
      return noCacheLevelTiling;
    }

    SmallVector<int64_t> defaultCacheTileSizes(numLoops - 3, 0);
    defaultCacheTileSizes.append({8, 128, 16});
    return defaultCacheTileSizes;
  }

  return noCacheLevelTiling;
}

static LogicalResult setMatmulPeelingRootConfig(
    func::FuncOp entryPointFn, linalg::ContractionOpInterface op,
    ArrayRef<int64_t> distTileSizes, ArrayRef<int64_t> cacheTileSizes,
    ArrayRef<int64_t> vecTileSizes, int vectorSize) {
  // The tiling for parallel dims (M and N) and reduction dim (K) should be
  // separated, so we move K dim from parallel tile sizes to reduction tile
  // sizes.
  int64_t numTilingDims = vecTileSizes.size();
  SmallVector<int64_t> cacheParallelTileSizes(cacheTileSizes.begin(),
                                              cacheTileSizes.end());
  SmallVector<int64_t> cacheReductionTileSizes(numTilingDims, 0);
  std::swap(cacheParallelTileSizes.back(), cacheReductionTileSizes.back());

  SmallVector<int64_t> vectorParallelTileSizes(vecTileSizes.begin(),
                                               vecTileSizes.end());
  SmallVector<int64_t> vectorReductionTileSizes(numTilingDims, 0);
  std::swap(vectorParallelTileSizes.back(), vectorReductionTileSizes.back());

  TileSizesListType tileSizes = {
      SmallVector<int64_t>(distTileSizes), cacheParallelTileSizes,
      cacheReductionTileSizes, vectorParallelTileSizes,
      vectorReductionTileSizes};
  // No need for tiling inner parallel dims.
  tileSizes.emplace_back(numTilingDims, 0);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes,
      DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert);
}

static LogicalResult
setMatmulRootConfig(func::FuncOp entryPointFn,
                    linalg::ContractionOpInterface op,
                    const TileSizesListTypeRef inputTileSizes,
                    const ScalableTileFlagsListTypeRef inputScalableTileFlags,
                    int vectorSize, VectorPreProcStrategy vecPreProcStrategy) {
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();

  // The tiling for parallel dims and reduction dims are separated.
  const SmallVectorImpl<int64_t> &vecTileSizes = inputTileSizes.back();
  const SmallVectorImpl<bool> &vecScalableDims = inputScalableTileFlags.back();
  SmallVector<int64_t> parallelTileSizes;
  SmallVector<bool> parallelScalableFlags;
  int numScalableDims = llvm::count(vecScalableDims, true);

  for (auto [index, tileSize] : llvm::enumerate(vecTileSizes)) {
    int64_t sz = tileSize;
    bool isScalable = vecScalableDims[index];
    // The backend struggles to legalize non-power-of-two scalable vectors.
    bool enforcePowerOfTwo = isScalable;

    // Ad-hoc: Don't attempt to resize scalable tiles when numScalableDims >= 2.
    // For ArmSME (the only current user of 2D scalable vectors), tile sizes
    // must match SME tiles (and cannot be arbitrarily resized).
    if (sz != 0 && (numScalableDims < 2 || !isScalable)) {
      sz = getMaxVectorTileSize(
          /*numElem=*/shape[index],
          /*tileSize=*/sz, vectorSize, enforcePowerOfTwo);
    }
    parallelTileSizes.push_back(sz);
    // 1x scalable vectors e.g. vector<[1]xty> are also poorly supported, so
    // fallback to fixed vectorization if they occur:
    parallelScalableFlags.push_back(sz > 1 ? isScalable : false);
  }
  SmallVector<int64_t> reductionTileSizes;
  SmallVector<bool> reductionScalableFlags;
  splitParallelAndReductionTiles(
      cast<linalg::LinalgOp>(op.getOperation()), parallelTileSizes,
      reductionTileSizes, &parallelScalableFlags, &reductionScalableFlags);

  if (vecPreProcStrategy == VectorPreProcStrategy::None) {
    setVectorSizesForDynamicShapes(cast<linalg::LinalgOp>(op.getOperation()),
                                   vecPreProcStrategy, parallelTileSizes,
                                   reductionTileSizes);
  }

  // Ensure there's no zero scalable dims.
  int64_t numTilingDims = parallelTileSizes.size();
  for (unsigned i = 0; i < numTilingDims; i++) {
    if (reductionTileSizes[i] == 0)
      reductionScalableFlags[i] = false;
    if (parallelTileSizes[i] == 0)
      parallelScalableFlags[i] = false;
  }

  TileSizesListType newTileSizes;
  // Copy all the tile size levels except the distribution which will be split
  // into parallel and reduction.
  std::copy(inputTileSizes.begin(), inputTileSizes.end() - 1,
            std::back_inserter(newTileSizes));
  newTileSizes.push_back(parallelTileSizes);
  newTileSizes.push_back(reductionTileSizes);
  // No need for tiling inner parallel dims.
  newTileSizes.emplace_back(numTilingDims, 0);

  // Mirror the same layout for the scalable dims.
  ScalableTileFlagsListType newScalableTileFlags;
  std::copy(inputScalableTileFlags.begin(), inputScalableTileFlags.end() - 1,
            std::back_inserter(newScalableTileFlags));
  newScalableTileFlags.push_back(parallelScalableFlags);
  newScalableTileFlags.push_back(reductionScalableFlags);
  // No scalable inner parallel dims.
  newScalableTileFlags.emplace_back(numTilingDims, false);

  LLVM_DEBUG(KD_DBGS() << "Final tile sizes for contraction: " << newTileSizes
                       << "\n"
                       << "Final tile scalable flags for contraction: "
                       << newScalableTileFlags << "\n");

  auto pipeline = DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
  if (vecPreProcStrategy == VectorPreProcStrategy::Peeling) {
    pipeline = DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert;
  }
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, op, newTileSizes,
                                               newScalableTileFlags, pipeline);
}

/// Returns default hard-coded vector sizes for a give target. No smartness
/// should be introduced in this utility.
static void getDefaultMatmulVectorSizes(
    linalg::LinalgOp op, SmallVectorImpl<int64_t> &sizes,
    SmallVectorImpl<bool> &scalableSizeFlags, int64_t vectorSize) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (isX86(targetAttr)) {
    sizes.append({8, 32, 16});
    return;
  }

  if (isAArch64(targetAttr)) {
    sizes.append({8, 16, 1});

    // Specialisation for SVE.
    if (hasAnySVEFeature(targetAttr)) {
      // Mark middle dimensions as scalable, so sizes are (8, [16], 1).
      scalableSizeFlags.append({false, true, false});
    }
    return;
  }

  if (isRISCV(targetAttr)) {
    // RISC-V natively supports scalar x vector operations so we don't have to
    // vectorize dimension k. Vectorizing dimension k results in a vector load
    // and a sequence of vrgather ops to implemement the broadcast explicitly.
    // We should tile and/or unroll that dimension without vectorization, which
    // is not possible right now.
    sizes.append({8, 32, 1});
    return;
  }

  // Fallback to use vectorSize for unknown arch.
  sizes.append(3, vectorSize);
  return;
}

/// Checks if the input and output types of a linalg op are the same, and if so
/// returns the type. Otherwise, returns failure.
static FailureOr<Type> nonWideningLinalgElementType(linalg::LinalgOp op) {
  SmallVector<Type, 3> inputAndOutputElementTypes;
  for (Value v :
       llvm::concat<Value>(op.getRegionInputArgs(), op.getRegionOutputArgs())) {
    inputAndOutputElementTypes.push_back(v.getType());
  }
  assert(!inputAndOutputElementTypes.empty() &&
         "expected linalg op to have input and output types");
  if (!llvm::all_equal(inputAndOutputElementTypes))
    return failure();
  return inputAndOutputElementTypes[0];
}

/// Utility to compute the tile sizes for AArch64 SME. Unlike other targets, the
/// tile sizes picked here must exactly match the SME hardware virtual tiles, as
/// there is currently no support for lowering non-standard shapes.
static void
getMatmulAArch64SMEVectorSizes(linalg::LinalgOp op,
                               SmallVectorImpl<int64_t> &sizes,
                               SmallVectorImpl<bool> &scalableSizeFlags) {
  // Double-check the operation is one that is supported for lowering to ArmSME.
  if (!llvm::isa<linalg::MatmulOp, linalg::MatmulTransposeAOp>(op))
    return;

  auto elementType = nonWideningLinalgElementType(op);
  if (failed(elementType))
    return;

  if (elementType->isF32()) {
    sizes.append({4, 4, 1});
    scalableSizeFlags.append({true, true, false});
  }

  if (elementType->isF64()) {
    sizes.append({2, 2, 1});
    scalableSizeFlags.append({true, true, false});
  }

  // TODO(macdue): Other element types (there is little support for anything
  // other than f32 and f64 yet).
}

/// Main utility to compute the vectorization/unrolling tile sizes.
static SizesAndScalableFlags getMatmulVectorSizes(func::FuncOp entryPointFn,
                                                  linalg::LinalgOp op,
                                                  int64_t vectorSize,
                                                  bool isQuantized) {
  SmallVector<int64_t> matmulTileSizes;
  SmallVector<bool> matmulScalableFlags;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);

  // TODO: Compute vector tile sizes using heuristics.

  if (isAArch64(targetAttr) && hasSMEFeature(targetAttr)) {
    // Note: This may not pick any sizes (which will fallback to the default
    // SVE) sizes below.
    getMatmulAArch64SMEVectorSizes(op, matmulTileSizes, matmulScalableFlags);
  }

  // Get default hard-coded tile sizes if we couldn't compute anything better.
  if (matmulTileSizes.empty()) {
    getDefaultMatmulVectorSizes(op, matmulTileSizes, matmulScalableFlags,
                                vectorSize);
  }
  // Pad the scalable flags with false to match the tile sizes.
  matmulScalableFlags.resize(matmulTileSizes.size());

  SmallVector<int64_t> tileSizes;
  SmallVector<bool> scalableTileFlags;
  unsigned numLoops = op.getNumLoops();
  if (numLoops > 3) {
    tileSizes.append(numLoops - 3, 1);
    tileSizes.append(matmulTileSizes.begin(), matmulTileSizes.end());
    scalableTileFlags.append(numLoops - 3, false);
    scalableTileFlags.append(matmulScalableFlags.begin(),
                             matmulScalableFlags.end());
  } else {
    tileSizes.append(matmulTileSizes.begin() + (3 - numLoops),
                     matmulTileSizes.end());
    scalableTileFlags.append(matmulScalableFlags.begin() + (3 - numLoops),
                             matmulScalableFlags.end());
  }

  // For proper 2-D or higher order matmuls, make sure we don't use a tile size
  // greater than the static dim size for dims that are only unrolled, i.e., N
  // and batch dims.

  int numScalableDims = llvm::count(scalableTileFlags, true);
  SmallVector<int64_t> staticShape = op.getStaticLoopRanges();
  if (numLoops >= 3) {
    for (int i = 0; i < (numLoops - 2); ++i) {
      int64_t dimSize = staticShape[i];
      int64_t tileSize = tileSizes[i];
      if (tileSize == 0 || ShapedType::isDynamic(dimSize)) {
        continue;
      }
      // Ad-hoc: Don't attempt to resize scalable tiles when numScalableDims
      // >= 2. For ArmSME (the only current user of 2D scalable vectors), tile
      // sizes must match SME tiles (and cannot be arbitrarily resized).
      if (numScalableDims >= 2 && scalableTileFlags[i]) {
        continue;
      }
      tileSizes[i] = std::min<int64_t>(tileSize, dimSize);
    }
  }

  LLVM_DEBUG(KD_DBGS() << "Matmul vector sizes: " << tileSizes << "\n");
  LLVM_DEBUG(KD_DBGS() << "Matmul vector scalable flags: " << scalableTileFlags
                       << "\n");
  return std::make_pair(tileSizes, scalableTileFlags);
}

/// Adjust cache-level tile sizes based on the op shape.
static SmallVector<int64_t>
getMatmulCacheTileSizesForShape(ArrayRef<int64_t> inputTileSizes,
                                ArrayRef<int64_t> inputShape) {
  // Make sure the tile sizes are not larger than the dim sizes.
  int numDims = inputShape.size();
  SmallVector<int64_t> outputTileSizes(numDims);
  for (int i = 0, end = numDims; i < end; ++i) {
    outputTileSizes[i] =
        (ShapedType::isDynamic(inputShape[i]) || inputShape[i] == 0)
            ? inputTileSizes[i]
            : std::min(inputTileSizes[i], inputShape[i]);
  }

  // TODO: Enable caching for reduction dims.
  outputTileSizes.back() = 0;

  return outputTileSizes;
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult
setRootConfig(func::FuncOp entryPointFn,
              linalg::ContractionOpInterface contractionOp) {
  assert(!getLoweringConfig(contractionOp) &&
         "expected lowering_config is not set");
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
  auto lhsShapedType = llvm::cast<ShapedType>(contractionOp.lhs().getType());
  auto rhsShapedType = llvm::cast<ShapedType>(contractionOp.rhs().getType());
  auto resShapedType =
      llvm::cast<ShapedType>(linalgOp.getDpsInitOperand(0)->get().getType());
  int64_t vectorSize = getVectorSize(entryPointFn, lhsShapedType);
  vectorSize = std::min(vectorSize, getVectorSize(entryPointFn, rhsShapedType));
  vectorSize = std::min(vectorSize, getVectorSize(entryPointFn, resShapedType));
  bool isQuantized =
      lhsShapedType.getElementType() != resShapedType.getElementType();

  auto [vecTileSizes, vecScalableFlags] =
      getMatmulVectorSizes(entryPointFn, linalgOp, vectorSize, isQuantized);

  DistributionHeuristicConfig distConfig;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);

  // Use the default distribution for the matmul loops.
  int64_t defaultMaxSize = clDefaultDistTileSize;
  if (isX86(targetAttr) || isRISCV(targetAttr) ||
      (isAArch64(targetAttr) && hasAnySVEFeature(targetAttr))) {
    defaultMaxSize = 128;
  }

  bool isBM = isa<linalg::BatchMatmulOp>(contractionOp.getOperation());
  distConfig.maxTileSizes.resize(numLoops, defaultMaxSize);
  if (isBM) {
    distConfig.maxTileSizes[0] = 1;
  }

  // Compute cache-level tile sizes. Cache a dimension only if there are
  // enough iterations.
  SmallVector<int64_t> cacheTileSizes;
  cacheTileSizes = getDefaultMatmulCacheSizes(linalgOp, isQuantized);
  cacheTileSizes = getMatmulCacheTileSizesForShape(
      cacheTileSizes, linalgOp.getStaticLoopRanges());

  // Choose the next non-zero tile size immediately after the distribution
  // level to help compute the distribution tile sizes.
  SmallVector<int64_t> distTileSizes;
  auto vecPreProcStrategy = getVectorPreProcStrategy(linalgOp);
  bool usePeelingPipeline =
      vecPreProcStrategy == VectorPreProcStrategy::Peeling;

  LLVM_DEBUG(KD_DBGS() << "Vector pre-processing strategy: "
                       << vecPreProcStrategy << "\n");

  if (usePeelingPipeline && isX86(targetAttr)) {
    // It's inspired from https://github.com/iree-org/iree-llvm-sandbox repo.
    // Sandbox has [[288, 128, 512], [12, 32, 1]] setup. We scale 288 to 192
    // because 288/12*8=192
    if (numLoops == 3) {
      distConfig.maxTileSizes[0] = 192;
      distConfig.maxTileSizes[1] = 128;
    }
  }

  distConfig.minTileSizes = vecTileSizes;
  distConfig.allowIncompleteTile = true;
  distConfig.vectorSizeHints.resize(numLoops, vectorSize);
  if (isBM) {
    distConfig.vectorSizeHints[0] = 1;
  }
  distTileSizes = getDefaultDistributedLevelTileSizes(linalgOp, distConfig);

  // TODO: We set cache tile sizes to the distribution sizes for now (no-op) to
  // make sure there are no performance changes. This will let us change the
  // distribution sizes while still preserving the cache behavior of the
  // original sizes. When we set proper sizes, we should call again
  // `getMatmulCacheTileSizesForShape(cacheTileSizes, distTileSizes);` here as
  // the `getDefaultDistributedLevelTileSizes` above may return sizes that are
  // smaller than `minTileSizes`, so we have to adjust the cache sizes again.
  cacheTileSizes = distTileSizes;

  LLVM_DEBUG(KD_DBGS() << "Distribution tile sizes: " << distTileSizes << "\n");
  LLVM_DEBUG(KD_DBGS() << "Cache tile sizes: " << cacheTileSizes << "\n");
  LLVM_DEBUG(KD_DBGS() << "Vector tile sizes: " << vecTileSizes << "\n");
  LLVM_DEBUG(KD_DBGS() << "Vector scalable tile flags: " << vecScalableFlags
                       << "\n");
  LLVM_DEBUG(KD_DBGS() << "Vector size: " << vectorSize << "\n");

  if (usePeelingPipeline) {
    // TODO: Use scalable vector sizes.
    return setMatmulPeelingRootConfig(entryPointFn, contractionOp,
                                      distTileSizes, cacheTileSizes,
                                      vecTileSizes, vectorSize);
  }

  SmallVector<bool> distScalableTileFlags(distTileSizes.size(), false);
  TileSizesListType tileSizes = {distTileSizes, vecTileSizes};
  ScalableTileFlagsListType scalableTileFlags = {distScalableTileFlags,
                                                 vecScalableFlags};
  return setMatmulRootConfig(entryPointFn, contractionOp, tileSizes,
                             scalableTileFlags, vectorSize, vecPreProcStrategy);
}

static TileSizesListType getMmt4dTileSizes(linalg::LinalgOp op) {
  DistributionHeuristicConfig distConfig;
  distConfig.allowIncompleteTile = true;
  distConfig.minTileSizes.resize(op.getNumLoops(), 0);
  distConfig.maxTileSizes.resize(op.getNumLoops(), 0);

  Value lhs = op.getDpsInputs()[0];
  Value rhs = op.getDpsInputs()[1];
  ShapedType lhsType = cast<ShapedType>(lhs.getType());
  ShapedType rhsType = cast<ShapedType>(rhs.getType());
  int mmt4dDimBase = 0;
  if (isa<linalg::BatchMmt4DOp>(op)) {
    mmt4dDimBase = 1;
    distConfig.minTileSizes[0] = 1;
    distConfig.maxTileSizes[0] = 1; // Force batch dimension tile size 1.
  }
  distConfig.minTileSizes[mmt4dDimBase + 0] = 1;
  distConfig.minTileSizes[mmt4dDimBase + 1] = 1;
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  int64_t M1 = lhsShape[mmt4dDimBase + 0];
  int64_t N1 = rhsShape[mmt4dDimBase + 0];
  int64_t K1 = lhsShape[mmt4dDimBase + 1];
  int64_t M0 = lhsShape[mmt4dDimBase + 2];
  int64_t N0 = rhsShape[mmt4dDimBase + 2];
  int64_t K0 = lhsShape[mmt4dDimBase + 3];
  // Unfortunately we have to compute some tile size at compile-time, even
  // though that can't be done meaningfully in general, unless specializing the
  // compilation to fine details of the runtime workload including number of
  // threads and cache sizes. Another thing that we need to know and can't
  // really know at compile time is the values of dynamic sizes. Here we have to
  // guess a reasonable default for the reduction dimension size.
  int64_t reductionSize = ShapedType::isDynamic(K1) ? 1024 : K0 * K1;
  auto getMatmulTileSize = [](int64_t targetTileBytes, int bitWidth,
                              int64_t reductionSize, int64_t tile0Size) {
    int64_t targetRhsTileElems = targetTileBytes * 8 / bitWidth;
    int64_t targetRhsTileNSize = targetRhsTileElems / reductionSize;
    int64_t tileSize = llvm::divideCeil(targetRhsTileNSize, tile0Size);
    tileSize = std::max<int64_t>(tileSize, 1);
    return tileSize;
  };
  int64_t tileBytes =
      (M1 == 1 || N1 == 1) ? clNarrowMatmulTileBytes : clGeneralMatmulTileBytes;
  distConfig.maxTileSizes[mmt4dDimBase + 0] =
      M1 == 1 ? 1
              : getMatmulTileSize(tileBytes, lhsType.getElementTypeBitWidth(),
                                  reductionSize, M0);
  distConfig.maxTileSizes[mmt4dDimBase + 1] =
      N1 == 1 ? 1
              : getMatmulTileSize(tileBytes, rhsType.getElementTypeBitWidth(),
                                  reductionSize, N0);

  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(op, distConfig);
  SmallVector<int64_t> parallelTileSizes(op.getNumLoops(), 1);
  assert(parallelTileSizes.size() == mmt4dDimBase + 6);
  parallelTileSizes[mmt4dDimBase + 3] = M0;
  parallelTileSizes[mmt4dDimBase + 4] = N0;
  parallelTileSizes[mmt4dDimBase + 5] = K0;
  SmallVector<int64_t> reductionTileSizes;
  splitParallelAndReductionTiles(op, parallelTileSizes, reductionTileSizes);
  return {distTileSizes, parallelTileSizes, reductionTileSizes};
}

/// Sets the lowering configuration for dispatch region for linalg.mmt4d
/// root op
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::Mmt4DOp Mmt4dOp) {
  assert(!getLoweringConfig(Mmt4dOp) && "expected lowering_config is not set");
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, Mmt4dOp, getMmt4dTileSizes(Mmt4dOp),
      DispatchLoweringPassPipeline::Mmt4dTilingExpert);
}

/// Sets the lowering configuration for dispatch region for linalg.batch_mmt4d
/// root op
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::BatchMmt4DOp batchMmt4dOp) {
  assert(!getLoweringConfig(batchMmt4dOp) &&
         "expected lowering_config is not set");
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, batchMmt4dOp, getMmt4dTileSizes(batchMmt4dOp),
      DispatchLoweringPassPipeline::Mmt4dTilingExpert);
}

static bool isPackMatmulLHS(tensor::PackOp op) {
  // linalg.batch_matmul LHS shape
  if (op.getSourceRank() == 3 && op.getInnerDimsPos().size() == 2 &&
      op.getInnerDimsPos()[0] == 1 && op.getInnerDimsPos()[1] == 2) {
    return true;
  }
  // linalg.matmul LHS shape
  return op.getSourceRank() == 2 && op.getInnerDimsPos().size() == 2 &&
         op.getInnerDimsPos()[0] == 0 && op.getInnerDimsPos()[1] == 1;
}

/// Returns vectorization tile sizes for a pack op. It is driven by pack op
/// configurations and target CPU features.
static SmallVector<int64_t> getPackVectorTileSizes(func::FuncOp entryPointFn,
                                                   tensor::PackOp op) {
  SmallVector<int64_t> tileSizes(op.getSourceRank(), 1);
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  int64_t vectorSize = getVectorSize(entryPointFn, op.getSourceType());
  // TODO(#15421): Improve tile sizes selection for non f32 cases.
  if (op.getSourceType().getElementType().isF32() &&
      hasAVX512fFeature(targetAttr) && isPackMatmulLHS(op)) {
    tileSizes.back() = vectorSize;
  }
  return tileSizes;
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   tensor::PackOp op) {
  assert(!getLoweringConfig(op) && "expected lowering_config is not set");

  int64_t vectorSize = getVectorSize(entryPointFn, op.getSourceType());
  DistributionHeuristicConfig distConfig;
  distConfig.allowIncompleteTile = true;
  distConfig.vectorSizeHints.resize(op.getSourceRank(), 1);
  for (auto dim : op.getInnerDimsPos()) {
    distConfig.vectorSizeHints[dim] = vectorSize;
  }
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(op, distConfig);
  SmallVector<int64_t> workload(op.getSourceType().getShape());
  reduceDistributionWorkgroups(workload, distTileSizes,
                               /*maxTileSizes=*/std::nullopt,
                               distConfig.vectorSizeHints);

  // The default function aims to returns the number of workload per workgroup,
  // but it does not know that it is working on packed domain. We need to take
  // inner tile sizes into account and adjust the distribution tile sizes.
  SmallVector<int64_t> innerTiles = op.getStaticTiles();
  ArrayRef<int64_t> dimPos = op.getInnerDimsPos();
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    if (distTileSizes[pos] == 0 || ShapedType::isDynamic(size))
      continue;
    distTileSizes[pos] = distTileSizes[pos] / size;
    distTileSizes[pos] = std::max<int64_t>(distTileSizes[pos], 1);
  }

  SmallVector<int64_t> vecTileSizes = getPackVectorTileSizes(entryPointFn, op);
  TileSizesListType tileSizesList = {distTileSizes, vecTileSizes};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizesList,
      DispatchLoweringPassPipeline::CPUDataTiling);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   tensor::UnPackOp op) {
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(op, DistributionHeuristicConfig{});
  SmallVector<int64_t> workload(op.getDestType().getShape());
  reduceDistributionWorkgroups(workload, distTileSizes);

  // Fixup for making distTileSizes be multiple of inner_tile_sizes.
  SmallVector<int64_t> innerTiles = op.getStaticTiles();
  ArrayRef<int64_t> dimPos = op.getInnerDimsPos();
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    if (distTileSizes[pos] == 0 || ShapedType::isDynamic(size))
      continue;
    distTileSizes[pos] = llvm::alignTo(distTileSizes[pos], size);
  }

  SmallVector<int64_t> tileSizes(op.getDestRank(), 1);
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    tileSizes[pos] = ShapedType::isDynamic(size) ? 1 : size;
  }

  TileSizesListType tileSizesList = {distTileSizes, tileSizes};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizesList,
      DispatchLoweringPassPipeline::CPUDataTiling);
}

/// Sets the lowering configuration for dispatch region for linalg_ext.fft
/// root op.
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   IREE::LinalgExt::FftOp fftOp) {
  assert(!getLoweringConfig(fftOp) && "expected lowering_config is not set");
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(fftOp, DistributionHeuristicConfig{});
  auto rank = fftOp.getOperandRank();
  if (distTileSizes.size() >= rank && distTileSizes[rank - 1] != 0) {
    APInt value;
    if (matchPattern(fftOp.getStage(), m_ConstantInt(&value))) {
      distTileSizes[rank - 1] = 1ll << value.getSExtValue();
      distTileSizes[rank - 1] = std::max(
          distTileSizes[rank - 1], static_cast<int64_t>(clDefaultDistTileSize));
    } else {
      return fftOp.emitOpError("non-constant stage might not work for fft op");
    }
  }
  TileSizesListType tileSizes = {distTileSizes};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, fftOp, tileSizes, DispatchLoweringPassPipeline::CPUDefault);
}

static void setX86VectorTileSizes(linalg::GenericOp genericOp,
                                  unsigned numLoops,
                                  ArrayRef<int64_t> distTileSizes,
                                  ArrayRef<int64_t> minTileSizes,
                                  ArrayRef<int64_t> maxTileSizes,
                                  VectorPreProcStrategy vecPreProcStrategy,
                                  SmallVectorImpl<int64_t> &vecTileSizes) {
  vecTileSizes.append(numLoops, 0);
  SmallVector<int64_t> staticLoopRanges = genericOp.getStaticLoopRanges();
  for (auto loopNum : llvm::seq<unsigned>(0, numLoops)) {
    if (distTileSizes[loopNum]) {
      vecTileSizes[loopNum] = getMaxVectorTileSize(
          distTileSizes[loopNum], minTileSizes[loopNum], minTileSizes[loopNum],
          /*enforcePowerOfTwo=*/vecPreProcStrategy ==
              VectorPreProcStrategy::Masking);
    } else {
      // If the distribution tile size is zero, and static loop range is 0 as
      // well, set the tile sizes here to zero as well.
      vecTileSizes[loopNum] =
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
  if ((genericOp.getNumDpsInputs() != 1) || (genericOp.getNumDpsInits() != 1)) {
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

/// Sets the default lowering configuration for a generic op to use
/// CPUDoubleTilingExpert pipeline.
static LogicalResult
setDefaultGenericOpRootConfig(func::FuncOp entryPointFn,
                              linalg::GenericOp genericOp,
                              const LinalgOpInfo &linalgOpInfo,
                              const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");
  LLVM_DEBUG(KD_DBGS() << "Setting default generic op root configuration\n");

  // If there are no loops, there is nothing to do.
  unsigned numLoops = genericOp.getNumLoops();
  if (numLoops == 0) {
    return setOpConfigAndEntryPointFnTranslation(
        entryPointFn, genericOp, {{}},
        DispatchLoweringPassPipeline::CPUDefault);
  }

  DistributionHeuristicConfig distConfig;
  distConfig.minTileSizes = getMinTilingSizesForEachDim(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
  // For generic ops we'll use the default divided by 2 to control the stack
  // allocation limit See #9469 for example.
  distConfig.maxTileSizes.append(numLoops, clDefaultDistTileSize / 2);

  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(genericOp, distConfig);

  LLVM_DEBUG(KD_DBGS() << "Final tile sizes for distribution: " << distTileSizes
                       << "\n");

  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  LLVM_DEBUG(KD_DBGS() << "Vectorization pre-processing strategy "
                       << vecPreProcStrategy << "\n");

  // Set the next level tile sizes.
  SmallVector<int64_t> parallelTileSizes;
  SmallVector<int64_t> reductionTileSizes;
  setX86VectorTileSizes(genericOp, numLoops, distTileSizes,
                        distConfig.minTileSizes, distConfig.maxTileSizes,
                        vecPreProcStrategy, parallelTileSizes);
  splitParallelAndReductionTiles(genericOp, parallelTileSizes,
                                 reductionTileSizes);
  setVectorSizesForDynamicShapes(genericOp, vecPreProcStrategy,
                                 parallelTileSizes, reductionTileSizes);

  LLVM_DEBUG(KD_DBGS() << "Vectorization/unrolling tile sizes (parallel): "
                       << parallelTileSizes << "\n");
  LLVM_DEBUG(KD_DBGS() << "Vectorization/unrolling tile sizes (reduction): "
                       << reductionTileSizes << "\n");

  TileSizesListType tileSizes = {distTileSizes, parallelTileSizes,
                                 reductionTileSizes};
  // No need for tiling inner parallel dims.
  tileSizes.emplace_back(numLoops, 0);

  // For non-tensor based ops use the Buffer ops pipeline.
  DispatchLoweringPassPipeline passPipeline;
  if (genericOp.hasTensorSemantics()) {
    passPipeline =
        vecPreProcStrategy == VectorPreProcStrategy::Peeling
            ? DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert
            : DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
  } else {
    passPipeline = DispatchLoweringPassPipeline::CPUBufferOpsTileAndVectorize;
  }

  return setOpConfigAndEntryPointFnTranslation(entryPointFn, genericOp,
                                               tileSizes, passPipeline);
}

/// Set lowering info to be used by the transform dialect jitter.
static LogicalResult
setTransformStrategyRootConfig(func::FuncOp entryPointFn,
                               linalg::GenericOp genericOp,
                               const LinalgOpInfo &linalgOpInfo,
                               const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");
  if (!clEnableTransformDialectJit)
    return failure();
  cpu::CPUModel cpuModel;
  if (failed(
          cpu::matchAndSetReductionStrategy(entryPointFn, genericOp, cpuModel)))
    return failure();
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      entryPointFn->getContext(),
      IREE::Codegen::DispatchLoweringPassPipeline::TransformDialectCodegen);
  if (failed(setTranslationInfo(entryPointFn, translationInfo)))
    return failure();
  return success();
}

/// Sets the lowering configuration for a generic op implementing a
/// transposition to use CPUDoubleTilingExpert pipeline.
static LogicalResult
setTransposeLikeOpRootConfig(func::FuncOp entryPointFn,
                             linalg::GenericOp genericOp,
                             const LinalgOpInfo &linalgOpInfo,
                             const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  if (!hasAVX2Feature(targetAttr) || !isSupportedTransposeOp(genericOp)) {
    return failure();
  }

  unsigned numLoops = genericOp.getNumLoops();
  DistributionHeuristicConfig distConfig;
  distConfig.minTileSizes = getMinTilingSizesForEachDim(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
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
      getDefaultDistributedLevelTileSizes(genericOp, distConfig);

  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  LLVM_DEBUG(KD_DBGS() << "Vectorization pre-processing strategy "
                       << vecPreProcStrategy << "\n");

  // Set the next level tile sizes.
  SmallVector<int64_t> parallelTileSizes;
  setX86VectorTileSizes(genericOp, numLoops, distTileSizes,
                        distConfig.minTileSizes, distConfig.maxTileSizes,
                        vecPreProcStrategy, parallelTileSizes);

  TileSizesListType tileSizes = {distTileSizes, parallelTileSizes};
  // No need for tiling reduction dims and inner parallel dims.
  int64_t numTilingDims = parallelTileSizes.size();
  tileSizes.emplace_back(numTilingDims, 0);
  tileSizes.emplace_back(numTilingDims, 0);

  // For non-tensor based ops use the Buffer ops pipeline.
  auto passPipeline =
      genericOp.hasTensorSemantics()
          ? DispatchLoweringPassPipeline::CPUDoubleTilingExpert
          : DispatchLoweringPassPipeline::CPUBufferOpsTileAndVectorize;
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, genericOp,
                                               tileSizes, passPipeline);
}

/// Sets elementwise dispatches to use peeling approach. It scales the number of
/// workload per workgroup to a larger number, which prevents runtime overheads
/// from tiny dispatches.
static LogicalResult setElementwiseGenericOpRootConfig(
    func::FuncOp entryPointFn, linalg::GenericOp genericOp,
    const LinalgOpInfo &linalgOpInfo,
    const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");
  unsigned numLoops = genericOp.getNumLoops();
  if (numLoops == 0)
    return failure();
  if (!linalg::isElementwise(genericOp))
    return failure();

  DistributionHeuristicConfig distConfig;
  distConfig.allowIncompleteTile = true;
  distConfig.minTileSizes = getMinTilingSizesForEachDim(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
  distConfig.maxTileSizes.append(numLoops, clDefaultDistTileSize);
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(genericOp, distConfig);

  // TODO(dcaballe): The logic below is disconnected from the main tile size
  // selection logic in getMaxTileSize. We should either port it there or remove
  // it.
  // Adjust the number of workload per workgroup to at least 4096. This
  // prevents the runtime overheads domiating the execution time. The number is
  // derived from experimients. We should be able to make it related to target.
  constexpr int64_t kMinimumWorkload = 4096;
  auto shape = genericOp.getStaticLoopRanges();
  int64_t numWorkload = 1;
  for (const auto &[index, size] : llvm::enumerate(shape)) {
    if (ShapedType::isDynamic(size)) {
      numWorkload = ShapedType::kDynamic;
      break;
    }
    numWorkload *= distTileSizes[index] ? distTileSizes[index] : size;
  }
  for (unsigned currDim = 0;
       numWorkload < kMinimumWorkload && currDim < numLoops;) {
    int64_t currSize = distTileSizes[currDim];
    if (currSize == shape[currDim] || currSize == 0 ||
        ShapedType::isDynamic(shape[currDim]) ||
        ShapedType::isDynamic(numWorkload)) {
      currDim++;
      continue;
    }
    int64_t newSize = std::min<int64_t>(currSize * 2, shape[currDim]);
    numWorkload = numWorkload / currSize * newSize;
    distTileSizes[currDim] = newSize;
  }

  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  LLVM_DEBUG(KD_DBGS() << "Vector pre-processing strategy: "
                       << vecPreProcStrategy << "\n");

  // Adjust tiling sizes of vector levels to avoid large unroll factors. Most of
  // the cases are f32 and i32, so we divide it by 4.
  int64_t vecSize = getNativeVectorSizeInBytes(entryPointFn) / 4;
  SmallVector<int64_t> vecTileSizes = distConfig.minTileSizes;
  for (auto &i : vecTileSizes) {
    i = roundUpToPow2(std::min(i, vecSize),
                      vecPreProcStrategy == VectorPreProcStrategy::Masking);
  }

  // Setting reduction tile sizes is a workaround to kick in peeling transform.
  // The tiling won't happen because the sizes are zeros. Also, no need for
  // further tiling inner parallel dims, so the 4-th list is also zeros.
  SmallVector<int64_t> zeros(numLoops, 0);
  TileSizesListType tileSizes = {distTileSizes, vecTileSizes, zeros, zeros};

  LLVM_DEBUG(KD_DBGS() << "Final tile sizes for element-wise op: " << tileSizes
                       << "\n");

  DispatchLoweringPassPipeline passPipeline;
  if (genericOp.hasBufferSemantics()) {
    passPipeline = DispatchLoweringPassPipeline::CPUBufferOpsTileAndVectorize;
  } else if (vecPreProcStrategy == VectorPreProcStrategy::Peeling) {
    passPipeline = DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert;
  } else {
    passPipeline = DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
  }

  return setOpConfigAndEntryPointFnTranslation(entryPointFn, genericOp,
                                               tileSizes, passPipeline);
}

/// Sets the lowering configuration for a generic op to use
/// CPUDoubleTilingExpert pipeline.
static LogicalResult
setRootConfig(func::FuncOp entryPointFn, linalg::GenericOp genericOp,
              const LinalgOpInfo &linalgOpInfo,
              const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");
  // First, try to apply the transform dialect strategy, if defined.
  if (succeeded(setTransformStrategyRootConfig(
          entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo))) {
    return success();
  }

  if (succeeded(setTransposeLikeOpRootConfig(
          entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo))) {
    return success();
  }
  if (succeeded(setElementwiseGenericOpRootConfig(
          entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo))) {
    return success();
  }
  if (succeeded(setDefaultGenericOpRootConfig(
          entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo))) {
    return success();
  }
  return failure();
}

static bool is2DConvOp(Operation *op) {
  return isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp>(op);
}

static bool is2DDepthConvOp(Operation *op) {
  return isa<linalg::DepthwiseConv2DNhwcHwcOp>(op);
}

static bool is2DPoolingOp(Operation *op) {
  return isa<linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
             linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
             linalg::PoolingNhwcMinUnsignedOp, linalg::PoolingNchwSumOp,
             linalg::PoolingNchwMaxOp>(op);
}

/// Helper enum to represent conv2d input traversal order.
enum class Conv2DDimOrder {
  // Corresponds to operation that traverses the input in (n, c, h, w) order.
  Nchw,
  // Corresponds to operation that traverses the input in (n, h, w, c) order.
  Nhwc
};

static Conv2DDimOrder getConv2DDimOrder(Operation *op) {
  if (isa<linalg::Conv2DNchwFchwOp, linalg::PoolingNchwSumOp,
          linalg::PoolingNchwMaxOp>(op))
    return Conv2DDimOrder::Nchw;
  if (isa<linalg::Conv2DNhwcHwcfOp, linalg::PoolingNhwcSumOp,
          linalg::PoolingNhwcMaxOp, linalg::PoolingNhwcMaxUnsignedOp,
          linalg::PoolingNhwcMinOp, linalg::PoolingNhwcMinUnsignedOp,
          linalg::DepthwiseConv2DNhwcHwcOp>(op))
    return Conv2DDimOrder::Nhwc;
  llvm::llvm_unreachable_internal("unsupported conv op");
}

/// Sets lowering configuration for conv ops. See below for supported conv ops.
static LogicalResult setConvRootConfig(func::FuncOp entryPointFn,
                                       linalg::LinalgOp convOp,
                                       ArrayRef<int64_t> targetTileSizes,
                                       int64_t vectorSize) {
  if (!is2DConvOp(convOp) && !is2DDepthConvOp(convOp) &&
      !is2DPoolingOp(convOp)) {
    return failure();
  }
  assert(!getLoweringConfig(convOp) && "expected lowering_config is not set");

  unsigned numLoops = convOp.getNumLoops();
  DistributionHeuristicConfig distConfig;

  // Give the vector size hint on OC.
  distConfig.vectorSizeHints.append(numLoops, 1);
  distConfig.vectorSizeHints[3] = vectorSize;

  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(convOp, distConfig);

  // Shapes of N, OH, OW, OC, KH, KW, (IC)
  SmallVector<int64_t> shapes = convOp.getStaticLoopRanges();
  SmallVector<int64_t> parallelTileSizes(targetTileSizes.begin(),
                                         targetTileSizes.end());
  for (auto i : llvm::seq<unsigned>(0, parallelTileSizes.size())) {
    auto tileSize = distTileSizes[i] ? distTileSizes[i] : shapes[i];
    // If the tile size is intended to be 1, do not adjust it to `vectorSize`.
    // The ops will be decomposed to lower-rank named ops.
    if (parallelTileSizes[i] != 1) {
      parallelTileSizes[i] =
          getMaxVectorTileSize(tileSize, parallelTileSizes[i], vectorSize);
    }
  }
  SmallVector<int64_t> reductionTileSizes;
  splitParallelAndReductionTiles(convOp, parallelTileSizes, reductionTileSizes);
  setAlwaysVectorizeSizes(convOp, parallelTileSizes, reductionTileSizes);

  TileSizesListType tileSizes = {distTileSizes, parallelTileSizes,
                                 reductionTileSizes};
  // No need for tiling inner parallel dims.
  int64_t numTilingDims = parallelTileSizes.size();
  tileSizes.emplace_back(numTilingDims, 0);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, convOp, tileSizes,
      DispatchLoweringPassPipeline::CPUConvTileAndDecomposeExpert);
}

/// Main utility to compute the vectorization/unrolling tile sizes.
/// Note that this only works for NHWC input and HWCF kernel/filter
/// convolutions, where the shape is [N, OH, OW, OC, KH, KW, (IC)].
static SmallVector<int64_t>
getNhwcConvVectorSizes(func::FuncOp entryPointFn,
                       linalg::ConvolutionOpInterface op, int64_t vectorSize) {
  bool isSupported = is2DConvOp(op) || is2DDepthConvOp(op) || is2DPoolingOp(op);
  (void)isSupported;
  assert(isSupported && "conv op is not supported");

  SmallVector<int64_t> tileSizes;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);

  if (isX86(targetAttr)) {
    if (is2DConvOp(op))
      return {1, 1, 8, vectorSize, 1, 1, 8};
    if (is2DDepthConvOp(op))
      return {1, 1, 8, vectorSize, 1, 3};
    if (is2DPoolingOp(op))
      return {1, 1, 8, vectorSize, 1, 8};
    llvm_unreachable("unsupported conv");
  }
  if (isRISCV(targetAttr)) {
    if (is2DConvOp(op))
      return {1, 1, 8, vectorSize * 2, 1, 1, 8};
    if (is2DDepthConvOp(op))
      return {1, 1, 8, vectorSize, 1, 3};
    if (is2DPoolingOp(op))
      return {1, 1, 8, vectorSize * 2, 1, 8};
    llvm_unreachable("unsupported conv");
  }
  if (isAArch64(targetAttr)) {
    if (is2DConvOp(op))
      return {1, 1, 32, 64, 1, 1, 16};
    if (is2DDepthConvOp(op))
      return {1, 1, 4, 4, 1, 4};
    if (is2DPoolingOp(op))
      return {1, 1, 32, 64, 1, 16};
    llvm_unreachable("unsupported conv");
  }

  // Get default hard-coded tile sizes if we couldn't compute anything
  // better.
  if (is2DConvOp(op))
    return {1, 1, vectorSize, vectorSize, 1, 1, vectorSize};
  if (is2DDepthConvOp(op))
    return {1, 1, vectorSize, vectorSize, 1, vectorSize};
  if (is2DPoolingOp(op))
    return {1, 1, vectorSize, vectorSize, 1, vectorSize};
  llvm_unreachable("unsupported conv");
}

static LogicalResult
setConvInterfaceRootConfig(func::FuncOp entryPointFn,
                           linalg::ConvolutionOpInterface convOp) {
  int64_t vectorSize = getVectorSize(
      entryPointFn, cast<ShapedType>(convOp->getResultTypes()[0]));
  SmallVector<int64_t> targetTileSizes =
      getNhwcConvVectorSizes(entryPointFn, convOp, vectorSize);

  // The tiling sizes are for NHWC layout. We need to apply a permutation if
  // they are in other layout format.
  Conv2DDimOrder order = getConv2DDimOrder(convOp);
  switch (order) {
  case Conv2DDimOrder::Nhwc:
    break;
  case Conv2DDimOrder::Nchw:
    SmallVector<int64_t> perm;
    if (is2DConvOp(convOp)) {
      // D.n, D.oh, D.ow,  D.f, D.kh, D.kw, D.c ->
      // D.n,  D.f, D.oh, D.ow,  D.c, D.kh, D.kw
      perm = {0, 3, 1, 2, 6, 4, 5};
    } else if (is2DPoolingOp(convOp)) {
      // D.n, D.oh, D.ow, D.c, D.kh, D.kw ->
      // D.n, D.c, D.oh, D.ow, D.kh, D.kw
      perm = {0, 3, 1, 2, 4, 5};
    } else if (is2DDepthConvOp(convOp)) {
      llvm_unreachable("Not implemented yet");
    }
    applyPermutationToVector(targetTileSizes, perm);
    break;
  }

  return setConvRootConfig(entryPointFn,
                           cast<linalg::LinalgOp>(convOp.getOperation()),
                           targetTileSizes, vectorSize);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   tensor::PadOp padOp) {
  SmallVector<int64_t> lbs, ubs;
  getRangeBounds(cast<TilingInterface>(padOp.getOperation()), lbs, ubs);

  int64_t numLoops = lbs.size();
  unsigned typeWidthInBytes = IREE::Util::getRoundedElementByteWidth(
      padOp.getResultType().getElementType());
  int64_t typeVectorSize = getVectorSize(entryPointFn, typeWidthInBytes);
  DistributionHeuristicConfig distConfig;
  distConfig.vectorSizeHints.append(numLoops, 1);
  if (!ShapedType::isDynamic(ubs.back())) {
    distConfig.vectorSizeHints.back() = std::min(typeVectorSize, ubs.back());
  }

  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(padOp, distConfig);
  // No further tiling for reduction and inner parallel loops.
  SmallVector<int64_t> zeros(numLoops, 0);
  TileSizesListType tileSizes = {distTileSizes, distConfig.vectorSizeHints,
                                 zeros, zeros};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, padOp, tileSizes,
      DispatchLoweringPassPipeline::CPUDoubleTilingExpert);
}

/// Set the default configuration for operations that implement the
/// `TiledOpInterface`.
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   TilingInterface op) {
  assert(!getLoweringConfig(op) && "expected lowering_config is not set");
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(op, DistributionHeuristicConfig{});
  TileSizesListType tileSizes = {distTileSizes};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes, DispatchLoweringPassPipeline::CPUDefault);
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult
setRootConfigImpl(func::FuncOp entryPointFn, Operation *op,
                  const TargetMLTransformInfo &targetMLTransInfo) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<linalg::GenericOp>([&](auto op) {
          return setRootConfig(entryPointFn, op, LinalgOpInfo(op),
                               targetMLTransInfo);
        })
        .Case<IREE::LinalgExt::FftOp, tensor::PackOp, tensor::PadOp,
              tensor::UnPackOp, linalg::Mmt4DOp, linalg::BatchMmt4DOp>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp,
              linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
              linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
              linalg::PoolingNhwcMinUnsignedOp, linalg::PoolingNchwSumOp,
              linalg::PoolingNchwMaxOp, linalg::DepthwiseConv2DNhwcHwcOp>(
            [&](auto op) {
              return setConvInterfaceRootConfig(entryPointFn, op);
            })
        .Case<linalg::ContractionOpInterface>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<TilingInterface>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Update the distribution tile sizes and parallel vector tile sizes to ensure:
/// 1. Distribution tile sizes and parallel vector tile sizes are aligned to the
///    inner tile sizes of the pack op.
/// 2. Parallel vector tile sizes are set with getPackVectorTileSizes to get
///    good performance on the pack op (e.g. 16x16 tile size on AVX512 for good
///    transpose codegen on the pack op).
/// For example:
/// Given the tile sizes for a Matmul RHS pack op is [1, 1, 1] and its inner
/// tile size is 16x1. We set the parallel vector tile sizes to [1, 1, 16],
/// which will be translated to tile sizes [1, 1, 1] on the pack op in
/// setLoweringConfigForComputeOps due to its affine map. At the same time,
/// its producer will have the parallel tile sizes [1, 1, 16], which is how the
/// pack op wants to tile-and-fuse it.
static LogicalResult
adjustTileSizesForPackOp(func::FuncOp entryPointFn, tensor::PackOp packOp,
                         SmallVector<int64_t> &distTileSizes,
                         SmallVector<int64_t> &parallelVecTileSizes) {

  ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();
  ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
  // Currently we only handle pack op with static inner tile sizes.
  if (llvm::any_of(innerTiles,
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    return failure();
  }
  // Pack op requires special vector tile sizes to achieve good performance.
  // Override the parallel vector tile sizes from pack op.
  auto vecTileSizes = getPackVectorTileSizes(entryPointFn, packOp);
  auto outerDimsPerm = packOp.getOuterDimsPerm();
  if (!outerDimsPerm.empty()) {
    auto invertedPerm = invertPermutationVector(outerDimsPerm);
    applyPermutationToVector(vecTileSizes, invertedPerm);
  }
  // Scale to actual tile sizes with the pack op's inner tile sizes.
  for (auto [pos, size] : llvm::zip_equal(innerDimsPos, innerTiles)) {
    vecTileSizes[pos] *= size;
  }
  for (auto [pos, size] : llvm::enumerate(vecTileSizes)) {
    if (!size)
      continue;
    if (!parallelVecTileSizes[pos]) {
      parallelVecTileSizes[pos] = size;
      continue;
    }
    // If other ops already set a smaller tile size, don't override it to avoid
    // too large tile size on them.
    parallelVecTileSizes[pos] = std::min(parallelVecTileSizes[pos], size);
  }
  // Align the tile sizes to the pack op's inner tile sizes, so we can derive
  // the outer tile sizes for pack ops later in setLoweringConfigForComputeOps
  // by dividing with inner tile sizes.
  for (auto [pos, size] : llvm::zip_equal(innerDimsPos, innerTiles)) {
    if (distTileSizes[pos])
      distTileSizes[pos] = llvm::alignTo(distTileSizes[pos], size);
    if (parallelVecTileSizes[pos])
      parallelVecTileSizes[pos] =
          llvm::alignTo(parallelVecTileSizes[pos], size);
  }
  return success();
}

/// Adjusts the tile sizes (carried by `rootOp`) to be aligned with
/// tensor.unpack inner tile sizes, if there are tensor.unpack producers. If the
/// tile sizes are not aligned, a stack buffer is needed because of
/// tensor.unpack tiling implementations.
static LogicalResult adjustTileSizesForUnPackOp(func::FuncOp entryPointFn,
                                                Operation *rootOp) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp)
    return success();

  auto loweringConfig = getLoweringConfig(linalgOp);
  TileSizesListType tileSizesList = loweringConfig.getTileSizeVals();

  bool foundUnPackOp = false;
  SmallVector<int64_t> alignedSizes(linalgOp.getNumLoops(), 1);
  for (OpOperand *opOperand : linalgOp.getDpsInputOperands()) {
    auto unpackOp = opOperand->get().getDefiningOp<tensor::UnPackOp>();
    if (!unpackOp)
      continue;

    foundUnPackOp = true;
    auto idxMap = linalgOp.getMatchingIndexingMap(opOperand);
    LLVM_DEBUG(KD_DBGS() << "Find unpack op candidate: " << unpackOp << "\n"
                         << "The corresponding indexing map is: " << idxMap
                         << "\n");

    SmallVector<int64_t> innerTiles = unpackOp.getStaticTiles();
    ArrayRef<int64_t> dimPos = unpackOp.getInnerDimsPos();
    for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
      if (ShapedType::isDynamic(size))
        continue;
      auto dimExpr = dyn_cast<AffineDimExpr>(idxMap.getResult(pos));
      if (!dimExpr)
        return failure();
      int mappedPos = dimExpr.getPosition();
      alignedSizes[mappedPos] = std::lcm(alignedSizes[mappedPos], size);
    }
  }

  if (!foundUnPackOp)
    return success();

  LLVM_DEBUG(
      KD_DBGS() << "The tile sizes for each dimension should be aligned to "
                << alignedSizes);

  // Fixup for making tileSizes be multiple of inner_tile_sizes.
  for (SmallVectorImpl<int64_t> &tileSizes : tileSizesList) {
    for (auto idx : llvm::seq<int64_t>(0, tileSizes.size())) {
      if (tileSizes[idx] == 0)
        continue;
      tileSizes[idx] = llvm::alignTo(tileSizes[idx], alignedSizes[idx]);
    }
  }

  auto pipeline = getTranslationInfo(entryPointFn).getPassPipeline().getValue();
  if (pipeline == DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert) {
    LLVM_DEBUG(KD_DBGS() << "unpack fusion does not work with peeling, falling "
                            "back to non-peeling path");
    pipeline = DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
  }

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, rootOp, tileSizesList,
      loweringConfig.getScalableTileFlagVals(), pipeline);
}

/// Get tile sizes for the generic op and fill into the parallel vector tile
/// sizes if the tile size on a dimension is missing. Also get the tile sizes on
/// the reduction dimensions. This makes sure there is a tile size set for each
/// dimension of the generic op.
/// For example:
/// The root op has iterator types: parallel, reduction, reduction and the
/// parallel tile sizes from the root op is [X, 0, 0]. The generic op's iterator
/// types are: parallel, parallel, reduction. After the update, the parallel
/// tile sizes become [X, Y, 0] while the Y is set by the generic op. The
/// function also returns the reduction tile sizes for the generic op [0, 0, Z].
static LogicalResult
adjustTileSizesForGenericOp(func::FuncOp entryPointFn,
                            linalg::GenericOp genericOp,
                            SmallVector<int64_t> &parallelVecTileSizes,
                            SmallVector<int64_t> &reductionTileSizes) {
  auto linalgOpInfo = LinalgOpInfo(genericOp);
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  auto targetMLTransInfo =
      TargetMLTransformInfo::getTargetMLTransformInfo(targetAttr);
  SmallVector<int64_t> vecTileSizes = getMinTilingSizesForEachDim(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);

  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  int64_t vecSize = getNativeVectorSizeInBytes(entryPointFn) / 4;

  for (auto &vecTileSize : vecTileSizes) {
    vecTileSize =
        roundUpToPow2(std::min(vecTileSize, vecSize),
                      vecPreProcStrategy == VectorPreProcStrategy::Masking);
  }

  splitParallelAndReductionTiles(genericOp, vecTileSizes, reductionTileSizes);
  setVectorSizesForDynamicShapes(genericOp, vecPreProcStrategy, vecTileSizes,
                                 reductionTileSizes);
  for (auto [pos, tileSize] : llvm::enumerate(vecTileSizes)) {
    // Generic op vector parallel tile size is low priority. Only use if no
    // other op has set the tile size.
    if (tileSize == 0 || parallelVecTileSizes[pos] != 0)
      continue;
    parallelVecTileSizes[pos] = tileSize;
  }
  return success();
}

/// Set the lowering configs for all the compute ops. The lowering config is
/// already set on `rootOperation`. We will duplicate the tile sizes of
/// distribution and common parallel dims to other compute ops (so they have
/// consistent configurations); set the tile size for the rest of dims. E.g., it
/// sets the lowering_config for reduction + broadcast + pack op like:
///
///   %11 = linalg.fill ... -> tensor<384xf32>
///   %12 = linalg.generic {
///     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
///                      affine_map<(d0, d1) -> (d0)>,
///                      affine_map<(d0, d1) -> (d0)>],
///     iterator_types = ["parallel", "reduction"]}
///     ins(%5, %6 : tensor<384x528xf32>, tensor<384xf32>)
///     outs(%11 : tensor<384xf32>) {
///   ^bb0(%in: f32, %in_2: f32, %out: f32):
///     ...
///   } -> tensor<384xf32>
///   %13 = linalg.generic {
///     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
///                      affine_map<(d0, d1) -> (d0)>,
///                      affine_map<(d0, d1) -> (d0, d1)>],
///     iterator_types = ["parallel", "parallel"]}
///     ins(%4, %12 : tensor<384x1024xf32>, tensor<384xf32>)
///     outs(%9 : tensor<384x1024xf32>) {
///   ^bb0(%in: f32, %in_2: f32, %out: f32):
///     ...
///   } -> tensor<384x1024xf32>
///   %pack = tensor.pack %13
///     inner_dims_pos = [0, 1]
///     inner_tiles = [16, 1]
///     into %14 : tensor<384x1024xf32> -> tensor<24x1024x16x1xf32>
///
/// The lowering config is set on `rootOperation`, which is the reduction ops in
/// this case. The configuration is [[X, 0], [Y, 0], [0, Z], [0, 0]]. The `d0`
/// is the common parallel dims for all the ops. The lowering config from
/// `rootOperation` sets the tile sizes for `d0` and the rest of dims in
/// reduction. But it does not have tile sizes for the rest of dims in
/// elementwise op and pack ops. This method sets the vectorization tile sizes
/// for other compute ops. E.g., [[X, 0], [Y, 0], [0, 0], [0, 16]] for the
/// elementwise operations and [[X, 0], [Y, 0], [0, 0], [0, 1]] for the pack op.
static LogicalResult
setLoweringConfigForComputeOps(func::FuncOp entryPointFn,
                               ArrayRef<Operation *> computeOps,
                               Operation *rootOperation) {
  if (isa<linalg::ConvolutionOpInterface>(rootOperation)) {
    // TODO(dcaballe): We don't know yet how to properly propagate the lowering
    // config of a convolution.
    return success();
  }

  auto ctx = entryPointFn.getContext();
  auto rootLoweringConfig = getLoweringConfig(rootOperation);
  TilingConfig tilingConfig(rootLoweringConfig);
  SmallVector<int64_t> distTileSizes, parallelVecTileSizes;
  if (tilingConfig.getNumTilingLevels() > 0) {
    distTileSizes = tilingConfig.getDistributionTileSizes();
  }
  if (tilingConfig.getNumTilingLevels() > 1) {
    // TODO: Handle scalable tiles.
    std::tie(parallelVecTileSizes, std::ignore) =
        tilingConfig.getVectorCommonParallelSizes();
  }

  size_t maxLoopNums = 0;
  for (auto op : computeOps) {
    // Multi-lowering config works only if all the operations can share the same
    // distribution and parallel tile sizes from the root op.
    auto iterTypes = cast<TilingInterface>(op).getLoopIteratorTypes();
    for (auto [idx, iterType] : llvm::enumerate(iterTypes)) {
      if (idx >= parallelVecTileSizes.size())
        break;
      if (iterType == utils::IteratorType::parallel)
        continue;
      if (distTileSizes[idx] || parallelVecTileSizes[idx])
        return success();
    }
    maxLoopNums = std::max(maxLoopNums, iterTypes.size());
  }

  // Adjust the distribution tile sizes and join parallel vector tile sizes from
  // other ops. The results of parallel vector tile sizes might overlap
  // reduction dimensions on some ops, so it will be splitted into common vector
  // tile sizes and inner vector tile sizes later.
  //
  // This step is to ensure all ops are using an equivalent set of parallel tile
  // sizes.
  //
  // Here we use the assumption in FormDispatchRegions that all ops in a
  // dispatch have identity mapping between their parallel dimensions. So we
  // don't need to handle the permutation on dimensions between ops except for
  // the pack op.
  //
  // For example:
  // Given there are 3 generic ops in the dispatch:
  // %rootOp = linalg.generic {iterator_types = ["reduction", "parallel"]} ...
  // %2 = linalg.generic {iterator_types = ["parallel", "parallel"]}
  // %3 = tensor.pack %2
  // Assume the distribution and parallel vector tile sizes from %rootOp is:
  // [[X1, 0], [X2, 0]]
  // Then the generic op %2 set the missing parallel vector tile sizes on its
  // parallel dims:
  // [[X1, 0], [X2, Y2]]
  // Then the pack op %3 updates the distribution and parallel vector tile sizes
  // based on its requirement:
  // [[X1', Z1], [X2', Y2']]
  // which is the final parallel tile sizes for all ops.
  llvm::SmallDenseMap<Operation *, SmallVector<int64_t>> reductionTileSizeMap;
  distTileSizes.resize(maxLoopNums);
  parallelVecTileSizes.resize(maxLoopNums);
  bool hasSeenPackOp = false;
  for (auto op : computeOps) {
    assert(!hasSeenPackOp && "Pack op must be the last op");
    if (hasSeenPackOp)
      return failure();

    // Tile sizes have been initialized from rootOperation, so we skip it.
    if (op == rootOperation)
      continue;

    if (auto packOp = dyn_cast<tensor::PackOp>(op)) {
      if (failed(adjustTileSizesForPackOp(entryPointFn, packOp, distTileSizes,
                                          parallelVecTileSizes))) {
        return failure();
      }
      hasSeenPackOp = true;
    } else if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      SmallVector<int64_t> reductionTileSizes;
      if (failed(adjustTileSizesForGenericOp(entryPointFn, genericOp,
                                             parallelVecTileSizes,
                                             reductionTileSizes))) {
        return failure();
      }
      reductionTileSizeMap[op] = reductionTileSizes;
    }
  }

  LLVM_DEBUG(KD_DBGS() << "Parallel vector tile sizes: " << parallelVecTileSizes
                       << "\n");

  // Split parallel vector tile sizes into common parts and op-specific parts.
  SmallVector<int64_t> commonVecTileSizes = parallelVecTileSizes;
  SmallVector<int64_t> innerVecTileSizes(maxLoopNums, 0);
  for (auto op : computeOps) {
    auto iterTypes = cast<TilingInterface>(op).getLoopIteratorTypes();
    for (auto [idx, iterType] : llvm::enumerate(iterTypes)) {
      if (iterType == utils::IteratorType::reduction) {
        innerVecTileSizes[idx] = parallelVecTileSizes[idx];
        commonVecTileSizes[idx] = 0;
      }
    }
  }

  // Set the lowering configs with new tile sizes.
  for (auto op : computeOps) {
    int numLoops = cast<TilingInterface>(op).getLoopIteratorTypes().size();
    TileSizesListType tileSizesList;
    ScalableTileFlagsListType scalableTileFlagsList;

    // For root op, we patch the adjusted tile sizes on its original tiling
    // config.
    if (op == rootOperation) {
      tileSizesList = rootLoweringConfig.getTileSizeVals();
      scalableTileFlagsList = rootLoweringConfig.getScalableTileFlagVals();
      if (tilingConfig.getNumTilingLevels() > 0) {
        tileSizesList[tilingConfig.getDistributionLevel()] = distTileSizes;
      }
      if (tilingConfig.getNumTilingLevels() > 1) {
        tileSizesList[tilingConfig.getVectorCommonParallelLevel()] =
            commonVecTileSizes;
      }
    } else {
      // Build 4-level lowering configs for other ops.
      tileSizesList = {distTileSizes, commonVecTileSizes};
      SmallVector<int64_t> zeros(numLoops, 0);
      TypeSwitch<Operation *>(op)
          .Case<tensor::PackOp>([&](auto packOp) {
            tileSizesList.push_back(zeros);
            tileSizesList.push_back(innerVecTileSizes);
            // Scale and permutate the outer dim tiles for pack op.
            ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
            ArrayRef<int64_t> dimPos = packOp.getInnerDimsPos();
            auto outerDimsPerm = packOp.getOuterDimsPerm();
            for (auto &tileSizes : tileSizesList) {
              for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
                if (tileSizes[pos] == 0 || ShapedType::isDynamic(size))
                  continue;
                tileSizes[pos] = tileSizes[pos] / size;
              }
              if (!outerDimsPerm.empty()) {
                tileSizes.resize(numLoops, 0);
                applyPermutationToVector(tileSizes, outerDimsPerm);
              }
            }
          })
          .Default([&](auto) {
            if (reductionTileSizeMap.contains(op)) {
              tileSizesList.push_back(reductionTileSizeMap[op]);
            } else {
              tileSizesList.push_back(zeros);
            }
            // Only copy the inner vector tile sizes on parallel dims.
            SmallVector<int64_t> vecTileSizes(numLoops, 0);
            auto iterTypes = cast<TilingInterface>(op).getLoopIteratorTypes();
            for (auto [idx, iterType] : llvm::enumerate(iterTypes)) {
              if (iterType == utils::IteratorType::parallel)
                vecTileSizes[idx] = innerVecTileSizes[idx];
            }
            tileSizesList.push_back(vecTileSizes);
          });
    }

    for (auto &ts : tileSizesList)
      ts.resize(numLoops, 0);
    auto config = IREE::Codegen::LoweringConfigAttr::get(ctx, tileSizesList,
                                                         scalableTileFlagsList);
    setLoweringConfig(op, config);
  }

  return success();
}

/// Helper method to set the dispatch to be lowered through the default
/// pipeline.
static LogicalResult lowerUsingDefaultPipeline(func::FuncOp entryPointFn) {
  // If there is a translation info set, do nothing.
  if (getTranslationInfo(entryPointFn)) {
    return success();
  }
  // Otherwise lower using default pipeline.
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      entryPointFn->getContext(), DispatchLoweringPassPipeline::CPUDefault);
  return setTranslationInfo(entryPointFn, translationInfo);
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult
setTranslationInfoAndRootConfig(func::FuncOp entryPointFn,
                                ArrayRef<Operation *> computeOps) {
  // Make sure that lowering_config is not preset on any compute ops.
  for (auto computeOp : computeOps) {
    if (getLoweringConfig(computeOp))
      return failure();
  }

  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp))
    return failure();
  Operation *rootOperation = rootOp.value();

  // Handle the case with no known root operation.
  if (!rootOperation) {
    return lowerUsingDefaultPipeline(entryPointFn);
  }

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  auto targetMLTransInfo =
      TargetMLTransformInfo::getTargetMLTransformInfo(targetAttr);
  if (failed(
          setRootConfigImpl(entryPointFn, rootOperation, targetMLTransInfo))) {
    return failure();
  }

  // The transform dialect codegen has differnet logics and codegen flow.
  // Ignore the tile sizes adjustment.
  auto pipeline = getTranslationInfo(entryPointFn).getPassPipeline().getValue();
  if (pipeline != DispatchLoweringPassPipeline::TransformDialectCodegen) {
    if (failed(adjustTileSizesForUnPackOp(entryPointFn, rootOperation))) {
      return failure();
    }

    // Set vector level tile sizes for other operations individually.
    if (failed(setLoweringConfigForComputeOps(entryPointFn, computeOps,
                                              rootOperation))) {
      return failure();
    }
  }

  return success();
}

LogicalResult initCPULaunchConfig(ModuleOp moduleOp) {
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp)
      continue;
    if (getTranslationInfo(exportOp))
      continue;

    // For now pick the default for functions with control flow, cause
    // the currently built pipelines dont work so well with control flow.
    if (funcOp.getBody().empty() || !llvm::hasSingleElement(funcOp.getBody())) {
      return lowerUsingDefaultPipeline(funcOp);
    }

    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    if (failed(setTranslationInfoAndRootConfig(funcOp, computeOps))) {
      return failure();
    }
  }

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(moduleOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

} // namespace mlir::iree_compiler
