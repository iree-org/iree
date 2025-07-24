// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenEnums.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/LLVMCPU/TargetMLTransformInfo.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

#define DEBUG_TYPE "kernel-dispatch"

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

static llvm::cl::opt<bool> clDisableArmSMETiling(
    "iree-llvmcpu-disable-arm-sme-tiling",
    llvm::cl::desc("Disables tiling for SME even if it is supported by the "
                   "target (i.e., when the +sme feature flag is present)"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableRiscvAggressiveDist(
    "iree-llvmcpu-riscv-aggressive-distribution",
    llvm::cl::desc(
        "Enable aggressive method for distribution tile size. "
        "It is only applied for linalg contraction ops now. "
        "If distConfig.minTileSizes[i] >= distConfig.maxTileSizes[i], "
        "set distConfig.maxTileSizes[i] to 2 * distConfig.minTileSizes[i]."),
    llvm::cl::init(false));

using IREE::Codegen::DispatchLoweringPassPipeline;
using IREE::CPU::TilingLevel;

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
  None
};

// Use this flag to override IREE's heuristics for selecting the pre-processing
// strategy.
static llvm::cl::opt<VectorPreProcStrategy> clPProcStrategy(
    "iree-llvmcpu-vector-pproc-strategy",
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
            "Do not apply any vectorization pre-processing transformation.")));

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
    os << llvm::interleaved_array(tuple);
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

/// Returns the vectorization pre-processing strategy (peeling, masking) for the
/// given LinalgOp. It is based on either (in the priority order):
///   * user-specified value, or
///   * IREE's heuristics (e.g. the op traits and the target architecture).
static VectorPreProcStrategy
getVectorPreProcStrategy(linalg::LinalgOp linalgOp) {
  // If set, use the strategy selected by a user.
  if (clPProcStrategy.getNumOccurrences()) {
    return clPProcStrategy;
  }

  // TODO: Implement heuristics for Convs
  if (isa<linalg::ConvolutionOpInterface>(linalgOp.getOperation())) {
    return VectorPreProcStrategy::None;
  }

  // Select a strategy based on heuristics.
  if (linalgOp.hasPureBufferSemantics()) {
    return VectorPreProcStrategy::None;
  }

  // Walk the linalgOp code, if there is any instruction that could result in
  // undefined behavior in mask strategy, fall back to using peel strategy.
  bool usePeelingStrategy = false;
  linalgOp.walk([&](Operation *op) -> WalkResult {
    if (mlir::iree_compiler::mayHaveUndefinedBehaviorInMasking(op)) {
      usePeelingStrategy = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (usePeelingStrategy) {
    return VectorPreProcStrategy::Peeling;
  }

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(linalgOp);
  bool isLinalgGeneric = isa<linalg::GenericOp>(linalgOp.getOperation());

  // Default X86 specific strategy.
  if (targetAttr && isX86(targetAttr.getConfiguration())) {
    if (isLinalgGeneric) {
      return VectorPreProcStrategy::Masking;
    }

    if (!clDisableVectorPeeling) {
      return VectorPreProcStrategy::Peeling;
    }
  }

  // Default RISC-V specific strategies.
  if (targetAttr && isRISCV(targetAttr.getConfiguration())) {
    if (isLinalgGeneric) {
      return VectorPreProcStrategy::Masking;
    }

    if (!clDisableVectorPeeling) {
      return VectorPreProcStrategy::Peeling;
    }
  }

  // Default AArch64 specific strategies.
  if (targetAttr && isAArch64(targetAttr.getConfiguration())) {
    if (isScalableVectorizationEnabled() &&
        hasAnySVEFeature(targetAttr.getConfiguration())) {
      return VectorPreProcStrategy::Masking;
    }

    if (!clDisableVectorPeeling) {
      return VectorPreProcStrategy::Peeling;
    }
  }

  return VectorPreProcStrategy::None;
}

static DictionaryAttr getPipelineConfWithPeelingAttr(MLIRContext *context) {
  auto enableLoopPeelingAttrName = getEnableLoopPeelingAttrName(context);
  auto unitAttr = UnitAttr::get(context);

  return DictionaryAttr::get(
      context, ArrayRef<NamedAttribute>({enableLoopPeelingAttrName, unitAttr}));
}

static DictionaryAttr
getPipelineConfWithDecompositionAttr(MLIRContext *context) {
  auto attrName = getEnableDecompositionAttrName(context);
  auto unitAttr = UnitAttr::get(context);
  return DictionaryAttr::get(context,
                             ArrayRef<NamedAttribute>({attrName, unitAttr}));
}

/// Looks for the `native_vector_size` attribute in the hal.executable.target
/// looked up from this op.
static int64_t
getNativeVectorSizeInBytes(mlir::FunctionOpInterface entryPointFn) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  std::optional<int64_t> nativeVectorSize =
      getConfigNativeVectorSize(targetAttr.getConfiguration());
  if (nativeVectorSize) {
    int64_t nativeVectorSizeVal = nativeVectorSize.value();
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
static int64_t getVectorSize(mlir::FunctionOpInterface entryPointFn,
                             unsigned byteWidth) {
  return getNativeVectorSizeInBytes(entryPointFn) / byteWidth;
}
static int64_t getVectorSize(mlir::FunctionOpInterface entryPointFn,
                             ShapedType shapedType) {
  Type elementType = shapedType.getElementType();
  if (!elementType.isIntOrFloat())
    return 1;
  unsigned byteWidth = IREE::Util::getRoundedElementByteWidth(elementType);
  return getVectorSize(entryPointFn, byteWidth);
}

/// Returns true if the operation is a GenericOp implementing a supported
/// transposition:
///   1. The op has a single input and a single output.
///   2. One of the indexing_map is identity and the other is a permutation.
static bool x86TransposeLoweringPrecondition(linalg::GenericOp genericOp) {
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

/// Returns minimum tiling sizes for each dimension. One dimension is possible
/// to access at different element types. It determines the tiling sizes by
/// looking into all the operands.
// TODO(diegocaballero): Refactor this logic to a method that computes the final
// tile sizes for vectorization/unrolling in one shot.
static SmallVector<int64_t>
getMinTilingSizesForEachDim(mlir::FunctionOpInterface entryPointFn,
                            linalg::LinalgOp op,
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
  // unrolling. The util limits the second rightmost non-one tiled dimension
  // to be not larger than `maxUnrollFactor` and others tiled dimension to 1.
  auto limitUnrollFactor = [&](int64_t maxUnrollFactor) {
    int vecDim;
    for (vecDim = minTileSizes.size() - 1; vecDim >= 0; --vecDim) {
      if (minTileSizes[vecDim] > 1) {
        break;
      }
    }
    bool seen = false;
    for (int unrollDim = vecDim - 1; unrollDim >= 0; --unrollDim) {
      if (minTileSizes[unrollDim] <= 1) {
        continue;
      }
      int64_t factor = seen ? 1LL : maxUnrollFactor;
      seen = true;
      LDBG() << "Adjusted min tile sizes: " << minTileSizes[unrollDim]
             << " with factor=" << factor << "\n";
      minTileSizes[unrollDim] =
          std::min<int64_t>(minTileSizes[unrollDim], factor);
    }
  };

  auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (linalgOpInfo.isTranspose() && genericOp &&
      x86TransposeLoweringPrecondition(genericOp)) {
    // TODO(dcaballe): Consider input and output transposes.
    limitUnrollFactor(targetMLTransInfo.defaultMaxTransposeUnrollFactor);
  } else if (linalgOpInfo.isReduction()) {
    limitUnrollFactor(targetMLTransInfo.defaultMaxReductionUnrollFactor);
  } else {
    limitUnrollFactor(targetMLTransInfo.defaultMaxElementwiseUnrollFactor);
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
    int64_t targetSize =
        std::min(workload[i] / clNumberOfRuntimeThreads, maxTileSizes[i]);
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

// Returns true if `map` is a function of `dim` and is not a function of any
// more inner dimensions past `dim`.
static bool isInnerMostDimThatMapIsFunctionOf(AffineMap map, int dim) {
  if (!map.isFunctionOfDim(dim)) {
    return false;
  }
  for (int d = dim + 1; d < map.getNumDims(); ++d) {
    if (map.isFunctionOfDim(d)) {
      return false;
    }
  }
  return true;
}

static void limitVectorTileSizes(SmallVectorImpl<int64_t> &vecTileSizes,
                                 int64_t eachOperandMaxTileBits,
                                 int64_t allOperandsMaxTileBits,
                                 TypeRange operandTypes,
                                 ArrayRef<AffineMap> indexingMaps,
                                 ArrayRef<int64_t> bounds = {}) {
  int64_t numLoops = vecTileSizes.size();
  int numOperands = operandTypes.size();

  SmallVector<int64_t> operandElemBits =
      llvm::map_to_vector(operandTypes, [](Type t) -> int64_t {
        return IREE::Util::getTypeBitWidth(getElementTypeOrSelf(t));
      });

  // For each operand, we track how big the tile is going to be based on the
  // dimensions that we have already accounted for. Here we initialize this to
  // just elemBitWidth, having not yet accounted for any array dimension.
  SmallVector<int64_t> tileBits;
  for (auto bits : operandElemBits) {
    if (bits > eachOperandMaxTileBits) {
      // eachOperandMaxTileBits is too small for this requirement to be
      // satisfied. Early-returning. This might conceivably happen due to the
      // tail-recursion in this function, where eachOperandMaxTileBits is
      // halved.
      return;
    }
    tileBits.push_back(bits);
  }

  // For each loop, we must make sure that if it's the inner-most dimension that
  // concerns an operand with a not-multiple-of-8-bit element type, we preserve
  // the requirement that the inner-most tile dimension is a multiple of 8 bits.
  // For example, if the element type is `i2`, the minimum inner tile size is
  // 8/2 = 4. If the element type if i6, since gcd(8, 6) = 2, we are in the same
  // situation and the minimum inner tile size is again 8/2 = 4.
  // TODO(#17178): check this actually works when we have such element types.
  SmallVector<int64_t> minVecTileSizes(numLoops, 1);
  for (int loopNum : llvm::seq<int>(0, numLoops)) {
    for (int i : llvm::seq<int>(0, numOperands)) {
      if (isInnerMostDimThatMapIsFunctionOf(indexingMaps[i], loopNum)) {
        if (operandElemBits[i] % 8) {
          minVecTileSizes[loopNum] = std::max(
              minVecTileSizes[loopNum], 8 / std::gcd(8, operandElemBits[i]));
        }
      }
    }
  }

  // For each loop starting from the inner-most, clamp vecTileSizes to honor
  // `eachOperandMaxTileBits`, to the extent permitted by `minVecTileSizes`
  // (which is a harder requirement).
  for (int loopNum : llvm::reverse(llvm::seq<int>(0, numLoops))) {
    for (int i : llvm::seq<int>(0, numOperands)) {
      // Check if this operand is concerned with this loop.
      if (!indexingMaps[i].isFunctionOfDim(loopNum)) {
        continue;
      }
      int64_t oldVal = vecTileSizes[loopNum];
      int64_t maxVal = std::max(minVecTileSizes[loopNum],
                                eachOperandMaxTileBits / tileBits[i]);
      int64_t adjustedVal = std::min(oldVal, maxVal);
      // If we are adjusting a tile size, make sure that the adjusted tile size
      // is a power-of-two, as introducing non-power-of-two tile sizes can be a
      // major performance regression, e.g. a 30% e2e regression in MobileNetV2
      // just from a single occurrence in a DepthwiseConv (example discussed
      // below in this comment).
      //
      // We round the adjustedVal to the nearest power of two /below/ not
      // /above/ because doing that could make the entire adjustment a
      // no-operation (e.g. from 8 to 5 back to 8) and then would trigger the
      // tail recursion with halved eachOperandMaxTileBits, which would result
      // in more aggressive tile size limitation. E.g. in MobileNetV2 on AVX-512
      // we have a DepthwiseConv with
      //   vecTileSizes = [1 1 8 16 1 3],
      // without this adjustment, this becomes
      //   vecTileSizes = [1 1 5 16 1 3].
      // If we round this 5 back to 8, letting this function tail-recurse, we
      // end up with this much narrower final result:
      //   vecTileSizes = [1 1 2 16 1 3].
      // It is best to round this 5 to 4, obtaining
      //   vecTileSizes = [1 1 4 16 1 3].
      //
      // The above example also shows that some non-power-of-two tile sizes
      // are not getting adjusted, and should not be rounded to nearest
      // power-of-two: the inner-most dimension size 3 above.
      if (adjustedVal != oldVal) {
        // Round to nearest power of 2, rounding down.
        adjustedVal = 1ll << llvm::Log2_64(adjustedVal);
      }

      if (oldVal == 0) {
        // Skip updating tile sizes of 0, we want to preserve these values.
        if (!bounds.empty()) {
          // If we have enough information about the upper bound of this tile
          // size, use it.
          tileBits[i] *= bounds[loopNum];
        }
      } else {
        vecTileSizes[loopNum] = adjustedVal;
        tileBits[i] *= adjustedVal;
      }
    }
  }

  // At this point we have satisfied our entry-wise requirements on
  // `vecTileSizes`, but there is still the possibility that the sum of all
  // operand tiles' bit sizes exceeds `allOperandsMaxTileBits`. Tail-recurse
  // with halved `eachOperandMaxTileBits` until that requirement is satisfied.
  // Note that if `eachOperandMaxTileBits` falls below some element type bit
  // width, it will trigger an early-return above, so we don't need to worry
  // about that here.
  if (std::reduce(tileBits.begin(), tileBits.end()) > allOperandsMaxTileBits) {
    limitVectorTileSizes(vecTileSizes, eachOperandMaxTileBits / 2,
                         allOperandsMaxTileBits, operandTypes, indexingMaps,
                         bounds);
  }
}

// Clamps in-place `vecTileSizes`, ensuring that the resulting vector tile sizes
// for each opearand of `op` satisfy two requirements:
// 1. No resulting operand tile size exceeds `eachOperandMaxTileBits`.
// 2. The sum of all resulting operand tile size does not exceed
// `allOperandsMaxTileBits`.
static void limitVectorTileSizes(Operation *inputOp,
                                 SmallVectorImpl<int64_t> &vecTileSizes,
                                 int64_t eachOperandMaxTileBits,
                                 int64_t allOperandsMaxTileBits) {
  auto op = dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(inputOp);
  if (!op) {
    return;
  }

  limitVectorTileSizes(vecTileSizes, eachOperandMaxTileBits,
                       allOperandsMaxTileBits, op->getOperandTypes(),
                       op.getIndexingMapsArray());
}

// Returns the size in bits of SIMD register space, or 0 if it can't be
// determined (e.g. Arm SVE).
static int getRegisterSpaceBitsIfKnown(IREE::HAL::ExecutableTargetAttr target) {
  if (!target) {
    return 0;
  }
  DictionaryAttr targetConfig = target.getConfiguration();
  if (isX86(targetConfig)) {
    if (hasFeature(targetConfig, "+avx512f")) {
      return 32 * 512;
    } else if (hasFeature(targetConfig, "+avx") ||
               hasFeature(targetConfig, "+avx2")) {
      return 16 * 256;
    } else {
      return 16 * 128;
    }
  } else if (isAArch64(targetConfig)) {
    // 32 NEON/SVE registers (at least 128-bit each, returns the base size for
    // SVE).
    return 32 * 128;
  } else {
    // Don't know register space size as a compile-time constant on other
    // architectures.
    return 0;
  }
}

// Clamps in-place vecTileSizes to ensure that the tile sizes of all operands of
// `op` can simultaneously be allocated in SIMD registers. Does nothing when
// SIMD register space can't be determined as a compile-time constant (e.g. Arm
// SVE).
static void limitVectorTileSizes(Operation *op,
                                 SmallVectorImpl<int64_t> &vecTileSizes) {
  if (int registerSpaceBits = getRegisterSpaceBitsIfKnown(
          IREE::HAL::ExecutableTargetAttr::lookup(op))) {
    limitVectorTileSizes(op, vecTileSizes, registerSpaceBits,
                         registerSpaceBits);
  }
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
      cast<PartitionableLoopsInterface>(op).getPartitionableLoops(std::nullopt);
  for (auto i : partitionableLoops) {
    adjustedMinTileSizes[i] =
        config.minTileSizes.empty() ? 1 : config.minTileSizes[i];
    adjustedMaxTileSizes[i] = config.maxTileSizes.empty()
                                  ? clDefaultDistTileSize
                                  : config.maxTileSizes[i];
    adjustedVectorSizeHints[i] =
        config.vectorSizeHints.empty() ? 1 : config.vectorSizeHints[i];
  }

  LDBG() << "Adjusted min tile sizes: " << adjustedMinTileSizes;
  LDBG() << "Adjusted max tile sizes: " << adjustedMaxTileSizes;
  LDBG() << "Adjusted vector size hints: " << adjustedVectorSizeHints;

  SmallVector<int64_t> distributedTileSizes = getDefaultDistributionTileSizes(
      lbs, ubs, adjustedMinTileSizes, adjustedMaxTileSizes,
      adjustedVectorSizeHints);

  LDBG() << "Distributed tile sizes before fixups: " << distributedTileSizes;

  // Final fix up of the tile sizes to make sure that they divide the problem
  // size to make it vectorizable.
  for (auto i : llvm::seq<unsigned>(0, distributedTileSizes.size())) {
    if (!distributedTileSizes[i])
      continue;
    distributedTileSizes[i] = getMaxDistributionTileSize(
        lbs[i], ubs[i], distributedTileSizes[i], adjustedMinTileSizes[i],
        config.allowIncompleteTile);
  }
  LDBG() << "Distributed tile sizes after fixups: " << distributedTileSizes;
  return distributedTileSizes;
}

/// Splits the tile sizes in `parallelSizes` into `reductionSizes` for the
/// reduction loops.
static void splitParallelAndReductionTiles(
    Operation *op, SmallVectorImpl<int64_t> &parallelSizes,
    SmallVectorImpl<int64_t> &reductionSizes,
    SmallVectorImpl<bool> *parallelScalableFlags = nullptr,
    SmallVectorImpl<bool> *reductionScalableFlags = nullptr) {
  reductionSizes.assign(parallelSizes.begin(), parallelSizes.end());
  if (reductionScalableFlags) {
    assert(parallelScalableFlags && "expected parallel scalable flags!");
    reductionScalableFlags->assign(parallelScalableFlags->begin(),
                                   parallelScalableFlags->end());
  }
  TilingInterface tilingOp = cast<TilingInterface>(op);
  for (auto [index, iteratorType] :
       llvm::enumerate(tilingOp.getLoopIteratorTypes())) {
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
                                    SmallVectorImpl<int64_t> &vecTileSizes) {
  SmallVector<int64_t> staticLoopRanges = op.getStaticLoopRanges();
  for (auto [index, size, iterType] :
       llvm::enumerate(staticLoopRanges, op.getIteratorTypesArray())) {
    if (ShapedType::isStatic(size))
      continue;
    vecTileSizes[index] = 1;
  }
  LDBG() << "Set always-vectorize sizes: " << vecTileSizes;
}

/// A helper class to record different level tiling sizes and generate
/// corresponding IREE::CPU::LoweringConfigAttr for the rootOp. It implies that
/// the distribution tiling level is always set, even if tile sizes are all
/// zeros. Because a rootOp must have distribution tiling level.
/// Only vector level supports scalable tile sizes for now.
class LoweringConfigGenerator {
public:
  explicit LoweringConfigGenerator(Operation *op)
      : ctx(op->getContext()), rootOp(op) {}

  void setDistributionTileSizes(ArrayRef<int64_t> tileSizes) {
    assert(distTileSizes.empty() && "expected to set only once");
    distTileSizes.assign(tileSizes.begin(), tileSizes.end());
  }

  void setCacheTileSizes(ArrayRef<int64_t> tileSizes) {
    assert(cacheTileSizes.empty() && "expected to set only once");
    cacheTileSizes.assign(tileSizes.begin(), tileSizes.end());
  }

  void setVectorTileSizes(ArrayRef<int64_t> tileSizes,
                          ArrayRef<bool> scalableFlags = {}) {
    assert(vectorTileSizes.empty() && "expected to set only once");
    vectorTileSizes.assign(tileSizes.begin(), tileSizes.end());
    vectorScalableFlags.assign(scalableFlags.begin(), scalableFlags.end());
  }

  /// Returns a `IREE::CPU::LoweringConfigAttr` that is constructed by the
  /// existing values. By default, it will always contain distribution tile
  /// sizes, unless the rootOp does not implement TilingInterface.
  IREE::CPU::LoweringConfigAttr generateCPULoweringConfig() {
    SmallVector<NamedAttribute> items;
    if (!distTileSizes.empty()) {
      appendLoweringConfigLevelAttr(items, TilingLevel::DistributionTiles,
                                    distTileSizes);
    } else if (auto op = dyn_cast<TilingInterface>(rootOp)) {
      size_t numTilingDims = op.getLoopIteratorTypes().size();
      appendLoweringConfigLevelAttr(items, TilingLevel::DistributionTiles,
                                    SmallVector<int64_t>(numTilingDims, 0));
    }
    if (!cacheTileSizes.empty()) {
      SmallVector<int64_t> parallelTileSizes = cacheTileSizes;
      SmallVector<int64_t> reductionTileSizes;
      splitParallelAndReductionTiles(rootOp, parallelTileSizes,
                                     reductionTileSizes);
      appendLoweringConfigLevelAttr(items, TilingLevel::CacheParallelTiles,
                                    parallelTileSizes);
      appendLoweringConfigLevelAttr(items, TilingLevel::CacheReductionTiles,
                                    reductionTileSizes);
    }
    if (!vectorTileSizes.empty()) {
      SmallVector<int64_t> parallelTileSizes = vectorTileSizes;
      SmallVector<int64_t> reductionTileSizes;
      SmallVector<bool> parallelScalableFlags = vectorScalableFlags;
      SmallVector<bool> reductionScalableFlags;
      parallelScalableFlags.resize(parallelTileSizes.size(), false);
      splitParallelAndReductionTiles(rootOp, parallelTileSizes,
                                     reductionTileSizes, &parallelScalableFlags,
                                     &reductionScalableFlags);
      appendLoweringConfigLevelAttr(items,
                                    TilingLevel::VectorCommonParallelTiles,
                                    parallelTileSizes, parallelScalableFlags);
      appendLoweringConfigLevelAttr(items, TilingLevel::VectorReductionTiles,
                                    reductionTileSizes, reductionScalableFlags);
    }
    return IREE::CPU::LoweringConfigAttr::get(ctx, items);
  }

private:
  /// Appends the `level` with (`tileSizes`, `scalableFlags`) tiling config to
  /// `items`, if it is not a NOP config. E.g., if all the tile sizes are zeros,
  /// it means no tiling at all. Only the distribution tiling level is
  /// unconditionally added because a root op expects the level to be present.
  void appendLoweringConfigLevelAttr(SmallVectorImpl<NamedAttribute> &items,
                                     TilingLevel level,
                                     ArrayRef<int64_t> tileSizes,
                                     ArrayRef<bool> scalableFlags = {}) {
    if (level != TilingLevel::DistributionTiles &&
        llvm::all_of(tileSizes, [](int64_t v) { return v == 0; })) {
      return;
    }
    items.emplace_back(IREE::CPU::getTilingLevelName(level),
                       IREE::CPU::LoweringConfigAttr::getTilingLevelAttr(
                           ctx, tileSizes, scalableFlags));
  }

  MLIRContext *ctx;
  Operation *rootOp;

  // The tile sizes for distribution from the `rootOp`'s perspective.
  SmallVector<int64_t> distTileSizes;

  // The tile sizes for cache level tiling from the `rootOp`'s perspective.
  SmallVector<int64_t> cacheTileSizes;

  // The tile sizes and scalable flags for vector level tiling from the
  // `rootOp`'s perspective.
  SmallVector<int64_t> vectorTileSizes;
  SmallVector<bool> vectorScalableFlags;
};

/// A helper class that tracks dimension mappings both within individual
/// operations and across multiple operations by analyzing the producer-consumer
/// relationships of SSA values. This tracking is established by assigning a
/// global dimension index to all loop dimensions encountered. Dimensions
/// sharing the same global index are considered equivalent.
class IterationDimTracker {
public:
  explicit IterationDimTracker(ArrayRef<Operation *> operations)
      : operations(operations.begin(), operations.end()) {
    // Ensure operations are processed in topological order.
    mlir::computeTopologicalSorting(this->operations);
    buildDimMapping();
  }

  /// Returns true if the given dimension of `op` is common across all
  /// operations.
  bool isCommonDim(Operation *op, unsigned pos) {
    assert(operationToGlobalDimMaps.contains(op));
    int64_t dim = operationToGlobalDimMaps[op][pos];
    for ([[maybe_unused]] auto &[_, dims] : operationToGlobalDimMaps) {
      if (!llvm::is_contained(dims, dim)) {
        return false;
      }
    }
    return true;
  }

private:
  /// Builds and unifies dimension index mappings for all operations,
  /// using producer–consumer SSA value relationships.
  void buildDimMapping() {
    // Tracks equivalent global dimension indices.
    llvm::EquivalenceClasses<int64_t> indicesEquivalence;
    // For each SSA value, maps its local dimension index to a global index.
    // Value -> (local dim index -> global dim index)
    llvm::SmallDenseMap<Value, SmallVector<int64_t>> valueToGlobalDimMaps;

    for (Operation *op : operations) {
      auto tilingOp = cast<TilingInterface>(op);
      int64_t numLoops = tilingOp.getLoopIteratorTypes().size();
      // Unconditionally assign new global indices, to be unified later.
      for (int64_t i = 0; i < numLoops; ++i) {
        int64_t globalIndex = totalLoopNum++;
        indicesEquivalence.insert(globalIndex);
        operationToGlobalDimMaps[op].push_back(globalIndex);
      }
      // The assigned global dimension indices are now unified based on
      // producer–consumer SSA value relationships:
      // - For operations implementing `IndexingMapOpInterface`, unify
      // dimensions by iterating over their indexing maps.
      // - For pack/unpack operations, use an identity mapping, since tiling
      // applies to the outer (unpacked) dimensions.
      // - For all other (unknown) operations, assume an identity mapping for
      // any value whose rank matches the operation’s loop count.
      TypeSwitch<Operation *>(op)
          .Case<IndexingMapOpInterface>([&](auto op) {
            propagateOnIndexingMapOp(op, indicesEquivalence,
                                     valueToGlobalDimMaps);
          })
          .Case<linalg::PackOp, linalg::UnPackOp>([&](auto op) {
            propagateOnPackUnpackOp(op, indicesEquivalence,
                                    valueToGlobalDimMaps, numLoops);
          })
          .Default([&](auto op) {
            propagateOnUnknownOp(op, indicesEquivalence, valueToGlobalDimMaps,
                                 numLoops);
          });
    }

    // Remap the global dimension indices in two steps:
    // 1. Assign the same temporary index to all equivalent dimensions.
    // 2. Convert these temporary indices to a compact, zero-based range.
    auto applyReplaceMap = [&](llvm::SmallDenseMap<int64_t, int64_t> &map) {
      for (auto &opEntry : operationToGlobalDimMaps) {
        for (auto &dim : opEntry.second) {
          dim = map.lookup(dim);
        }
      }
    };
    llvm::SmallDenseMap<int64_t, int64_t> replaceMap0, replaceMap1;
    int64_t tempDimIndex = totalLoopNum;
    totalLoopNum = 0;
    for (auto it = indicesEquivalence.begin(); it != indicesEquivalence.end();
         ++it) {
      if (!(*it)->isLeader()) {
        continue;
      }
      for (auto mit = indicesEquivalence.member_begin(**it);
           mit != indicesEquivalence.member_end(); ++mit) {
        replaceMap0[*mit] = tempDimIndex;
      }
      replaceMap1[tempDimIndex] = totalLoopNum;
      tempDimIndex++;
      totalLoopNum++;
    }
    applyReplaceMap(replaceMap0);
    applyReplaceMap(replaceMap1);
  }

  /// Ties loop dimensions together based on the operation’s indexing maps,
  /// considering only simple result dimension expressions (`AffineDimExpr`).
  ///
  /// Complex expressions (e.g., `affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2,
  /// d1 * 3 + d3)>`) are ignored because they fall outside the "loop dimension"
  /// concept. Such expressions describe how indices are computed within the
  /// innermost loop body, but they do not directly identify which loop
  /// dimensions correspond or should be tied.
  void propagateOnIndexingMapOp(
      IndexingMapOpInterface indexingMapOp,
      llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
      llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueToGlobalDimMaps) {
    Operation *op = indexingMapOp.getOperation();
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      // Skip operands that have no known mapping from their producers.
      if (!valueToGlobalDimMaps.contains(value)) {
        continue;
      }
      AffineMap map = indexingMapOp.getMatchingIndexingMap(&operand);
      for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
        // Stop if the current dimension exceeds the number of mapped ones.
        if (dim >= valueToGlobalDimMaps[value].size()) {
          break;
        }
        // Skip on complex expressions.
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr) {
          continue;
        }
        int64_t pos = dimExpr.getPosition();
        // Unify the dimension index between the producer and the current op.
        indicesEquivalence.unionSets(valueToGlobalDimMaps[value][dim],
                                     operationToGlobalDimMaps[op][pos]);
      }
    }
    // Propogate to results.
    auto dsOp = cast<DestinationStyleOpInterface>(op);
    for (OpResult result : op->getResults()) {
      OpOperand *operand = dsOp.getTiedOpOperand(result);
      AffineMap map = indexingMapOp.getMatchingIndexingMap(operand);
      for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
        // Skip on complex expressions.
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr) {
          continue;
        }
        int64_t pos = dimExpr.getPosition();
        valueToGlobalDimMaps[result].push_back(
            operationToGlobalDimMaps[op][pos]);
      }
    }
  }

  /// Ties the dimensions of pack and unpack operations with their operands in
  /// the outer (unpacked) dimensions.
  void propagateOnPackUnpackOp(
      Operation *op, llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
      llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueToGlobalDimMaps,
      int64_t numLoops) {
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      if (!valueToGlobalDimMaps.contains(value)) {
        continue;
      }
      int64_t rank = cast<ShapedType>(value.getType()).getRank();
      int64_t outDimSize = std::min(rank, numLoops);
      for (int64_t i = 0; i < outDimSize; ++i) {
        indicesEquivalence.unionSets(valueToGlobalDimMaps[value][i],
                                     operationToGlobalDimMaps[op][i]);
      }
    }
    // Propagate to results.
    for (Value result : op->getResults()) {
      valueToGlobalDimMaps[result] = operationToGlobalDimMaps[op];
    }
  }

  /// Ties the dimensions of operations with their operands, if the operand rank
  /// matches the operation’s loop count.
  void propagateOnUnknownOp(
      Operation *op, llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
      llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueToGlobalDimMaps,
      int64_t numLoops) {
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      if (!valueToGlobalDimMaps.contains(value) ||
          numLoops != cast<ShapedType>(value.getType()).getRank()) {
        continue;
      }
      for (int64_t i = 0; i < numLoops; ++i) {
        indicesEquivalence.unionSets(valueToGlobalDimMaps[value][i],
                                     operationToGlobalDimMaps[op][i]);
      }
    }
    // Propagate to results.
    for (Value result : op->getResults()) {
      if (numLoops == cast<ShapedType>(result.getType()).getRank()) {
        valueToGlobalDimMaps[result] = operationToGlobalDimMaps[op];
      }
    }
  }

  SmallVector<Operation *> operations;
  // Tracks the total number of unique loop dimensions among the given set of
  // operations.
  int64_t totalLoopNum = 0;
  // For each compute operation, maps its local loop dimension index to the
  // global index. Operation -> (local dim index -> global dim
  // index)
  llvm::SmallDenseMap<Operation *, SmallVector<int64_t>>
      operationToGlobalDimMaps;
};

/// Returns the same lowering_config attribute with the updated tile sizes and
/// scalable tile flags. The distribution tiling sizes is not set if it is
/// false.
static IREE::Codegen::LoweringConfigAttrInterface
getNewLoweringConfig(MLIRContext *ctx,
                     ArrayRef<IREE::CPU::LoweringConfigLevelInfo> tilingInfo,
                     bool setDistributionConfig) {
  SmallVector<NamedAttribute> newItems;
  for (auto [level, tileSizes, scalableFlags] : tilingInfo) {
    if (!setDistributionConfig && level == TilingLevel::DistributionTiles) {
      continue;
    }
    // Distribution tile sizes is a must for rootOp, because it is the
    // definition of root op. An operation that has distribution tile sizes is
    // the root op. Other level can be dropped if all the tile sizes are zeros.
    if (level != TilingLevel::DistributionTiles &&
        llvm::all_of(tileSizes, [](int64_t val) { return val == 0; })) {
      continue;
    }
    newItems.emplace_back(IREE::CPU::getTilingLevelName(level),
                          IREE::CPU::LoweringConfigAttr::getTilingLevelAttr(
                              ctx, tileSizes, scalableFlags));
  }
  return IREE::CPU::LoweringConfigAttr::get(ctx,
                                            DictionaryAttr::get(ctx, newItems));
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
  if (targetAttr && isX86(targetAttr.getConfiguration())) {
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
    mlir::FunctionOpInterface entryPointFn, linalg::ContractionOpInterface op,
    ArrayRef<int64_t> distTileSizes, ArrayRef<int64_t> cacheTileSizes,
    ArrayRef<bool> inputVecScalableTileFlags, ArrayRef<int64_t> vecTileSizes,
    int vectorSize) {

  // 0. Preprocess for scalable vectors
  SmallVector<int64_t> roundedVecTileSizes(vecTileSizes);

  // The LLVM backend struggles to legalize non-power-of-two scalable vectors,
  // hence the extra rounding up.
  for (const auto &[index, size] : llvm::enumerate(roundedVecTileSizes)) {
    if (!size)
      continue;
    roundedVecTileSizes[index] =
        roundUpToPow2(size,
                      /*predicate=*/inputVecScalableTileFlags[index]);
  }

  SmallVector<int64_t> vectorTileSizes(roundedVecTileSizes.begin(),
                                       roundedVecTileSizes.end());
  SmallVector<bool> vectorScalableFlags(inputVecScalableTileFlags.begin(),
                                        inputVecScalableTileFlags.end());
  vectorScalableFlags.back() = false;

  LoweringConfigGenerator generator(op);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setCacheTileSizes(cacheTileSizes);
  generator.setVectorTileSizes(vecTileSizes, vectorScalableFlags);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  LDBG() << "Final tile sizes and scalable flags for contraction: "
         << loweringConfig;

  DictionaryAttr pipelineConfig =
      getPipelineConfWithPeelingAttr(op.getContext());
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, loweringConfig,
      DispatchLoweringPassPipeline::CPUDoubleTilingExpert,
      /*workgroupSize=*/{}, /*subgroupSize=*/{}, pipelineConfig);
}

static LogicalResult setMatmulRootConfig(
    mlir::FunctionOpInterface entryPointFn, linalg::ContractionOpInterface op,
    ArrayRef<int64_t> distTileSizes, ArrayRef<bool> inputVecScalableDims,
    ArrayRef<int64_t> inputVecTileSizes, int vectorSize,
    VectorPreProcStrategy vecPreProcStrategy) {
  assert(vecPreProcStrategy != VectorPreProcStrategy::Peeling &&
         "peeling should go to the setMatmulPeelingRootConfig method");
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();

  SmallVector<int64_t> vecTileSizes;
  SmallVector<bool> vecScalableFlags;
  int numScalableDims = llvm::count(inputVecScalableDims, true);
  for (auto [index, tileSize] : llvm::enumerate(inputVecTileSizes)) {
    int64_t sz = tileSize;
    bool isScalable = inputVecScalableDims[index];
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
    vecTileSizes.push_back(sz);
    // 1x scalable vectors e.g. vector<[1]xty> are also poorly supported, so
    // fallback to fixed vectorization if they occur:
    vecScalableFlags.push_back(sz > 1 ? isScalable : false);
  }
  limitVectorTileSizes(cast<linalg::LinalgOp>(op.getOperation()), vecTileSizes);

  LoweringConfigGenerator generator(op);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecTileSizes, vecScalableFlags);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  LDBG() << "Final tile sizes and scalable flags for contraction: "
         << loweringConfig;

  auto pipeline = DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, op, loweringConfig,
                                               pipeline);
}

/// Returns default hard-coded vector sizes for a give target. No smartness
/// should be introduced in this utility.
static void
getDefaultMatmulVectorSizes(linalg::LinalgOp op, int64_t vectorSize,
                            SmallVectorImpl<int64_t> &sizes,
                            SmallVectorImpl<bool> &scalableSizeFlags) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (targetAttr && isX86(targetAttr.getConfiguration())) {
    if (hasAVX512fFeature(targetAttr.getConfiguration())) {
      sizes.append({8, 32, 16});
    } else {
      sizes.append({1, 1, vectorSize});
    }
    return;
  }

  if (targetAttr && isAArch64(targetAttr.getConfiguration())) {
    sizes.append({8, 16, 1});

    // Specialisation for scalable vectorization.
    if (isScalableVectorizationEnabled() &&
        hasAnySVEFeature(targetAttr.getConfiguration())) {
      // Mark middle dimensions as scalable, so sizes are (8, [16], 1).
      scalableSizeFlags.append({false, true, false});
    }
    return;
  }

  if (targetAttr && isRISCV(targetAttr.getConfiguration())) {
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

/// Compute or adjust existing vector sizes using a generic heuristic that will
/// aim to fill at least one full vector register for all the element types of
/// the matmul. For now, the current heuristics only look at the N dimension but
/// we would introduce logic to also consider unrolling trade-offs between the
/// M, N and K.
///
/// Example: for an (i32 <- i8, i8) matmul and a 128-bit vector register, vector
/// size N would be at least 128/8=16.
///
/// NOTE: This function should not contain target-specific conditional code.
/// TODO: Currently it's only use on Aarch64. We should generalize it to other
/// targets.
static void getMatmulVectorSizesUsingFullVectorHeuristics(
    mlir::FunctionOpInterface entryPointFn, linalg::LinalgOp op,
    int64_t vectorSize, SmallVectorImpl<int64_t> &sizes,
    SmallVectorImpl<bool> &scalableSizeFlags) {
  if (sizes.empty())
    getDefaultMatmulVectorSizes(op, vectorSize, sizes, scalableSizeFlags);

  // Find the smallest type size in the matmul.
  SmallVector<Type> matmulTypes;
  auto operandTypes = op->getOperandTypes();
  matmulTypes.append(operandTypes.begin(), operandTypes.end());
  auto resultTypes = op->getResultTypes();
  matmulTypes.append(resultTypes.begin(), resultTypes.end());

  int64_t minSize = std::numeric_limits<int64_t>::max();
  for (Type mmType : matmulTypes) {
    if (auto shType = dyn_cast<ShapedType>(mmType))
      mmType = shType.getElementType();

    if (mmType.isSignlessIntOrFloat())
      minSize = std::min<int64_t>(minSize, mmType.getIntOrFloatBitWidth());
  }

  LDBG() << "Smallest type found: " << minSize << " bits";
  assert(minSize > 0 && minSize < std::numeric_limits<int64_t>::max() &&
         "Min size couldn't be computed");

  // Make sure that the smallest type can at least fill a full vector register
  // given the tile size of the main vector dimension (N).
  constexpr int64_t byteSizeInBits = 8;
  int64_t minNumElements =
      (getNativeVectorSizeInBytes(entryPointFn) * byteSizeInBits) / minSize;
  sizes[1] = std::max<int64_t>(sizes[1], minNumElements);
}

/// Utility to compute the tile sizes for RISC-V Vector.
/// For now, it only supports nonWideningLinalgElementType float.
/// TileSize is set to m = 7, n = maxNumberElementsForLMUL4, and k = 1.
///
/// Example: for an pure f32-matmul and a 512-bit vector register.
/// nativeVectorSize is equal to VLEN * LMUL2 / 8, so it's 128.
/// maxNumberElementsForLMUL4 = 128 * 2 * 8 / 32 = 64.
///
/// TODO: Currently it only supports for nonWideningLinalgElementType.
static void
getMatmulRISCVVectorSizes(mlir::FunctionOpInterface entryPointFn,
                          linalg::LinalgOp op, int64_t vectorSize,
                          SmallVectorImpl<int64_t> &sizes,
                          SmallVectorImpl<bool> &scalableSizeFlags) {
  if (sizes.empty())
    getDefaultMatmulVectorSizes(op, vectorSize, sizes, scalableSizeFlags);
  // TODO: support widening matmul.
  // Determines n dimension tile size with VLEN for
  // nonWideningLinalgElementType.
  FailureOr<Type> elementType = nonWideningLinalgElementType(op);
  if (failed(elementType))
    return;

  // nativeVectorSize is cacluated with VLEN and LMUL=2.
  int64_t nativeVectorSize = getNativeVectorSizeInBytes(entryPointFn);
  int64_t elementSize;
  if (elementType->isF16()) {
    elementSize = 16;
  } else if (elementType->isF32()) {
    elementSize = 32;
  } else if (elementType->isF64()) {
    elementSize = 64;
  } else {
    // TODO: support int data type
    return;
  }
  FailureOr<linalg::ContractionDimensions> cDims =
      linalg::inferContractionDims(op);
  if (failed(cDims) || cDims->m.size() != 1)
    return;
  // Use 7 x lmul4 to fully utilize vector registers.
  sizes[0] = 7;
  // Calculate tile size for the main vector dimension (N).
  constexpr int64_t kByteSizeInBits = 8;
  int64_t maxNumberElementsForLMUL4 =
      (nativeVectorSize * 2 * kByteSizeInBits) / elementSize;
  sizes[1] = maxNumberElementsForLMUL4;
  sizes[2] = 1;
  ArrayRef<int64_t> lhsShape = op.getShape(op.getDpsInputOperand(0));
  // If m = 1, set tile size to 1 x lmul8
  if (lhsShape[cDims->m[0]] == 1) {
    sizes[0] = 1;
    sizes[1] *= 2;
  }
}

/// Utility to compute the tile sizes for AArch64 SME. Unlike other targets, the
/// tile sizes picked here must exactly match multiples of the SME hardware
/// virtual tiles, as there is currently no support for lowering non-standard
/// shapes.
static void
getMatmulAArch64SMEVectorSizes(linalg::LinalgOp op,
                               SmallVectorImpl<int64_t> &sizes,
                               SmallVectorImpl<bool> &scalableSizeFlags) {
  // Double-check the operation is one that is supported for lowering to ArmSME.
  Operation *rawOp = op.getOperation();
  if (!(IREE::LinalgExt::isPureMatmul(rawOp) ||
        isa<linalg::MatmulTransposeAOp>(rawOp)))
    return;

  auto elementType = nonWideningLinalgElementType(op);
  if (failed(elementType))
    return;

  // TODO(macdue): Come up with some heuristics to pick the appropriate tiling
  // for SME, i.e. optimal layout based on static sizes.

  if (elementType->isF32()) {
    // Tile for [8]x[8], this results in equal loads from both the A and B
    // matrices and will use all four [4]x[4] 32-bit SME accumulators.
    sizes.append({8, 8, 1});
    scalableSizeFlags.append({true, true, false});
  }

  if (elementType->isF64()) {
    // Tile for [4]x[8], this results in loading twice as much from matrix B
    // than and will use all eight [2]x[2] 64-bit SME accumulators.
    // The B dimension is larger as it is known to be contiguous.
    sizes.append({4, 8, 1});
    scalableSizeFlags.append({true, true, false});
  }

  // TODO(macdue): Other element types (there is little support for anything
  // other than f32 and f64 yet).
}

/// Main utility to compute the vectorization/unrolling tile sizes.
static SizesAndScalableFlags
getMatmulVectorSizes(mlir::FunctionOpInterface entryPointFn,
                     linalg::LinalgOp op, int64_t vectorSize,
                     bool isQuantized) {
  SmallVector<int64_t> matmulTileSizes;
  SmallVector<bool> matmulScalableFlags;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);

  // TODO: Compute vector tile sizes using heuristics.

  if (targetAttr && isAArch64(targetAttr.getConfiguration())) {
    if (isScalableVectorizationEnabled() && !clDisableArmSMETiling &&
        hasSMEFeature(targetAttr.getConfiguration())) {
      // Note: This may not pick any sizes (which will fallback to the scalable
      // vectorization heuristics below).
      getMatmulAArch64SMEVectorSizes(op, matmulTileSizes, matmulScalableFlags);
    }

    // Try to maximize the vector register utilization for all the matmul
    // element types.
    if (matmulTileSizes.empty()) {
      getMatmulVectorSizesUsingFullVectorHeuristics(
          entryPointFn, op, vectorSize, matmulTileSizes, matmulScalableFlags);
    }
  }

  if (targetAttr && isRISCV(targetAttr.getConfiguration()) &&
      hasAnyVFeature(targetAttr.getConfiguration())) {
    // Use default tile size for matmul_transpose_b &
    // batch_matmul_transpose_b to avoid performance drop.
    if (!isa<linalg::MatmulTransposeBOp, linalg::BatchMatmulTransposeBOp>(
            op.getOperation())) {
      // Try to maximize the vector register utilization rate for matmul.
      getMatmulRISCVVectorSizes(entryPointFn, op, vectorSize, matmulTileSizes,
                                matmulScalableFlags);
    }
  }

  // If tile sizes were not computed by previous heuristics, use default
  // hard-coded tile sizes.
  if (matmulTileSizes.empty()) {
    getDefaultMatmulVectorSizes(op, vectorSize, matmulTileSizes,
                                matmulScalableFlags);
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

  LDBG() << "Matmul vector sizes: " << tileSizes;
  LDBG() << "Matmul vector scalable flags: " << scalableTileFlags;
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
setRootConfig(mlir::FunctionOpInterface entryPointFn,
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

  auto vecPreProcStrategy = getVectorPreProcStrategy(linalgOp);
  bool usePeelingPipeline =
      vecPreProcStrategy == VectorPreProcStrategy::Peeling;

  LDBG() << "Vector pre-processing strategy: " << vecPreProcStrategy;

  DistributionHeuristicConfig distConfig;
  distConfig.maxTileSizes.resize(numLoops, clDefaultDistTileSize);
  distConfig.allowIncompleteTile =
      vecPreProcStrategy != VectorPreProcStrategy::None;
  distConfig.vectorSizeHints.resize(numLoops, vectorSize);
  bool isBM = isa<linalg::BatchMatmulOp>(contractionOp.getOperation());
  if (isBM) {
    distConfig.maxTileSizes[0] = 1;
    distConfig.vectorSizeHints[0] = 1;
  }

  // Compute cache-level tile sizes. Cache a dimension only if there are
  // enough iterations.
  SmallVector<int64_t> cacheTileSizes;
  cacheTileSizes = getDefaultMatmulCacheSizes(linalgOp, isQuantized);
  cacheTileSizes = getMatmulCacheTileSizesForShape(
      cacheTileSizes, linalgOp.getStaticLoopRanges());

  // Choose the next non-zero tile size immediately after the distribution
  // level to help compute the distribution tile sizes.
  for (auto [cacheTileSize, vecTileSize] :
       llvm::zip_equal(cacheTileSizes, vecTileSizes)) {
    int64_t minTileSize = cacheTileSize != 0 ? cacheTileSize : vecTileSize;
    distConfig.minTileSizes.push_back(minTileSize);
  }
  // FIXME: Apply maxTileSize modification for all targets.
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  if (targetAttr && isRISCV(targetAttr.getConfiguration()) &&
      hasAnyVFeature(targetAttr.getConfiguration())) {
    LDBG() << "RISC-V Aggressive Distribution: " << clEnableRiscvAggressiveDist;
    for (auto loopNum :
         llvm::seq<unsigned>(static_cast<unsigned>(isBM), numLoops)) {
      if (clEnableRiscvAggressiveDist) {
        if (distConfig.maxTileSizes[loopNum] <=
            distConfig.minTileSizes[loopNum]) {
          distConfig.maxTileSizes[loopNum] =
              2 * distConfig.minTileSizes[loopNum];
        }
      } else {
        distConfig.maxTileSizes[loopNum] = std::max(
            distConfig.maxTileSizes[loopNum], distConfig.minTileSizes[loopNum]);
      }
    }
  }
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(linalgOp, distConfig);

  // TODO: We set cache tile sizes to the distribution sizes for now (no-op) to
  // make sure there are no performance changes. This will let us change the
  // distribution sizes while still preserving the cache behavior of the
  // original sizes. When we set proper sizes, we should call again
  // `getMatmulCacheTileSizesForShape(cacheTileSizes, distTileSizes);` here as
  // the `getDefaultDistributedLevelTileSizes` above may return sizes that are
  // smaller than `minTileSizes`, so we have to adjust the cache sizes again.
  cacheTileSizes = distTileSizes;

  LDBG() << "Distribution tile sizes: " << distTileSizes;
  LDBG() << "Cache tile sizes: " << cacheTileSizes;
  LDBG() << "Vector tile sizes: " << vecTileSizes;
  LDBG() << "Vector scalable tile flags: " << vecScalableFlags;
  LDBG() << "Vector size: " << vectorSize;

  if (usePeelingPipeline) {
    return setMatmulPeelingRootConfig(
        entryPointFn, contractionOp, distTileSizes, cacheTileSizes,
        vecScalableFlags, vecTileSizes, vectorSize);
  }

  return setMatmulRootConfig(entryPointFn, contractionOp, distTileSizes,
                             vecScalableFlags, vecTileSizes, vectorSize,
                             vecPreProcStrategy);
}

static IREE::Codegen::LoweringConfigAttrInterface
getMmt4dLoweringConfig(linalg::LinalgOp op) {
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
  unsigned numLoops = op.getNumLoops();
  SmallVector<int64_t> vecTileSizes(numLoops, 1);
  assert(vecTileSizes.size() == mmt4dDimBase + 6);
  vecTileSizes[mmt4dDimBase + 3] = M0;
  vecTileSizes[mmt4dDimBase + 4] = N0;
  vecTileSizes[mmt4dDimBase + 5] = K0;
  limitVectorTileSizes(op, vecTileSizes);

  LoweringConfigGenerator generator(op);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecTileSizes);
  return generator.generateCPULoweringConfig();
}

/// Sets the lowering configuration for dispatch region for linalg.mmt4d
/// root op
static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::Mmt4DOp mmt4dOp) {
  assert(!getLoweringConfig(mmt4dOp) && "expected lowering_config is not set");
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, mmt4dOp, getMmt4dLoweringConfig(mmt4dOp),
      DispatchLoweringPassPipeline::Mmt4dTilingExpert);
}

/// Sets the lowering configuration for dispatch region for linalg.batch_mmt4d
/// root op
static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::BatchMmt4DOp batchMmt4dOp) {
  assert(!getLoweringConfig(batchMmt4dOp) &&
         "expected lowering_config is not set");
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, batchMmt4dOp, getMmt4dLoweringConfig(batchMmt4dOp),
      DispatchLoweringPassPipeline::Mmt4dTilingExpert);
}

static bool isPackMatmulLHS(linalg::PackOp op) {
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
static SmallVector<int64_t>
getPackVectorTileSizes(mlir::FunctionOpInterface entryPointFn,
                       linalg::PackOp op) {
  SmallVector<int64_t> tileSizes(op.getSourceRank(), 1);
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  int64_t vectorSize = getVectorSize(entryPointFn, op.getSourceType());
  if (!targetAttr || !hasAVX512fFeature(targetAttr.getConfiguration()) ||
      !isPackMatmulLHS(op)) {
    return tileSizes;
  }
  if (op.getSourceType().getElementType().isF32()) {
    tileSizes.back() = vectorSize;
  }
  // TODO(#16314): Generate efficient tile sizes for non-f32 cases.
  if (op.getSourceType().getElementType().isF16()) {
    // We adjust the vector size to half to use the same lowering strategy as
    // f32.
    tileSizes.back() = vectorSize / 2;
  }
  return tileSizes;
}

static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::PackOp op) {
  assert(!getLoweringConfig(op) && "expected lowering_config is not set");

  int srcRank = op.getSourceRank();
  SmallVector<int64_t> innerTiles = op.getStaticTiles();
  ArrayRef<int64_t> dimPos = op.getInnerDimsPos();
  int64_t vectorSize = getVectorSize(entryPointFn, op.getSourceType());

  DistributionHeuristicConfig distConfig;
  distConfig.maxTileSizes.resize(srcRank, clDefaultDistTileSize);
  distConfig.allowIncompleteTile = true;
  distConfig.vectorSizeHints.resize(srcRank, 1);
  for (auto pos : dimPos) {
    distConfig.vectorSizeHints[pos] = vectorSize;
  }
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(op, distConfig);

  // The default function aims to returns the number of workload per workgroup,
  // but it does not know that it is working on packed domain. We need to take
  // inner tile sizes into account and adjust the distribution tile sizes.
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    if (distTileSizes[pos] == 0 || ShapedType::isDynamic(size))
      continue;
    distTileSizes[pos] = distTileSizes[pos] / size;
    distTileSizes[pos] = std::max<int64_t>(distTileSizes[pos], 1);
  }

  // Dynamic inner tiles lead to unbounded stack allocation (which is introduced
  // by tensor.pad op), so we do not decompose the cases. The x86 and risc-v
  // backends prefer to not decompose the ops.
  DictionaryAttr pipelineConfig;
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  DictionaryAttr targetConfig = target ? target.getConfiguration() : nullptr;
  bool hasDynamicInnerTile =
      llvm::any_of(op.getMixedTiles(), llvm::IsaPred<Value>);
  if (!hasDynamicInnerTile && targetConfig && !isX86(targetConfig) &&
      !isRISCV(targetConfig)) {
    pipelineConfig = getPipelineConfWithDecompositionAttr(op.getContext());
  }

  SmallVector<int64_t> vecTileSizes = getPackVectorTileSizes(entryPointFn, op);
  LoweringConfigGenerator generator(op);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecTileSizes);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, loweringConfig,
      DispatchLoweringPassPipeline::CPUDataTiling, /*workgroupSize=*/{},
      /*subgroupSize=*/{}, pipelineConfig);
}

static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::UnPackOp op) {
  DistributionHeuristicConfig distConfig;
  distConfig.maxTileSizes.resize(op.getDestRank(), clDefaultDistTileSize);
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(op, distConfig);

  // Fixup for making distTileSizes be multiple of inner_tile_sizes.
  SmallVector<int64_t> innerTiles = op.getStaticTiles();
  ArrayRef<int64_t> dimPos = op.getInnerDimsPos();
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    if (distTileSizes[pos] == 0 || ShapedType::isDynamic(size))
      continue;
    distTileSizes[pos] = llvm::alignTo(distTileSizes[pos], size);
  }

  SmallVector<int64_t> vecTileSizes(op.getDestRank(), 1);
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    vecTileSizes[pos] = ShapedType::isDynamic(size) ? 1 : size;
  }

  // Dynamic inner tiles lead to unbounded stack allocation (which is introduced
  // by tensor.pad op), so we do not decompose the cases. The x86 and risc-v
  // backends prefer to not decompose the ops.
  DictionaryAttr pipelineConfig;
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  bool hasDynamicInnerTile =
      llvm::any_of(op.getMixedTiles(), llvm::IsaPred<Value>);
  if (!hasDynamicInnerTile && target && !isX86(target.getConfiguration()) &&
      !isRISCV(target.getConfiguration())) {
    pipelineConfig = getPipelineConfWithDecompositionAttr(op.getContext());
  }
  LoweringConfigGenerator generator(op);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecTileSizes);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, loweringConfig,
      DispatchLoweringPassPipeline::CPUDataTiling, /*workgroupSize=*/{},
      /*subgroupSize=*/{}, pipelineConfig);
}

static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   IREE::LinalgExt::AttentionOp attnOp) {
  FailureOr<IREE::LinalgExt::AttentionOpDetail> maybeOpInfo =
      IREE::LinalgExt::AttentionOpDetail::get(
          attnOp.getQueryMap(), attnOp.getKeyMap(), attnOp.getValueMap(),
          attnOp.getOutputMap());
  assert(succeeded(maybeOpInfo) && "failed to infer attention dims");
  auto opInfo = maybeOpInfo.value();

  SmallVector<int64_t> lbs, ubs;
  getRangeBounds(attnOp, lbs, ubs);

  LDBG() << "Attention Detail:";
  LDBG() << "Batch: " << llvm::interleaved_array(opInfo.getBatchDims());
  LDBG() << "M: " << llvm::interleaved_array(opInfo.getMDims());
  LDBG() << "K1: " << llvm::interleaved_array(opInfo.getK1Dims());
  LDBG() << "K2: " << llvm::interleaved_array(opInfo.getK2Dims());
  LDBG() << "N: " << llvm::interleaved_array(opInfo.getNDims());

  // Batch, M and N (parallel dimensions) are distributed on workgroups.
  DistributionHeuristicConfig config;
  int64_t vectorSize =
      getVectorSize(entryPointFn, attnOp.getOutput().getType());
  config.maxTileSizes.resize(opInfo.getDomainRank(), clDefaultDistTileSize);
  config.vectorSizeHints.resize(opInfo.getDomainRank(), vectorSize);
  // Distribute batch dimensions completely on workgroups (tile_size = 1).
  for (int batch : opInfo.getBatchDims()) {
    config.maxTileSizes[batch] = 1;
    config.vectorSizeHints[batch] = 1;
  }
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(attnOp, config);

  // Batch, M and N (parallel dimensions) are distributed on workgroups.
  SmallVector<int64_t> vecTileSizeBounds(attnOp.getIterationDomainRank());
  for (auto i : llvm::seq<int64_t>(0, vecTileSizeBounds.size())) {
    vecTileSizeBounds[i] = distTileSizes[i] ? distTileSizes[i] : ubs[i];
  }

  SmallVector<int64_t> vecTileSizes(vecTileSizeBounds.size(), 1);
  // Due to the way attention works, K1 dimensions cannot be tiled. Mark k1
  // reduction dimensions not to distribute.
  for (int i : opInfo.getK1Dims()) {
    vecTileSizes[i] = 0;
  }

  for (auto i : llvm::seq<int64_t>(0, vecTileSizeBounds.size())) {
    if (vecTileSizes[i] == 0) {
      continue;
    }
    // TODO: Use native tile size here once bufferization is fixed for scf.
    vecTileSizes[i] = getMaxVectorTileSize(
        /*numElem=*/vecTileSizeBounds[i], vectorSize, vectorSize);
  }

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(attnOp);

  // Limit large vector sizes.
  {
    // TODO: These aren't the best heuristics and we can do much much better if
    // we reuse the matmul heuristics here. limitVectorTileSizes doesn't try to
    // convert the matmul into a square matmul, which is bad.

    // Limit vector sizes based on register space.
    if (int64_t registerSpaceBits = getRegisterSpaceBitsIfKnown(targetAttr)) {
      SmallVector<Type> pvOperandTypes({Float32Type::get(attnOp.getContext()),
                                        attnOp.getValue().getType(),
                                        Float32Type::get(attnOp.getContext())});
      SmallVector<AffineMap> pvMaps(
          {opInfo.getSMap(), attnOp.getValueMap(), attnOp.getOutputMap()});

      // Limit vector sizes based on register space available.
      limitVectorTileSizes(vecTileSizes, registerSpaceBits * 2,
                           registerSpaceBits * 2, pvOperandTypes, pvMaps,
                           vecTileSizeBounds);

      SmallVector<Type> qkOperandTypes({attnOp.getQuery().getType(),
                                        attnOp.getKey().getType(),
                                        Float32Type::get(attnOp.getContext())});
      SmallVector<AffineMap> qkMaps(
          {attnOp.getQueryMap(), attnOp.getKeyMap(), opInfo.getSMap()});

      // Limit vector sizes based on register space available.
      limitVectorTileSizes(vecTileSizes, registerSpaceBits * 2,
                           registerSpaceBits * 2, qkOperandTypes, qkMaps,
                           vecTileSizeBounds);
    }

    // Limit vector sizes based on large vector sizes check.
    if (targetAttr) {
      int64_t maxVectorSizeBits =
          getMaxVectorSizeForLargeVectorCheck(targetAttr) * 8;
      SmallVector<Type> operandTypes;
      SmallVector<AffineMap> maps;
      // The large vector size check also has a restriction that the entire
      // iteration space of a contraction cannot exceed the maxVectorSizeBits,
      // so add that restriction here.
      //
      // Add restriction on QK matmul.
      auto identity = AffineMap::getMultiDimIdentityMap(opInfo.getDomainRank(),
                                                        attnOp.getContext());
      operandTypes.push_back(Float32Type::get(attnOp.getContext()));
      maps.push_back(identity.dropResults(opInfo.getNDims()));
      // Add restriction on PV matmul.
      operandTypes.push_back(Float32Type::get(attnOp.getContext()));
      maps.push_back(identity.dropResults(opInfo.getK1Dims()));
      // Limit vector sizes based on constraints.
      limitVectorTileSizes(vecTileSizes, maxVectorSizeBits, maxVectorSizeBits,
                           operandTypes, maps, vecTileSizeBounds);
    }
  }

  LoweringConfigGenerator generator(attnOp);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecTileSizes);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  LDBG() << "Set lowering_config for attnOp: " << loweringConfig;
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, attnOp, loweringConfig,
      DispatchLoweringPassPipeline::CPULinalgExtTileAndVectorize);
}

/// Sets the lowering configuration for dispatch region for linalg_ext.fft
/// root op.
static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
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
  // Append vector level tiling sizes using zero values, which means no tiling
  // in the pipeline.
  LoweringConfigGenerator generator(fftOp);
  generator.setDistributionTileSizes(distTileSizes);
  SmallVector<int64_t> zeros(rank, 0);
  generator.setVectorTileSizes(zeros);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, fftOp, loweringConfig,
      DispatchLoweringPassPipeline::CPULinalgExtTileAndVectorize);
}

/// Sets the lowering configuration for dispatch region for winograd ops:
///   linalg_ext.winograd.filter_transform
///   linalg_ext.winograd.input_transform
///   linalg_ext.winograd.output_transform
/// The vector tile sizes should be 1 for each dim here, because
/// the winograd decomposition relies on these unit dimensions.
template <typename WinogradOp>
static LogicalResult
setWinogradRootConfig(mlir::FunctionOpInterface entryPointFn,
                      WinogradOp winogradOp) {
  static_assert(
      std::is_same<WinogradOp, IREE::LinalgExt::WinogradInputTransformOp>() ||
          std::is_same<WinogradOp,
                       IREE::LinalgExt::WinogradOutputTransformOp>() ||
          std::is_same<WinogradOp,
                       IREE::LinalgExt::WinogradFilterTransformOp>(),
      "op expected to be a winograd op");
  assert(!getLoweringConfig(winogradOp) &&
         "expected lowering_config is not set");
  auto iterationRank = winogradOp.getIterationDomainRank();
  SmallVector<int64_t> vecSizeHints(iterationRank, 1);
  DistributionHeuristicConfig distConfig;
  distConfig.vectorSizeHints = vecSizeHints;
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(winogradOp, distConfig);
  SmallVector<int64_t> vecTileSizes(iterationRank, 1);
  LoweringConfigGenerator generator(winogradOp);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecTileSizes);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, winogradOp, loweringConfig,
      DispatchLoweringPassPipeline::CPULinalgExtTileAndVectorize);
}

static void setVectorTileSizes(linalg::LinalgOp op,
                               ArrayRef<int64_t> distTileSizes,
                               ArrayRef<int64_t> minTileSizes,
                               ArrayRef<int64_t> maxTileSizes,
                               VectorPreProcStrategy vecPreProcStrategy,
                               SmallVectorImpl<int64_t> &vecTileSizes) {
  int numLoops = op.getNumLoops();
  vecTileSizes.append(numLoops, 0);
  SmallVector<int64_t> staticLoopRanges = op.getStaticLoopRanges();
  for (auto loopNum : llvm::seq<int>(0, numLoops)) {
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

/// Sets the default lowering configuration for a generic op to use
/// CPUDoubleTilingExpert pipeline.
static LogicalResult
setDefaultGenericOpRootConfig(mlir::FunctionOpInterface entryPointFn,
                              linalg::GenericOp genericOp,
                              const LinalgOpInfo &linalgOpInfo,
                              const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");
  LDBG() << "Setting default generic op root configuration";

  // If there are no loops, there is nothing to do.
  unsigned numLoops = genericOp.getNumLoops();
  if (numLoops == 0) {
    LoweringConfigGenerator generator(genericOp);
    return setOpConfigAndEntryPointFnTranslation(
        entryPointFn, genericOp, generator.generateCPULoweringConfig(),
        DispatchLoweringPassPipeline::CPUDefault);
  }

  DistributionHeuristicConfig distConfig;
  // For generic ops we'll use the default divided by 2 to control the stack
  // allocation limit See #9469 for example.
  distConfig.maxTileSizes.append(numLoops, clDefaultDistTileSize / 2);

  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(genericOp, distConfig);
  LDBG() << "Final tile sizes for distribution: " << distTileSizes;

  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  LDBG() << "Vectorization pre-processing strategy " << vecPreProcStrategy;

  // Set the next level tile sizes.
  SmallVector<int64_t> vecTileSizes;
  setVectorTileSizes(genericOp, distTileSizes,
                     getMinTilingSizesForEachDim(entryPointFn, genericOp,
                                                 linalgOpInfo,
                                                 targetMLTransInfo),
                     distConfig.maxTileSizes, vecPreProcStrategy, vecTileSizes);
  limitVectorTileSizes(genericOp, vecTileSizes);

  LoweringConfigGenerator generator(genericOp);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecTileSizes);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  LDBG() << "Set lowering_config: " << loweringConfig;

  // For non-tensor based ops use the Buffer ops pipeline.
  DispatchLoweringPassPipeline passPipeline;
  DictionaryAttr pipelineConfig;
  if (genericOp.hasPureTensorSemantics()) {
    passPipeline = DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
    if (vecPreProcStrategy == VectorPreProcStrategy::Peeling) {
      pipelineConfig = getPipelineConfWithPeelingAttr(genericOp.getContext());
    }
  } else {
    passPipeline = DispatchLoweringPassPipeline::CPUBufferOpsTileAndVectorize;
  }

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, genericOp, loweringConfig, passPipeline,
      /*workgroupSize=*/{},
      /*subgroupSize=*/{}, pipelineConfig);
}

/// Utility to return the transpose vector `sizes` for X86. Empty `sizes` on
/// return indicates failure.
static void getTransposeX86VectorSizes(
    linalg::GenericOp genericOp, IREE::HAL::ExecutableTargetAttr targetAttr,
    ArrayRef<int64_t> minTileSizes, SmallVectorImpl<int64_t> &sizes) {
  if (!targetAttr || !hasAVX2Feature(targetAttr.getConfiguration()) ||
      !x86TransposeLoweringPrecondition(genericOp))
    return;

  if (llvm::count_if(minTileSizes,
                     [](int64_t tileSize) { return tileSize > 1; }) != 2) {
    // Transpose patterns are not applicable if vectorizing more or less than
    // two dims.
    return;
  }

  // Make sure that the original tile sizes are greater than or equal to the
  // tile sizes to be used for the transpose op (e.g., 8x8, 16x16, etc).
  int64_t targetVectorSize = 8;
  if (llvm::any_of(minTileSizes, [&](int64_t tileSize) {
        return tileSize > 1 && tileSize < targetVectorSize;
      })) {
    return;
  }

  // Target 16x16 tile sizes if there are AVX512 features and all the tile sizes
  // are greater than or equal to 16.
  if (hasAVX512fFeature(targetAttr.getConfiguration()) &&
      llvm::all_of(minTileSizes, [](int64_t tileSize) {
        return tileSize == 1 || tileSize >= 16;
      })) {
    targetVectorSize = 16;
  }

  // Replace dims to be vectorized with the new tile sizes.
  sizes.assign(minTileSizes.begin(), minTileSizes.end());
  std::replace_if(
      sizes.begin(), sizes.end(), [](int64_t tileSize) { return tileSize > 1; },
      targetVectorSize);
}

/// Utility to return the transpose vector `sizes` for AArch64. Empty `sizes` on
/// return indicates failure.
/// NOTE: only SME is currently supported.
static void getTransposeAArch64VectorSizes(
    linalg::GenericOp genericOp, IREE::HAL::ExecutableTargetAttr targetAttr,
    SmallVectorImpl<int64_t> &sizes, SmallVectorImpl<bool> &scalableFlags) {
  if (!targetAttr || !isLinalgGeneric2DTranspose(genericOp))
    return;

  auto elementType = nonWideningLinalgElementType(genericOp);
  if (failed(elementType))
    return;

  if (hasSMEFeature(targetAttr.getConfiguration()) &&
      isScalableVectorizationEnabled() && !clDisableArmSMETiling) {
    if (elementType->isF32()) {
      sizes.append({4, 4});
    } else if (elementType->isF64()) {
      sizes.append({2, 2});
    } else {
      return;
    }
    scalableFlags.append({true, true});
  }
}

/// Main utility to compute the transpose vector sizes.
static std::optional<SizesAndScalableFlags>
getTransposeVectorSizes(mlir::FunctionOpInterface entryPointFn,
                        linalg::GenericOp genericOp,
                        const LinalgOpInfo &linalgOpInfo,
                        const TargetMLTransformInfo &targetMLTransInfo) {
  SmallVector<int64_t> tileSizes;
  SmallVector<bool> scalableFlags;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  if (targetAttr && isX86(targetAttr.getConfiguration())) {
    SmallVector<int64_t> minTileSizes = getMinTilingSizesForEachDim(
        entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
    getTransposeX86VectorSizes(genericOp, targetAttr, minTileSizes, tileSizes);
  } else if (targetAttr && isAArch64(targetAttr.getConfiguration())) {
    getTransposeAArch64VectorSizes(genericOp, targetAttr, tileSizes,
                                   scalableFlags);
  }

  if (tileSizes.empty())
    return std::nullopt;

  // If scalable flags are empty, assume target doesn't care about scalability.
  if (scalableFlags.empty())
    scalableFlags = SmallVector<bool>(tileSizes.size(), false);

  LDBG() << "Transpose vector sizes: " << tileSizes;
  LDBG() << "Transpose vector scalable flags: " << scalableFlags;
  return std::make_pair(tileSizes, scalableFlags);
}

/// Sets the lowering configuration for a generic op implementing a
/// transposition to use CPUDoubleTilingExpert pipeline.
static LogicalResult
setTransposeLikeOpRootConfig(mlir::FunctionOpInterface entryPointFn,
                             linalg::GenericOp genericOp,
                             const LinalgOpInfo &linalgOpInfo,
                             const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");

  if (!linalgOpInfo.isTranspose())
    return failure();

  LDBG() << "Setting transpose-like op root configuration";

  std::optional<SizesAndScalableFlags> vecDims = getTransposeVectorSizes(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
  if (!vecDims)
    return failure();

  auto [vecSizes, vecScalableDims] = *vecDims;

  DistributionHeuristicConfig distConfig;
  distConfig.minTileSizes = vecSizes;
  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  LDBG() << "Vectorization pre-processing strategy " << vecPreProcStrategy;
  if (vecPreProcStrategy != VectorPreProcStrategy::None) {
    distConfig.allowIncompleteTile = true;
  }
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(genericOp, distConfig);

  LoweringConfigGenerator generator(genericOp);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecSizes, vecScalableDims);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  LDBG() << "Set lowering_config: " << loweringConfig;

  auto passPipeline = DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, genericOp,
                                               loweringConfig, passPipeline);
}

/// Sets elementwise dispatches to use peeling approach. It scales the number of
/// workload per workgroup to a larger number, which prevents runtime overheads
/// from tiny dispatches.
static LogicalResult setElementwiseGenericOpRootConfig(
    mlir::FunctionOpInterface entryPointFn, linalg::GenericOp genericOp,
    const LinalgOpInfo &linalgOpInfo,
    const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");
  LDBG() << "Setting elementwise generic op root configuration";

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
  LDBG() << "Vector pre-processing strategy: " << vecPreProcStrategy;

  // Adjust tiling sizes of vector levels to avoid large unroll factors. Most of
  // the cases are f32 and i32, so we divide it by 4.
  int64_t vecSize = getNativeVectorSizeInBytes(entryPointFn) / 4;
  SmallVector<int64_t> vecTileSizes = distConfig.minTileSizes;
  for (auto &i : vecTileSizes) {
    i = roundUpToPow2(std::min(i, vecSize),
                      vecPreProcStrategy == VectorPreProcStrategy::Masking);
  }

  LoweringConfigGenerator generator(genericOp);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecTileSizes);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  LDBG() << "Set lowering_config for element-wise op: " << loweringConfig;

  DispatchLoweringPassPipeline passPipeline;
  DictionaryAttr pipelineConfig;
  if (genericOp.hasPureBufferSemantics()) {
    passPipeline = DispatchLoweringPassPipeline::CPUBufferOpsTileAndVectorize;
  } else {
    passPipeline = DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
  }

  if (vecPreProcStrategy == VectorPreProcStrategy::Peeling) {
    pipelineConfig = getPipelineConfWithPeelingAttr(genericOp.getContext());
  }

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, genericOp, loweringConfig, passPipeline,
      /*workgroupSize=*/{},
      /*subgroupSize=*/{}, pipelineConfig);
}

/// Sets the lowering configuration for a generic op to use
/// CPUDoubleTilingExpert pipeline.
static LogicalResult
setRootConfig(mlir::FunctionOpInterface entryPointFn,
              linalg::GenericOp genericOp, const LinalgOpInfo &linalgOpInfo,
              const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");

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
static LogicalResult
setConvRootConfig(mlir::FunctionOpInterface entryPointFn,
                  linalg::LinalgOp convOp, ArrayRef<int64_t> targetTileSizes,
                  int64_t vectorSize,
                  VectorPreProcStrategy vecPreProcStrategy) {
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
  SmallVector<int64_t> vecTileSizes(targetTileSizes.begin(),
                                    targetTileSizes.end());
  for (auto i : llvm::seq<unsigned>(0, vecTileSizes.size())) {
    auto tileSize = distTileSizes[i] ? distTileSizes[i] : shapes[i];
    // If the tile size is intended to be 1, do not adjust it to `vectorSize`.
    // The ops will be decomposed to lower-rank named ops.
    if (vecTileSizes[i] != 1) {
      vecTileSizes[i] =
          getMaxVectorTileSize(tileSize, vecTileSizes[i], vectorSize);
    }
  }
  limitVectorTileSizes(convOp, vecTileSizes);
  setAlwaysVectorizeSizes(convOp, vecTileSizes);

  // Set "scalable" flags.
  int64_t numTilingDims = vecTileSizes.size();
  SmallVector<bool> vecScalableFlags(numTilingDims, false);
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  if (targetAttr && isAArch64(targetAttr.getConfiguration()) &&
      hasAnySVEFeature(targetAttr.getConfiguration()) &&
      isScalableVectorizationEnabled() &&
      isa<linalg::DepthwiseConv2DNhwcHwcOp>(convOp)) {
    auto dims = linalg::inferConvolutionDims(convOp);
    // Make the channel dim scalable
    vecScalableFlags[dims->depth[0]] = true;
  }

  DictionaryAttr pipelineConfig;
  if (vecPreProcStrategy == VectorPreProcStrategy::Peeling) {
    pipelineConfig = getPipelineConfWithPeelingAttr(convOp.getContext());
  }

  LoweringConfigGenerator generator(convOp);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecTileSizes, vecScalableFlags);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, convOp, loweringConfig,
      DispatchLoweringPassPipeline::CPUConvTileAndDecomposeExpert,
      /*workgroupSize=*/{}, /*subgroupSize=*/{}, pipelineConfig);
}

/// Main utility to compute the vectorization/unrolling tile sizes.
/// Note that this only works for NHWC input and HWCF kernel/filter
/// convolutions, where the shape is [N, OH, OW, OC, KH, KW, (IC)].
static SmallVector<int64_t>
getNhwcConvVectorSizes(mlir::FunctionOpInterface entryPointFn,
                       linalg::ConvolutionOpInterface op, int64_t vectorSize) {
  bool isSupported = is2DConvOp(op) || is2DDepthConvOp(op) || is2DPoolingOp(op);
  (void)isSupported;
  assert(isSupported && "conv op is not supported");

  SmallVector<int64_t> tileSizes;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);

  if (targetAttr) {
    DictionaryAttr targetConfig = targetAttr.getConfiguration();
    if (isX86(targetConfig)) {
      if (is2DConvOp(op))
        return {1, 1, 8, vectorSize, 1, 1, 8};
      if (is2DDepthConvOp(op))
        return {1, 1, 8, vectorSize, 1, 3};
      if (is2DPoolingOp(op))
        return {1, 1, 8, vectorSize, 1, 8};
      llvm_unreachable("unsupported conv");
    }
    if (isRISCV(targetConfig)) {
      if (is2DConvOp(op))
        return {1, 1, 8, vectorSize * 2, 1, 1, 8};
      if (is2DDepthConvOp(op))
        return {1, 1, 8, vectorSize, 1, 3};
      if (is2DPoolingOp(op))
        return {1, 1, 8, vectorSize * 2, 1, 8};
      llvm_unreachable("unsupported conv");
    }
    if (isAArch64(targetConfig)) {
      if (is2DConvOp(op))
        return {1, 1, 32, 64, 1, 1, 16};
      if (is2DDepthConvOp(op))
        return {1, 1, 4, 4, 1, 4};
      if (is2DPoolingOp(op))
        return {1, 1, 32, 64, 1, 16};
      llvm_unreachable("unsupported conv");
    }
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
setConvInterfaceRootConfig(mlir::FunctionOpInterface entryPointFn,
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

  auto vecPreProcStrategy =
      getVectorPreProcStrategy(cast<linalg::LinalgOp>(convOp.getOperation()));
  return setConvRootConfig(entryPointFn,
                           cast<linalg::LinalgOp>(convOp.getOperation()),
                           targetTileSizes, vectorSize, vecPreProcStrategy);
}

static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   tensor::PadOp padOp) {
  SmallVector<int64_t> lbs, ubs;
  getRangeBounds(cast<TilingInterface>(padOp.getOperation()), lbs, ubs);

  int64_t numLoops = lbs.size();
  unsigned typeWidthInBytes = IREE::Util::getRoundedElementByteWidth(
      padOp.getResultType().getElementType());
  int64_t typeVectorSize = getVectorSize(entryPointFn, typeWidthInBytes);
  DistributionHeuristicConfig distConfig;
  distConfig.vectorSizeHints.append(numLoops, 1);
  if (ShapedType::isStatic(ubs.back())) {
    distConfig.vectorSizeHints.back() = std::min(typeVectorSize, ubs.back());
  }

  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(padOp, distConfig);
  LoweringConfigGenerator generator(padOp);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(distConfig.vectorSizeHints);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  LDBG() << "Set lowering_config for tensor.pad op: " << loweringConfig;

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, padOp, loweringConfig,
      DispatchLoweringPassPipeline::CPUDoubleTilingExpert);
}

/// Set the default configuration for operations that implement the
/// `TiledOpInterface`.
static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   TilingInterface op) {
  assert(!getLoweringConfig(op) && "expected lowering_config is not set");
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributedLevelTileSizes(op, DistributionHeuristicConfig{});

  // Add an extra level of tiling.
  // TODO: Limit vector tile sizes for other TilingInterface ops.
  SmallVector<int64_t> vecTileSizes = distTileSizes;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(*op)) {
    limitVectorTileSizes(linalgOp, vecTileSizes);
  }

  LoweringConfigGenerator generator(op);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vecTileSizes);
  IREE::CPU::LoweringConfigAttr loweringConfig =
      generator.generateCPULoweringConfig();
  LDBG() << "Set lowering_config for tensor.pad op: " << loweringConfig;
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, loweringConfig,
      DispatchLoweringPassPipeline::CPUDefault);
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult
setRootConfigImpl(mlir::FunctionOpInterface entryPointFn, Operation *op,
                  const TargetMLTransformInfo &targetMLTransInfo) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<linalg::GenericOp>([&](auto op) {
          return setRootConfig(entryPointFn, op, LinalgOpInfo(op),
                               targetMLTransInfo);
        })
        .Case<IREE::LinalgExt::CustomOp>([&](auto op) {
          return setDefaultCustomOpLoweringConfig(entryPointFn, op,
                                                  initCPULaunchConfig);
        })
        .Case<IREE::LinalgExt::AttentionOp, IREE::LinalgExt::FftOp,
              linalg::PackOp, tensor::PadOp, linalg::UnPackOp, linalg::Mmt4DOp,
              linalg::BatchMmt4DOp>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<IREE::LinalgExt::WinogradFilterTransformOp,
              IREE::LinalgExt::WinogradInputTransformOp,
              IREE::LinalgExt::WinogradOutputTransformOp>(
            [&](auto op) { return setWinogradRootConfig(entryPointFn, op); })
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
adjustTileSizesForPackOp(mlir::FunctionOpInterface entryPointFn,
                         linalg::PackOp packOp,
                         SmallVector<int64_t> &distTileSizes,
                         SmallVector<int64_t> &parallelVecTileSizes) {

  ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();
  ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
  // Currently we only handle pack op with static inner tile sizes.
  if (ShapedType::isDynamicShape(innerTiles)) {
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
/// linalg.unpack inner tile sizes, if there are linalg.unpack producers. If the
/// tile sizes are not aligned, a stack buffer is needed because of
/// linalg.unpack tiling implementations.
static LogicalResult
adjustTileSizesForUnPackOp(mlir::FunctionOpInterface entryPointFn,
                           Operation *rootOp) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp) {
    return success();
  }
  auto loweringConfig =
      getLoweringConfig<IREE::CPU::LoweringConfigAttr>(linalgOp);
  if (!loweringConfig) {
    // Tile size adjustment is only available when the rootOp uses
    // IREE::CPU::LoweringConfigAttr.
    return success();
  }

  bool foundUnPackOp = false;
  SmallVector<int64_t> alignedSizes(linalgOp.getNumLoops(), 1);
  for (OpOperand *opOperand : linalgOp.getDpsInputOperands()) {
    auto unpackOp = opOperand->get().getDefiningOp<linalg::UnPackOp>();
    if (!unpackOp)
      continue;

    foundUnPackOp = true;
    auto idxMap = linalgOp.getMatchingIndexingMap(opOperand);
    LDBG() << "Find unpack op candidate: " << unpackOp;
    LDBG() << "The corresponding indexing map is: " << idxMap;

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

  LDBG() << "The tile sizes for each dimension should be aligned to "
         << alignedSizes;

  // Fixup for making tileSizes be multiple of inner_tile_sizes.
  SmallVector<IREE::CPU::LoweringConfigLevelInfo> tilingInfo =
      loweringConfig.getAvailableTilingInfo();
  for (IREE::CPU::LoweringConfigLevelInfo &info : tilingInfo) {
    SmallVector<int64_t> &tileSizes = info.sizes;
    for (auto idx : llvm::seq<int64_t>(0, tileSizes.size())) {
      if (tileSizes[idx] == 0)
        continue;
      tileSizes[idx] = llvm::alignTo(tileSizes[idx], alignedSizes[idx]);
    }
  }

  auto tInfo = getTranslationInfo(entryPointFn);
  auto pipeline = tInfo.getPassPipeline().getValue();
  auto pipelineConfig = tInfo.getConfiguration();
  if (isOptEnabled(entryPointFn, getEnableLoopPeelingStr())) {
    // See #16406
    LDBG() << "unpack fusion does not work with peeling, falling back to "
              "non-peeling path";
    pipeline = DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

    // Remove the "enable_loop_peeling" attr from pipelineConfig
    auto enableLoopPeelingAttrName =
        getEnableLoopPeelingAttrName(rootOp->getContext());
    auto newPipelineConfigEntries = llvm::filter_to_vector(
        pipelineConfig.getValue(), [&](NamedAttribute entry) {
          return entry.getName() != enableLoopPeelingAttrName;
        });

    pipelineConfig =
        DictionaryAttr::get(rootOp->getContext(), newPipelineConfigEntries);
  }

  IREE::Codegen::LoweringConfigAttrInterface newLoweringConfig =
      getNewLoweringConfig(rootOp->getContext(), tilingInfo,
                           /*setDistributionConfig=*/true);
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, rootOp, newLoweringConfig, pipeline, /*workgroupSize=*/{},
      /*subgroupSize=*/{}, pipelineConfig);
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
adjustTileSizesForGenericOp(mlir::FunctionOpInterface entryPointFn,
                            linalg::GenericOp genericOp,
                            SmallVector<int64_t> &parallelVecTileSizes,
                            SmallVector<int64_t> &reductionTileSizes,
                            SmallVector<bool> &parallelScalableFlags,
                            SmallVector<bool> &reductionScalableFlags) {
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
  limitVectorTileSizes(genericOp, vecTileSizes);
  splitParallelAndReductionTiles(genericOp, vecTileSizes, reductionTileSizes,
                                 &parallelScalableFlags,
                                 &reductionScalableFlags);
  for (auto [pos, tileSize] : llvm::enumerate(vecTileSizes)) {
    // Generic op vector parallel tile size is low priority. Only use if no
    // other op has set the tile size.
    if (tileSize == 0 || parallelVecTileSizes[pos] != 0)
      continue;
    parallelVecTileSizes[pos] = tileSize;
  }
  return success();
}

/// Updates the tile sizes and scalable flags in the `tilingInfo` with given
/// `level`, if it is present. Otherwise, adds a new item to the vector.
static void updateOrAddTilingLevelInfo(
    SmallVectorImpl<IREE::CPU::LoweringConfigLevelInfo> &tilingInfo,
    TilingLevel level, ArrayRef<int64_t> tileSizes,
    ArrayRef<bool> scalableFlags) {
  for (IREE::CPU::LoweringConfigLevelInfo &info : tilingInfo) {
    if (info.level == level) {
      info.sizes.assign(tileSizes.begin(), tileSizes.end());
      info.scalableFlags.assign(scalableFlags.begin(), scalableFlags.end());
      return;
    }
  }
  IREE::CPU::LoweringConfigLevelInfo newInfo;
  newInfo.level = level;
  newInfo.sizes.assign(tileSizes.begin(), tileSizes.end());
  newInfo.scalableFlags.assign(scalableFlags.begin(), scalableFlags.end());
  tilingInfo.push_back(newInfo);
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
///   %pack = linalg.pack %13
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
setLoweringConfigForComputeOps(mlir::FunctionOpInterface entryPointFn,
                               ArrayRef<Operation *> computeOps,
                               Operation *rootOperation) {
  if (isa<linalg::ConvolutionOpInterface>(rootOperation)) {
    // TODO(dcaballe): We don't know yet how to properly propagate the lowering
    // config of a convolution.
    return success();
  }

  auto rootLoweringConfig =
      getLoweringConfig<IREE::CPU::LoweringConfigAttr>(rootOperation);
  if (!rootLoweringConfig) {
    // Propagation is only available for IREE::CPU::LoweringConfigAttr.
    return success();
  }

  SmallVector<int64_t> distTileSizes, parallelVecTileSizes;
  SmallVector<bool> distScalableTileSizes, parallelVecScalableTileSizes;
  assert(rootLoweringConfig.hasWorkgroupTilingLevel());
  distTileSizes = rootLoweringConfig.getWorkgroupTileSizes();
  if (rootLoweringConfig.hasTilingLevel(
          TilingLevel::VectorCommonParallelTiles)) {
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        rootLoweringConfig.getTilingLevelAttr(
            TilingLevel::VectorCommonParallelTiles));
    parallelVecTileSizes.assign(attr.getSizes().begin(), attr.getSizes().end());
    parallelVecScalableTileSizes.assign(attr.getScalableFlags().begin(),
                                        attr.getScalableFlags().end());
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
  // %3 = linalg.pack %2
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
  llvm::SmallDenseMap<Operation *, SmallVector<bool>>
      reductionScalableFlagseMap;
  distTileSizes.resize(maxLoopNums);
  parallelVecTileSizes.resize(maxLoopNums);
  parallelVecScalableTileSizes.resize(maxLoopNums);
  bool hasSeenPackOp = false;
  for (auto op : computeOps) {
    assert(!hasSeenPackOp && "Pack op must be the last op");
    if (hasSeenPackOp)
      return failure();

    // Tile sizes have been initialized from rootOperation, so we skip it.
    if (op == rootOperation)
      continue;

    if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
      if (failed(adjustTileSizesForPackOp(entryPointFn, packOp, distTileSizes,
                                          parallelVecTileSizes))) {
        return failure();
      }
      hasSeenPackOp = true;
    } else if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      SmallVector<int64_t> reductionTileSizes;
      SmallVector<bool> reductionScalableFlags;
      if (failed(adjustTileSizesForGenericOp(
              entryPointFn, genericOp, parallelVecTileSizes, reductionTileSizes,
              parallelVecScalableTileSizes, reductionScalableFlags))) {
        return failure();
      }
      reductionTileSizeMap[op] = reductionTileSizes;
      reductionScalableFlagseMap[op] = reductionScalableFlags;
    }
  }

  LDBG() << "Parallel vector tile sizes: " << parallelVecTileSizes;

  // Split parallel vector tile sizes into common parts and op-specific parts.
  SmallVector<int64_t> commonVecTileSizes = parallelVecTileSizes;
  SmallVector<bool> commonVecScalableTileFlags = parallelVecScalableTileSizes;
  SmallVector<int64_t> innerVecTileSizes(maxLoopNums, 0);
  SmallVector<bool> innerVecScalableTileFlags(maxLoopNums, false);
  IterationDimTracker dimTracker(computeOps);
  for (auto op : computeOps) {
    auto iterTypes = cast<TilingInterface>(op).getLoopIteratorTypes();
    for (auto [idx, iterType] : llvm::enumerate(iterTypes)) {
      if (iterType == utils::IteratorType::reduction ||
          !dimTracker.isCommonDim(op, idx)) {
        innerVecTileSizes[idx] = parallelVecTileSizes[idx];
        innerVecScalableTileFlags[idx] = parallelVecScalableTileSizes[idx];
        commonVecTileSizes[idx] = 0;
        commonVecScalableTileFlags[idx] = false;
      }
    }
  }

  // Make sure the innermost tile size times element size is multiple
  // of byte bits. This is required for now because we do not fully
  // support sub-byte vector stores. Once vector stores are supported
  // then this can be eliminated. Note that emulating sub-byte sized vector
  // loads and stores will have a performance impact.
  auto resultTypes = rootOperation->getResultTypes();
  if (commonVecTileSizes.size() != 0 && !resultTypes.empty()) {
    Type elementType = cast<ShapedType>(resultTypes[0]).getElementType();
    unsigned int elementTypeSize;
    if (auto complexType = llvm::dyn_cast<ComplexType>(elementType)) {
      elementTypeSize =
          2 * complexType.getElementType().getIntOrFloatBitWidth();
    } else {
      elementTypeSize = elementType.getIntOrFloatBitWidth();
    }
    // for now just enable for i1
    if (elementTypeSize == 1) {
      auto innermostTileSize = commonVecTileSizes.back();
      commonVecTileSizes.back() =
          llvm::alignTo(innermostTileSize * elementTypeSize, 8) /
          elementTypeSize;
    }
  }

  // Set the lowering configs with new tile sizes.
  for (auto op : computeOps) {
    int numLoops = cast<TilingInterface>(op).getLoopIteratorTypes().size();
    SmallVector<IREE::CPU::LoweringConfigLevelInfo> newTilingInfo;
    // For root op, we patch the adjusted tile sizes on its original tiling
    // config.
    if (op == rootOperation) {
      newTilingInfo = rootLoweringConfig.getAvailableTilingInfo();
      updateOrAddTilingLevelInfo(newTilingInfo, IREE::CPU::DistributionTiles,
                                 distTileSizes, distScalableTileSizes);
      if (rootLoweringConfig.hasTilingLevel(
              IREE::CPU::VectorCommonParallelTiles)) {
        updateOrAddTilingLevelInfo(
            newTilingInfo, IREE::CPU::VectorCommonParallelTiles,
            commonVecTileSizes, commonVecScalableTileFlags);
      }
    } else {
      // Build 4-level lowering configs for other ops. No scalable tiling for
      // the distribution.
      SmallVector<int64_t> zeros(numLoops, 0);
      SmallVector<bool> falseVec(numLoops, 0);
      updateOrAddTilingLevelInfo(newTilingInfo, IREE::CPU::DistributionTiles,
                                 distTileSizes, falseVec);
      // The cache level tiling sizes are not adjusted, so we use the
      // config from the rootOp directly.
      if (rootLoweringConfig.hasTilingLevel(IREE::CPU::CacheParallelTiles)) {
        updateOrAddTilingLevelInfo(
            newTilingInfo, IREE::CPU::CacheParallelTiles,
            rootLoweringConfig.getStaticTilingLevelSizes(
                IREE::CPU::CacheParallelTiles, rootOperation),
            falseVec);
      }
      if (rootLoweringConfig.hasTilingLevel(IREE::CPU::CacheReductionTiles)) {
        updateOrAddTilingLevelInfo(
            newTilingInfo, IREE::CPU::CacheReductionTiles,
            rootLoweringConfig.getStaticTilingLevelSizes(
                IREE::CPU::CacheReductionTiles, rootOperation),
            falseVec);
      }
      updateOrAddTilingLevelInfo(
          newTilingInfo, IREE::CPU::VectorCommonParallelTiles,
          commonVecTileSizes, commonVecScalableTileFlags);
      bool setUpOK =
          TypeSwitch<Operation *, bool>(op)
              .Case<linalg::PackOp>([&](auto packOp) {
                // TODO: Handle scalable flags
                if (llvm::any_of(rootLoweringConfig.getVectorScalableFlags(),
                                 [&](bool flag) { return flag; })) {
                  return false;
                }
                updateOrAddTilingLevelInfo(newTilingInfo,
                                           IREE::CPU::VectorReductionTiles,
                                           zeros, falseVec);
                updateOrAddTilingLevelInfo(newTilingInfo,
                                           IREE::CPU::VectorInnerParallelTiles,
                                           innerVecTileSizes, falseVec);
                // Scale and permutate the outer dim tiles for pack op.
                ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
                ArrayRef<int64_t> dimPos = packOp.getInnerDimsPos();
                auto outerDimsPerm = packOp.getOuterDimsPerm();
                for (IREE::CPU::LoweringConfigLevelInfo &info : newTilingInfo) {
                  SmallVector<int64_t> &tileSizes = info.sizes;
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

                return true;
              })
              .Default([&](auto) {
                if (reductionTileSizeMap.contains(op)) {
                  updateOrAddTilingLevelInfo(
                      newTilingInfo, IREE::CPU::VectorReductionTiles,
                      reductionTileSizeMap[op], reductionScalableFlagseMap[op]);
                } else {
                  updateOrAddTilingLevelInfo(newTilingInfo,
                                             IREE::CPU::VectorReductionTiles,
                                             zeros, falseVec);
                }
                // Only copy the inner vector tile sizes on parallel dims.
                SmallVector<int64_t> vecTileSizes(numLoops, 0);
                SmallVector<bool> vecScalableTileFlags(numLoops, false);
                auto iterTypes =
                    cast<TilingInterface>(op).getLoopIteratorTypes();
                for (auto [idx, iterType] : llvm::enumerate(iterTypes)) {
                  if (iterType == utils::IteratorType::parallel) {
                    vecTileSizes[idx] = innerVecTileSizes[idx];
                    vecScalableTileFlags[idx] = innerVecScalableTileFlags[idx];
                  }
                }
                updateOrAddTilingLevelInfo(newTilingInfo,
                                           IREE::CPU::VectorInnerParallelTiles,
                                           vecTileSizes, vecScalableTileFlags);
                return true;
              });

      // TODO: (awarzynski) This is effectively tracking the case of
      // linalg.pack + scalable flags, which is not support ATM (see TODO
      // above). Remove once that's implemented.
      if (!setUpOK)
        return failure();
    }

    for (auto &[level, tileSizes, scalableFlags] : newTilingInfo) {
      (void)level;
      tileSizes.resize(numLoops, 0);
      scalableFlags.resize(numLoops, 0);
    }
    std::sort(newTilingInfo.begin(), newTilingInfo.end(),
              [](const IREE::CPU::LoweringConfigLevelInfo &lhs,
                 const IREE::CPU::LoweringConfigLevelInfo &rhs) {
                return lhs.level < rhs.level;
              });
    IREE::Codegen::LoweringConfigAttrInterface config =
        getNewLoweringConfig(rootOperation->getContext(), newTilingInfo,
                             /*setDistributionConfig=*/op == rootOperation);
    setLoweringConfig(op, config);
  }

  return success();
}

/// Helper method to set the dispatch to be lowered through the default
/// pipeline.
static LogicalResult
lowerUsingDefaultPipeline(mlir::FunctionOpInterface entryPointFn) {
  // If there is a translation info set, do nothing.
  if (getTranslationInfo(entryPointFn)) {
    return success();
  }
  // Otherwise lower using default pipeline.
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      entryPointFn->getContext(), DispatchLoweringPassPipeline::CPUDefault);
  return setTranslationInfo(entryPointFn, translationInfo);
}

/// Returns true if the given operation should have a lowering config set.
///
/// This predicate excludes:
///   - Ops inside a `CustomOp` that already have a lowering config.
///   - `linalg.pack` ops whose producer is a `tensor.collapse_shape`,
///     as they will be lowered together into a `map_scatter` later in the
///     pipeline.
static bool shouldSetLoweringConfig(Operation *op) {
  if (isa_and_nonnull<IREE::LinalgExt::CustomOp>(op->getParentOp()) &&
      getLoweringConfig(op) != nullptr) {
    return false;
  }

  if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
    if (isa_and_nonnull<tensor::CollapseShapeOp>(
            packOp.getSource().getDefiningOp())) {
      return false;
    }
  }

  return true;
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult
setTranslationInfoAndRootConfig(mlir::FunctionOpInterface entryPointFn,
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

  LDBG() << "Root op: " << *rootOperation;

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

    auto prunedComputeOps = llvm::to_vector(
        llvm::make_filter_range(computeOps, shouldSetLoweringConfig));
    if (failed(setLoweringConfigForComputeOps(entryPointFn, prunedComputeOps,
                                              rootOperation))) {
      return failure();
    }
  }

  return success();
}

LogicalResult initCPULaunchConfig(FunctionOpInterface funcOp) {
  if (getTranslationInfo(funcOp)) {
    return success();
  }

  // For now pick the default for functions with control flow, cause
  // the currently built pipelines dont work so well with control flow.
  if (funcOp.empty() || !llvm::hasSingleElement(funcOp.getFunctionBody())) {
    return lowerUsingDefaultPipeline(funcOp);
  }

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  if (failed(setTranslationInfoAndRootConfig(funcOp, computeOps))) {
    return failure();
  }

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  return applyPatternsGreedily(funcOp, std::move(patterns));
}

} // namespace mlir::iree_compiler
