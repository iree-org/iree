// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"

#include <numeric>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/LLVMCPU/TargetMLTransformInfo.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/CPU/Common.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "kernel-dispatch"
#define KD_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

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
    llvm::cl::desc("number of threads that are used at runtime if codegen "
                   "thread distribution is enabled"),
    llvm::cl::init(8));

static llvm::cl::opt<bool> clDisableDistribution(
    "iree-codegen-llvm-disable-distribution",
    llvm::cl::desc("disable thread distribution in codegen"),
    llvm::cl::init(false));

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

// TODO(hanchung): Remove the flag. This is the flag for fastly falling back to
// the previous snapshot.

static llvm::cl::opt<bool> enableVectorPadding(
    "iree-codegen-enable-vector-padding",
    llvm::cl::desc("Enable padding for vectorization"), llvm::cl::init(true));

static llvm::cl::opt<bool> enableVectorPeeling(
    "iree-codegen-enable-vector-peeling",
    llvm::cl::desc("Enable peeling for vectorization"), llvm::cl::init(true));

// Non-static options are used in other places.
llvm::cl::opt<std::string> clCPUCodegenTransformDialectFileName(
    "iree-codegen-llvmcpu-use-transform-dialect",
    llvm::cl::desc(
        "MLIR file containing a transform dialect specification to apply"),
    llvm::cl::init(""));
llvm::cl::opt<bool> clCPUEnableTransformDialectJit(
    "iree-codegen-llvmcpu-enable-transform-dialect-jit",
    llvm::cl::desc("enable the usage of the transform dialect JIT"),
    llvm::cl::init(false));
llvm::cl::opt<std::string> clCPUCodegenTransformDialectDebugPayloadTag(
    "iree-codegen-llvmcpu-transform-dialect-debug-payload-tag",
    llvm::cl::desc("tag attribute value for the transform dialect interpreter "
                   "payload root operation"),
    llvm::cl::init(""));

llvm::cl::opt<std::string> clCPUCodegenTransformDialectDebugTransformTag(
    "iree-codegen-llvmcpu-transform-dialect-debug-transform-tag",
    llvm::cl::desc(
        "tag attribute value for the transform dialect transform op container"),
    llvm::cl::init(""));

using IREE::Codegen::DispatchLoweringPassPipeline;

// Encodes the pre-processing strategy to be applied on a Linalg operation
// before vectorization.
enum class VectorPreProcStrategy {
  // Pad vector dimensions of tensors so that they are multiple of the vector
  // length.
  Padding,
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

// TODO(dcaballe): Move operator<< to DebugUtils.h.
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const VectorPreProcStrategy &strategy) {
  switch (strategy) {
    case VectorPreProcStrategy::Padding:
      os << "Padding";
      break;
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

static llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os,
    const mlir::iree_compiler::TileSizesListType &tileSizeList) {
  os << "[";
  for (auto &tuple : tileSizeList) {
    os << "[" << tuple << "]";
  }
  os << "]";

  return os;
}

/// Returns true if all the input and output tensor operands of 'op' are fully
/// dynamic.
static bool isFullyDynamicOp(linalg::LinalgOp op) {
  SmallVector<int64_t, 4> loopRanges = op.getStaticLoopRanges();
  return llvm::all_of(loopRanges,
                      [](int64_t size) { return ShapedType::isDynamic(size); });
}

/// Returns the vectorization pre-processing strategy (padding, peeling) for the
/// given LinalgOp, depending on the op traits and the target architecture.
static VectorPreProcStrategy getVectorPreProcStrategy(
    linalg::LinalgOp linalgOp) {
  // Generic strategies.

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

    if (isFullyDynamicOp(linalgOp) && enableVectorPeeling) {
      return VectorPreProcStrategy::Peeling;
    }

    if (enableVectorPadding) {
      // Padding is only enabled on x86. It leads to too much overhead on
      // RISC-V and ARM.
      return VectorPreProcStrategy::Padding;
    }
  }

  // Default RISC-V specific strategies.
  if (isRISCV(targetAttr)) {
    if (isLinalgGeneric) {
      return VectorPreProcStrategy::Masking;
    }

    if (enableVectorPeeling) {
      return VectorPreProcStrategy::Peeling;
    }
  }

  // Default AArch64 specific strategies.
  if (isAArch64(targetAttr)) {
    if ((linalg::isElementwise(linalgOp) || isFullyDynamicOp(linalgOp)) &&
        enableVectorPeeling) {
      return VectorPreProcStrategy::Peeling;
    }
  }

  return VectorPreProcStrategy::None;
}

/// Looks for the `native_vector_size` attribute in the hal.executable.target
/// looked up from this op.
static std::optional<int64_t> getNativeVectorSizeInBytes(
    func::FuncOp entryPointFn) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  auto nativeVectorSizeAttr =
      getConfigIntegerAttr(targetAttr, "native_vector_size");
  if (!nativeVectorSizeAttr) return std::nullopt;
  int64_t nativeVectorSizeVal = nativeVectorSizeAttr->getInt();
  if (!nativeVectorSizeVal) return std::nullopt;
  return nativeVectorSizeVal;
}

/// For a given `shapedType` or (`byteWidth` of element type) return the number
/// of elements that correspond to the native vector size. Returns 1 as the
/// fallback.
static int64_t getVectorSize(func::FuncOp entryPointFn, unsigned byteWidth) {
  if (std::optional<int64_t> nativeVectorSize =
          getNativeVectorSizeInBytes(entryPointFn)) {
    return nativeVectorSize.value() / byteWidth;
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
// TODO(diegocaballero): Refactor this logic to a method that computes the final
// tile sizes for vectorization/unrolling in one shot.
static SmallVector<int64_t> getMinTilingSizesForEachDim(
    func::FuncOp entryPointFn, linalg::LinalgOp op,
    const LinalgOpInfo &linalgOpInfo,
    const TargetMLTransformInfo &targetMLTransInfo) {
  unsigned numLoops = op.getNumLoops();
  SmallVector<int64_t> minTileSizes(numLoops, 1);
  auto inputOutputOpOperands = op->getOpOperands();

  for (auto [index, map] : llvm::enumerate(op.getIndexingMapsArray())) {
    // Check the fastest varying dimension of the operand. Set the vector size
    // of the corresponding loop to the vector size.
    if (map.getNumResults() == 0) continue;
    auto fastestVaryingDimExpr =
        map.getResults().back().dyn_cast<AffineDimExpr>();
    if (!fastestVaryingDimExpr) continue;
    unsigned fastestVaryingDim = fastestVaryingDimExpr.getPosition();

    // If the indexing map has result it has to be a shaped type.
    auto operandType =
        inputOutputOpOperands[index].get().getType().cast<ShapedType>();
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

/// Returns the type length in bytes. Looks through all the interface binding
/// ops to see the ABI types and guess-timates the type size to use. This is
/// used to convert the vector size in bytes to vector size in number of
/// elements.
static unsigned getReferenceTypeLengthInBytes(func::FuncOp entryPointFn) {
  unsigned referenceTypeLengthInBytes = 4;
  entryPointFn.walk([&](IREE::HAL::InterfaceBindingSubspanOp subSpanOp) {
    Type type = subSpanOp.getResult().getType();
    Type elementType =
        TypeSwitch<Type, Type>(type)
            .Case<IREE::Flow::DispatchTensorType>(
                [&](auto dispatchTensorType) -> Type {
                  // Ignore operands that are 0D tensors. These
                  // are not vector-loadable, so using these to
                  // get vector length would be a pessimization.
                  if (!dispatchTensorType.getRank()) return nullptr;
                  return dispatchTensorType.getBoundElementType();
                })
            .Case<ShapedType>([&](auto shapedType) -> Type {
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
  // Set all the distribution tile sizes to zero if thread distribution is
  // disabled.
  if (clDisableDistribution) {
    return SmallVector<int64_t>(numDims, 0);
  }

  SmallVector<int64_t> distributedTileSizes(numDims, 1);
  SmallVector<int64_t> numWorkgroupsPerDim(numDims, 1);
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
    if (workload[index] == ShapedType::kDynamic ||
        currSize >= maxTileSizes[index] || currSize >= workload[index]) {
      currDim--;
      continue;
    }

    int64_t newSize = std::min<int64_t>(currSize * 2, workload[index]);
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
  if (ub == ShapedType::kDynamic || lb == ShapedType::kDynamic) {
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
/// dimension based on its number of iterations and the native vector size of
/// the target. The resulting tile size will be a multiple of the provided
/// vector size, except when `allowIncompleteTile` is set to true. If
/// `enforcePowerOfTwo` is set to true, the resulting tile size will be a power
/// of two.
static int64_t getMaxVectorTileSize(int64_t lb, int64_t ub, int64_t maxSize,
                                    int64_t vectorSize,
                                    bool allowIncompleteTile = false,
                                    bool enforcePowerOfTwo = false) {
  if (ub == ShapedType::kDynamic || lb == ShapedType::kDynamic) {
    return roundUpToPow2(maxSize, enforcePowerOfTwo);
  }
  int64_t numIters = ub - lb;
  if (numIters <= maxSize && numIters < vectorSize) {
    return roundUpToPow2(numIters, enforcePowerOfTwo);
  }

  // Return the largest suitable power of two if power of two is enforced.
  if (enforcePowerOfTwo) {
    return roundUpToPow2(std::min(maxSize, numIters), enforcePowerOfTwo);
  }

  // Try to find a tile size that is multiple of the vector size.
  int64_t scaledUB = std::min(maxSize, numIters) / vectorSize * vectorSize;
  for (int64_t i = scaledUB; i > 0; i -= vectorSize) {
    if (numIters % i == 0) {
      return i;
    }
  }
  if (allowIncompleteTile) {
    // Try to find a tile size that is not multiple of the vector size but
    // multiple of the number of iterations. Otherwise, return `maxSize`.
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

/// Returns the tile size to use for the Flow level.
///
/// The vectorSizeHints can be empty or as many as the number of loops. When not
/// empty, each hint should be 1 or the vector size. On the dimensions where the
/// hints != 1, it will try to find the tile sizes which are multipliers of the
/// hints.
///
/// TODO(hanchung): Remove `allowIncompleteTile` option after codegen can handle
/// padding/peeling for all the kernels. Allowing incomplete tile is critical
/// for odd shapes (e.g., some dim sizes could be prime number).
static SmallVector<int64_t> getDefaultDistributedLevelTileSizes(
    ArrayRef<unsigned> partitionableLoops, ArrayRef<int64_t> lbs,
    ArrayRef<int64_t> ubs, ArrayRef<int64_t> minTileSizes,
    ArrayRef<int64_t> maxTileSizes, bool allowIncompleteTile = false,
    ArrayRef<int64_t> vectorSizeHints = {}) {
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
    distributedTileSizes[i] =
        getMaxDistributionTileSize(lbs[i], ubs[i], distributedTileSizes[i],
                                   minTileSizes[i], allowIncompleteTile);
  }
  return distributedTileSizes;
}

static SmallVector<int64_t> getDefaultDistributedLevelTileSizes(
    linalg::LinalgOp linalgOp, ArrayRef<int64_t> minTileSizes,
    ArrayRef<int64_t> maxTileSizes, bool allowIncompleteTile = false,
    ArrayRef<int64_t> vectorSizeHints = {}) {
  OpBuilder builder(linalgOp.getContext());
  builder.setInsertionPoint(linalgOp);
  SmallVector<int64_t> lbs(linalgOp.getNumLoops(), 0);
  SmallVector<int64_t> ubs = linalgOp.getStaticLoopRanges();
  auto loops = cast<PartitionableLoopsInterface>(linalgOp.getOperation())
                   .getPartitionableLoops(kNumMaxParallelDims);
  return getDefaultDistributedLevelTileSizes(loops, lbs, ubs, minTileSizes,
                                             maxTileSizes, allowIncompleteTile,
                                             vectorSizeHints);
}

/// Splits the tile sizes in `parallelSizes` into `reductionSizes` for the
/// reduction loops.
static void splitParallelAndReductionTiles(
    linalg::LinalgOp op, SmallVectorImpl<int64_t> &parallelSizes,
    SmallVectorImpl<int64_t> &reductionSizes) {
  reductionSizes.assign(parallelSizes.begin(), parallelSizes.end());
  for (auto [index, iteratorType] :
       llvm::enumerate(op.getIteratorTypesArray())) {
    if (iteratorType == utils::IteratorType::parallel) {
      reductionSizes[index] = 0;
    } else {
      parallelSizes[index] = 0;
    }
  }
}

static void setAlwaysVectorizeSizes(linalg::LinalgOp op,
                                    SmallVectorImpl<int64_t> &parallelSizes,
                                    SmallVectorImpl<int64_t> &reductionSizes) {
  SmallVector<int64_t, 4> staticLoopRanges = op.getStaticLoopRanges();
  for (auto [index, valuePair] : llvm::enumerate(
           llvm::zip_equal(staticLoopRanges, op.getIteratorTypesArray()))) {
    auto [size, iterType] = valuePair;
    if (!ShapedType::isDynamic(size)) continue;
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

static void setVectorSizesForDynamicShapes(
    linalg::LinalgOp op, VectorPreProcStrategy vecPreProcStrategy,
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

  // If peeling is enabled and the 'op' is fully dynamic, we only vectorize the
  // lowest order parallel dimension for now to avoid peeling higher level
  // dimensions. If no parallel dimension is found to be vectorized, we try to
  // vectorize the lowest order reduction dimension.

  if (!isFullyDynamicOp(op) ||
      vecPreProcStrategy != VectorPreProcStrategy::Peeling) {
    return;
  }

  bool isParallelDimVectorized = false;
  for (int i = origParallelSizes.size() - 1; i >= 0; --i) {
    if (origParallelSizes[i] > 1) {
      assert(parallelSizes[i] == 1 &&
             "This tile size should have been set to one");
      parallelSizes[i] = origParallelSizes[i];
      isParallelDimVectorized = true;
      break;
    }
  }

  if (isParallelDimVectorized) {
    return;
  }

  for (int i = origReductionSizes.size() - 1; i >= 0; --i) {
    if (origReductionSizes[i] > 1) {
      assert(reductionSizes[i] == 1 &&
             "This tile size should have been set to one");
      reductionSizes[i] = origReductionSizes[i];
      break;
    }
  }

  LLVM_DEBUG(KD_DBGS() << "Parallel sizes for dynamic sizes: " << parallelSizes
                       << "\n");
  LLVM_DEBUG(KD_DBGS() << "Reduction sizes for dynamic sizes: "
                       << reductionSizes << "\n");

  return;
}

/// Sets the default configuration to use for an operation that implements the
/// `PartitionableLoopsInterface`, given the `lbs` and `ubs` of all the loops.
static LogicalResult setDefaultRootConfig(
    func::FuncOp entryPointFn,
    PartitionableLoopsInterface partitionableLoopsInterfaceOp,
    ArrayRef<int64_t> lbs, ArrayRef<int64_t> ubs) {
  assert(!getLoweringConfig(partitionableLoopsInterfaceOp) &&
         "expected lowering_config is not set");
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
  auto loweringConfig = IREE::Codegen::LoweringConfigAttr::get(
      entryPointFn.getContext(), tileSizes);
  setLoweringConfig(partitionableLoopsInterfaceOp, loweringConfig);
  return success();
}

static LogicalResult setMatmulPadRootConfig(
    func::FuncOp entryPointFn, linalg::ContractionOpInterface op,
    ArrayRef<int64_t> flowTileSizes, ArrayRef<int64_t> workgroupTileSizes,
    int vectorSize) {
  // The tiling for parallel dims and reduction dims should be separated.
  SmallVector<int64_t> parallelTileSizes(workgroupTileSizes.begin(),
                                         workgroupTileSizes.end());
  parallelTileSizes.back() = 0;

  // Clamp inner tiling sizes to avoid masking. The vector masking takes the
  // last level of tiling to create masks. It would lead to incorrect masking if
  // the inner tiling sizes are not clamped. Because padding won't be applied
  // along those dimensions.
  for (const auto &[index, size] : llvm::enumerate(flowTileSizes)) {
    if (!size) continue;
    parallelTileSizes[index] = std::min(parallelTileSizes[index], size);
  }

  // TODO(hanchung): Make logic more heuristic. Padding hurts performance a lot
  // if the dim size is small (e.g., K=24).
  SmallVector<int64_t> reductionTileSizes(workgroupTileSizes.size() - 1, 0);
  auto lhsShapedType = op.lhs().getType().cast<ShapedType>();
  int64_t K = lhsShapedType.getShape().back();
  reductionTileSizes.push_back(
      getMaxVectorTileSize(0, K, workgroupTileSizes.back(), vectorSize));

  TileSizesListType tileSizes;
  tileSizes.emplace_back(flowTileSizes.begin(), flowTileSizes.end());
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes,
      DispatchLoweringPassPipeline::CPUDoubleTilingPadExpert);
}

static DispatchLoweringPassPipeline getNoPadTilingExpert(
    VectorPreProcStrategy strategy) {
  if (strategy == VectorPreProcStrategy::Peeling) {
    return DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert;
  }
  return DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
}

static LogicalResult setMatmulNoPadRootConfig(
    func::FuncOp entryPointFn, linalg::ContractionOpInterface op,
    const TileSizesListTypeRef inputTileSizes, int vectorSize,
    VectorPreProcStrategy vecPreProcStrategy) {
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();
  // Iterate over the inner tile size tuples to check that their sizes divides
  // the sizes of the iteration space.
  for (auto tileSizeTuple :
       llvm::make_range(inputTileSizes.begin(), inputTileSizes.end() - 1)) {
    for (const auto &[idx, tileSize] : llvm::enumerate(tileSizeTuple)) {
      // Quantized cases are not fully evaluated yet, so it might go with NoPad
      // approach.
      if (tileSize == 0 || shape[idx] == ShapedType::kDynamic) continue;
      assert(shape[idx] % tileSize == 0);
      shape[idx] = tileSize;
    }
  }

  // TODO(hanchung): Create an addtional pass to handle such cases.
  // The tiling for parallel dims and reduction dims should be separated.
  const SmallVectorImpl<int64_t> &workgroupTileSizes = inputTileSizes.back();
  SmallVector<int64_t> parallelTileSizes;
  for (auto [index, tileSize] : llvm::enumerate(workgroupTileSizes)) {
    int64_t sz = tileSize;
    bool allowIncompleteTile =
        vecPreProcStrategy == VectorPreProcStrategy::Peeling ||
        vecPreProcStrategy == VectorPreProcStrategy::Masking;

    if (sz != 0) {
      sz = getMaxVectorTileSize(
          /*lb=*/0, /*ub=*/shape[index],
          /*maxTileSize=*/sz, vectorSize, allowIncompleteTile);
    }
    parallelTileSizes.push_back(sz);
  }
  SmallVector<int64_t> reductionTileSizes;
  splitParallelAndReductionTiles(cast<linalg::LinalgOp>(op.getOperation()),
                                 parallelTileSizes, reductionTileSizes);

  setVectorSizesForDynamicShapes(cast<linalg::LinalgOp>(op.getOperation()),
                                 vecPreProcStrategy, parallelTileSizes,
                                 reductionTileSizes);

  TileSizesListType newTileSizes;
  // Copy all the tile size levels except the workgroup one which will be split
  // into parallel and reduction.
  std::copy(inputTileSizes.begin(), inputTileSizes.end() - 1,
            std::back_inserter(newTileSizes));
  newTileSizes.push_back(parallelTileSizes);
  newTileSizes.push_back(reductionTileSizes);

  LLVM_DEBUG(KD_DBGS() << "Final tile sizes for no-padding contraction: "
                       << newTileSizes << "\n");

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, newTileSizes, getNoPadTilingExpert(vecPreProcStrategy));
}

static LogicalResult setAArch64RootConfig(func::FuncOp entryPointFn,
                                          linalg::ContractionOpInterface op,
                                          ArrayRef<int64_t> flowTileSizes,
                                          ArrayRef<int64_t> workgroupTileSizes,
                                          int vectorSize) {
  assert(flowTileSizes.size() == workgroupTileSizes.size());
  SmallVector<int64_t> parallelTileSizes;
  auto shape = cast<linalg::LinalgOp>(op.getOperation()).getStaticLoopRanges();
  for (auto [index, tileSize] : llvm::enumerate(flowTileSizes.drop_back())) {
    parallelTileSizes.push_back(
        getMaxVectorTileSize(0, tileSize ? tileSize : shape[index],
                             workgroupTileSizes[index], vectorSize));
  }

  auto lhsShapedType = op.lhs().getType().cast<ShapedType>();
  int64_t K = lhsShapedType.getShape().back();
  parallelTileSizes.push_back(
      getMaxVectorTileSize(0, K, workgroupTileSizes.back(), vectorSize));

  SmallVector<int64_t> reductionTileSizes;
  splitParallelAndReductionTiles(cast<linalg::LinalgOp>(op.getOperation()),
                                 parallelTileSizes, reductionTileSizes);

  TileSizesListType tileSizes;
  tileSizes.emplace_back(flowTileSizes.begin(), flowTileSizes.end());
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes,
      DispatchLoweringPassPipeline::Mmt4dTilingExpert);
}

/// Returns default hard-coded workgroup sizes for a give target. No smartness
/// should be introduced in this utility.
static void getDefaultMatmulWorkgroupSizes(linalg::LinalgOp op,
                                           SmallVectorImpl<int64_t> &sizes,
                                           int64_t vectorSize) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (isX86(targetAttr)) {
    sizes.append({8, 32, 16});
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

/// Main utility to compute the workgroup (vectorization/unrolling) tile sizes.
static SmallVector<int64_t> getMatmulWorkgroupSizes(func::FuncOp entryPointFn,
                                                    linalg::LinalgOp op,
                                                    int64_t vectorSize,
                                                    bool isQuantized) {
  SmallVector<int64_t> matmulTileSizes;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);

  // Compute workgroup tile sizes using heuristics.
  // TODO: if (isX86(targetAttr) || isRISCV(targetAttr)) {

  if (isAArch64(targetAttr)) {
    if (isQuantized) {
      matmulTileSizes = {vectorSize, vectorSize * 4, vectorSize};
    } else {
      matmulTileSizes = {5 * vectorSize, vectorSize, vectorSize * 16};
    }
  }

  // Get default hard-coded tile sizes if we couldn't compute anything better.
  if (matmulTileSizes.empty())
    getDefaultMatmulWorkgroupSizes(op, matmulTileSizes, vectorSize);

  SmallVector<int64_t> tileSizes;
  unsigned numLoops = op.getNumLoops();
  if (numLoops > 3) {
    tileSizes.append(numLoops - 3, 1);
    tileSizes.append(matmulTileSizes.begin(), matmulTileSizes.end());
  } else {
    tileSizes.append(matmulTileSizes.begin() + (3 - numLoops),
                     matmulTileSizes.end());
  }

  LLVM_DEBUG(KD_DBGS() << "Matmul workgroup sizes: " << tileSizes << "\n");

  return tileSizes;
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, linalg::ContractionOpInterface contractionOp) {
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
  auto lhsShapedType = contractionOp.lhs().getType().cast<ShapedType>();
  auto rhsShapedType = contractionOp.rhs().getType().cast<ShapedType>();
  auto resShapedType =
      linalgOp.getDpsInitOperand(0)->get().getType().cast<ShapedType>();
  int64_t vectorSize = getVectorSize(entryPointFn, lhsShapedType);
  vectorSize = std::min(vectorSize, getVectorSize(entryPointFn, rhsShapedType));
  vectorSize = std::min(vectorSize, getVectorSize(entryPointFn, resShapedType));
  bool isQuantized =
      lhsShapedType.getElementType() != resShapedType.getElementType();

  SmallVector<int64_t> workgroupTileSizes =
      getMatmulWorkgroupSizes(entryPointFn, linalgOp, vectorSize, isQuantized);

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);

  // Use the default distribution for the matmul loops.
  int64_t defaultMaxSize = defaultWorkgroupTileSize;
  if (isX86(targetAttr) || isRISCV(targetAttr)) {
    defaultMaxSize = 128;
  }

  bool isBM = isa<linalg::BatchMatmulOp>(contractionOp.getOperation());
  SmallVector<int64_t> maxTileSizes(numLoops, defaultMaxSize);
  if (isBM) {
    maxTileSizes[0] = 1;
  }

  // There are hard-coded configurations in DoubleTilingPadExpert, so it only
  // works for linalg.matmul cases. We can relax it once we have better
  // scheduling, e.g., transform dialect.
  SmallVector<int64_t> flowTileSizes;
  auto vecPreProcStrategy = getVectorPreProcStrategy(linalgOp);
  bool usePaddingPipeline =
      vecPreProcStrategy == VectorPreProcStrategy::Padding;

  LLVM_DEBUG(KD_DBGS() << "Vector pre-processing strategy: "
                       << vecPreProcStrategy << "\n");

  if (usePaddingPipeline) {
    // It's inspired from https://github.com/iree-org/iree-llvm-sandbox repo.
    // Sandbox has [[288, 128, 512], [12, 32, 1]] setup. We scale 288 to 192
    // because 288/12*8=192
    if (numLoops == 3) {
      maxTileSizes[0] = 192;
      maxTileSizes[1] = 128;
    }
    flowTileSizes = getDefaultDistributedLevelTileSizes(
        linalgOp, workgroupTileSizes, maxTileSizes,
        /*allowIncompleteTile=*/true);
  } else {
    flowTileSizes = getDefaultDistributedLevelTileSizes(
        linalgOp, workgroupTileSizes, maxTileSizes);
  }

  LLVM_DEBUG(KD_DBGS() << "Flow tile sizes: " << flowTileSizes << "\n");
  LLVM_DEBUG(KD_DBGS() << "Workgroup tile sizes: " << workgroupTileSizes
                       << "\n");
  LLVM_DEBUG(KD_DBGS() << "Vector size: " << vectorSize << "\n");

  // ARM codgen does not switch to use codegen driver based approach, so we have
  // special logic for it. All the new pipeline is expected to use codegen
  // driver based approach.
  if (isAArch64(targetAttr) && !isQuantized) {
    return setAArch64RootConfig(entryPointFn, contractionOp, flowTileSizes,
                                workgroupTileSizes, vectorSize);
  }

  TileSizesListType tileSizes = {flowTileSizes, workgroupTileSizes};
  if (usePaddingPipeline) {
    return setMatmulPadRootConfig(entryPointFn, contractionOp, flowTileSizes,
                                  workgroupTileSizes, vectorSize);
  }
  return setMatmulNoPadRootConfig(entryPointFn, contractionOp, tileSizes,
                                  vectorSize, vecPreProcStrategy);
}

/// Sets the lowering configuration for dispatch region for linalg.mmt4d root
/// op
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::Mmt4DOp mmt4dOp) {
  assert(!getLoweringConfig(mmt4dOp) && "expected lowering_config is not set");
  auto getWorkgroupTileSizes = [&]() -> SmallVector<int64_t> {
    if (!mmt4dWorkgroupTileSizes.empty()) {
      return SmallVector<int64_t>(mmt4dWorkgroupTileSizes.begin(),
                                  mmt4dWorkgroupTileSizes.end());
    }
    unsigned numLoops = mmt4dOp.getNumLoops();
    SmallVector<int64_t> minTileSizes(numLoops, 0);
    SmallVector<int64_t> maxTileSizes(numLoops, 0);
    minTileSizes[0] = 4;
    minTileSizes[1] = 4;
    maxTileSizes[0] = 48;
    maxTileSizes[1] = 32;
    SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
        mmt4dOp, minTileSizes, maxTileSizes);
    return flowTileSizes;
  };

  auto getL1TileSizes = [&]() -> SmallVector<int64_t> {
    auto lhsShape =
        mmt4dOp.getInputs()[0].getType().cast<ShapedType>().getShape();
    auto rhsShape =
        mmt4dOp.getInputs()[1].getType().cast<ShapedType>().getShape();
    int M0 = lhsShape[2];
    int N0 = rhsShape[2];
    int K0 = lhsShape[3];
    if (!mmt4dL1TileSizes.empty()) {
      return SmallVector<int64_t>(mmt4dL1TileSizes.begin(),
                                  mmt4dL1TileSizes.end());
    }
    return {1, 1, 1, M0, N0, K0};
  };

  SmallVector<int64_t> parallelTileSizes = getL1TileSizes();
  SmallVector<int64_t> reductionTileSizes;
  splitParallelAndReductionTiles(cast<linalg::LinalgOp>(mmt4dOp.getOperation()),
                                 parallelTileSizes, reductionTileSizes);

  TileSizesListType tileSizes = {getWorkgroupTileSizes(), parallelTileSizes,
                                 reductionTileSizes};

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, mmt4dOp, tileSizes,
      DispatchLoweringPassPipeline::Mmt4dTilingExpert);
}

static SmallVector<int64_t> getLinalgExtDefaultWorkgroupTileSizes(
    TilingInterface op) {
  unsigned numLoops = op.getLoopIteratorTypes().size();
  // Set all the distribution tile sizes to zero if thread distribution is
  // disabled.
  if (clDisableDistribution) {
    return SmallVector<int64_t>(numLoops, 0);
  }

  auto partitionedLoops = cast<PartitionableLoopsInterface>(op.getOperation())
                              .getPartitionableLoops(kNumMaxParallelDims);
  SmallVector<int64_t> workgroupTileSizes(numLoops, defaultWorkgroupTileSize);
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto dim : llvm::seq<int64_t>(0, workgroupTileSizes.size())) {
    if (!partitionedLoopsSet.count(dim)) {
      workgroupTileSizes[dim] = 0;
    }
  }

  return workgroupTileSizes;
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   tensor::PackOp op) {
  assert(!getLoweringConfig(op) && "expected lowering_config is not set");
  SmallVector<int64_t> distTileSizes = getLinalgExtDefaultWorkgroupTileSizes(
      cast<TilingInterface>(op.getOperation()));

  // The default function aims to returns the number of workload per workgroup,
  // but it does not know that it is working on packed domain. We need to take
  // inner tile sizes into account and adjust the distribution tile sizes.
  SmallVector<int64_t> innerTiles = op.getStaticTiles();
  ArrayRef<int64_t> dimPos = op.getInnerDimsPos();
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    if (distTileSizes[pos] == 0 || ShapedType::isDynamic(size)) continue;
    distTileSizes[pos] = distTileSizes[pos] / size;
    distTileSizes[pos] = std::max<int64_t>(distTileSizes[pos], 1);
  }

  SmallVector<int64_t> tileSizes(op.getSourceRank(), 1);
  TileSizesListType tileSizesList = {distTileSizes};
  tileSizesList.push_back(tileSizes);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizesList,
      DispatchLoweringPassPipeline::CPUDataTiling);
}

static LogicalResult setUnPackOpRootConfig(
    func::FuncOp entryPointFn, tensor::UnPackOp op,
    DispatchLoweringPassPipeline pipeline =
        DispatchLoweringPassPipeline::CPUDataTiling) {
  SmallVector<int64_t> distTileSizes = getLinalgExtDefaultWorkgroupTileSizes(
      cast<TilingInterface>(op.getOperation()));

  // Fixup for making distTileSizes be multiple of inner_tile_sizes.
  SmallVector<int64_t> innerTiles = op.getStaticTiles();
  ArrayRef<int64_t> dimPos = op.getInnerDimsPos();
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    if (distTileSizes[pos] == 0 || ShapedType::isDynamic(size)) continue;
    distTileSizes[pos] = llvm::alignTo(distTileSizes[pos], size);
  }

  SmallVector<int64_t> tileSizes(op.getDestRank(), 1);
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    tileSizes[pos] = ShapedType::isDynamic(size) ? 1 : size;
  }

  TileSizesListType tileSizesList = {distTileSizes};
  tileSizesList.push_back(tileSizes);

  return setOpConfigAndEntryPointFnTranslation(entryPointFn, op, tileSizesList,
                                               pipeline);
}

/// Sets the lowering configuration for dispatch region for linalg_ext.fft
/// root op.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, IREE::LinalgExt::FftOp fftOp,
    DispatchLoweringPassPipeline pipeline =
        DispatchLoweringPassPipeline::CPUDefault) {
  assert(!getLoweringConfig(fftOp) && "expected lowering_config is not set");
  SmallVector<int64_t> workgroupTileSizes =
      getLinalgExtDefaultWorkgroupTileSizes(fftOp);
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
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, fftOp, tileSizes,
                                               pipeline);
}

static void setX86WorkgroupTileSizes(
    linalg::GenericOp genericOp, unsigned numLoops,
    ArrayRef<int64_t> flowTileSizes, ArrayRef<int64_t> minTileSizes,
    ArrayRef<int64_t> maxTileSizes, VectorPreProcStrategy vecPreProcStrategy,
    SmallVectorImpl<int64_t> &workgroupTileSizes) {
  workgroupTileSizes.append(numLoops, 0);
  SmallVector<int64_t, 4> staticLoopRanges = genericOp.getStaticLoopRanges();
  for (auto loopNum : llvm::seq<unsigned>(0, numLoops)) {
    if (flowTileSizes[loopNum]) {
      workgroupTileSizes[loopNum] = getMaxVectorTileSize(
          0, flowTileSizes[loopNum], minTileSizes[loopNum],
          minTileSizes[loopNum], /*allowIncompleteTile=*/false,
          /*enforcePowerOfTwo=*/vecPreProcStrategy ==
              VectorPreProcStrategy::Masking);
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
  if ((genericOp.getNumDpsInputs() != 1) || (genericOp.getNumDpsInits() != 1)) {
    return false;
  }

  // Check that all the iterators are parallel.
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }

  // Check that the two indexing maps are a permutation of each other.
  auto indexing_maps = genericOp.getIndexingMapsArray();
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

  SmallVector<int64_t> minTileSizes = getMinTilingSizesForEachDim(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
  // For generic ops we'll use the default divided by 2 to control the stack
  // allocation limit See #9469 for example.
  SmallVector<int64_t> maxTileSizes(numLoops, defaultWorkgroupTileSize / 2);

  LLVM_DEBUG(KD_DBGS() << "Min tile sizes for distribution: " << minTileSizes
                       << "\n");
  LLVM_DEBUG(KD_DBGS() << "Max tile sizes for distribution: " << maxTileSizes
                       << "\n");

  // Set the flow level tiling to the default.
  SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
      genericOp, minTileSizes, maxTileSizes);

  LLVM_DEBUG(KD_DBGS() << "Final tile sizes for distribution: " << flowTileSizes
                       << "\n");

  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  LLVM_DEBUG(KD_DBGS() << "Vectorization pre-processing strategy "
                       << vecPreProcStrategy << "\n");

  // Set the next level tile sizes.
  SmallVector<int64_t> parallelTileSizes;
  SmallVector<int64_t> reductionTileSizes;
  setX86WorkgroupTileSizes(genericOp, numLoops, flowTileSizes, minTileSizes,
                           maxTileSizes, vecPreProcStrategy, parallelTileSizes);
  splitParallelAndReductionTiles(genericOp, parallelTileSizes,
                                 reductionTileSizes);
  setVectorSizesForDynamicShapes(genericOp, vecPreProcStrategy,
                                 parallelTileSizes, reductionTileSizes);

  LLVM_DEBUG(KD_DBGS() << "Vectorization/unrolling tile sizes (parallel): "
                       << parallelTileSizes << "\n");
  LLVM_DEBUG(KD_DBGS() << "Vectorization/unrolling tile sizes (reduction): "
                       << reductionTileSizes << "\n");

  TileSizesListType tileSizes;
  tileSizes.push_back(flowTileSizes);
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);

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
static LogicalResult setTransformStrategyRootConfig(
    func::FuncOp entryPointFn, linalg::GenericOp genericOp,
    const LinalgOpInfo &linalgOpInfo,
    const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");
  if (!clCPUEnableTransformDialectJit) return failure();
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
static LogicalResult setTransposeLikeOpRootConfig(
    func::FuncOp entryPointFn, linalg::GenericOp genericOp,
    const LinalgOpInfo &linalgOpInfo,
    const TargetMLTransformInfo &targetMLTransInfo) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  if (!hasAVX2Feature(targetAttr) || !isSupportedTransposeOp(genericOp)) {
    return failure();
  }

  unsigned numLoops = genericOp.getNumLoops();
  SmallVector<int64_t> minTileSizes = getMinTilingSizesForEachDim(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultWorkgroupTileSize);
  if (llvm::all_of(minTileSizes, [](int64_t vs) { return vs == 1; })) {
    // Nothing to vectorize just lower to loops.
    return failure();
  }

  if (llvm::count_if(minTileSizes,
                     [](int64_t tileSize) { return tileSize > 1; }) != 2) {
    // Transpose patterns are not applicable if vectorizing more or less than
    // two dims.
    return failure();
  }

  // Make sure that the original tile sizes are multiple of the tile sizes
  // to be used for the transpose op (i.e., 8x8).
  // TODO(diegocaballero): Enable 4x8 tile sizes if we find it useful.
  if (llvm::any_of(minTileSizes, [](int64_t tileSize) {
        return tileSize > 1 && (tileSize % 8) != 0;
      })) {
    return failure();
  }

  // Replace dims to be vectorized with the new 8x8 tile sizes.
  std::replace_if(
      minTileSizes.begin(), minTileSizes.end(),
      [](int64_t tileSize) { return tileSize > 1; }, 8);

  // Set the flow level tiling to the default.
  SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
      genericOp, minTileSizes, maxTileSizes);

  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  LLVM_DEBUG(KD_DBGS() << "Vectorization pre-processing strategy "
                       << vecPreProcStrategy << "\n");

  // Set the next level tile sizes.
  SmallVector<int64_t> parallelTileSizes;
  setX86WorkgroupTileSizes(genericOp, numLoops, flowTileSizes, minTileSizes,
                           maxTileSizes, vecPreProcStrategy, parallelTileSizes);

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
  if (numLoops == 0) return failure();
  if (!linalg::isElementwise(genericOp)) return failure();

  // Set the flow level tiling to the default.
  SmallVector<int64_t> minTileSizes = getMinTilingSizesForEachDim(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultWorkgroupTileSize);
  SmallVector<int64_t> flowTileSizes =
      getDefaultDistributedLevelTileSizes(genericOp, minTileSizes, maxTileSizes,
                                          /*allowIncompleteTile=*/true);

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
    if (size == ShapedType::kDynamic) {
      numWorkload = ShapedType::kDynamic;
      break;
    }
    numWorkload *= flowTileSizes[index] ? flowTileSizes[index] : size;
  }
  for (unsigned currDim = 0;
       numWorkload < kMinimumWorkload && currDim < numLoops;) {
    int64_t currSize = flowTileSizes[currDim];
    if (currSize == shape[currDim] || currSize == 0 ||
        shape[currDim] == ShapedType::kDynamic ||
        numWorkload == ShapedType::kDynamic) {
      currDim++;
      continue;
    }
    int64_t newSize = std::min<int64_t>(currSize * 2, shape[currDim]);
    numWorkload = numWorkload / currSize * newSize;
    flowTileSizes[currDim] = newSize;
  }

  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  LLVM_DEBUG(KD_DBGS() << "Vector pre-processing strategy: "
                       << vecPreProcStrategy << "\n");

  // Adjust tiling sizes of vector levels to avoid large unroll factors. Most of
  // the cases are f32 and i32, so we divide it by 4.
  auto nativeVecSize = getNativeVectorSizeInBytes(entryPointFn);
  int64_t vecSize =
      nativeVecSize ? nativeVecSize.value() : clNativeVectorSizeInBytes;
  vecSize /= 4;
  SmallVector<int64_t> vecTileSizes(minTileSizes.begin(), minTileSizes.end());
  for (auto &i : vecTileSizes) {
    i = roundUpToPow2(std::min(i, vecSize),
                      vecPreProcStrategy == VectorPreProcStrategy::Masking);
  }

  // Setting reduction tile sizes is a workaround to kick in peeling transform.
  // The tiling won't happen because the sizes are zeros.
  SmallVector<int64_t> zeros(numLoops, 0);

  TileSizesListType tileSizes;
  tileSizes.push_back(flowTileSizes);
  tileSizes.push_back(vecTileSizes);
  tileSizes.push_back(zeros);

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
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, linalg::GenericOp genericOp,
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

namespace {
bool is2DPoolingOp(linalg::LinalgOp op) {
  return isa<linalg::PoolingNhwcSumOp, linalg::PoolingNchwSumOp,
             linalg::PoolingNhwcMaxOp, linalg::PoolingNhwcMaxUnsignedOp,
             linalg::PoolingNhwcMinOp, linalg::PoolingNhwcMinUnsignedOp,
             linalg::PoolingNchwMaxOp>(op.getOperation());
}
}  // namespace

/// Sets lowering configuration for conv ops. See below for supported conv ops.
static LogicalResult setConvRootConfig(func::FuncOp entryPointFn,
                                       linalg::LinalgOp convOp,
                                       ArrayRef<int64_t> targetTileSizes,
                                       int64_t vectorSize) {
  if (!isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp,
           linalg::DepthwiseConv2DNhwcHwcOp>(convOp.getOperation()) &&
      !is2DPoolingOp(convOp)) {
    return failure();
  }
  assert(!getLoweringConfig(convOp) && "expected lowering_config is not set");

  // Use the default distribution for the conv loops.
  unsigned numLoops = convOp.getNumLoops();
  SmallVector<int64_t> minTileSizes(numLoops, 1);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultWorkgroupTileSize);
  SmallVector<int64_t> vectorSizeHints(numLoops, 1);

  // Give the vector size hint on OC.
  vectorSizeHints[3] = vectorSize;

  // Set the flow level tiling to the default.
  SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
      convOp, minTileSizes, maxTileSizes, /*allowIncompleteTile=*/false,
      vectorSizeHints);

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
          getMaxVectorTileSize(0, tileSize, parallelTileSizes[i], vectorSize);
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

/// Main utility to compute the workgroup (vectorization/unrolling) tile sizes.
/// Note that this only works for NHWC input and HWCF kernel/filter
/// convolutions, where the shape is [N, OH, OW, OC, KH, KW, (IC)].
/// TODO(hanchung): Drive the tiling sizes through heuristics. The parameters
/// are derived from limit experiments.
static SmallVector<int64_t> getConvWorkgroupSizes(func::FuncOp entryPointFn,
                                                  linalg::LinalgOp op,
                                                  int64_t vectorSize) {
  bool isSupported = isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp,
                         linalg::DepthwiseConv2DNhwcHwcOp>(op.getOperation()) ||
                     is2DPoolingOp(op);
  (void)isSupported;
  assert(isSupported && "conv op is not supported");

  SmallVector<int64_t> tileSizes;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);

  if (isX86(targetAttr)) {
    TypeSwitch<Operation *>(op.getOperation())
        .Case<linalg::Conv2DNhwcHwcfOp>(
            [&](auto op) { tileSizes = {1, 1, 8, vectorSize * 2, 1, 1, 8}; })
        .Case<linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
              linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
              linalg::PoolingNhwcMinUnsignedOp>(
            [&](auto op) { tileSizes = {1, 1, 8, vectorSize * 2, 1, 8}; })
        .Case<linalg::DepthwiseConv2DNhwcHwcOp>(
            [&](auto op) { tileSizes = {1, 1, 8, vectorSize * 2, 1, 3}; })
        .Default([&](Operation *op) { llvm_unreachable("unsupported conv"); });
  } else if (isRISCV(targetAttr)) {
    TypeSwitch<Operation *>(op.getOperation())
        .Case<linalg::Conv2DNhwcHwcfOp>(
            [&](auto op) { tileSizes = {1, 1, 8, vectorSize * 2, 1, 1, 8}; })
        .Case<linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
              linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
              linalg::PoolingNhwcMinUnsignedOp>(
            [&](auto op) { tileSizes = {1, 1, 8, vectorSize * 2, 1, 8}; })
        .Case<linalg::DepthwiseConv2DNhwcHwcOp>(
            [&](auto op) { tileSizes = {1, 1, 8, vectorSize, 1, 3}; })
        .Default([&](Operation *op) { llvm_unreachable("unsupported conv"); });
  } else if (isAArch64(targetAttr)) {
    TypeSwitch<Operation *>(op.getOperation())
        .Case<linalg::Conv2DNhwcHwcfOp>(
            [&](auto op) { tileSizes = {1, 1, 32, 64, 1, 1, 16}; })
        .Case<linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
              linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
              linalg::PoolingNhwcMinUnsignedOp>(
            [&](auto op) { tileSizes = {1, 1, 32, 64, 1, 16}; })
        .Case<linalg::DepthwiseConv2DNhwcHwcOp>(
            [&](auto op) { tileSizes = {1, 1, 4, 4, 1, 4}; })
        .Default([&](Operation *op) { llvm_unreachable("unsupported conv"); });
  } else {
    // Get default hard-coded tile sizes if we couldn't compute anything better.
    TypeSwitch<Operation *>(op.getOperation())
        .Case<linalg::Conv2DNhwcHwcfOp>([&](auto op) {
          tileSizes = {1, 1, vectorSize, vectorSize, 1, 1, vectorSize};
        })
        .Case<linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
              linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
              linalg::PoolingNhwcMinUnsignedOp>([&](auto op) {
          tileSizes = {1, 1, vectorSize, vectorSize, 1, vectorSize};
        })
        .Case<linalg::DepthwiseConv2DNhwcHwcOp>([&](auto op) {
          tileSizes = {1, 1, vectorSize, vectorSize, 1, vectorSize};
        })
        .Default([&](Operation *op) { llvm_unreachable("unsupported conv"); });
  }

  return tileSizes;
}

static LogicalResult setConvNhwcRootConfigImpl(func::FuncOp entryPointFn,
                                               linalg::LinalgOp convOp) {
  int64_t vectorSize = getVectorSize(
      entryPointFn,
      cast<ShapedType>(convOp.getDpsInitOperand(0)->get().getType()));
  SmallVector<int64_t> targetTileSizes =
      getConvWorkgroupSizes(entryPointFn, convOp, vectorSize);
  return setConvRootConfig(entryPointFn, convOp, targetTileSizes, vectorSize);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::Conv2DNhwcHwcfOp convOp) {
  return setConvNhwcRootConfigImpl(entryPointFn, convOp);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::PoolingNhwcSumOp convOp) {
  return setConvNhwcRootConfigImpl(entryPointFn, convOp);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::PoolingNhwcMaxOp convOp) {
  return setConvNhwcRootConfigImpl(entryPointFn, convOp);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::PoolingNhwcMaxUnsignedOp convOp) {
  return setConvNhwcRootConfigImpl(entryPointFn, convOp);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::PoolingNhwcMinOp convOp) {
  return setConvNhwcRootConfigImpl(entryPointFn, convOp);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::PoolingNhwcMinUnsignedOp convOp) {
  return setConvNhwcRootConfigImpl(entryPointFn, convOp);
}

/// Sets the lowering configuration for linalg.conv_2d_nchw_fchw
/// operations.
static LogicalResult setConvNchwRootConfigImpl(func::FuncOp entryPointFn,
                                               linalg::LinalgOp convOp) {
  int64_t vectorSize = getVectorSize(
      entryPointFn,
      cast<ShapedType>(convOp.getDpsInitOperand(0)->get().getType()));
  SmallVector<int64_t> targetTileSizes = {1, vectorSize * 2, 1, 8, 8, 1, 1};
  return setConvRootConfig(entryPointFn, convOp, targetTileSizes, vectorSize);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::Conv2DNchwFchwOp convOp) {
  return setConvNchwRootConfigImpl(entryPointFn, convOp);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::PoolingNchwSumOp convOp) {
  return setConvNchwRootConfigImpl(entryPointFn, convOp);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::PoolingNchwMaxOp convOp) {
  return setConvNchwRootConfigImpl(entryPointFn, convOp);
}

/// Sets the lowering configuration for linalg.depthwise_conv_2d_nhwc_hwc
/// operations.
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::DepthwiseConv2DNhwcHwcOp convOp) {
  int64_t vectorSize = getVectorSize(
      entryPointFn, cast<ShapedType>(convOp.getResult(0).getType()));
  SmallVector<int64_t> targetTileSizes =
      getConvWorkgroupSizes(entryPointFn, convOp, vectorSize);
  return setConvRootConfig(entryPointFn, convOp, targetTileSizes, vectorSize);
}

/// Set default configuration for Linalg ops.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, linalg::LinalgOp linalgOp,
    DispatchLoweringPassPipeline pipeline =
        DispatchLoweringPassPipeline::CPUDefault) {
  auto partitionableLoopOp =
      cast<PartitionableLoopsInterface>(linalgOp.getOperation());
  SmallVector<int64_t> lbs(linalgOp.getNumLoops(), 0);
  SmallVector<int64_t> ubs = linalgOp.getStaticLoopRanges();
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      entryPointFn->getContext(), pipeline);

  if (failed(setTranslationInfo(entryPointFn, translationInfo))) {
    return failure();
  }
  return setDefaultRootConfig(entryPointFn, partitionableLoopOp, lbs, ubs);
}

/// Set the default configuration for operations that implement the
/// `TiledOpInterface`.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, TilingInterface tilingInterfaceOp,
    DispatchLoweringPassPipeline pipeline =
        DispatchLoweringPassPipeline::CPUDefault) {
  assert(!getLoweringConfig(tilingInterfaceOp) &&
         "expected lowering_config is not set");
  auto partitionableLoopOp =
      cast<PartitionableLoopsInterface>(tilingInterfaceOp.getOperation());

  // TODO(hanchung): Implement getStaticLoopRanges method for TiledOpInterface.
  OpBuilder builder(tilingInterfaceOp.getContext());
  builder.setInsertionPoint(tilingInterfaceOp);
  SmallVector<Range> iterationDomain =
      tilingInterfaceOp.getIterationDomain(builder);
  auto getStaticValue = [](OpFoldResult ofr) -> int64_t {
    std::optional<int64_t> intVal = getConstantIntValue(ofr);
    if (!intVal) return ShapedType::kDynamic;
    return intVal.value();
  };
  auto lbs = llvm::to_vector(llvm::map_range(
      iterationDomain, [&](Range r) { return getStaticValue(r.offset); }));
  auto ubs = llvm::to_vector(llvm::map_range(
      iterationDomain, [&](Range r) { return getStaticValue(r.size); }));
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      entryPointFn->getContext(), pipeline);
  if (failed(setTranslationInfo(entryPointFn, translationInfo))) {
    return failure();
  }
  return setDefaultRootConfig(entryPointFn, partitionableLoopOp, lbs, ubs);
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult setRootConfigImpl(
    func::FuncOp entryPointFn, Operation *op,
    const TargetMLTransformInfo &targetMLTransInfo) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<linalg::GenericOp>([&](auto op) {
          return setRootConfig(entryPointFn, op, LinalgOpInfo(op),
                               targetMLTransInfo);
        })
        .Case<IREE::LinalgExt::FftOp, tensor::PackOp, linalg::Mmt4DOp,
              linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp,
              linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
              linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
              linalg::PoolingNhwcMinUnsignedOp, linalg::PoolingNchwSumOp,
              linalg::PoolingNchwMaxOp, linalg::DepthwiseConv2DNhwcHwcOp>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<tensor::UnPackOp>(
            [&](auto op) { return setUnPackOpRootConfig(entryPointFn, op); })
        .Case<linalg::ContractionOpInterface>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<linalg::LinalgOp>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<TilingInterface>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Redirects to methods that set the configuration based on operation type for
/// VMVX backend.
static LogicalResult setVMVXRootConfigImpl(func::FuncOp entryPointFn,
                                           Operation *op) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<IREE::LinalgExt::FftOp>([&](auto op) {
          return setRootConfig(entryPointFn, op,
                               DispatchLoweringPassPipeline::VMVXDefault);
        })
        .Case<linalg::LinalgOp>([&](auto op) {
          return setRootConfig(entryPointFn, op,
                               DispatchLoweringPassPipeline::VMVXDefault);
        })
        .Case<tensor::UnPackOp>([&](auto op) {
          return setUnPackOpRootConfig(
              entryPointFn, op, DispatchLoweringPassPipeline::VMVXDefault);
        })
        .Case<TilingInterface>([&](auto op) {
          return setRootConfig(entryPointFn, op,
                               DispatchLoweringPassPipeline::VMVXDefault);
        })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Find the root operation for the dispatch region. The priority is:
///   1. A Linalg operation that has reduction loops.
///   2. Any other Lainlg op or LinalgExt op.
///   3. An operation that implements TilingInterface.
/// If there are multiple operations meeting the same priority, the one closer
/// to the end of the function is the root op.
static FailureOr<Operation *> getRootOperation(
    ArrayRef<Operation *> computeOps) {
  for (auto op : llvm::reverse(computeOps)) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) continue;
    if (linalgOp.getNumReductionLoops()) return op;
  }

  for (auto op : llvm::reverse(computeOps)) {
    if (isa<linalg::LinalgOp, IREE::LinalgExt::LinalgExtOp>(op)) return op;
  }

  for (auto op : llvm::reverse(computeOps)) {
    if (isa<TilingInterface>(op)) return op;
  }

  return nullptr;
}

static LogicalResult adjustTileSizesForPackOp(func::FuncOp entryPointFn,
                                              Operation *rootOp) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp) return success();

  // The transform dialect codegen has differnet logics and codegen flow. Ignore
  // the adjustment for it.
  auto pipeline = getTranslationInfo(entryPointFn).getPassPipeline().getValue();
  if (pipeline == DispatchLoweringPassPipeline::TransformDialectCodegen) {
    return success();
  }
  TileSizesListType tileSizesList =
      getLoweringConfig(linalgOp).getTileSizeVals();

  bool hasChanged = false;
  auto res = entryPointFn.walk([&](tensor::PackOp packOp) -> WalkResult {
    // Multiple pack ops case is not supported.
    if (hasChanged) return WalkResult::interrupt();

    // TODO(hanchung): Support chain cases. It needs following use-def chain
    // until rootOp.
    if (packOp.getSource().getDefiningOp() != rootOp) {
      return WalkResult::advance();
    }
    hasChanged = true;
    OpResult result = packOp.getSource().cast<OpResult>();
    auto idxMap = linalgOp.getMatchingIndexingMap(
        linalgOp.getDpsInitOperand(result.getResultNumber()));
    (void)idxMap;
    LLVM_DEBUG(KD_DBGS() << "Find pack op candidate: " << packOp << "\n"
                         << "The corresponding indexing map is: " << idxMap
                         << "\n");
    assert(idxMap.isIdentity() && "unexpected codegen input");

    for (SmallVectorImpl<int64_t> &tileSizes : tileSizesList) {
      SmallVector<int64_t> innerTiles = packOp.getStaticTiles();
      ArrayRef<int64_t> dimPos = packOp.getInnerDimsPos();
      for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
        if (tileSizes[pos] == 0 || ShapedType::isDynamic(size)) continue;
        tileSizes[pos] = tileSizes[pos] / size;
        tileSizes[pos] = std::max<int64_t>(tileSizes[pos], 1);
        LLVM_DEBUG(KD_DBGS() << "Scale # " << pos << " tile size to "
                             << tileSizes[pos] << "\n");
      }
    }

    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();

  return setOpConfigAndEntryPointFnTranslation(entryPointFn, rootOp,
                                               tileSizesList, pipeline);
}

static LogicalResult adjustTileSizesForUnPackOp(func::FuncOp entryPointFn,
                                                Operation *rootOp) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp) return success();

  // The transform dialect codegen has differnet logics and codegen flow. Ignore
  // the adjustment for it.
  auto pipeline = getTranslationInfo(entryPointFn).getPassPipeline().getValue();
  if (pipeline == DispatchLoweringPassPipeline::TransformDialectCodegen) {
    return success();
  }
  TileSizesListType tileSizesList =
      getLoweringConfig(linalgOp).getTileSizeVals();

  bool foundUnPackOp = false;
  SmallVector<int64_t> alignedSizes(linalgOp.getNumLoops(), 1);
  for (OpOperand *opOperand : linalgOp.getDpsInputOperands()) {
    auto unpackOp = opOperand->get().getDefiningOp<tensor::UnPackOp>();
    if (!unpackOp) continue;

    foundUnPackOp = true;
    auto idxMap = linalgOp.getMatchingIndexingMap(opOperand);
    LLVM_DEBUG(KD_DBGS() << "Find unpack op candidate: " << unpackOp << "\n"
                         << "The corresponding indexing map is: " << idxMap
                         << "\n");

    SmallVector<int64_t> innerTiles = unpackOp.getStaticTiles();
    ArrayRef<int64_t> dimPos = unpackOp.getInnerDimsPos();
    for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
      if (ShapedType::isDynamic(size)) continue;
      auto dimExpr = idxMap.getResult(pos).dyn_cast<AffineDimExpr>();
      if (!dimExpr) return failure();
      int mappedPos = dimExpr.getPosition();
      alignedSizes[mappedPos] = std::lcm(alignedSizes[mappedPos], size);
    }
  }

  if (!foundUnPackOp) return success();

  LLVM_DEBUG(
      KD_DBGS() << "The tile sizes for each dimension should be aligned to "
                << alignedSizes);

  // Fixup for making tileSizes be multiple of inner_tile_sizes.
  for (SmallVectorImpl<int64_t> &tileSizes : tileSizesList) {
    for (auto idx : llvm::seq<int64_t>(0, tileSizes.size())) {
      if (tileSizes[idx] == 0) continue;
      tileSizes[idx] = llvm::alignTo(tileSizes[idx], alignedSizes[idx]);
    }
  }

  if (pipeline == DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert) {
    LLVM_DEBUG(KD_DBGS() << "unpack fusion does not work with peeling, falling "
                            "back to non-peeling path");
    pipeline = DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
  }

  return setOpConfigAndEntryPointFnTranslation(entryPointFn, rootOp,
                                               tileSizesList, pipeline);
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult setTranslationInfoAndRootConfig(
    func::FuncOp entryPointFn, ArrayRef<Operation *> computeOps) {
  if (computeOps.empty()) {
    // No compute operations found. Allow to pass through without a config.
    return success();
  }

  // First check if the operations have a preset pipeline. If the config is
  // preset, do not overwrite it.
  for (auto computeOp : computeOps) {
    if (IREE::Codegen::CompilationInfoAttr compilationInfo =
            getCompilationInfo(computeOp)) {
      return setUserConfig(entryPointFn, computeOp, compilationInfo);
    }
  }

  // Make sure that lowering_config is not preset on any compute ops.
  for (auto computeOp : computeOps) {
    if (getLoweringConfig(computeOp)) return failure();
  }

  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp)) return failure();
  Operation *rootOperation = rootOp.value();

  // Handle the case with no known root operation.
  if (!rootOperation) {
    // If there is no translation info set, just set it to default.
    if (!getTranslationInfo(entryPointFn)) {
      auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
          entryPointFn->getContext(), DispatchLoweringPassPipeline::CPUDefault);
      // Fall back, just set the translation to CPUDefault.
      if (failed(setTranslationInfo(entryPointFn, translationInfo))) {
        return failure();
      }
    }
    return success();
  }

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  if (isVMVXBackend(targetAttr)) {
    if (failed(setVMVXRootConfigImpl(entryPointFn, rootOperation))) {
      return failure();
    }
  } else {
    auto targetMLTransInfo =
        TargetMLTransformInfo::getTargetMLTransformInfo(targetAttr);
    if (failed(setRootConfigImpl(entryPointFn, rootOperation,
                                 targetMLTransInfo))) {
      return failure();
    }
  }

  if (failed(adjustTileSizesForUnPackOp(entryPointFn, rootOperation))) {
    return failure();
  }

  if (failed(adjustTileSizesForPackOp(entryPointFn, rootOperation))) {
    return failure();
  }

  return success();
}

LogicalResult initCPULaunchConfig(ModuleOp moduleOp) {
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp) continue;
    if (getTranslationInfo(exportOp)) continue;

    // If using the transform dialect with a script file, intercept early.
    if (!clCPUCodegenTransformDialectFileName.empty()) {
      assert(!clCPUEnableTransformDialectJit &&
             "Can't use both transform dialect interpreted and jitted modes");
      auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
          moduleOp.getContext(),
          IREE::Codegen::DispatchLoweringPassPipeline::TransformDialectCodegen);
      if (failed(setTranslationInfo(funcOp, translationInfo))) return failure();
      continue;
    }

    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    if (failed(setTranslationInfoAndRootConfig(funcOp, computeOps))) {
      return failure();
    }
  }

  // The root configuration setting introduces `tensor.dim` operations. Resolve
  // those away.
  RewritePatternSet patterns(moduleOp.getContext());
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

}  // namespace iree_compiler
}  // namespace mlir
