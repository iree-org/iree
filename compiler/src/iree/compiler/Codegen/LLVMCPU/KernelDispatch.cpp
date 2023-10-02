// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"

#include <numeric>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/LLVMCPU/TargetMLTransformInfo.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/TransformStrategies/CPU/Common.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
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
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
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

static llvm::cl::opt<int> clNumberOfRuntimeThreads(
    "iree-codegen-llvm-number-of-threads",
    llvm::cl::desc("number of threads that are used at runtime if codegen "
                   "thread distribution is enabled"),
    llvm::cl::init(8));

static llvm::cl::opt<bool> clDisableDistribution(
    "iree-codegen-llvm-disable-distribution",
    llvm::cl::desc("disable thread distribution in codegen"),
    llvm::cl::init(false));

static llvm::cl::list<int> mmt4dDistributionTileSizes(
    "iree-codegen-llvm-mmt4d-distribution-tile-sizes",
    llvm::cl::desc("linalg.mmt4d distribution tile size"),
    llvm::cl::ZeroOrMore);

static llvm::cl::list<int>
    mmt4dL1TileSizes("iree-codegen-llvm-mmt4d-l1-tile-size",
                     llvm::cl::desc("linalg.mmt4d L1 tile size"),
                     llvm::cl::ZeroOrMore);

static llvm::cl::list<int>
    mmt4dVectorSizes("iree-codegen-llvm-mmt4d-vector-size",
                     llvm::cl::desc("linalg.mmt4d vector tile size"),
                     llvm::cl::ZeroOrMore);

static llvm::cl::opt<int>
    defaultDistTileSize("iree-codegen-llvm-distribution-size",
                        llvm::cl::desc("default distribution tile size"),
                        llvm::cl::init(64));

// TODO(hanchung): Remove the flag. This is the flag for fastly falling back to
// the previous snapshot.

static llvm::cl::opt<bool>
    enableVectorPadding("iree-codegen-enable-vector-padding",
                        llvm::cl::desc("Enable padding for vectorization"),
                        llvm::cl::init(true));

static llvm::cl::opt<bool>
    enableVectorPeeling("iree-codegen-enable-vector-peeling",
                        llvm::cl::desc("Enable peeling for vectorization"),
                        llvm::cl::init(true));

// Non-static options are used in other places.
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

/// Splits the given `Range` vector and returns the `lbs` and the `ubs` as
/// separate lists.
static void getBoundsFromRange(ArrayRef<Range> loopRange,
                               SmallVector<int64_t> &lb,
                               SmallVector<int64_t> &ub) {
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

/// Returns the vectorization pre-processing strategy (padding, peeling) for the
/// given LinalgOp, depending on the op traits and the target architecture.
static VectorPreProcStrategy
getVectorPreProcStrategy(linalg::LinalgOp linalgOp) {
  // Generic strategies.

  if (linalgOp.hasBufferSemantics()) {
    return VectorPreProcStrategy::None;
  }

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(linalgOp);
  bool isLinalgGeneric = isa<linalg::GenericOp>(linalgOp.getOperation());
  bool isByteAligned = hasByteAlignedElementTypes(linalgOp);

  // Default X86 specific strategy.
  if (isX86(targetAttr)) {
    if (isLinalgGeneric && isByteAligned) {
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
    if (isLinalgGeneric && isByteAligned) {
      return VectorPreProcStrategy::Masking;
    }

    if (enableVectorPeeling) {
      return VectorPreProcStrategy::Peeling;
    }
  }

  // Default AArch64 specific strategies.
  if (isAArch64(targetAttr)) {
    if (hasAnySVEFeature(targetAttr) && isByteAligned) {
      return VectorPreProcStrategy::Masking;
    }
    if ((linalg::isElementwise(linalgOp) || isFullyDynamicOp(linalgOp)) &&
        enableVectorPeeling) {
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

  // TODO(dcaballe): Remove this workaround for VMVX.
  if (isVMVXBackend(targetAttr)) {
    constexpr int64_t defaultNativeVectorSizeforVMVX = 16;
    return defaultNativeVectorSizeforVMVX;
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
        map.getResults().back().dyn_cast<AffineDimExpr>();
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
                  if (!dispatchTensorType.getRank())
                    return nullptr;
                  return dispatchTensorType.getBoundElementType();
                })
            .Case<ShapedType>([&](auto shapedType) -> Type {
              // Ignore operands that are 0D tensors. These
              // are not vector-loadable, so using these to
              // get vector length would be a pessimization.
              if (!shapedType.getRank())
                return nullptr;
              return shapedType.getElementType();
            })
            .Default([&](Type t) -> Type { return nullptr; });
    if (!elementType || !elementType.isIntOrFloat())
      return;
    unsigned typeWidthInBytes =
        IREE::Util::getRoundedElementByteWidth(elementType);
    referenceTypeLengthInBytes =
        std::min<unsigned>(referenceTypeLengthInBytes, typeWidthInBytes);
  });
  return referenceTypeLengthInBytes;
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

/// Returns the tile size to use for distribution.
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
  assert((vectorSizeHints.empty() || vectorSizeHints.size() == numLoops) &&
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

  SmallVector<int64_t> distributedTileSizes = getDefaultDistributionTileSizes(
      lbs, ubs, adjustedMinTileSizes, adjustedMaxTileSizes,
      adjustedVectorSizeHints);
  // Final fix up of the tile sizes to make sure that they divide the problem
  // size to make it vectorizable.
  for (auto i : llvm::seq<unsigned>(0, distributedTileSizes.size())) {
    if (!distributedTileSizes[i])
      continue;
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
static LogicalResult
setDefaultRootConfig(func::FuncOp entryPointFn,
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
      maxTileSizes[partitionableLoopId] = defaultDistTileSize;
    }
  }

  SmallVector<int64_t> distTileSizes = getDefaultDistributedLevelTileSizes(
      partitionableLoops, lbs, ubs, minTileSizes, maxTileSizes);
  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(distTileSizes));
  auto loweringConfig = IREE::Codegen::LoweringConfigAttr::get(
      entryPointFn.getContext(), tileSizes);
  setLoweringConfig(partitionableLoopsInterfaceOp, loweringConfig);
  return success();
}

static LogicalResult setMatmulPadRootConfig(func::FuncOp entryPointFn,
                                            linalg::ContractionOpInterface op,
                                            ArrayRef<int64_t> distTileSizes,
                                            ArrayRef<int64_t> vecTileSizes,
                                            int vectorSize) {
  // The tiling for parallel dims and reduction dims should be separated.
  SmallVector<int64_t> parallelTileSizes(vecTileSizes.begin(),
                                         vecTileSizes.end());
  parallelTileSizes.back() = 0;

  // Clamp inner tiling sizes to avoid masking. The vector masking takes the
  // last level of tiling to create masks. It would lead to incorrect masking if
  // the inner tiling sizes are not clamped. Because padding won't be applied
  // along those dimensions.
  for (const auto &[index, size] : llvm::enumerate(distTileSizes)) {
    if (!size)
      continue;
    parallelTileSizes[index] = std::min(parallelTileSizes[index], size);
  }

  // TODO(hanchung): Make logic more heuristic. Padding hurts performance a lot
  // if the dim size is small (e.g., K=24).
  int64_t numTilingDims = vecTileSizes.size();
  SmallVector<int64_t> reductionTileSizes(numTilingDims - 1, 0);
  auto lhsShapedType = llvm::cast<ShapedType>(op.lhs().getType());
  int64_t K = lhsShapedType.getShape().back();
  reductionTileSizes.push_back(
      getMaxVectorTileSize(0, K, vecTileSizes.back(), vectorSize));

  TileSizesListType tileSizes;
  tileSizes.emplace_back(distTileSizes.begin(), distTileSizes.end());
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);
  // No need for tiling inner parallel dims.
  tileSizes.emplace_back(numTilingDims, 0);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes,
      DispatchLoweringPassPipeline::CPUDoubleTilingPadExpert);
}

static DispatchLoweringPassPipeline
getNoPadTilingExpert(VectorPreProcStrategy strategy) {
  if (strategy == VectorPreProcStrategy::Peeling) {
    return DispatchLoweringPassPipeline::CPUDoubleTilingPeelingExpert;
  }
  return DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
}

static LogicalResult setMatmulNoPadRootConfig(
    func::FuncOp entryPointFn, linalg::ContractionOpInterface op,
    const TileSizesListTypeRef inputTileSizes,
    const ScalableTileFlagsListTypeRef inputScalableTileFlags, int vectorSize,
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
      if (tileSize == 0 || shape[idx] == ShapedType::kDynamic)
        continue;
      assert(shape[idx] % tileSize == 0);
      shape[idx] = tileSize;
    }
  }

  // The tiling for parallel dims and reduction dims are separated.
  const SmallVectorImpl<int64_t> &vecTileSizes = inputTileSizes.back();
  const SmallVectorImpl<bool> &vecScalableDims = inputScalableTileFlags.back();
  SmallVector<int64_t> parallelTileSizes;
  SmallVector<bool> parallelScalableFlags;
  for (auto [index, tileSize] : llvm::enumerate(vecTileSizes)) {
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
    // TODO: How to handle scalable sizes with getMaxVectorTileSize()?
    parallelScalableFlags.push_back(vecScalableDims[index]);
  }
  SmallVector<int64_t> reductionTileSizes;
  SmallVector<bool> reductionScalableFlags;
  splitParallelAndReductionTiles(
      cast<linalg::LinalgOp>(op.getOperation()), parallelTileSizes,
      reductionTileSizes, &parallelScalableFlags, &reductionScalableFlags);

  setVectorSizesForDynamicShapes(cast<linalg::LinalgOp>(op.getOperation()),
                                 vecPreProcStrategy, parallelTileSizes,
                                 reductionTileSizes);

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

  LLVM_DEBUG(
      KD_DBGS() << "Final tile sizes for no-padding contraction: "
                << newTileSizes << "\n"
                << "Final tile scalable flags for no-padding contraction: "
                << newScalableTileFlags << "\n");

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, newTileSizes, newScalableTileFlags,
      getNoPadTilingExpert(vecPreProcStrategy));
}

/// Configure the Mmt4d tiling expert for AArch64
static LogicalResult
setMmt4dAArch64RootConfig(func::FuncOp entryPointFn,
                          linalg::ContractionOpInterface op,
                          ArrayRef<int64_t> distTileSizes,
                          ArrayRef<int64_t> vecTileSizes, int vectorSize) {
  assert(distTileSizes.size() == vecTileSizes.size());
  SmallVector<int64_t> parallelTileSizes;
  auto shape = cast<linalg::LinalgOp>(op.getOperation()).getStaticLoopRanges();
  for (auto [index, tileSize] : llvm::enumerate(distTileSizes.drop_back())) {
    parallelTileSizes.push_back(
        getMaxVectorTileSize(0, tileSize ? tileSize : shape[index],
                             vecTileSizes[index], vectorSize));
  }

  auto lhsShapedType = llvm::cast<ShapedType>(op.lhs().getType());
  int64_t K = lhsShapedType.getShape().back();
  parallelTileSizes.push_back(
      getMaxVectorTileSize(0, K, vecTileSizes.back(), vectorSize));

  SmallVector<int64_t> reductionTileSizes;
  splitParallelAndReductionTiles(cast<linalg::LinalgOp>(op.getOperation()),
                                 parallelTileSizes, reductionTileSizes);

  TileSizesListType tileSizes;
  tileSizes.emplace_back(distTileSizes.begin(), distTileSizes.end());
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);
  // No need for tiling inner parallel dims.
  int64_t numTilingDims = parallelTileSizes.size();
  tileSizes.emplace_back(numTilingDims, 0);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes,
      DispatchLoweringPassPipeline::Mmt4dTilingExpert);
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

  // Specialisation for SVE.
  if (isAArch64(targetAttr) && hasAnySVEFeature(targetAttr)) {
    // Mark middle dimensions as scalable, so sizes are (8, [32], 16).
    sizes.append({8, 32, 16});
    scalableSizeFlags.append({false, true, false});
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

/// Main utility to compute the vectorization/unrolling tile sizes.
static SizesAndScalableFlags getMatmulVectorSizes(func::FuncOp entryPointFn,
                                                  linalg::LinalgOp op,
                                                  int64_t vectorSize,
                                                  bool isQuantized) {
  SmallVector<int64_t> matmulTileSizes;
  SmallVector<bool> matmulScalableFlags;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);

  // Compute vector tile sizes using heuristics.
  // TODO: if (isX86(targetAttr) || isRISCV(targetAttr)) {

  // FIXME: Introduce a more structured way to specialise for SVE
  if (isAArch64(targetAttr) && !hasAnySVEFeature(targetAttr)) {
    if (isQuantized) {
      matmulTileSizes = {vectorSize, vectorSize * 4, vectorSize};
    } else {
      matmulTileSizes = {5 * vectorSize, vectorSize, vectorSize * 16};
    }
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
  SmallVector<int64_t> staticShape = op.getStaticLoopRanges();
  if (numLoops >= 3) {
    for (int i = 0; i < (numLoops - 2); ++i) {
      int64_t dimSize = staticShape[i];
      int64_t tileSize = tileSizes[i];
      if (tileSize == 0 || ShapedType::isDynamic(dimSize)) {
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

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);

  // Use the default distribution for the matmul loops.
  int64_t defaultMaxSize = defaultDistTileSize;
  if (isX86(targetAttr) || isRISCV(targetAttr) ||
      (isAArch64(targetAttr) && hasAnySVEFeature(targetAttr))) {
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
  SmallVector<int64_t> distTileSizes;
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
    distTileSizes = getDefaultDistributedLevelTileSizes(
        linalgOp, vecTileSizes, maxTileSizes,
        /*allowIncompleteTile=*/true);
  } else {
    distTileSizes = getDefaultDistributedLevelTileSizes(linalgOp, vecTileSizes,
                                                        maxTileSizes);
  }

  LLVM_DEBUG(KD_DBGS() << "Distribution tile sizes: " << distTileSizes << "\n");
  LLVM_DEBUG(KD_DBGS() << "Vector tile sizes: " << vecTileSizes << "\n");
  LLVM_DEBUG(KD_DBGS() << "Vector scalable tile flags: " << vecScalableFlags
                       << "\n");
  LLVM_DEBUG(KD_DBGS() << "Vector size: " << vectorSize << "\n");

  // ARM SVE codgen switches to use codegen driver based approach. In non-SVE
  // cases we use special logic instead. All the new pipeline is expected to use
  // codegen driver based approach.
  if (isAArch64(targetAttr) && !isQuantized && !hasAnySVEFeature(targetAttr)) {
    return setMmt4dAArch64RootConfig(entryPointFn, contractionOp, distTileSizes,
                                     vecTileSizes, vectorSize);
  }

  if (usePaddingPipeline) {
    // TODO: Use scalable vector sizes.
    return setMatmulPadRootConfig(entryPointFn, contractionOp, distTileSizes,
                                  vecTileSizes, vectorSize);
  }
  SmallVector<bool> distScalableTileFlags(distTileSizes.size(), false);
  TileSizesListType tileSizes = {distTileSizes, vecTileSizes};
  ScalableTileFlagsListType scalableTileFlags = {distScalableTileFlags,
                                                 vecScalableFlags};
  return setMatmulNoPadRootConfig(entryPointFn, contractionOp, tileSizes,
                                  scalableTileFlags, vectorSize,
                                  vecPreProcStrategy);
}

/// Sets the lowering configuration for dispatch region for linalg.mmt4d root
/// op
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::Mmt4DOp mmt4dOp) {
  assert(!getLoweringConfig(mmt4dOp) && "expected lowering_config is not set");
  auto getDistTileSizes = [&]() -> SmallVector<int64_t> {
    if (!mmt4dDistributionTileSizes.empty()) {
      return SmallVector<int64_t>(mmt4dDistributionTileSizes.begin(),
                                  mmt4dDistributionTileSizes.end());
    }
    unsigned numLoops = mmt4dOp.getNumLoops();
    SmallVector<int64_t> minTileSizes(numLoops, 0);
    SmallVector<int64_t> maxTileSizes(numLoops, 0);
    minTileSizes[0] = 4;
    minTileSizes[1] = 4;
    maxTileSizes[0] = 48;
    maxTileSizes[1] = 32;
    SmallVector<int64_t> distTileSizes = getDefaultDistributedLevelTileSizes(
        mmt4dOp, minTileSizes, maxTileSizes);
    return distTileSizes;
  };

  auto getL1TileSizes = [&]() -> SmallVector<int64_t> {
    auto lhsShape =
        llvm::cast<ShapedType>(mmt4dOp.getInputs()[0].getType()).getShape();
    auto rhsShape =
        llvm::cast<ShapedType>(mmt4dOp.getInputs()[1].getType()).getShape();
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

  TileSizesListType tileSizes = {getDistTileSizes(), parallelTileSizes,
                                 reductionTileSizes};

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, mmt4dOp, tileSizes,
      DispatchLoweringPassPipeline::Mmt4dTilingExpert);
}

/// Sets the lowering configuration for dispatch region for linalg.batch_mmt4d
/// root op
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::BatchMmt4DOp batchMmt4dOp) {
  assert(!getLoweringConfig(batchMmt4dOp) &&
         "expected lowering_config is not set");
  auto getDistTileSizes = [&]() -> SmallVector<int64_t> {
    if (!mmt4dDistributionTileSizes.empty()) {
      SmallVector<int64_t> tileSizes;
      // If mmt4dDistributionTileSizes is set, tile batch dim to 1 + the
      // specified mmt4d tile sizes.
      tileSizes.push_back(1);
      tileSizes.append(mmt4dDistributionTileSizes.begin(),
                       mmt4dDistributionTileSizes.end());
      return tileSizes;
    }
    unsigned numLoops = batchMmt4dOp.getNumLoops();
    SmallVector<int64_t> minTileSizes(numLoops, 0);
    SmallVector<int64_t> maxTileSizes(numLoops, 0);
    minTileSizes[0] = 1;
    minTileSizes[1] = 4;
    minTileSizes[2] = 4;
    maxTileSizes[0] = 1;
    maxTileSizes[1] = 48;
    maxTileSizes[2] = 32;
    SmallVector<int64_t> distTileSizes = getDefaultDistributedLevelTileSizes(
        batchMmt4dOp, minTileSizes, maxTileSizes);
    return distTileSizes;
  };

  auto getL1TileSizes = [&]() -> SmallVector<int64_t> {
    SmallVector<int64_t> tileSizes;
    // Tile batch dim to 1
    tileSizes.push_back(1);

    // If mmt4dL1TileSizes is set, use them as mmt4d L1 tile sizes.
    if (!mmt4dL1TileSizes.empty()) {
      tileSizes.append(mmt4dL1TileSizes.begin(), mmt4dL1TileSizes.end());
      return tileSizes;
    }

    auto lhsShape =
        llvm::cast<ShapedType>(batchMmt4dOp.getInputs()[0].getType())
            .getShape();
    auto rhsShape =
        llvm::cast<ShapedType>(batchMmt4dOp.getInputs()[1].getType())
            .getShape();
    int M0 = lhsShape[3];
    int N0 = rhsShape[3];
    int K0 = lhsShape[4];
    tileSizes.append({1, 1, 1, M0, N0, K0});
    return tileSizes;
  };

  SmallVector<int64_t> parallelTileSizes = getL1TileSizes();
  SmallVector<int64_t> reductionTileSizes;
  splitParallelAndReductionTiles(
      cast<linalg::LinalgOp>(batchMmt4dOp.getOperation()), parallelTileSizes,
      reductionTileSizes);

  TileSizesListType tileSizes = {getDistTileSizes(), parallelTileSizes,
                                 reductionTileSizes};

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, batchMmt4dOp, tileSizes,
      DispatchLoweringPassPipeline::Mmt4dTilingExpert);
}

static SmallVector<int64_t>
getDefaultDistributionTileSizes(TilingInterface op) {
  unsigned numLoops = op.getLoopIteratorTypes().size();
  // Set all the distribution tile sizes to zero if thread distribution is
  // disabled.
  if (clDisableDistribution) {
    return SmallVector<int64_t>(numLoops, 0);
  }

  auto partitionedLoops = cast<PartitionableLoopsInterface>(op.getOperation())
                              .getPartitionableLoops(kNumMaxParallelDims);
  SmallVector<int64_t> distTileSizes(numLoops, defaultDistTileSize);
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto dim : llvm::seq<int64_t>(0, distTileSizes.size())) {
    if (!partitionedLoopsSet.count(dim))
      distTileSizes[dim] = 0;
  }

  return distTileSizes;
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
  if (hasAVX512fFeature(targetAttr) && isPackMatmulLHS(op)) {
    tileSizes.back() = vectorSize;
  }
  return tileSizes;
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   tensor::PackOp op) {
  assert(!getLoweringConfig(op) && "expected lowering_config is not set");
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributionTileSizes(cast<TilingInterface>(op.getOperation()));

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
  TileSizesListType tileSizesList = {distTileSizes};
  tileSizesList.push_back(vecTileSizes);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizesList,
      DispatchLoweringPassPipeline::CPUDataTiling);
}

static LogicalResult
setUnPackOpRootConfig(func::FuncOp entryPointFn, tensor::UnPackOp op,
                      DispatchLoweringPassPipeline pipeline =
                          DispatchLoweringPassPipeline::CPUDataTiling) {
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributionTileSizes(cast<TilingInterface>(op.getOperation()));

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

  TileSizesListType tileSizesList = {distTileSizes};
  tileSizesList.push_back(tileSizes);

  return setOpConfigAndEntryPointFnTranslation(entryPointFn, op, tileSizesList,
                                               pipeline);
}

/// Sets the lowering configuration for dispatch region for linalg_ext.fft
/// root op.
static LogicalResult
setRootConfig(func::FuncOp entryPointFn, IREE::LinalgExt::FftOp fftOp,
              DispatchLoweringPassPipeline pipeline =
                  DispatchLoweringPassPipeline::CPUDefault) {
  assert(!getLoweringConfig(fftOp) && "expected lowering_config is not set");
  SmallVector<int64_t> distTileSizes = getDefaultDistributionTileSizes(fftOp);
  auto rank = fftOp.getOperandRank();
  if (distTileSizes.size() >= rank && distTileSizes[rank - 1] != 0) {
    APInt value;
    if (matchPattern(fftOp.getStage(), m_ConstantInt(&value))) {
      distTileSizes[rank - 1] = 1ll << value.getSExtValue();
      distTileSizes[rank - 1] = std::max(
          distTileSizes[rank - 1], static_cast<int64_t>(defaultDistTileSize));
    } else {
      return fftOp.emitOpError("non-constant stage might not work for fft op");
    }
  }
  TileSizesListType tileSizes = {distTileSizes};
  return setOpConfigAndEntryPointFnTranslation(entryPointFn, fftOp, tileSizes,
                                               pipeline);
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
          0, distTileSizes[loopNum], minTileSizes[loopNum],
          minTileSizes[loopNum], /*allowIncompleteTile=*/false,
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
  auto indexing_maps = genericOp.getIndexingMapsArray();
  return !indexing_maps[0].isEmpty() && !indexing_maps[1].isEmpty() &&
         ((indexing_maps[0].isIdentity() && !indexing_maps[1].isIdentity() &&
           indexing_maps[1].isPermutation()) ||
          (!indexing_maps[0].isIdentity() && indexing_maps[0].isPermutation() &&
           indexing_maps[1].isIdentity()));
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

  SmallVector<int64_t> minTileSizes = getMinTilingSizesForEachDim(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
  // For generic ops we'll use the default divided by 2 to control the stack
  // allocation limit See #9469 for example.
  SmallVector<int64_t> maxTileSizes(numLoops, defaultDistTileSize / 2);

  LLVM_DEBUG(KD_DBGS() << "Min tile sizes for distribution: " << minTileSizes
                       << "\n");
  LLVM_DEBUG(KD_DBGS() << "Max tile sizes for distribution: " << maxTileSizes
                       << "\n");

  SmallVector<int64_t> distTileSizes = getDefaultDistributedLevelTileSizes(
      genericOp, minTileSizes, maxTileSizes);

  LLVM_DEBUG(KD_DBGS() << "Final tile sizes for distribution: " << distTileSizes
                       << "\n");

  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  LLVM_DEBUG(KD_DBGS() << "Vectorization pre-processing strategy "
                       << vecPreProcStrategy << "\n");

  // Set the next level tile sizes.
  SmallVector<int64_t> parallelTileSizes;
  SmallVector<int64_t> reductionTileSizes;
  setX86VectorTileSizes(genericOp, numLoops, distTileSizes, minTileSizes,
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
  tileSizes.push_back(distTileSizes);
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);
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
  if (!clCPUEnableTransformDialectJit)
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
  SmallVector<int64_t> minTileSizes = getMinTilingSizesForEachDim(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultDistTileSize);
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

  SmallVector<int64_t> distTileSizes = getDefaultDistributedLevelTileSizes(
      genericOp, minTileSizes, maxTileSizes);

  auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
  LLVM_DEBUG(KD_DBGS() << "Vectorization pre-processing strategy "
                       << vecPreProcStrategy << "\n");

  // Set the next level tile sizes.
  SmallVector<int64_t> parallelTileSizes;
  setX86VectorTileSizes(genericOp, numLoops, distTileSizes, minTileSizes,
                        maxTileSizes, vecPreProcStrategy, parallelTileSizes);

  TileSizesListType tileSizes;
  tileSizes.push_back(distTileSizes);
  tileSizes.push_back(parallelTileSizes);
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

  SmallVector<int64_t> minTileSizes = getMinTilingSizesForEachDim(
      entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultDistTileSize);
  SmallVector<int64_t> distTileSizes =
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
    numWorkload *= distTileSizes[index] ? distTileSizes[index] : size;
  }
  for (unsigned currDim = 0;
       numWorkload < kMinimumWorkload && currDim < numLoops;) {
    int64_t currSize = distTileSizes[currDim];
    if (currSize == shape[currDim] || currSize == 0 ||
        shape[currDim] == ShapedType::kDynamic ||
        numWorkload == ShapedType::kDynamic) {
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
  SmallVector<int64_t> vecTileSizes(minTileSizes.begin(), minTileSizes.end());
  for (auto &i : vecTileSizes) {
    i = roundUpToPow2(std::min(i, vecSize),
                      vecPreProcStrategy == VectorPreProcStrategy::Masking);
  }

  // Setting reduction tile sizes is a workaround to kick in peeling transform.
  // The tiling won't happen because the sizes are zeros.
  SmallVector<int64_t> zeros(numLoops, 0);

  TileSizesListType tileSizes;
  tileSizes.push_back(distTileSizes);
  tileSizes.push_back(vecTileSizes);
  tileSizes.push_back(zeros);
  // No need for further tiling inner parallel dims.
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

  // Use the default distribution for the conv loops.
  unsigned numLoops = convOp.getNumLoops();
  SmallVector<int64_t> minTileSizes(numLoops, 1);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultDistTileSize);
  SmallVector<int64_t> vectorSizeHints(numLoops, 1);

  // Give the vector size hint on OC.
  vectorSizeHints[3] = vectorSize;

  SmallVector<int64_t> distTileSizes = getDefaultDistributedLevelTileSizes(
      convOp, minTileSizes, maxTileSizes, /*allowIncompleteTile=*/false,
      vectorSizeHints);

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
          getMaxVectorTileSize(0, tileSize, parallelTileSizes[i], vectorSize);
    }
  }
  SmallVector<int64_t> reductionTileSizes;
  splitParallelAndReductionTiles(convOp, parallelTileSizes, reductionTileSizes);
  setAlwaysVectorizeSizes(convOp, parallelTileSizes, reductionTileSizes);

  TileSizesListType tileSizes;
  tileSizes.push_back(distTileSizes);
  tileSizes.push_back(parallelTileSizes);
  tileSizes.push_back(reductionTileSizes);
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
  OpBuilder builder(padOp.getContext());
  builder.setInsertionPoint(padOp);
  SmallVector<Range> iterationDomain =
      cast<TilingInterface>(padOp.getOperation()).getIterationDomain(builder);
  SmallVector<int64_t> lbs, ubs;
  getBoundsFromRange(iterationDomain, lbs, ubs);

  SmallVector<int64_t> minTileSizes(lbs.size(), 1);
  SmallVector<int64_t> maxTileSizes(ubs.size(), defaultDistTileSize);
  SmallVector<int64_t> vectorTileSizes(lbs.size(), 1);

  unsigned typeWidthInBytes = IREE::Util::getRoundedElementByteWidth(
      padOp.getResultType().getElementType());
  int64_t typeVectorSize = getVectorSize(entryPointFn, typeWidthInBytes);
  vectorTileSizes.back() = (ubs.back() == ShapedType::kDynamic
                                ? 1
                                : std::min(typeVectorSize, ubs.back()));
  minTileSizes.back() = vectorTileSizes.back();

  SmallVector<unsigned> partitionableLoops =
      cast<PartitionableLoopsInterface>(padOp.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  SmallVector<int64_t> distributedTileSizes =
      getDefaultDistributedLevelTileSizes(partitionableLoops, lbs, ubs,
                                          minTileSizes, maxTileSizes);
  TileSizesListType tileSizes;
  // Distribution tiling
  tileSizes.emplace_back(std::move(distributedTileSizes));
  // Tiling for vectorization.
  tileSizes.emplace_back(std::move(vectorTileSizes));
  // No further tiling.
  int64_t numTilingDims = vectorTileSizes.size();
  SmallVector<int64_t> zeros(numTilingDims, 0);
  tileSizes.push_back(zeros);
  tileSizes.push_back(zeros);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, padOp, tileSizes,
      DispatchLoweringPassPipeline::CPUDoubleTilingExpert);
}

/// Set default configuration for Linalg ops.
static LogicalResult
setRootConfig(func::FuncOp entryPointFn, linalg::LinalgOp linalgOp,
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
static LogicalResult
setRootConfig(func::FuncOp entryPointFn, TilingInterface tilingInterfaceOp,
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
    if (!intVal)
      return ShapedType::kDynamic;
    return intVal.value();
  };
  auto lbs = llvm::map_to_vector(
      iterationDomain, [&](Range r) { return getStaticValue(r.offset); });
  auto ubs = llvm::map_to_vector(
      iterationDomain, [&](Range r) { return getStaticValue(r.size); });
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      entryPointFn->getContext(), pipeline);
  if (failed(setTranslationInfo(entryPointFn, translationInfo))) {
    return failure();
  }
  return setDefaultRootConfig(entryPointFn, partitionableLoopOp, lbs, ubs);
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
              linalg::Mmt4DOp, linalg::BatchMmt4DOp>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp,
              linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
              linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
              linalg::PoolingNhwcMinUnsignedOp, linalg::PoolingNchwSumOp,
              linalg::PoolingNchwMaxOp, linalg::DepthwiseConv2DNhwcHwcOp>(
            [&](auto op) {
              return setConvInterfaceRootConfig(entryPointFn, op);
            })
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

static LogicalResult adjustTileSizesForPackOp(func::FuncOp entryPointFn,
                                              Operation *rootOp) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp)
    return success();

  auto loweringConfig = getLoweringConfig(linalgOp);
  TileSizesListType tileSizesList = loweringConfig.getTileSizeVals();

  bool hasChanged = false;
  auto res = entryPointFn.walk([&](tensor::PackOp packOp) -> WalkResult {
    // Multiple pack ops case is not supported.
    if (hasChanged)
      return WalkResult::interrupt();

    hasChanged = true;
    LLVM_DEBUG(KD_DBGS() << "Find pack op candidate: " << packOp << "\n");

    // Only adjust tile sizes for distribution and TileAndFuse, which are the
    // first two tile lists.
    // Align the tile sizes of the root op to the pack op's inner tile sizes, so
    // we can derive the outer tile sizes for pack ops later in
    // setLoweringConfigForComputeOps by dividing with inner tile sizes.
    for (int i = 0, e = std::min<int>(tileSizesList.size(), 2); i < e; ++i) {
      auto &tileSizes = tileSizesList[i];
      ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
      ArrayRef<int64_t> dimPos = packOp.getInnerDimsPos();
      for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
        if (tileSizes[pos] == 0 || ShapedType::isDynamic(size))
          continue;
        tileSizes[pos] = llvm::alignTo(tileSizes[pos], size);
        LLVM_DEBUG(KD_DBGS() << "Align # " << pos << " tile size to "
                             << tileSizes[pos] << "\n");
      }
    }

    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return failure();

  auto pipeline = getTranslationInfo(entryPointFn).getPassPipeline().getValue();
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, rootOp, tileSizesList,
      loweringConfig.getScalableTileFlagVals(), pipeline);
}

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
      auto dimExpr = idxMap.getResult(pos).dyn_cast<AffineDimExpr>();
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
/// for other compute ops. E.g., [[X, 0], [Y, 0], [0, 0], [0, 4]] for the
/// elementwise operations and [[X, 0], [Y, 0], [0, 0], [0, 16]] for the pack
/// op.
static void setLoweringConfigForComputeOps(func::FuncOp entryPointFn,
                                           ArrayRef<Operation *> computeOps,
                                           Operation *rootOperation) {
  if (isa<linalg::ConvolutionOpInterface>(rootOperation)) {
    // TODO(dcaballe): We don't know yet how to properly propagate the lowering
    // config of a convolution.
    return;
  }

  auto ctx = entryPointFn.getContext();
  TilingConfig tilingConfig(getLoweringConfig(rootOperation));
  SmallVector<int64_t> distTileSizes, tileAndFuseSizes;
  if (tilingConfig.getNumTilingLevels() > 0) {
    distTileSizes = tilingConfig.getDistributionTileSizes();
  }
  if (tilingConfig.getNumTilingLevels() > 1) {
    // TODO: Handle scalable tiles.
    std::tie(tileAndFuseSizes, std::ignore) =
        tilingConfig.getVectorCommonParallelSizes();
  }

  // Multi-lowering config works only if all the operations can share the same
  // distribution and TileAndFuse tile sizes.
  for (auto op : computeOps) {
    auto iterTypes = cast<TilingInterface>(op).getLoopIteratorTypes();
    for (auto [idx, iterType] : llvm::enumerate(iterTypes)) {
      if (idx >= tileAndFuseSizes.size())
        break;
      if (iterType == utils::IteratorType::parallel)
        continue;
      if (distTileSizes[idx] || tileAndFuseSizes[idx])
        return;
    }
  }

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(entryPointFn);
  auto targetMLTransInfo =
      TargetMLTransformInfo::getTargetMLTransformInfo(targetAttr);
  for (auto op : computeOps) {
    // The lowering config is already set on rootOperation, so we skip it.
    if (op == rootOperation)
      continue;

    int numLoops = cast<TilingInterface>(op).getLoopIteratorTypes().size();
    SmallVector<int64_t> zeros(numLoops, 0);
    TileSizesListType tileSizesList = {distTileSizes, tileAndFuseSizes};
    TypeSwitch<Operation *>(op)
        .Case<tensor::PackOp>([&](auto packOp) {
          ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
          ArrayRef<int64_t> dimPos = packOp.getInnerDimsPos();
          auto outerDimsPerm = packOp.getOuterDimsPerm();
          // Scale the outer dim tiles for pack op.
          for (int i = 0, e = std::min<int>(tileSizesList.size(), 2); i < e;
               ++i) {
            auto &tileSizes = tileSizesList[i];
            for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
              if (tileSizes[pos] == 0 || ShapedType::isDynamic(size))
                continue;
              tileSizes[pos] = tileSizes[pos] / size;
            }
            if (!outerDimsPerm.empty())
              applyPermutationToVector(tileSizes, outerDimsPerm);
          }

          SmallVector<int64_t> vecTileSizes =
              getPackVectorTileSizes(entryPointFn, packOp);
          // tensor.pack op does not have reduction loops.
          tileSizesList.push_back(zeros);
          tileSizesList.push_back(vecTileSizes);
        })
        .Case<linalg::GenericOp>([&](auto genericOp) {
          auto vecPreProcStrategy = getVectorPreProcStrategy(genericOp);
          auto linalgOpInfo = LinalgOpInfo(genericOp);
          int64_t vecSize = getNativeVectorSizeInBytes(entryPointFn) / 4;
          SmallVector<int64_t> vecTileSizes = getMinTilingSizesForEachDim(
              entryPointFn, genericOp, linalgOpInfo, targetMLTransInfo);
          for (auto &i : vecTileSizes) {
            i = roundUpToPow2(std::min(i, vecSize),
                              vecPreProcStrategy ==
                                  VectorPreProcStrategy::Masking);
          }

          // If the dimension is already tiled, we don't tile it again. This
          // prevents the mismatch common vector sizes between producer and
          // consumers. E.g., the convolution vectorization does not support
          // masking yet, while the strategy for generic op could use masking.
          // This introduces odd behavior like convolution takes 12 as tile size
          // while generic op takes 8 as tile size. It would introduce an
          // inefficient loop and only apply masking on generic op, which hurts
          // performance a lot. Thus, we do not tile it again, so they have
          // consistent vector tile sizes.
          for (auto i : llvm::seq<int64_t>(
                   0, std::min(tileAndFuseSizes.size(), vecTileSizes.size()))) {
            if (tileAndFuseSizes[i])
              vecTileSizes[i] = 0;
          }

          SmallVector<int64_t> reductionTiles;
          splitParallelAndReductionTiles(genericOp, vecTileSizes,
                                         reductionTiles);
          setVectorSizesForDynamicShapes(genericOp, vecPreProcStrategy,
                                         vecTileSizes, reductionTiles);

          tileSizesList.push_back(reductionTiles);
          tileSizesList.push_back(vecTileSizes);
        })
        .Default([&](auto) {
          // Do nothing for unknown ops.
          tileSizesList.push_back(zeros);
          tileSizesList.push_back(zeros);
        });
    for (auto &ts : tileSizesList)
      ts.resize(numLoops, 0);
    auto config = IREE::Codegen::LoweringConfigAttr::get(ctx, tileSizesList);
    setLoweringConfig(op, config);
  }
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

  // The transform dialect codegen has differnet logics and codegen flow. Ignore
  // the tile sizes adjustment.
  auto pipeline = getTranslationInfo(entryPointFn).getPassPipeline().getValue();
  if (pipeline != DispatchLoweringPassPipeline::TransformDialectCodegen) {
    if (failed(adjustTileSizesForUnPackOp(entryPointFn, rootOperation))) {
      return failure();
    }

    if (failed(adjustTileSizesForPackOp(entryPointFn, rootOperation))) {
      return failure();
    }

    // Set vector level tile sizes for other operations individually.
    setLoweringConfigForComputeOps(entryPointFn, computeOps, rootOperation);
  }

  return success();
}

LogicalResult initCPULaunchConfig(ModuleOp moduleOp) {
  ModuleOp transformModule =
      transform::detail::getPreloadedTransformModule(moduleOp.getContext());
  if (transformModule && clCPUEnableTransformDialectJit) {
    return moduleOp.emitError()
           << "option clash in transform dialect lowering config: a preloaded "
              "transform library cannot be provided when the jit option is "
              "set.";
  }

  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp)
      continue;
    if (getTranslationInfo(exportOp))
      continue;

    if (transformModule) {
      auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
          moduleOp.getContext(),
          IREE::Codegen::DispatchLoweringPassPipeline::TransformDialectCodegen);
      if (failed(setTranslationInfo(funcOp, translationInfo)))
        return failure();
      continue;
    }

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

  // The root configuration setting introduces `tensor.dim` operations. Resolve
  // those away.
  RewritePatternSet patterns(moduleOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

} // namespace iree_compiler
} // namespace mlir
