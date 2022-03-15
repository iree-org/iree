// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"

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
    llvm::cl::desc("linalg.mmt4d workgroup tile size"), llvm::cl::ZeroOrMore,
    llvm::cl::MiscFlags::CommaSeparated);

static llvm::cl::list<int> mmt4dL1TileSizes(
    "iree-codegen-llvm-mmt4d-l1-tile-size",
    llvm::cl::desc("linalg.mmt4d L1 tile size"), llvm::cl::ZeroOrMore,
    llvm::cl::MiscFlags::CommaSeparated);

static llvm::cl::list<int> mmt4dVectorSizes(
    "iree-codegen-llvm-mmt4d-vector-size",
    llvm::cl::desc("linalg.mmt4d vector tile size"), llvm::cl::ZeroOrMore,
    llvm::cl::MiscFlags::CommaSeparated);

static llvm::cl::opt<int> defaultWorkgroupTileSize(
    "iree-codegen-llvm-generic-ops-workgroup-size",
    llvm::cl::desc(
        "linalg.generic and linalg.indexed_generic workgroup tile size"),
    llvm::cl::init(64));

using IREE::Codegen::DispatchLoweringPassPipeline;

/// Looks for the `native_vector_size` attribute in the hal.executable.variant
/// op.
static Optional<int64_t> getNativeVectorSizeInBytes(FuncOp entryPointFn) {
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
static int64_t getVectorSize(FuncOp entryPointFn, unsigned byteWidth) {
  if (Optional<int64_t> nativeVectorSize =
          getNativeVectorSizeInBytes(entryPointFn)) {
    return nativeVectorSize.getValue() / byteWidth;
  }
  return clNativeVectorSizeInBytes / byteWidth;
}
static int64_t getVectorSize(FuncOp entryPointFn, ShapedType shapedType) {
  Type elementType = shapedType.getElementType();
  if (!elementType.isIntOrFloat()) return 1;
  unsigned byteWidth = IREE::Util::getRoundedElementByteWidth(elementType);
  return getVectorSize(entryPointFn, byteWidth);
}

/// Returns minimum tiling sizes for each dimension. One dimension is possible
/// to access at different element types. It determines the tiling sizes by
/// looking into all the operands.
static SmallVector<int64_t> getMinTilingSizesForEachDim(FuncOp entryPointFn,
                                                        linalg::LinalgOp op) {
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
static unsigned getReferenceTypeLengthInBytes(FuncOp entryPointFn) {
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
    ArrayRef<int64_t> minTileSizes, ArrayRef<int64_t> maxTileSizes) {
  assert(lbs.size() == ubs.size() && lbs.size() == minTileSizes.size() &&
         lbs.size() == maxTileSizes.size() &&
         "expected all vectors to be of equal size");
  if (lbs.empty()) {
    return {};
  }
  size_t numDims = lbs.size();
  SmallVector<int64_t> distributedTileSizes(numDims, 1);
  SmallVector<int64_t> numWorkgroupsPerDim(numDims, 1);
  SmallVector<int64_t> workload(numDims, 1);
  auto ceilFn = [](int64_t a, int64_t b) { return (a + b - 1) / b; };

  for (auto i : llvm::seq<size_t>(0, numDims)) {
    if (ShapedType::isDynamic(lbs[i]) || ShapedType::isDynamic(ubs[i])) {
      distributedTileSizes[i] = maxTileSizes[i];
      workload[i] = ShapedType::kDynamicSize;
      continue;
    }
    int64_t candidateTileSize = 1;
    if (ubs[i] > lbs[i]) {
      // Pick a value that evenly distributes the workload.
      candidateTileSize = std::max<int64_t>(
          llvm::PowerOf2Floor(static_cast<uint64_t>(ubs[i] - lbs[i]) / 2),
          minTileSizes[i]);
    }

    // Limit the workload per workgroup to the default being the max to keep the
    // work per invocation reasonable.
    distributedTileSizes[i] =
        std::min<int64_t>(candidateTileSize, maxTileSizes[i]);
    workload[i] = (ubs[i] <= lbs[i] ? 1 : ubs[i] - lbs[i]);
    numWorkgroupsPerDim[i] = ceilFn(workload[i], distributedTileSizes[i]);
  }

  // Reduce the number of workgroups in cases where we are dividing the work too
  // much. Over-provision the number of workgroups to twice the number of
  // threads.
  int64_t numWorkgroupsLimit = 2 * clNumberOfRuntimeThreads;
  int64_t numWorkgroups = 1;
  for (auto ng : numWorkgroupsPerDim) {
    numWorkgroups *= ng;
  }
  unsigned currDim = numDims;
  while (numWorkgroups > numWorkgroupsLimit && currDim > 0) {
    if (distributedTileSizes[currDim - 1] >= maxTileSizes[currDim - 1] ||
        workload[currDim - 1] == ShapedType::kDynamicSize ||
        distributedTileSizes[currDim - 1] >= workload[currDim - 1]) {
      currDim--;
      continue;
    }
    distributedTileSizes[currDim - 1] = std::min<int64_t>(
        distributedTileSizes[currDim - 1] * 2, maxTileSizes[currDim - 1]);
    int64_t nwg =
        ceilFn(workload[currDim - 1], distributedTileSizes[currDim - 1]);
    if (nwg < numWorkgroupsPerDim[currDim - 1]) {
      numWorkgroups /= numWorkgroupsPerDim[currDim - 1];
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

/// Returns the tile size to use for the Flow level of an operation that
/// implements the `PartitionableLoopsInterface`.
static SmallVector<int64_t> getDefaultDistributedLevelTileSizes(
    ArrayRef<Range> iterationDomain,
    IREE::Flow::PartitionableLoopsInterface partitionableLoopInterfaceOp,
    ArrayRef<int64_t> minTileSizes, ArrayRef<int64_t> maxTileSizes) {
  assert(iterationDomain.size() == minTileSizes.size() &&
         "expected as many min tile sizes as number of loops");
  auto getStaticValue = [](Value v) -> int64_t {
    IntegerAttr attr;
    if (!matchPattern(v, m_Constant(&attr))) return ShapedType::kDynamicSize;
    return attr.getInt();
  };
  auto lbs = llvm::to_vector(llvm::map_range(
      iterationDomain, [&](Range r) { return getStaticValue(r.offset); }));
  auto ubs = llvm::to_vector(llvm::map_range(
      iterationDomain, [&](Range r) { return getStaticValue(r.size); }));

  SmallVector<unsigned> partitionableLoops =
      partitionableLoopInterfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  llvm::SmallDenseSet<unsigned, 4> partitionableLoopsSet;
  partitionableLoopsSet.insert(partitionableLoops.begin(),
                               partitionableLoops.end());

  size_t numPartitionedLoops = partitionableLoops.size();
  SmallVector<int64_t> distributedLoopLbs(numPartitionedLoops,
                                          ShapedType::kDynamicSize),
      distributedLoopUbs(numPartitionedLoops, ShapedType::kDynamicSize),
      minDistributedLoopTileSizes(numPartitionedLoops, 1),
      maxDistributedLoopTileSizes(numPartitionedLoops,
                                  defaultWorkgroupTileSize);
  // Find the bounds of the partitionable loops
  unsigned index = 0;
  for (auto range : llvm::enumerate(iterationDomain)) {
    if (!partitionableLoopsSet.count(range.index())) continue;

    minDistributedLoopTileSizes[index] = minTileSizes[range.index()];
    maxDistributedLoopTileSizes[index] = maxTileSizes[range.index()];
    distributedLoopLbs[index] = lbs[range.index()];
    distributedLoopUbs[index] = ubs[range.index()];
    index++;
  }

  SmallVector<int64_t> distributedTileSizes =
      getDefaultDistributedLoopTileSizes(distributedLoopLbs, distributedLoopUbs,
                                         minDistributedLoopTileSizes,
                                         maxDistributedLoopTileSizes);
  SmallVector<int64_t> distributedLevelTileSizes(iterationDomain.size(), 0);
  for (auto loopID : llvm::enumerate(partitionableLoops)) {
    distributedLevelTileSizes[loopID.value()] =
        distributedTileSizes[loopID.index()];
  }
  // Final fix up of the tile sizes to make sure that they divide the problem
  // size to make it vectorizable.
  for (auto i : llvm::seq<unsigned>(0, distributedLevelTileSizes.size())) {
    distributedLevelTileSizes[i] =
        distributedLevelTileSizes[i] != 0
            ? getMaxTileSize(lbs[i], ubs[i], distributedLevelTileSizes[i],
                             minTileSizes[i])
            : 0;
  }
  return distributedLevelTileSizes;
}

/// Splits the tile sizes in parallelSizes into reductionSizes for the reduction
/// loops.
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

/// Sets the default configuration to use for an operation that implements the
/// `PartitionableLoopsInterface`, given the iteration domain of all the loops.
static LogicalResult setDefaultRootConfig(
    FuncOp entryPointFn,
    IREE::Flow::PartitionableLoopsInterface partitionableLoopsInterfaceOp,
    ArrayRef<Range> iterationDomain) {
  if (getLoweringConfig(partitionableLoopsInterfaceOp)) return success();

  SmallVector<unsigned> partitionableLoops =
      partitionableLoopsInterfaceOp.getPartitionableLoops(kNumMaxParallelDims);

  SmallVector<int64_t> minTileSizes(iterationDomain.size(), 1);
  SmallVector<int64_t> maxTileSizes(iterationDomain.size(), 1);
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
      iterationDomain, partitionableLoopsInterfaceOp, minTileSizes,
      maxTileSizes);
  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(flowTileSizes));
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, partitionableLoopsInterfaceOp, tileSizes,
      DispatchLoweringPassPipeline::CPUDefault);
}

static LogicalResult setX86SandboxRootConfig(FuncOp entryPointFn,
                                             linalg::ContractionOpInterface op,
                                             ArrayRef<int64_t> flowTileSizes,
                                             int vectorSize) {
  // Hardcoded tiling sizes {1, 1, ..., 8, 32, 16}.
  // The tiling for parallel dims and reduction dims should be separated.
  SmallVector<int64_t> l1TileSizes;
  int64_t nLoops = cast<linalg::LinalgOp>(op.getOperation()).getNumLoops();
  l1TileSizes.append(nLoops - 3, 1);
  l1TileSizes.push_back(
      getMaxTileSize(0, flowTileSizes[nLoops - 3], 8, vectorSize));
  l1TileSizes.push_back(
      getMaxTileSize(0, flowTileSizes[nLoops - 2], 32, vectorSize));
  l1TileSizes.push_back(0);

  auto lhsShapedType = op.lhs().getType().cast<ShapedType>();
  int64_t K = lhsShapedType.getShape().back();
  SmallVector<int64_t> vectorTileSizes;
  vectorTileSizes.append(nLoops - 1, 0);
  vectorTileSizes.push_back(getMaxTileSize(0, K, 16, vectorSize));

  TileSizesListType tileSizes;
  tileSizes.emplace_back(flowTileSizes.begin(), flowTileSizes.end());
  tileSizes.push_back(l1TileSizes);
  tileSizes.push_back(vectorTileSizes);

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes,
      DispatchLoweringPassPipeline::CPUDoubleTilingExpert);
}

static LogicalResult setARMRootConfig(FuncOp entryPointFn,
                                      linalg::ContractionOpInterface op,
                                      ArrayRef<int64_t> flowTileSizes,
                                      int vectorSize) {
  // Hardcoded tile sizes, where v is the native vector size.
  // L1 tile sizes are {1, ..., 5v, v, 16v}.
  // Vector tile sizes are {1, ..., v, v, v}
  SmallVector<int64_t> l1TileSizes, vectorTileSizes;
  int64_t nLoops = cast<linalg::LinalgOp>(op.getOperation()).getNumLoops();
  l1TileSizes.append(nLoops - 3, 1);
  l1TileSizes.push_back(
      getMaxTileSize(0, flowTileSizes[nLoops - 3], 5 * vectorSize, vectorSize));
  l1TileSizes.push_back(
      getMaxTileSize(0, flowTileSizes[nLoops - 2], vectorSize, vectorSize));
  vectorTileSizes.append(nLoops - 3, 1);
  vectorTileSizes.push_back(vectorSize);
  vectorTileSizes.push_back(vectorSize);

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
    FuncOp entryPointFn, linalg::ContractionOpInterface contractionOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  auto linalgOp = cast<linalg::LinalgOp>(contractionOp.getOperation());
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
  unsigned numLoops = linalgOp.getNumLoops();
  SmallVector<int64_t> minTileSizes =
      getMinTilingSizesForEachDim(entryPointFn, linalgOp);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultWorkgroupTileSize);
  if (numLoops > 3) {
    minTileSizes[0] = 1;
    maxTileSizes[0] = 1;
  }

  OpBuilder builder(entryPointFn.getContext());
  builder.setInsertionPoint(contractionOp);
  SmallVector<Range> iterationDomain =
      linalgOp.createLoopRanges(builder, linalgOp->getLoc());
  SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
      iterationDomain,
      cast<IREE::Flow::PartitionableLoopsInterface>(
          contractionOp.getOperation()),
      minTileSizes, maxTileSizes);

  // TODO(dcaballe): Find better configurations for RISC-V backends.
  if (isX86(entryPointFn) || isRISCV(entryPointFn)) {
    // There is a tileInterchange option. If it needs to be configured, we can
    // only apply the pipeline to linalg.matmul. Because we don't know the
    // number of loops when adding the pass to pass manager.
    // TODO(hanchung): Embed options into attributes, so we can control options
    // more heuristically.
    return setX86SandboxRootConfig(entryPointFn, contractionOp, flowTileSizes,
                                   vectorSize);
  }

  // Fall back to ARM configurations.
  return setARMRootConfig(entryPointFn, contractionOp, flowTileSizes,
                          vectorSize);
}

/// Sets the lowering configuration for dispatch region for linalg.mmt4d root
/// op
static LogicalResult setRootConfig(
    FuncOp entryPointFn, linalg::Mmt4DOp mmt4dOp,
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
    FuncOp entryPointFn, IREE::LinalgExt::FftOp fftOp,
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

/// Sets the lowering configuration for a generic op to use DoubleTilingExpert.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, linalg::GenericOp genericOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
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
  OpBuilder builder(genericOp.getContext());
  builder.setInsertionPoint(genericOp);
  auto linalgOp = cast<linalg::LinalgOp>(genericOp.getOperation());
  SmallVector<Range> iterationDomain =
      linalgOp.createLoopRanges(builder, genericOp.getLoc());
  auto partitionableLoopsInterfaceOp =
      cast<IREE::Flow::PartitionableLoopsInterface>(genericOp.getOperation());
  SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
      iterationDomain, partitionableLoopsInterfaceOp, minTileSizes,
      maxTileSizes);

  // Set the Next level tile sizes.
  SmallVector<int64_t> l1TileSizes(numLoops, 0);
  Optional<SmallVector<int64_t, 4>> staticLoopRanges =
      linalgOp.getStaticLoopRanges();
  for (auto loopNum : llvm::seq<unsigned>(0, numLoops)) {
    if (flowTileSizes[loopNum]) {
      l1TileSizes[loopNum] =
          getMaxTileSize(0, flowTileSizes[loopNum], minTileSizes[loopNum],
                         minTileSizes[loopNum]);
    } else {
      // If the flow level tile size is zero, and static loop range is 0 as
      // well, set the tile sizes here to zero as well.
      l1TileSizes[loopNum] =
          (staticLoopRanges && staticLoopRanges.getValue()[loopNum] == 1)
              ? 0
              : minTileSizes[loopNum];
    }
  }
  SmallVector<int64_t> vectorTileSizes;
  splitParallelAndReductionTiles(linalgOp, l1TileSizes, vectorTileSizes);

  TileSizesListType tileSizes;
  tileSizes.push_back(flowTileSizes);
  tileSizes.push_back(l1TileSizes);
  tileSizes.push_back(vectorTileSizes);
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, genericOp, tileSizes,
      DispatchLoweringPassPipeline::CPUDoubleTilingExpert);
}

/// Sets the lowering configuration for linalg.conv_2d_nhwc_hwcf and
/// linalg.depthwise_conv_2d_nhwc_hwc operations.
static LogicalResult setConvRootConfig(
    FuncOp entryPointFn, linalg::LinalgOp convOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops,
    ArrayRef<int64_t> targetL1TileSizes, int64_t vectorSize) {
  if (!isa<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(
          convOp.getOperation())) {
    return failure();
  }

  // Use the default distribution for the matmul loops.
  unsigned numLoops = convOp.getNumLoops();
  SmallVector<int64_t> minTileSizes(numLoops, 1);
  SmallVector<int64_t> maxTileSizes(numLoops, defaultWorkgroupTileSize);

  // Set the flow level tiling to the default.
  OpBuilder builder(convOp.getContext());
  builder.setInsertionPoint(convOp);
  SmallVector<Range> iterationDomain =
      convOp.createLoopRanges(builder, convOp.getLoc());
  auto partitionableLoopsInterfaceOp =
      cast<IREE::Flow::PartitionableLoopsInterface>(convOp.getOperation());
  SmallVector<int64_t> flowTileSizes = getDefaultDistributedLevelTileSizes(
      iterationDomain, partitionableLoopsInterfaceOp, minTileSizes,
      maxTileSizes);

  // Shapes of N, OH, OW, OC, KH, KW, (IC)
  Optional<SmallVector<int64_t, 4>> shapes = convOp.getStaticLoopRanges();
  SmallVector<int64_t> l1TileSizes(targetL1TileSizes.begin(),
                                   targetL1TileSizes.end());
  for (auto i : llvm::seq<unsigned>(0, l1TileSizes.size())) {
    auto tileSize = flowTileSizes[i] ? flowTileSizes[i] : shapes.getValue()[i];
    // If the tile size is intended to be 1, do not adjust it to `vectorSize`.
    // The ops will be decomposed to lower-rank named ops.
    if (l1TileSizes[i] != 1) {
      l1TileSizes[i] = getMaxTileSize(0, tileSize, l1TileSizes[i], vectorSize);
    }
  }
  SmallVector<int64_t> vectorTileSizes;
  splitParallelAndReductionTiles(convOp, l1TileSizes, vectorTileSizes);

  TileSizesListType tileSizes;
  tileSizes.push_back(flowTileSizes);
  tileSizes.push_back(l1TileSizes);
  tileSizes.push_back(vectorTileSizes);
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, convOp, tileSizes,
      DispatchLoweringPassPipeline::CPUConvTileAndDecomposeExpert);
}

static LogicalResult setRootConfig(
    FuncOp entryPointFn, linalg::Conv2DNhwcHwcfOp convOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  auto linalgOp = cast<linalg::LinalgOp>(convOp.getOperation());
  int64_t vectorSize =
      getVectorSize(entryPointFn, convOp.getResult(0).getType());
  SmallVector<int64_t> l1TileSizes = {1, 1, 8, vectorSize * 2, 1, 1, 8};
  return setConvRootConfig(entryPointFn, linalgOp, tiledLoops, l1TileSizes,
                           vectorSize);
}

/// Sets the lowering configuration for linalg.depthwise_conv_2d_nhwc_hwc
/// operations.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, linalg::DepthwiseConv2DNhwcHwcOp convOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  auto linalgOp = cast<linalg::LinalgOp>(convOp.getOperation());
  int64_t vectorSize =
      getVectorSize(entryPointFn, convOp.getResult(0).getType());
  SmallVector<int64_t> l1TileSizes = {1, 1, 8, vectorSize * 2, 1, 3};
  return setConvRootConfig(entryPointFn, linalgOp, tiledLoops, l1TileSizes,
                           vectorSize);
}

/// Set default configuration for Linalg ops.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, linalg::LinalgOp linalgOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  if (getLoweringConfig(linalgOp)) return success();

  OpBuilder builder(linalgOp.getContext());
  builder.setInsertionPoint(linalgOp);
  SmallVector<Range> iterationDomain =
      linalgOp.createLoopRanges(builder, linalgOp.getLoc());

  auto partitionableLoopOp =
      cast<IREE::Flow::PartitionableLoopsInterface>(linalgOp.getOperation());
  return setDefaultRootConfig(entryPointFn, partitionableLoopOp,
                              iterationDomain);
}

/// Set the default configuration for operations that implement the
/// `TiledOpInterface`.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, IREE::LinalgExt::TiledOpInterface tiledOpInterfaceOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  if (getLoweringConfig(tiledOpInterfaceOp)) return success();

  OpBuilder builder(tiledOpInterfaceOp.getContext());
  builder.setInsertionPoint(tiledOpInterfaceOp);
  SmallVector<Range> iterationDomain =
      tiledOpInterfaceOp.getIterationDomain(builder);
  auto partitionableLoopInterfaceOp =
      cast<IREE::Flow::PartitionableLoopsInterface>(
          tiledOpInterfaceOp.getOperation());
  return setDefaultRootConfig(entryPointFn, partitionableLoopInterfaceOp,
                              iterationDomain);
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult setRootConfigImpl(
    FuncOp entryPointFn, Operation *op,
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
    FuncOp entryPointFn, Operation *op,
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
  for (auto op : computeOps) {
    if (isa<linalg::GenericOp>(op)) {
      if (failed(updateRootOperation(op))) return failure();
    }
  }
  if (rootOperation) return rootOperation;

  // TODO(ravishankarm): Currently there is a corner case of a dispatch region
  // with just a `tensor.extract_slice`/`tensor.insert_slice`. Those need to be
  // folded with `flow.dispatch.tensor.load`/`flow.dispatch.tensor.store` ops
  // respectively. This should go hand-in-hand with dropping the external model
  // implementation of the `TiledOpInterface` for these ops. Till we cross that
  // bridge, handle that case.
  // Throw in linalg.fill here as well, though that should never happen either.
  if (computeOps.size() == 1 &&
      isa<linalg::FillOp, tensor::ExtractSliceOp, tensor::InsertSliceOp>(
          computeOps[0])) {
    rootOperation = computeOps[0];
  }
  return rootOperation;
}

/// Finds the root operation in the given list of Linalg operations and sets
/// its configuration. Returns error for multiple root operations.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, ArrayRef<Operation *> computeOps,
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
    FuncOp entryPointFn, ArrayRef<Operation *> computeOps,
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
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    auto entryPointOp = entryPointOps.lookup(funcOp.getName());
    if (!entryPointOp) continue;
    if (getTranslationInfo(entryPointOp)) continue;
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
