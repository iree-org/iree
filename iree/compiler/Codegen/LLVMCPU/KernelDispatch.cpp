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
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

// TODO(hanchung): Enable the flag by default after addressing perf
// regresssions.
static llvm::cl::opt<bool> useDoubleTilingExpert(
    "iree-codegen-use-double-tiling-expert",
    llvm::cl::desc("DEVELOPMENT ONLY, DO NOT USE THE FLAG."),
    llvm::cl::init(false));

using IREE::Codegen::DispatchLoweringPassPipeline;

static bool isVMVX(FuncOp entryPointFn) {
  auto variantOp =
      entryPointFn->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.target();
  return targetAttr && targetAttr.getBackend().getValue() == "vmvx";
}

static Optional<llvm::Triple> getTargetTriple(FuncOp entryPointFn) {
  auto variantOp =
      entryPointFn->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.target();
  if (!targetAttr) return llvm::None;
  auto config = targetAttr.getConfiguration();
  if (!config) return llvm::None;
  auto triple = config.getAs<StringAttr>("target_triple");
  if (!triple) return llvm::None;
  return llvm::Triple(triple.getValue().str());
}

static DispatchLoweringPassPipeline getDispatchLoweringPassPipeline(
    FuncOp entryPointFn, Operation *op) {
  return TypeSwitch<Operation *, DispatchLoweringPassPipeline>(op)
      .Case<linalg::ContractionOpInterface, linalg::Mmt4DOp>([&](auto op) {
        return DispatchLoweringPassPipeline::CPUTileFuseAndVectorize;
      })
      .Default([&](Operation *op) {
        return DispatchLoweringPassPipeline::CPUDefault;
      });
}

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
  unsigned byteWidth =
      std::max<unsigned>(1, elementType.getIntOrFloatBitWidth() / 8);
  return getVectorSize(entryPointFn, byteWidth);
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
        std::max<unsigned>(elementType.getIntOrFloatBitWidth() / 8, 1);
    referenceTypeLengthInBytes =
        std::min<unsigned>(referenceTypeLengthInBytes, typeWidthInBytes);
  });
  return referenceTypeLengthInBytes;
}

static SmallVector<int64_t> getDefaultWorkloadPerWorkgroup(
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops,
    ArrayRef<int64_t> nativeVectorSizeInElements) {
  if (tiledLoops.empty()) {
    return {};
  }
  assert(tiledLoops.size() == nativeVectorSizeInElements.size());
  unsigned maxDim = 0;
  for (auto tiledLoop : tiledLoops) {
    maxDim = std::max<unsigned>(tiledLoop.processorDistributionDim, maxDim);
  }
  SmallVector<int64_t> workloadPerWorkgroup(maxDim + 1, 1);
  SmallVector<int64_t> numWorkgroupsPerDim(maxDim + 1, 1);
  SmallVector<int64_t> workload(maxDim + 1, 1);
  auto getStaticValue = [](OpFoldResult ofr) -> Optional<int64_t> {
    return (ofr ? getConstantIntValue(ofr) : llvm::None);
  };
  auto ceilFn = [](int64_t a, int64_t b) { return (a + b - 1) / b; };

  for (auto tiledLoop : enumerate(tiledLoops)) {
    Optional<int64_t> lb = getStaticValue(tiledLoop.value().untiledLowerBound);
    Optional<int64_t> ub = getStaticValue(tiledLoop.value().untiledUpperBound);
    unsigned dim = tiledLoop.value().processorDistributionDim;
    if (!lb || !ub) {
      workloadPerWorkgroup[dim] = defaultWorkgroupTileSize;
      workload[dim] = ShapedType::kDynamicSize;
      continue;
    }
    int64_t candidateTileSize = nativeVectorSizeInElements[tiledLoop.index()];
    if (*ub <= *lb) {
      // Should be avoiding tiling this loop, but use tile size of 1.
      candidateTileSize = 1;
    } else {
      // Pick a value that evenly distributes the workload.
      candidateTileSize = std::max<int64_t>(
          llvm::PowerOf2Floor(static_cast<uint64_t>(*ub - *lb) / 2),
          candidateTileSize);
    }

    // Limit the workload per workgroup to the default being the max to keep the
    // work per invocation reasonable.
    workloadPerWorkgroup[dim] =
        std::min<int64_t>(candidateTileSize, defaultWorkgroupTileSize);
    workload[dim] = (*ub <= *lb ? 1 : *ub - *lb);
    numWorkgroupsPerDim[dim] = ceilFn(workload[dim], workloadPerWorkgroup[dim]);
  }

  // Reduce the number of workgroups in cases where we are dividing the work too
  // much. Over-provision the number of workgroups to twice the number of
  // threads.
  int64_t numWorkgroupsLimit = 2 * clNumberOfRuntimeThreads;
  int64_t numWorkgroups = 1;
  for (auto ng : numWorkgroupsPerDim) {
    numWorkgroups *= ng;
  }
  unsigned currDim = 0;
  while (numWorkgroups > numWorkgroupsLimit &&
         currDim < numWorkgroupsPerDim.size()) {
    if (workloadPerWorkgroup[currDim] >= defaultWorkgroupTileSize ||
        workload[currDim] == ShapedType::kDynamicSize ||
        workloadPerWorkgroup[currDim] >= workload[currDim]) {
      currDim++;
      continue;
    }
    workloadPerWorkgroup[currDim] = std::min<int64_t>(
        workloadPerWorkgroup[currDim] * 2, defaultWorkgroupTileSize);
    int64_t nwg = ceilFn(workload[currDim], workloadPerWorkgroup[currDim]);
    if (nwg < numWorkgroupsPerDim[currDim]) {
      numWorkgroups /= numWorkgroupsPerDim[currDim];
      numWorkgroups *= nwg;
    } else {
      currDim++;
    }
  }
  return workloadPerWorkgroup;
}

/// Sets the default launch configuration to use for a tiled + distributed
/// dispatch region based on the `tiledLoops` found.
static LogicalResult setDefaultLaunchConfig(
    FuncOp entryPointFn, ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  SmallVector<int64_t> nativeVectorSizeInElements(tiledLoops.size(), 1);
  if (!tiledLoops.empty()) {
    unsigned typeWidthInBytes = getReferenceTypeLengthInBytes(entryPointFn);
    nativeVectorSizeInElements.back() =
        getVectorSize(entryPointFn, typeWidthInBytes);
  }

  SmallVector<int64_t> workloadPerWorkgroup =
      getDefaultWorkloadPerWorkgroup(tiledLoops, nativeVectorSizeInElements);

  setTranslationInfo(entryPointFn, DispatchLoweringPassPipeline::CPUDefault,
                     workloadPerWorkgroup,
                     /*workgroupSize =*/ArrayRef<int64_t>{});
  return success();
}

/// Adjusts the workload per workgroup to be a multiple of vector size to ensure
/// that the op vectorizes.
static int64_t getMaxTileSize(int64_t lb, int64_t ub, int64_t maxSize,
                              int64_t vectorSizeVal) {
  if (ub == ShapedType::kDynamicSize || lb == ShapedType::kDynamicSize) {
    return maxSize;
  }
  int64_t dim = ub - lb;
  if (dim < vectorSizeVal) return vectorSizeVal;
  for (int64_t i = std::min(maxSize, dim); i > 0; --i) {
    if (dim % i == 0 && i % vectorSizeVal == 0) {
      return i;
    }
  }
  return maxSize;
}

static LogicalResult setX86RootConfig(FuncOp entryPointFn,
                                      linalg::ContractionOpInterface op,
                                      SmallVector<int64_t> workloadPerWorkgroup,
                                      int vectorSize) {
  setTranslationInfo(entryPointFn,
                     getDispatchLoweringPassPipeline(entryPointFn, op),
                     workloadPerWorkgroup,
                     /*workgroupSize=*/ArrayRef<int64_t>{});

  // Hardcoded tile sizes, where v is the native vector size.
  // L1 tile sizes are {1, 1, ..., 8, 2v, 2v}.
  // Vector tile sizes are {1, ..., 1, v, v}
  SmallVector<int64_t> l1TileSizes, vectorTileSizes;
  int64_t nLoops = cast<linalg::LinalgOp>(op.getOperation()).getNumLoops();
  l1TileSizes.append(nLoops - 3, 1);
  l1TileSizes.push_back(
      getMaxTileSize(0, workloadPerWorkgroup[1], 8, vectorSize));
  l1TileSizes.push_back(
      getMaxTileSize(0, workloadPerWorkgroup[0], 2 * vectorSize, vectorSize));
  vectorTileSizes.append(nLoops - 2, 1);
  vectorTileSizes.push_back(vectorSize);

  // L1/vector tile size for k dimensions.
  auto lhsShapedType = op.lhs().getType().cast<ShapedType>();
  int64_t K = lhsShapedType.getShape().back();
  l1TileSizes.push_back(getMaxTileSize(0, K, 2 * vectorSize, vectorSize));
  vectorTileSizes.push_back(vectorSize);
  TileSizesListType tileSizes;
  tileSizes.push_back({});  // Empty here since there is nothing to do in first
                            // level tiling.
  tileSizes.push_back(l1TileSizes);
  tileSizes.push_back(vectorTileSizes);
  auto config = IREE::Codegen::LoweringConfigAttr::get(
      entryPointFn.getContext(), tileSizes, vectorTileSizes);
  setLoweringConfig(op, config);

  return success();
}

static LogicalResult setX86SandboxRootConfig(
    FuncOp entryPointFn, linalg::ContractionOpInterface op,
    SmallVector<int64_t> workloadPerWorkgroup, int vectorSize) {
  setTranslationInfo(entryPointFn,
                     DispatchLoweringPassPipeline::CPUDoubleTilingExpert,
                     workloadPerWorkgroup,
                     /*workgroupSize=*/ArrayRef<int64_t>{});

  // Hardcoded tile sizes. The configuration is derived from iree-llvm-sandbox.
  // L1 tile sizes are {1, 1, ..., 288, 128, 512}.
  // Vector tile sizes are {1, ..., 9, 32, 16}
  SmallVector<int64_t> l1TileSizes, vectorTileSizes;
  int64_t nLoops = cast<linalg::LinalgOp>(op.getOperation()).getNumLoops();
  l1TileSizes.append(nLoops - 3, 1);
  l1TileSizes.push_back(288);
  l1TileSizes.push_back(128);
  l1TileSizes.push_back(512);
  vectorTileSizes.append(nLoops - 3, 1);
  vectorTileSizes.push_back(9);
  vectorTileSizes.push_back(32);
  vectorTileSizes.push_back(16);

  TileSizesListType tileSizes;
  tileSizes.push_back({});
  tileSizes.push_back(l1TileSizes);
  tileSizes.push_back(vectorTileSizes);
  auto config = IREE::Codegen::LoweringConfigAttr::get(
      entryPointFn.getContext(), tileSizes, vectorTileSizes);
  setLoweringConfig(op, config);

  return success();
}

static LogicalResult setARMRootConfig(FuncOp entryPointFn,
                                      linalg::ContractionOpInterface op,
                                      SmallVector<int64_t> workloadPerWorkgroup,
                                      int vectorSize) {
  setTranslationInfo(entryPointFn,
                     getDispatchLoweringPassPipeline(entryPointFn, op),
                     workloadPerWorkgroup,
                     /*workgroupSize=*/ArrayRef<int64_t>{});

  // Hardcoded tile sizes, where v is the native vector size.
  // L1 tile sizes are {1, ..., 5v, v, 16v}.
  // Vector tile sizes are {1, ..., v, v, v}
  SmallVector<int64_t> l1TileSizes, vectorTileSizes;
  int64_t nLoops = cast<linalg::LinalgOp>(op.getOperation()).getNumLoops();
  l1TileSizes.append(nLoops - 3, 1);
  l1TileSizes.push_back(
      getMaxTileSize(0, workloadPerWorkgroup[1], 5 * vectorSize, vectorSize));
  l1TileSizes.push_back(
      getMaxTileSize(0, workloadPerWorkgroup[0], vectorSize, vectorSize));
  vectorTileSizes.append(nLoops - 3, 1);
  vectorTileSizes.push_back(vectorSize);
  vectorTileSizes.push_back(vectorSize);

  // L1/vector tile size for k dimensions.
  auto lhsShapedType = op.lhs().getType().cast<ShapedType>();
  int64_t K = lhsShapedType.getShape().back();
  l1TileSizes.push_back(getMaxTileSize(0, K, 16 * vectorSize, vectorSize));
  vectorTileSizes.push_back(vectorSize);
  TileSizesListType tileSizes;
  tileSizes.push_back({});  // Empty here since there is nothing to do in first
                            // level tiling.
  tileSizes.push_back(l1TileSizes);
  tileSizes.push_back(vectorTileSizes);
  auto config = IREE::Codegen::LoweringConfigAttr::get(
      entryPointFn.getContext(), tileSizes, vectorTileSizes);
  setLoweringConfig(op, config);

  return success();
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, linalg::ContractionOpInterface contractionOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  auto lhsShapedType = contractionOp.lhs().getType().cast<ShapedType>();
  // Use the default distribution for the matmul loops.
  int numBatchDims =
      cast<linalg::LinalgOp>(contractionOp.getOperation()).getNumLoops() - 3;

  int64_t vectorSize = getVectorSize(entryPointFn, lhsShapedType);
  SmallVector<int64_t> vectorSizeVals(tiledLoops.size(), 1);
  vectorSizeVals.back() = vectorSize;
  vectorSizeVals[vectorSizeVals.size() - 2] = vectorSize;

  SmallVector<int64_t> workloadPerWorkgroup = getDefaultWorkloadPerWorkgroup(
      tiledLoops.drop_front(numBatchDims),
      ArrayRef<int64_t>(vectorSizeVals).drop_front(numBatchDims));

  for (unsigned i = tiledLoops.size() - 2; i < tiledLoops.size(); ++i) {
    if (!tiledLoops[i].untiledLowerBound.is<Attribute>() ||
        !tiledLoops[i].untiledUpperBound.is<Attribute>()) {
      continue;
    }
    auto lb =
        tiledLoops[i].untiledLowerBound.get<Attribute>().cast<IntegerAttr>();
    auto ub =
        tiledLoops[i].untiledUpperBound.get<Attribute>().cast<IntegerAttr>();
    workloadPerWorkgroup[tiledLoops.size() - 1 - i] = getMaxTileSize(
        lb.getInt(), ub.getInt(),
        workloadPerWorkgroup[tiledLoops.size() - 1 - i], vectorSizeVals[i]);
  }
  workloadPerWorkgroup.append(numBatchDims, 1);

  Optional<llvm::Triple> triple = getTargetTriple(entryPointFn);
  if (triple && triple.getValue().isX86()) {
    // For DoubleTilingExpert, we will use LinalgSingleTilingExpertPassOptions
    // to control transforms. There is a tileInterchange option that needs to be
    // configured. However, we don't know the number of loops when adding the
    // pass to pass manager. Thus, we don't use double tiling expert for batch
    // gemms for now.
    if (!numBatchDims && useDoubleTilingExpert) {
      return setX86SandboxRootConfig(entryPointFn, contractionOp,
                                     workloadPerWorkgroup, vectorSize);
    } else {
      return setX86RootConfig(entryPointFn, contractionOp, workloadPerWorkgroup,
                              vectorSize);
    }
  }
  // Fall back to ARM configurations.
  return setARMRootConfig(entryPointFn, contractionOp, workloadPerWorkgroup,
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
    auto lhsShape = getUntiledShape(mmt4dOp.inputs()[0]);
    auto rhsShape = getUntiledShape(mmt4dOp.inputs()[1]);
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
    auto lhsShape = getUntiledShape(mmt4dOp.inputs()[0]);
    auto rhsShape = getUntiledShape(mmt4dOp.inputs()[1]);
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
      entryPointFn, mmt4dOp, tileSizes, nativeVectorSize,
      getDispatchLoweringPassPipeline(entryPointFn, mmt4dOp));
}

/// Sets the lowering configuration for dispatch region for linalg_ext.fft
/// root op.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, IREE::LinalgExt::FftOp fftOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  auto partitionedLoops = getPartitionedLoops(fftOp);
  unsigned maxDepth = partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSizes(maxDepth, defaultWorkgroupTileSize);
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
      fftOp.emitError("non-constant stage might not work for fft op");
      return failure();
    }
  }
  TileSizesListType tileSizes = {workgroupTileSizes};

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, fftOp, tileSizes,
      /*nativeVectorSizes=*/ArrayRef<int64_t>{},
      getDispatchLoweringPassPipeline(entryPointFn, fftOp));
}

/// Sets the lowering configuration for a generic op to use SingleTilingExpert.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, linalg::GenericOp genericOp,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  // If there are no loops, there is nothing to do.
  unsigned numLoops = genericOp.getNumLoops();
  if (numLoops == 0) return success();

  SmallVector<int64_t> nativeVectorSize(numLoops, 1);
  auto inputOutputOpOperands = genericOp.getInputAndOutputOperands();
  for (auto map : llvm::enumerate(genericOp.getIndexingMaps())) {
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
    nativeVectorSize[fastestVaryingDim] =
        std::max<int64_t>(nativeVectorSize[fastestVaryingDim],
                          getVectorSize(entryPointFn, operandType));
  }
  if (llvm::all_of(nativeVectorSize, [](int64_t vs) { return vs == 1; })) {
    // Nothing to vectorize just lower to loops.
    return success();
  }

  // Set the flow level tiling to the default.
  SmallVector<int64_t> prunedNativeVectorSize(tiledLoops.size(), 1);
  if (!tiledLoops.empty()) {
    SmallVector<unsigned> partitionedLoops = getPartitionedLoops(genericOp);
    for (auto loopDim : llvm::enumerate(partitionedLoops)) {
      prunedNativeVectorSize[loopDim.index()] =
          nativeVectorSize[loopDim.value()];
    }
  }
  SmallVector<int64_t> workloadPerWorkgroup =
      getDefaultWorkloadPerWorkgroup(tiledLoops, prunedNativeVectorSize);
  setTranslationInfo(entryPointFn,
                     DispatchLoweringPassPipeline::CPUSingleTilingExpert,
                     workloadPerWorkgroup,
                     /*workgroupSize=*/ArrayRef<int64_t>{});

  SmallVector<int64_t> l1TileSizes = nativeVectorSize;
  TileSizesListType tileSizes;
  tileSizes.push_back({});  // Empty since nothing to do for first level tiling.
  tileSizes.push_back(l1TileSizes);
  auto config = IREE::Codegen::LoweringConfigAttr::get(
      entryPointFn.getContext(), tileSizes, nativeVectorSize);
  setLoweringConfig(genericOp, config);

  return success();
}

static LogicalResult setRootConfigImpl(
    FuncOp entryPointFn, Operation *op,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  // Do not overwrite default configuration.
  if (getLoweringConfig(op)) return success();

  // Redirect to individual operations.
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<linalg::Mmt4DOp, linalg::ContractionOpInterface,
              IREE::LinalgExt::FftOp>([&](auto op) {
          return setRootConfig(entryPointFn, op, tiledLoops);
        })
        .Case<linalg::GenericOp>([&](auto genericOp) {
          if (genericOp.getNumLoops() == genericOp.getNumParallelLoops()) {
            // Ignore parallel elementwise operations now. They will be set as
            // roots ops if there are no other ops that can be treated as a
            // root op.
            return success();
          }
          return setRootConfig(entryPointFn, genericOp, tiledLoops);
        })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Finds the root operation in the given list of Linalg operations and sets
/// its configuration. Returns error for multiple root operations.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, ArrayRef<Operation *> computeOps,
    ArrayRef<LoopTilingAndDistributionInfo> tiledLoops) {
  Operation *rootOp = nullptr;
  for (auto computeOp : computeOps) {
    if (failed(setRootConfigImpl(entryPointFn, computeOp, tiledLoops))) {
      return failure();
    }
    if (getLoweringConfig(computeOp)) {
      if (rootOp) {
        return computeOp->emitOpError(
            "unhandled multiple roots in dispatch region");
      }
      rootOp = computeOp;
    }
  }
  if (rootOp) return success();

  // If there are any other ops other than `linalg.generic`, `linalg.copy` or
  // `linalg.fill` then just use the default.
  for (auto computeOp : computeOps) {
    if (!isa<linalg::GenericOp, linalg::CopyOp, linalg::FillOp>(computeOp)) {
      return success();
    }
  }

  // If there are no root ops, then check for a single `linalg.generic` op. Make
  // this the root, and vectorize the operation.
  for (auto computeOp : computeOps) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(computeOp)) {
      if (failed(setRootConfig(entryPointFn, genericOp, tiledLoops))) {
        return failure();
      }
      if (getLoweringConfig(computeOp)) {
        if (rootOp) {
          return computeOp->emitOpError(
              "unhanlded multiple parallel generic ops within a dispatch");
        }
        rootOp = computeOp;
      }
    }
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
  // For VMVX, do not use vectorization. Just lower as default.
  if (!isVMVXBackend(entryPointFn)) {
    if (failed(setRootConfig(entryPointFn, computeOps, tiledLoops))) {
      return failure();
    }
  }

  // Check if the translation info for the entry point is already set.
  if (!getTranslationInfo(entryPointFn)) {
    return setDefaultLaunchConfig(entryPointFn, tiledLoops);
  }
  return success();
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
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
