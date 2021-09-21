// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"

#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

// TODO(ravishankarm): This needs to be put in a common place for the CPU and
// GPU backends to use.
static llvm::cl::opt<int> clNativeVectorSizeInBytes(
    "iree-codegen-llvm-vector-size-in-bytes",
    llvm::cl::desc("linalg.matmul vector tile size"), llvm::cl::init(16));

static llvm::cl::opt<int> matmulWorkgroupTileSize(
    "iree-codegen-llvm-matmul-workgroup-size",
    llvm::cl::desc(
        "linalg.matmul tile size for workgroups spliting of M, N dimension"),
    llvm::cl::init(64));
static llvm::cl::opt<int> matmulL1TileSize(
    "iree-codegen-llvm-matmul-l1-size",
    llvm::cl::desc(
        "linalg.matmul tile size for L1 spliting of M, N, K dimension"),
    llvm::cl::init(32));
static llvm::cl::opt<int> matmulVectorSize(
    "iree-codegen-llvm-matmul-vector-size",
    llvm::cl::desc("linalg.matmul vector tile size"), llvm::cl::init(4));

static llvm::cl::opt<int> batchMatmulWorkgroupTileSize(
    "iree-codegen-llvm-batch-matmul-workgroup-size",
    llvm::cl::desc("linalg.batch_matmul tile size for workgroups spliting of "
                   "M, N dimension"),
    llvm::cl::init(32));
static llvm::cl::opt<int> batchMatmulL1TileSize(
    "iree-codegen-llvm-batch-matmul-l1-size",
    llvm::cl::desc("linalg.batch_matmul tile size for L1 spliting of M, N, K "
                   "dimensions"),
    llvm::cl::init(16));

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

/// Return the type length in bytes. Looks throuigh all the interface binding
/// ops to see the ABI types and guess-timates the type size to use.
static unsigned getReferenceTypeLengthInBytes(FuncOp entryPointFn) {
  unsigned referenceTypeLengthInBytes = 4;
  entryPointFn.walk([&](IREE::HAL::InterfaceBindingSubspanOp subSpanOp) {
    Type type = subSpanOp.getResult().getType();
    Type elementType = TypeSwitch<Type, Type>(type)
                           .Case<ShapedType, IREE::Flow::DispatchTensorType>(
                               [&](auto shapedType) -> Type {
                                 if (shapedType.getRank() > 0) {
                                   return shapedType.getElementType();
                                 }
                                 return nullptr;
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
    ArrayRef<TiledLoopInfo> tiledLoops, int64_t nativeVectorSizeInElements) {
  if (tiledLoops.empty()) {
    return {};
  }
  int64_t useDefaultWorkgroupTileSize = defaultWorkgroupTileSize;
  unsigned maxDim = 0;
  for (auto tiledLoop : tiledLoops) {
    maxDim = std::max<unsigned>(tiledLoop.distributionDim, maxDim);
  }
  SmallVector<int64_t> workloadPerWorkgroup(maxDim + 1, 1);
  if (tiledLoops.size() == 1) {
    useDefaultWorkgroupTileSize *= 2;
  } else if (tiledLoops.size() == 3) {
    useDefaultWorkgroupTileSize /= 2;
  }

  for (auto tiledLoop : tiledLoops) {
    if (!tiledLoop.ub || !tiledLoop.ub.is<Attribute>() || !tiledLoop.lb ||
        !tiledLoop.lb.is<Attribute>()) {
      workloadPerWorkgroup[tiledLoop.distributionDim] =
          defaultWorkgroupTileSize;
      continue;
    }
    int64_t lb = tiledLoop.lb.get<Attribute>().cast<IntegerAttr>().getInt();
    int64_t ub = tiledLoop.ub.get<Attribute>().cast<IntegerAttr>().getInt();
    int64_t candidateTileSize =
        (tiledLoop.distributionDim == 0 ? nativeVectorSizeInElements : 1);
    if (ub <= lb) {
      // Avoid tiling this loop.
      candidateTileSize = 0;
    } else {
      candidateTileSize = std::max<int64_t>(
          llvm::PowerOf2Floor(static_cast<uint64_t>(ub - lb) / 2),
          (tiledLoop.distributionDim == 0 ? nativeVectorSizeInElements : 1));
    }

    workloadPerWorkgroup[tiledLoop.distributionDim] =
        std::min<int64_t>(candidateTileSize, useDefaultWorkgroupTileSize);
  }
  return workloadPerWorkgroup;
}

/// Sets the default launch configuration to use for a tiled + distributed
/// dispatch region based on the `tiledLoops` found.
static LogicalResult setDefaultLaunchConfig(
    FuncOp entryPointFn, ArrayRef<TiledLoopInfo> tiledLoops) {
  unsigned typeWidthInBytes = getReferenceTypeLengthInBytes(entryPointFn);
  int64_t nativeVectorSizeInBytes = clNativeVectorSizeInBytes;
  if (auto fromConfig = getNativeVectorSizeInBytes(entryPointFn)) {
    nativeVectorSizeInBytes = fromConfig.getValue();
  }
  int64_t nativeVectorSizeInElements =
      nativeVectorSizeInBytes / typeWidthInBytes;
  SmallVector<int64_t> workloadPerWorkgroup =
      getDefaultWorkloadPerWorkgroup(tiledLoops, nativeVectorSizeInElements);

  setTranslationInfo(
      entryPointFn, IREE::HAL::DispatchLoweringPassPipeline::CPUDefault,
      /*workgroupSize =*/ArrayRef<int64_t>{}, workloadPerWorkgroup);
  return success();
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(FuncOp entryPointFn,
                                   linalg::ContractionOpInterface contractionOp,
                                   ArrayRef<TiledLoopInfo> tiledLoops) {
  if (getLoweringConfig(contractionOp)) return success();

  auto lhsShapedType = contractionOp.lhs().getType().cast<ShapedType>();
  Type elementType = lhsShapedType.getElementType();
  if (!elementType.isIntOrFloat()) return success();
  unsigned byteWidth = elementType.getIntOrFloatBitWidth() / 8;
  int64_t vectorSize;
  if (Optional<int64_t> nativeVectorSizeVal =
          getNativeVectorSizeInBytes(entryPointFn)) {
    vectorSize = nativeVectorSizeVal.getValue() / byteWidth;
  } else {
    vectorSize = matmulVectorSize;
  }

  // Use the default distribution for the matmul loops.
  bool isBatchMatmul = lhsShapedType.getRank() == 3;
  if (isBatchMatmul) {
    if (tiledLoops.size() != 3) {
      return contractionOp.emitOpError(
          "expected op to be distributed along 3 dimensions");
    }
  } else if (tiledLoops.size() != 2) {
    return contractionOp.emitOpError(
        "expected op tbe distributed along 2 dimensions");
  }
  SmallVector<int64_t> workloadPerWorkgroup = getDefaultWorkloadPerWorkgroup(
      isBatchMatmul ? tiledLoops.drop_front() : tiledLoops, vectorSize);
  setTranslationInfo(
      entryPointFn, IREE::HAL::DispatchLoweringPassPipeline::CPUTensorToVectors,
      /*workgroupSize =*/ArrayRef<int64_t>{}, workloadPerWorkgroup);

  SmallVector<int64_t, 4> l1TileSizes, vectorTileSizes, nativeVectorSize;
  if (isBatchMatmul) {
    l1TileSizes.push_back(1);
    vectorTileSizes.push_back(1);
    nativeVectorSize.push_back(1);
  }
  l1TileSizes.append(3, matmulL1TileSize);
  vectorTileSizes.append(3, vectorSize);
  nativeVectorSize.append(3, vectorSize);
  TileSizesListType tileSizes;
  tileSizes.push_back({});  // Empty here since there is nothing to do in first
                            // level tiling.
  tileSizes.emplace_back(std::move(l1TileSizes));
  tileSizes.emplace_back(std::move(vectorTileSizes));
  IREE::HAL::LoweringConfig config =
      buildConfigAttr(tileSizes, nativeVectorSize, entryPointFn.getContext());
  setLoweringConfig(contractionOp, config);
  return success();
}

/// Sets the lowering configuration for dispatch region for linalg.mmt4d root
/// op
static LogicalResult setRootConfig(FuncOp entryPointFn, linalg::Mmt4DOp mmt4dOp,
                                   ArrayRef<TiledLoopInfo> tiledLoops) {
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

  SmallVector<int64_t, 4> nativeVectorSize = getVectorSizes();

  TileSizesListType tileSizes = {getWorkgroupTileSizes(), getL1TileSizes(),
                                 nativeVectorSize};

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, mmt4dOp, tileSizes, nativeVectorSize,
      IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization);
}

/// Sets the lowering configuration for dispatch region for linalg_ext.fft
/// root op.
static LogicalResult setRootConfig(FuncOp entryPointFn, linalg_ext::FftOp fftOp,
                                   ArrayRef<TiledLoopInfo> tiledLoops) {
  auto partitionedLoops = getPartitionedLoops(fftOp);
  unsigned maxDepth = partitionedLoops.back() + 1;
  SmallVector<int64_t, 4> workgroupTileSizes(maxDepth,
                                             defaultWorkgroupTileSize);
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
      workgroupTileSizes[rank - 1] = 1 << value.getSExtValue();
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
      IREE::HAL::DispatchLoweringPassPipeline::CPUDefault);
}

/// Finds the root operation in the given list of linalg operations and sets
/// its configuration. Returns error for multiple root operations.
static LogicalResult setRootConfig(FuncOp entryPointFn,
                                   ArrayRef<Operation *> computeOps,
                                   ArrayRef<TiledLoopInfo> tiledLoops) {
  Operation *rootOp = nullptr;
  for (auto computeOp : computeOps) {
    if (!hasMarker(computeOp, getWorkgroupMarker())) continue;

    auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
      return TypeSwitch<Operation *, LogicalResult>(op)
          .Case<linalg::Mmt4DOp, linalg::ContractionOpInterface,
                linalg_ext::FftOp>([&](auto op) {
            return setRootConfig(entryPointFn, op, tiledLoops);
          })
          .Default([&](Operation *op) { return success(); });
    };
    if (failed(setRootConfigFn(computeOp))) {
      return failure();
    }
    if (getLoweringConfig(computeOp)) {
      if (rootOp) {
        return computeOp->emitError(
            "unhandled multiple roots in dispatch region");
      }
      rootOp = computeOp;
    }
  }
  return success();
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult setTranslationInfoAndRootConfig(
    FuncOp entryPointFn, ArrayRef<Operation *> computeOps,
    ArrayRef<TiledLoopInfo> tiledLoops) {
  // First check if the operations have a preset pipeline.
  for (auto computeOp : computeOps) {
    if (!hasMarker(computeOp, getWorkgroupMarker())) continue;

    if (auto config = getLoweringConfig(computeOp)) {
      // Check if the op has a preset pipeline.
      auto passPipeline = getLoweringPassPipeline(config);
      if (!passPipeline) continue;

      // If the function already has a translation, error out.
      if (auto translationInfo = getTranslationInfo(entryPointFn)) {
        return computeOp->emitOpError(
            "multiple ops within dispatch trying to set the translation "
            "info");
      }

      SmallVector<int64_t, 4> workgroupSize;
      if (auto workgroupSizeAttr = config.workgroupSize()) {
        workgroupSize = llvm::to_vector<4>(
            llvm::map_range(workgroupSizeAttr, [](Attribute intAttr) {
              return intAttr.cast<IntegerAttr>().getInt();
            }));
      }
      if (failed(setOpConfigAndEntryPointFnTranslation(
              entryPointFn, computeOp, config, *passPipeline, workgroupSize))) {
        return failure();
      }
    }
  }

  // Next set the configuration of the operations.
  if (failed(setRootConfig(entryPointFn, computeOps, tiledLoops))) {
    return failure();
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
    SmallVector<TiledLoopInfo> tiledLoops;

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
