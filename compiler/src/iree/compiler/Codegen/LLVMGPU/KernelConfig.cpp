// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"

#include <cstdint>
#include <numeric>
#include <optional>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler {

llvm::cl::opt<bool> clGPUEnableTransformDialectJit(
    "iree-codegen-llvmgpu-enable-transform-dialect-jit",
    llvm::cl::desc("enable the usage of the transform dialect JIT"),
    llvm::cl::init(true));

/// Flag to force using WMMA tensorcore operations.
llvm::cl::opt<bool>
    clGPUUseWMMA("iree-codegen-llvmgpu-use-wmma",
                 llvm::cl::desc("force use of wmma operations for tensorcore"),
                 llvm::cl::init(false));

/// Flag used to toggle using mma.sync vs wmma when targetting tensorcore.
llvm::cl::opt<bool>
    clGPUUseMMASync("iree-codegen-llvmgpu-use-mma-sync",
                    llvm::cl::desc("force use mma sync instead of wmma ops"),
                    llvm::cl::init(false));

namespace {

constexpr StringLiteral kCudaTarget = "cuda";
constexpr StringLiteral kRocmTarget = "rocm";

/// Structure to represent target features.
struct TargetInfo {
  // TODO: add finer grain control for other tensorcore types.
  bool hasTF32TensorCore = false;
  bool hasWarpShuffle = false;
  bool hasMmaSync = false;
  // These are listed in the order of preference, not necessarily monotonically.
  SmallVector<int64_t, 2> supportedSubgroupSizes = {32};
};

struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  std::array<int64_t, 3> workgroupSize;
  int64_t pipelineDepth;
};

// Simt codegen does not do software pipelining.
constexpr unsigned softwarePipelineDepthSimt = 0;

} // namespace

/// Return the best combination of tile size and wg size. It will then used to
/// pick the best size aligned with the shape dimension.
static void getMatmulConfig(SmallVectorImpl<TileWorkgroupSizePair> &tileSizes) {
  // Pick tile size so that M*K and K*N dividible by wgSize * \*vecSize=*\4.
  // This way workgroup memory copy don't need to be masked. Once we support
  // masked load we can get performance out of more configuration.
  tileSizes.push_back(TileWorkgroupSizePair({{32, 128, 32}, {32, 8, 1}, 1}));
  tileSizes.push_back(TileWorkgroupSizePair({{128, 64, 8}, {16, 8, 1}, 1}));
  tileSizes.push_back(TileWorkgroupSizePair({{16, 256, 32}, {64, 2, 1}, 1}));
  tileSizes.push_back(TileWorkgroupSizePair({{8, 32, 32}, {8, 8, 1}, 1}));

  tileSizes.push_back(TileWorkgroupSizePair({{32, 128, 4}, {32, 8, 1}, 1}));
  tileSizes.push_back(TileWorkgroupSizePair({{8, 128, 4}, {32, 1, 1}, 1}));
  tileSizes.push_back(TileWorkgroupSizePair({{16, 64, 4}, {16, 2, 1}, 1}));
  tileSizes.push_back(TileWorkgroupSizePair({{1, 128, 8}, {32, 1, 1}, 1}));
}

/// Return the best combination of tile size and wg size when using tensorcore
/// operations.
static void
getTensorCoreConfig(SmallVectorImpl<TileWorkgroupSizePair> &tileSizes,
                    Type elementType, int64_t M, int64_t N, int64_t K) {
  // Based on early analysis we found that 128x256x32_3 gives acceptable
  // performance across many of the large matrix sizes for f16 and fp32. This
  // needs to be refined into a better startegy based on empircal data but this
  // gives us a quick solution to achieve performance in the right order of
  // magnitude for large square like cases.
  int64_t parallelDim = M * N;
  static constexpr int64_t kLargDimThreashold = 1536;
  if (elementType.isF16()) {
    if (parallelDim >= kLargDimThreashold * kLargDimThreashold) {
      tileSizes.push_back(
          TileWorkgroupSizePair({{128, 256, 32}, {128, 2, 1}, 3}));
    }
    tileSizes.push_back(TileWorkgroupSizePair({{32, 32, 32}, {64, 2, 1}, 4}));
  } else {
    if (parallelDim >= kLargDimThreashold * kLargDimThreashold) {
      tileSizes.push_back(
          TileWorkgroupSizePair({{128, 256, 16}, {128, 2, 1}, 4}));
    }
    tileSizes.push_back(TileWorkgroupSizePair({{32, 32, 16}, {64, 2, 1}, 4}));
    tileSizes.push_back(TileWorkgroupSizePair({{16, 32, 16}, {64, 1, 1}, 4}));
    tileSizes.push_back(TileWorkgroupSizePair({{32, 16, 16}, {32, 2, 1}, 4}));
    tileSizes.push_back(TileWorkgroupSizePair({{16, 16, 16}, {32, 1, 1}, 4}));
  }
}

static StringRef getTargetArch(func::FuncOp entryPoint) {
  if (auto variantOp =
          entryPoint->getParentOfType<IREE::HAL::ExecutableVariantOp>()) {
    IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.getTarget();
    if (auto config = targetAttr.getConfiguration()) {
      if (auto attr = config.getAs<StringAttr>("target_arch")) {
        return attr.getValue();
      }
    }
  }
  return "";
}

bool isCudaTarget(func::FuncOp entryPoint) {
  if (auto variantOp =
          entryPoint->getParentOfType<IREE::HAL::ExecutableVariantOp>()) {
    IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.getTarget();
    if (auto backend = targetAttr.getBackend()) {
      return backend.getValue().str() == kCudaTarget;
    }
  }
  return false;
}

bool isRocmTarget(func::FuncOp entryPoint) {
  if (auto variantOp =
          entryPoint->getParentOfType<IREE::HAL::ExecutableVariantOp>()) {
    IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.getTarget();
    if (auto backend = targetAttr.getBackend()) {
      return backend.getValue().str() == kRocmTarget;
    }
  }
  return false;
}

static TargetInfo getCudaTargetInfo(func::FuncOp entryPoint) {
  TargetInfo info;
  // All the cuda target are assumed to have warp support.
  info.hasWarpShuffle = true;
  info.supportedSubgroupSizes = {32};
  StringRef targetName = getTargetArch(entryPoint);
  // If no target name is set assume all the features are off.
  if (targetName == "")
    return info;
  if (!StringRef(targetName).starts_with("sm_")) {
    entryPoint.emitError("unknown target name ") << targetName;
    return info;
  }
  APInt version;
  if (targetName.substr(3).getAsInteger(10, version)) {
    entryPoint.emitError("unknown target version ") << targetName;
    return info;
  }
  int64_t smVersion = version.getZExtValue();
  if (smVersion >= 80) {
    info.hasTF32TensorCore = true;
    info.hasMmaSync = true;
  }
  return info;
}

// TODO: Plumb in WarpSize into TargetInfo for wave64 systems.
static TargetInfo getRocmTargetInfo(func::FuncOp entryPoint) {
  TargetInfo info;
  StringRef targetName = getTargetArch(entryPoint);
  // If no target name is set assume all the features are off.
  if (targetName.empty())
    return info;

  if (!targetName.starts_with("gfx")) {
    entryPoint.emitError("unknown target name ") << targetName;
    return info;
  }

  // Assumes all gfx versions have warp shuffle.
  info.hasWarpShuffle = true;

  // RDNA supports wave32 and wave64, GCN and CDNA only wave64.
  if (targetName.starts_with("gfx10") || targetName.starts_with("gfx11"))
    info.supportedSubgroupSizes = {32, 64};
  else
    info.supportedSubgroupSizes = {64};

  // TODO: Check and enable for WMMA once pipeline is available.
  return info;
}

static TargetInfo getTargetInfo(func::FuncOp entryPoint) {
  // TODO: fill out target info for other vendors.
  if (isCudaTarget(entryPoint))
    return getCudaTargetInfo(entryPoint);
  if (isRocmTarget(entryPoint))
    return getRocmTargetInfo(entryPoint);
  return {};
}

static bool supportsTensorCore(func::FuncOp entryPoint, linalg::LinalgOp op,
                               const TargetInfo &targetInfo) {
  // Limit tensor core pipeline to matmul as not all combinations of transpose
  // are supported upstream.
  if (!targetInfo.hasTF32TensorCore)
    return false;
  if (!(isa<linalg::MatmulOp>(op) || isa<linalg::BatchMatmulOp>(op))) {
    assert(linalg::isaContractionOpInterface(op));
    // If this is not a named op matmul check some properties to make sure that
    // we can map it to tensorcore ops. We should have only mulAdd in the region
    // and the output map should have no permutation and the last dimension
    // should be a reduce.
    Region &body = op->getRegion(0);
    Region::OpIterator it = body.op_begin();
    if (it == body.op_end() || !isa<arith::MulFOp>(*(it++)))
      return false;
    if (it == body.op_end() || !isa<arith::AddFOp>(*(it++)))
      return false;
    if (it == body.op_end() || !isa<linalg::YieldOp>(*(it++)))
      return false;
    AffineMap outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
    if (outputMap.getNumResults() != outputMap.getNumDims() - 1)
      return false;
    OpBuilder b(op);
    for (unsigned i = 0, e = outputMap.getNumResults(); i < e - 1; i++) {
      if (outputMap.getResult(i) != b.getAffineDimExpr(i))
        return false;
    }
  }
  return true;
}

/// Decides which tensorcore operations to use.
static IREE::Codegen::DispatchLoweringPassPipeline
getTensorCorePipeline(Type elementType) {
  // Currently mma.sync is on by default for fp16 only.
  IREE::Codegen::DispatchLoweringPassPipeline codegenPipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulTensorCore;

  // For F16 and F32 use mmasync by default.
  if (elementType.isF16() || elementType.isF32()) {
    codegenPipeline = IREE::Codegen::DispatchLoweringPassPipeline::
        LLVMGPUMatmulTensorCoreMmaSync;
  }

  // Override the decision based on cl flags.
  assert(!(clGPUUseWMMA && clGPUUseMMASync) && "incompatible options.");
  if (clGPUUseMMASync) {
    codegenPipeline = IREE::Codegen::DispatchLoweringPassPipeline::
        LLVMGPUMatmulTensorCoreMmaSync;
  }
  if (clGPUUseWMMA) {
    codegenPipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulTensorCore;
  };
  return codegenPipeline;
}

static LogicalResult setContractConfig(func::FuncOp entryPoint,
                                       linalg::LinalgOp op,
                                       const TargetInfo &targetInfo) {
  if (!linalg::isaContractionOpInterface(op) || op.getNumParallelLoops() < 2) {
    return failure();
  }

  // Also exclude the case of matvec, which has only one non-unit parallel dim.
  // They should go down different pipelines.
  // Currently dynamic dimensions are tiled with size=1 in codegen.
  int staticNonUnitParallelDimCount = 0;
  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(op);
  assert(succeeded(contractionDims) && "Could not infer contraction dims");
  for (auto mDim : contractionDims->m) {
    staticNonUnitParallelDimCount +=
        bounds[mDim] != 1 && !ShapedType::isDynamic(bounds[mDim]);
  }
  for (auto nDim : contractionDims->n) {
    staticNonUnitParallelDimCount +=
        bounds[nDim] != 1 && !ShapedType::isDynamic(bounds[nDim]);
  }
  if (staticNonUnitParallelDimCount <= 1)
    return failure();

  // Don't consider operations that don't have a broadcast, those should go
  // through reductions.
  if (llvm::any_of(op.getIndexingMapsArray(),
                   [](AffineMap m) { return m.isPermutation(); })) {
    return failure();
  }

  // TODO: Properly rematerialize leading elementwise with shared memory
  // promotion.
  if (hasFusedLeadingOp(op)) {
    return failure();
  }

  auto setMatmulConfig =
      [&entryPoint, &op](int64_t tileX, int64_t tileY, int64_t tileK,
                         ArrayRef<int64_t> workgroupSize,
                         ArrayRef<int64_t> subgroupSizes,
                         unsigned softwarePipelineDepth,
                         IREE::Codegen::DispatchLoweringPassPipeline pipeline) {
        TileSizesListType tileSizes;
        unsigned numParallelLoops = op.getNumParallelLoops();
        SmallVector<int64_t> workgroupTileSizes(numParallelLoops - 2, 1);
        workgroupTileSizes.append({tileX, tileY});
        workgroupTileSizes.append(op.getNumReductionLoops(), tileK);

        SmallVector<unsigned> partitionedLoops =
            cast<PartitionableLoopsInterface>(op.getOperation())
                .getPartitionableLoops(/*maxNumPartitionedLoops=*/std::nullopt);
        llvm::SmallDenseSet<unsigned, 4> partitionedLoopsSet;
        partitionedLoopsSet.insert(partitionedLoops.begin(),
                                   partitionedLoops.end());
        for (auto loopID : llvm::seq<unsigned>(0, numParallelLoops)) {
          if (!partitionedLoopsSet.count(loopID)) {
            workgroupTileSizes[loopID] = 0;
          }
        }

        tileSizes.emplace_back(
            std::move(workgroupTileSizes)); // Workgroup level.
        std::optional<int64_t> subgroupSize = std::nullopt;
        if (!subgroupSizes.empty())
          subgroupSize = subgroupSizes.front();

        return setOpConfigAndEntryPointFnTranslation(
            entryPoint, op, tileSizes, pipeline, workgroupSize, subgroupSize,
            softwarePipelineDepth,
            /*softwarePipelineStoreStage=*/1);
      };
  // Infer the MxN size of the matmul based on operands and indexing maps.
  auto lhsShape =
      llvm::cast<ShapedType>(op.getDpsInputOperand(0)->get().getType())
          .getShape();
  auto rhsShape =
      llvm::cast<ShapedType>(op.getDpsInputOperand(1)->get().getType())
          .getShape();
  int64_t sizeM = ShapedType::kDynamic;
  int64_t sizeN = ShapedType::kDynamic;
  int64_t sizeK = ShapedType::kDynamic;
  auto outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  for (unsigned i = 0; i < lhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(0)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 2)) {
      sizeM = lhsShape[i];
      break;
    }
  }
  for (unsigned i = 0; i < rhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(1)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 1)) {
      sizeN = rhsShape[i];
      break;
    }
  }
  SmallVector<unsigned> exprs;
  op.getReductionDims(exprs);
  if (exprs.size() == 1) {
    for (unsigned i = 0; i < lhsShape.size(); i++) {
      if (op.getMatchingIndexingMap(op.getDpsInputOperand(0))
              .getDimPosition(i) == exprs[0]) {
        sizeK = lhsShape[i];
        break;
      }
    }
  }
  bool isStaticSize = !ShapedType::isDynamic(sizeM) &&
                      !ShapedType::isDynamic(sizeN) &&
                      !ShapedType::isDynamic(sizeK);
  if (isStaticSize) {
    /// Try tensorcore config first.
    if (supportsTensorCore(entryPoint, op, targetInfo)) {
      SmallVector<TileWorkgroupSizePair> TCtileSizeConfig;
      Type elementType = llvm::cast<RankedTensorType>(
                             op.getDpsInputOperand(0)->get().getType())
                             .getElementType();

      getTensorCoreConfig(TCtileSizeConfig, elementType, sizeM, sizeN, sizeK);
      // Pick the best configuration where the original shape is aligned on the
      // tile size.
      for (TileWorkgroupSizePair &config : TCtileSizeConfig) {
        if (sizeK % config.tileSize[2] == 0 &&
            sizeN % config.tileSize[1] == 0 &&
            sizeM % config.tileSize[0] == 0) {
          IREE::Codegen::DispatchLoweringPassPipeline codegenPipeline =
              getTensorCorePipeline(elementType);
          return setMatmulConfig(
              config.tileSize[0], config.tileSize[1], config.tileSize[2],
              config.workgroupSize, targetInfo.supportedSubgroupSizes,
              sizeK == config.tileSize[2] ? 1 : config.pipelineDepth,
              codegenPipeline);
        }
      }
    }
    // Special case for very small matrices.
    if (sizeM * sizeN <= targetInfo.supportedSubgroupSizes.front()) {
      return setMatmulConfig(
          sizeN, sizeM, 4, {sizeM, sizeN, 1}, targetInfo.supportedSubgroupSizes,
          softwarePipelineDepthSimt,
          IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt);
    }
    // simt matmul case
    SmallVector<TileWorkgroupSizePair> tileSizeConfig;
    // Query the best configuration.
    getMatmulConfig(tileSizeConfig);
    // Pick the best configuration where the original shape is aligned on the
    // tile size.
    for (TileWorkgroupSizePair &config : tileSizeConfig) {
      if (sizeN % config.tileSize[1] == 0 && sizeM % config.tileSize[0] == 0 &&
          sizeK % config.tileSize[2] == 0) {
        return setMatmulConfig(
            config.tileSize[0], config.tileSize[1], config.tileSize[2],
            config.workgroupSize, targetInfo.supportedSubgroupSizes,
            softwarePipelineDepthSimt,
            IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt);
      }
    }
  }
  // If we haven't found any config, use the best tile size hoping that
  // the workgroup specialization handles the main tile path efficiently.
  SmallVector<TileWorkgroupSizePair> tileSizeConfig;
  // Query the best configuration.
  getMatmulConfig(tileSizeConfig);
  constexpr size_t configIndex = 0;
  const TileWorkgroupSizePair &config = tileSizeConfig[configIndex];
  const int64_t tileX = config.tileSize[0];
  const int64_t tileY = config.tileSize[1];
  int64_t tileK = config.tileSize[2];
  // Since specialization doesn't work for K loop and peeling is not enabled yet
  // we pick a tileK size that is aligned on the K size.
  if (ShapedType::isDynamic(sizeK))
    tileK = 1;
  while (sizeK % tileK != 0) {
    tileK >>= 1;
  }
  const std::array<int64_t, 3> workgroupSize{config.workgroupSize[0],
                                             config.workgroupSize[1],
                                             config.workgroupSize[2]};
  return setMatmulConfig(
      tileX, tileY, tileK, workgroupSize, targetInfo.supportedSubgroupSizes,
      softwarePipelineDepthSimt,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt);
}

static LogicalResult setFftConfig(func::FuncOp entryPoint,
                                  IREE::LinalgExt::FftOp op,
                                  const TargetInfo &targetInfo) {
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  unsigned loopDepth = partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSize(loopDepth, 0);
  SmallVector<int64_t, 3> workgroupSize = {
      targetInfo.supportedSubgroupSizes.front(), 1, 1};

  // Tiling along partitioned loops with size 1.
  for (int64_t loopIndex : partitionedLoops) {
    workgroupTileSize[loopIndex] = 1;
  }
  auto rank = op.getOperandRank();
  if (workgroupTileSize.size() >= rank && workgroupTileSize[rank - 1] != 0) {
    APInt value;
    if (matchPattern(op.getStage(), m_ConstantInt(&value))) {
      workgroupTileSize[rank - 1] = 1ll << value.getSExtValue();
    } else {
      op.emitError("non-constant stage might not work for fft op");
      return failure();
    }
  }
  TileSizesListType tileSizes = {workgroupTileSize};
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
      workgroupSize);
}

static LogicalResult setSortConfig(func::FuncOp entryPoint, Operation *op,
                                   const TargetInfo &targetInfo) {
  TileSizesListType tileSizes;
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, tileSizes,
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
        {1, 1, 1});
  }
  size_t numLoops = partitionedLoops.back() + 1;
  // To get peak occupancy we need a workgroup size of at least two warps
  std::array<int64_t, 3> workgroupSize = {
      2 * targetInfo.supportedSubgroupSizes.front(), 1, 1};
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);
  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
    }
  }

  // Tile to have one element per thread.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.count(depth - 1)) {
      workgroupTileSizes[depth - 1] = workgroupSize[0];
      break;
    }
  }
  tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
      workgroupSize);
}

static SmallVector<int64_t>
getDefaultWorkgroupTileSizesForPackUnPack(TilingInterface op,
                                          int64_t defaultSize) {
  unsigned numLoops = op.getLoopIteratorTypes().size();
  auto partitionedLoops = cast<PartitionableLoopsInterface>(op.getOperation())
                              .getPartitionableLoops(kNumMaxParallelDims);
  SmallVector<int64_t> workgroupTileSizes(numLoops, defaultSize);
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto dim : llvm::seq<int64_t>(0, workgroupTileSizes.size())) {
    if (!partitionedLoopsSet.count(dim)) {
      workgroupTileSizes[dim] = 0;
    }
  }

  return workgroupTileSizes;
}

static LogicalResult setPackConfig(func::FuncOp entryPoint,
                                   tensor::PackOp packOp,
                                   const TargetInfo &targetInfo) {
  SmallVector<int64_t> tileSizes = getDefaultWorkgroupTileSizesForPackUnPack(
      cast<TilingInterface>(packOp.getOperation()),
      targetInfo.supportedSubgroupSizes.front());

  // The default function aims to returns the number of workload per workgroup,
  // but it does not know that it is working on packed domain. We need to take
  // inner tile sizes into account and adjust the distribution tile sizes.
  SmallVector<int64_t> innerTiles = packOp.getStaticTiles();
  ArrayRef<int64_t> dimPos = packOp.getInnerDimsPos();
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    if (tileSizes[pos] == 0 || ShapedType::isDynamic(size))
      continue;
    tileSizes[pos] = tileSizes[pos] / size;
    tileSizes[pos] = std::max<int64_t>(tileSizes[pos], 1);
  }

  TileSizesListType tileSizesList = {tileSizes};
  std::array<int64_t, 3> workgroupSizes = {
      targetInfo.supportedSubgroupSizes.front(), 1, 1};
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, packOp, tileSizesList,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUPackUnPack,
      workgroupSizes);
}

// Basic default properties for linalg ops that haven't been tuned.
static LogicalResult setRootDefaultConfig(func::FuncOp entryPoint,
                                          Operation *op,
                                          const TargetInfo &targetInfo) {
  IREE::Codegen::DispatchLoweringPassPipeline passPipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute;
  TileSizesListType tileSizes;
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops = interfaceOp.getPartitionableLoops(std::nullopt);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                                 passPipeline, {1, 1, 1});
  }

  size_t numLoops = partitionedLoops.back() + 1;
  // To get peak occupancy we need a workgroup size of at least two warps.
  std::array<int64_t, 3> workgroupSize = {
      2 * targetInfo.supportedSubgroupSizes.front(), 1, 1};
  unsigned vectorSize = 4;
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);
  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
    }
  }
  int64_t skipInnerTiling = 0;
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    for (auto [index, outputOperand] :
         llvm::enumerate(genericOp.getDpsInitsMutable())) {
      if (!genericOp.getMatchingIndexingMap(&outputOperand)
               .isProjectedPermutation()) {
        vectorSize = 1;
        break;
      }
      ArrayRef<int64_t> shape =
          llvm::cast<ShapedType>(outputOperand.get().getType()).getShape();
      if (llvm::any_of(shape, ShapedType::isDynamic)) {
        vectorSize = 1;
        break;
      }
      // Since we vectorize along the most inner dimension, make sure if can be
      // divided by number of threads * vectorSize.
      while (vectorSize > 1 &&
             shape.back() % (workgroupSize[0] * vectorSize) != 0) {
        vectorSize /= 2;
      }
      if (vectorSize == 1) // assume there is fastpath + slowpath
        vectorSize = 4;
      int64_t problemSize = std::accumulate(
          shape.begin(), shape.end(), 1,
          [](const int64_t &a, const int64_t &b) { return a * b; });
      if ((problemSize /
           (targetInfo.supportedSubgroupSizes.front() * vectorSize)) < 64) {
        vectorSize = 1;
        break;
      }
      // If the inner dimension is too small to have one element per thread
      // reduce the workgroup size try to distribute amongst more dimensions.
      if (shape.back() < vectorSize * workgroupSize[0]) {
        int64_t flatWG = workgroupSize[0];
        vectorSize = 1;
        int64_t id = 0;
        for (int64_t dim : llvm::reverse(shape)) {
          // Unit loops are already skipped.
          if (dim == 1)
            continue;
          if (dim < flatWG) {
            skipInnerTiling++;
            workgroupSize[id] = dim;
          } else {
            workgroupSize[id] = flatWG;
            break;
          }
          flatWG = flatWG / dim;
          id++;
          if (flatWG <= 1 || id >= workgroupSize.size())
            break;
        }
        break;
      }
    }
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // Pick a vectorSize of 1 for op that we know won't get vectorized.
  // Also skip vectorization for linalg on memref (no result) as the pipeline
  // relies on tensor level tiling.
  // TODO(thomasraoux): This could be improved by checking if the linalg op
  // would fail vectorization.
  if (!linalgOp || op->getNumResults() != 1 ||
      llvm::any_of(linalgOp.getIndexingMapsArray(),
                   [](AffineMap m) { return !m.isProjectedPermutation(); })) {
    vectorSize = 1;
  } else {
    passPipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorize;
  }

  int64_t id = 0;
  // Set the inner most parallel loop to `lowerTs`.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.count(depth - 1)) {
      if (skipInnerTiling > 0) {
        // For dimensions that don't need to be distributed across blocks skip
        // tiling by setting tile size to 0.
        workgroupTileSizes[depth - 1] = 0;
        skipInnerTiling--;
        id++;
        if (id >= workgroupSize.size())
          break;
        continue;
      }
      workgroupTileSizes[depth - 1] = workgroupSize[id] * vectorSize;
      break;
    }
  }

  if (linalgOp) {
    // Tile reduction dimension to 4 to allow doing load4 if the reduction size
    // is the most inner dimension.
    workgroupTileSizes.append(linalgOp.getNumReductionLoops(), 4);
  }
  tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, passPipeline, workgroupSize,
      targetInfo.supportedSubgroupSizes.front());
}

/// Set configuration for transform dialect based strategies.
static LogicalResult setTransformDialectConfig(func::FuncOp entryPoint,
                                               Operation *op,
                                               const TargetInfo &targetInfo) {
  if (!clGPUEnableTransformDialectJit) {
    return failure();
  }

  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      entryPoint.getContext(),
      IREE::Codegen::DispatchLoweringPassPipeline::TransformDialectCodegen);

  // TODO: unify the target informations into one structure.
  iree_compiler::gpu::GPUModel gpuModel;
  gpuModel.hasWarpShuffle = targetInfo.hasWarpShuffle;
  gpuModel.hasTF32TensorCore = targetInfo.hasTF32TensorCore;
  gpuModel.hasMmaSync = targetInfo.hasMmaSync;

  // Populates a subset of the fragment combinations supported in MLIR lowerings
  // to NVVM (which is itself a subset of what LLVM supports) based on what the
  // pipeline currently supports.
  // TODO: avoid hard coding this and populate based on hardware capabilities.
  // TODO: add missing supported configs once the pipeline supports it.
  MLIRContext *context = entryPoint.getContext();
  Type f32Type = Float32Type::get(context);
  Type f16Type = Float16Type::get(context);

  iree_compiler::gpu::MMAConfig f16f32AccConfig = {
      /*m=*/16,          /*n=*/16,          /*k=*/16,
      /*aType=*/f16Type, /*bType=*/f16Type, /*cType=*/f32Type};
  iree_compiler::gpu::MMAConfig f16f16AccConfig = {
      /*m=*/16,          /*n=*/16,          /*k=*/16,
      /*aType=*/f16Type, /*bType=*/f16Type, /*cType=*/f16Type};
  gpuModel.supportedWMMAConfigs = {f16f32AccConfig, f16f16AccConfig};

  if (targetInfo.hasTF32TensorCore) {
    iree_compiler::gpu::MMAConfig tf32WmmaConfig = {
        /*m=*/16,          /*n=*/16,          /*k=*/8,
        /*aType=*/f32Type, /*bType=*/f32Type, /*cType=*/f32Type};
    gpuModel.supportedWMMAConfigs.push_back(tf32WmmaConfig);
  }

  if (failed(iree_compiler::gpu::matchAndSetTransformStrategy(entryPoint, op,
                                                              gpuModel)))
    return failure();
  return setTranslationInfo(entryPoint, translationInfo);
}

static bool isMatvecLike(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumParallelLoops() != 2)
    return false;

  if (linalgOp.getNumReductionLoops() != 1)
    return false;

  // TODO: Allow for matvec with fused dequantization.
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(linalgOp);
  if (failed(dims))
    return false;

  // TODO: Support batch matvec.
  if (!dims->batch.empty())
    return false;

  for (ArrayRef indices : {dims->m, dims->n, dims->k}) {
    if (!llvm::hasSingleElement(indices))
      return false;
  }

  // Check if the first parallel dimension has bound 1, indicating we found a
  // vector shape.
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  if (bounds[dims->m.front()] != 1)
    return false;

  return true;
}

/// Set the configuration for reductions that can be mapped to warp reductions.
static LogicalResult setWarpReductionConfig(func::FuncOp entryPoint,
                                            linalg::LinalgOp op,
                                            const TargetInfo &targetInfo) {
  if (!targetInfo.hasWarpShuffle)
    return failure();

  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
  int64_t numParallelDims = op.getNumParallelLoops();

  if (reductionDims.empty())
    return failure();

  // Make sure reduction dimensions are static and innermost ones.
  int64_t numDynamicReductionDims = 0;
  for (unsigned dim : reductionDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      numDynamicReductionDims++;
    }
    if (dim < numParallelDims) {
      return failure();
    }
  }

  // Distribution of multi-dim masked writes currently aren't fully supported.
  if (numDynamicReductionDims > 1) {
    return failure();
  }

  if (op.getRegionOutputArgs().size() != 1)
    return failure();

  // Only support projected permutation, this could be extended to projected
  // permutated with broadcast.
  if (llvm::any_of(op.getDpsInputOperands(), [&](OpOperand *input) {
        return !op.getMatchingIndexingMap(input).isProjectedPermutation();
      }))
    return failure();

  bool foundSingleReductionOutput = false;
  for (auto [index, initOpOperand] : llvm::enumerate(op.getDpsInitsMutable())) {
    // Only single combiner operations are supported for now.
    SmallVector<Operation *> combinerOps;
    if (matchReduction(op.getRegionOutputArgs(), index, combinerOps) &&
        combinerOps.size() == 1) {
      if (foundSingleReductionOutput)
        return failure();
      foundSingleReductionOutput = true;
      continue;
    }
    if (!op.getMatchingIndexingMap(&initOpOperand).isIdentity())
      return failure();
  }
  if (!foundSingleReductionOutput)
    return failure();

  // Tile all the parallel dimension to 1.
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);

  // Without any bounds on dynamic reduction dims, we need specialization to
  // get peak performance. For now, just use the warp size.
  if (numDynamicReductionDims) {
    SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
    int64_t preferredSubgroupSize = targetInfo.supportedSubgroupSizes.front();
    reductionTileSizes[reductionDims[0]] = preferredSubgroupSize;
    TileSizesListType tileSizes;
    tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
    tileSizes.emplace_back(std::move(reductionTileSizes)); // Reduction level
    std::array<int64_t, 3> workgroupSize = {preferredSubgroupSize, 1, 1};
    if (failed(setOpConfigAndEntryPointFnTranslation(
            entryPoint, op, tileSizes,
            IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUWarpReduction,
            workgroupSize))) {
      return failure();
    }
    return success();
  }

  int64_t reductionSize = 1;
  for (int64_t dim : reductionDims)
    reductionSize *= bounds[dim];

  auto selectedSubgroupSizeIt = llvm::find_if(
      targetInfo.supportedSubgroupSizes, [reductionSize](int64_t subgroupSize) {
        return reductionSize % subgroupSize == 0;
      });
  if (selectedSubgroupSizeIt == targetInfo.supportedSubgroupSizes.end())
    return failure();
  int64_t subgroupSize = *selectedSubgroupSizeIt;

  const Type elementType =
      llvm::cast<ShapedType>(op.getDpsInitOperand(0)->get().getType())
          .getElementType();
  if (!elementType.isIntOrFloat())
    return failure();
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  // Reduction distribution only supports 8/16/32 bit types now.
  if (bitWidth != 32 && bitWidth != 16 && bitWidth != 8)
    return failure();

  const unsigned largestLoadSizeInBits = 128;
  unsigned vectorSize = largestLoadSizeInBits / bitWidth;
  while ((reductionSize / vectorSize) % subgroupSize != 0)
    vectorSize /= 2;

  // Deduce the workgroup size we should use for reduction. Currently a
  // workgroup processes all elements in reduction dimensions. Need to make sure
  // the workgroup size we use can divide the total reduction size, and it's
  // also within hardware limitations.
  const int64_t maxWorkgroupSize = 1024;
  int64_t groupSize = reductionSize / vectorSize;
  if (groupSize > maxWorkgroupSize) {
    groupSize = llvm::APIntOps::GreatestCommonDivisor(
                    {64, uint64_t(groupSize)}, {64, uint64_t(maxWorkgroupSize)})
                    .getZExtValue();
  }

  // Then we need to strike a balance--
  // 1) parallel dimensions are distributed to workgroups. If there are many
  //    workgroups dispatched, we'd want to have each GPU core hosting multiple
  //    of them for occupancy.
  // 2) we want each thread to read quite a few 128-bit vectors for better
  //    memory cache behavior.
  // Both means we cannot use a too large workgroup size.

  std::optional<int64_t> parallelSize = 1;
  for (int64_t dim : parallelDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      parallelSize = std::nullopt;
      break;
    }
    *parallelSize *= bounds[dim];
  }
  // Total parallel size that can fill the GPU with enough workgorups.
  // TODO: query from the target device; roughly 2x hardware compute unit.
  const int parallelThreshold = 256;
  // How many 128-bit vectors each thread should at least read.
  const int targetVectorCount = 8;
  while (parallelSize && *parallelSize > parallelThreshold &&
         (groupSize / 2) % subgroupSize == 0 &&
         reductionSize / (groupSize * vectorSize) < targetVectorCount) {
    // Use less subgroups per workgroup..
    groupSize /= 2;
    // in order to host more workgroups per hardware compute unit.
    *parallelSize /= 2;
  }

  // Current warp reduction pattern is a two step butterfly warp reduce.
  // First, do warp reductions along multiple subgroups.
  // Second, reduce results from multiple subgroups using single warp reduce.
  // The final warp reduce requires subgroup count <= subgroup size to work.
  if ((groupSize / subgroupSize) > subgroupSize)
    return failure();

  // With just one subgroup per workgroup, make each subgroup do more work and
  // process a few reductions (rows) along the last parallel dimension.
  //
  // TODO: This is enabled for matvec on ROCm for now. We should
  // validate this strategy and extend to more linalg generics and to CUDA.
  if (isRocmTarget(entryPoint) &&
      llvm::none_of(bounds, ShapedType::isDynamic) && isMatvecLike(op)) {
    int64_t lastParallelBound = bounds[parallelDims.back()];
    int64_t numParallelReductions = 1;
    const int64_t maxParallelFactor = groupSize / 4;
    for (int64_t parallelFactor = 2;
         (parallelFactor < maxParallelFactor) &&
         (lastParallelBound % parallelFactor == 0) &&
         (lastParallelBound > parallelFactor);
         parallelFactor *= 2) {
      numParallelReductions = parallelFactor;
    }
    workgroupTileSizes.back() = numParallelReductions;
  }

  std::array<int64_t, 3> workgroupSize = {groupSize, 1, 1};
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  int64_t remainingGroupSize = groupSize;
  for (int i = reductionDims.size() - 1; i >= 0; --i) {
    int64_t dim = reductionDims[i];
    int64_t bound = bounds[dim];
    if (i == reductionDims.size() - 1)
      bound /= vectorSize;
    APInt size = llvm::APIntOps::GreatestCommonDivisor(
        {64, uint64_t(remainingGroupSize)}, {64, uint64_t(bound)});
    reductionTileSizes[dim] = size.getSExtValue();
    if (i == reductionDims.size() - 1)
      reductionTileSizes[dim] *= vectorSize;
    remainingGroupSize /= size.getSExtValue();
  }
  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
  tileSizes.emplace_back(std::move(reductionTileSizes)); // Reduction level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUWarpReduction,
      workgroupSize, subgroupSize);
  return success();
}

static bool hasTwoOrThreeLoopsInfo(linalg::LinalgOp linalgOp) {
  return linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

static LogicalResult setTransposeConfig(func::FuncOp entryPoint,
                                        linalg::LinalgOp linalgOp) {
  LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);

  // Checks preconditions for shared mem transpose.
  if (!opInfo.isTranspose() || opInfo.isDynamic() || opInfo.isReduction() ||
      !isa<linalg::GenericOp>(linalgOp) || !hasTwoOrThreeLoopsInfo(linalgOp)) {
    return failure();
  }

  ArrayRef<OpOperand *> transposedOperands = opInfo.getTransposeOperands();

  // Determine the fastest moving dimensions for the source/destination indices
  // of each transpose. These inform the tile sizes.
  int64_t outputFastestDim = linalgOp.getNumLoops() - 1;
  int64_t inputFastestDim =
      linalgOp.getMatchingIndexingMap(transposedOperands[0])
          .getDimPosition(outputFastestDim);
  // Ensure the other transposed operands match
  for (int i = 1; i < transposedOperands.size(); ++i) {
    if (inputFastestDim !=
        linalgOp.getMatchingIndexingMap(transposedOperands[i])
            .getDimPosition(outputFastestDim)) {
      return failure();
    }
  }

  int32_t tileM = 32;
  int32_t tileN = 32;
  TileSizesListType tileSizes;
  // Set all tile sizes to 1 except for fastest moving dimensions.
  SmallVector<int64_t> tileSizesTemp(linalgOp.getNumLoops(), 1);
  tileSizesTemp[outputFastestDim] = 32;
  tileSizesTemp[inputFastestDim] = 32;
  tileSizes.push_back(tileSizesTemp);

  // Check alignment with tile size for each transpose. Only the fastest moving
  // dims need to match the transpose tile.
  auto loopRanges = linalgOp.getStaticLoopRanges();
  if (loopRanges[outputFastestDim] % tileM != 0 ||
      loopRanges[inputFastestDim] % tileN != 0) {
    return failure();
  }

  // Workgroup size contains 8 warps. Configured with 8 threads on fastest
  // moving dimension so each thread can execute a vectorized copy of 4
  // contigious elements at a time from the 32 block.
  std::array<int64_t, 3> workgroupSize = {8, 32, 1};

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, linalgOp, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTransposeSharedMem,
      workgroupSize);
}

/// Set the configuration for argmax that can be mapped to argmax uKernel.
/// Distribute all parallel dim across different workgroups, and only use single
/// subgroup per workgroup.
static LogicalResult setArgmaxUkernelConfig(func::FuncOp entryPoint,
                                            linalg::GenericOp op,
                                            const TargetInfo &targetInfo) {

  // Checks if UKernels are enabled.
  if (auto variantOp =
          entryPoint->getParentOfType<IREE::HAL::ExecutableVariantOp>()) {
    auto target = variantOp.getTarget();
    const char ukernelName[] = "argmax";
    if (!hasUkernel(target, ukernelName) ||
        !hasUkernelSupportedGpuArch(target)) {
      return failure();
    }
  }

  if (!targetInfo.hasWarpShuffle)
    return failure();

  if (failed(isArgmaxOp(op)))
    return failure();
  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  // Currently Argmax UKernel only support 1 reduction dim.
  if (reductionDims.size() != 1)
    return failure();

  // Make sure reduction dimensions are static and innermost ones.
  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
  int64_t numParallelDims = op.getNumParallelLoops();
  int64_t numDynamicReductionDims = 0;
  for (unsigned dim : reductionDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      numDynamicReductionDims++;
    }
    if (dim < numParallelDims) {
      return failure();
    }
  }

  // Distribution of multi-dim masked writes currently aren't fully supported.
  if (numDynamicReductionDims > 1) {
    return failure();
  }

  // Tile all the parallel dimension to 1.
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);

  // Currently Argmax Ukernel let's every thread reduce reductionDim/WarpSize
  // number of elements, and then it does a single step butterfly warp reduce.
  // Hence it expects workgroupSize to be warpSize/subgroupSize, and
  // reductionTileSize to be size of the reduction dim.
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  int64_t preferredSubgroupSize = targetInfo.supportedSubgroupSizes.front();
  reductionTileSizes[reductionDims[0]] = preferredSubgroupSize;
  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
  tileSizes.emplace_back(std::move(reductionTileSizes)); // Reduction level
  std::array<int64_t, 3> workgroupSize = {preferredSubgroupSize, 1, 1};
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPoint, op, tileSizes,
          IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDefault,
          workgroupSize))) {
    return failure();
  }
  return success();
}

/// Make UKernels take the LLVMGPUDefault lowering pipeline.
static LogicalResult
setUKernelConfig(func::FuncOp entryPoint,
                 IREE::Codegen::UKernelOpInterface ukernelOp) {
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      entryPoint->getContext(),
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDefault);
  return setTranslationInfo(entryPoint, translationInfo);
}

/// Decides the tiling and distribution parameters for one convolution
/// dimension. Returns true if we can succesfully deduce.
///
/// - `inputDim` is the size of the dimension to be distributed.
/// - `residualThreads` is the remaining threads we can distribute.
/// - `residualTilingFactor` indicates the remaining tiling scale factor.
/// - `wgDimSize` will be updated with the decided workgroup dimension size.
/// - `wgTileSize` will be updated with the decided workgroup tile size.
/// - `invoTileSize` will be updated with the decided invocation tile size.
static bool distributeToOneDim(const int64_t inputDim,
                               const bool isInnerMostDim,
                               int64_t &residualThreads,
                               int64_t &residualTilingFactor,
                               int64_t &wgDimSize, int64_t &wgTileSize) {
  const int64_t lb = isInnerMostDim ? 2 : 1;
  for (int64_t dim = residualThreads; dim >= lb; dim >>= 1) {
    int64_t chosenTileSize = 0;
    if (isInnerMostDim) {
      // Handle 4 elements per thread for the innermost dimension. We need
      // this for vectorized load.
      chosenTileSize = 4;
      if (inputDim % (dim * chosenTileSize) != 0)
        continue;
    } else {
      for (int64_t t = residualTilingFactor; t >= 1; t >>= 1)
        if (inputDim % (dim * t) == 0) {
          chosenTileSize = t;
          break;
        }
    }
    if (chosenTileSize) {
      wgDimSize = dim;
      wgTileSize = dim * chosenTileSize;
      residualThreads /= dim;
      residualTilingFactor /= chosenTileSize;
      return true;
    }
  }
  return false;
};

/// Decides the tiling and distribution parameters for two convolution window
/// dimensions to two workgroup dimensions as a square. Returns true if we can
/// succesfully deduce.
static bool distributeToSquare(const int64_t oh, const int64_t ow,
                               int64_t &residualThreads,
                               int64_t &residualTilingFactor,
                               MutableArrayRef<int64_t> wgDimSizes,
                               MutableArrayRef<int64_t> wgTileSizes) {
  assert(wgDimSizes.size() == 2 && wgTileSizes.size() == 2);

  const unsigned log2Threads = llvm::Log2_64(residualThreads);
  if (oh == ow && residualThreads != 1 && log2Threads % 2 == 0) {
    const int64_t yz = 1ll << (log2Threads / 2);

    int64_t chosenTileSize = 1ll << (llvm::Log2_64(residualTilingFactor) / 2);
    while (chosenTileSize >= 1 && ow % (yz * chosenTileSize) != 0) {
      chosenTileSize >>= 1;
    }

    if (chosenTileSize != 0) {
      wgDimSizes.front() = wgDimSizes.back() = yz;
      wgTileSizes.front() = wgTileSizes.back() = yz * chosenTileSize;
      return true;
    }
  }
  return false;
}

static LogicalResult setConvolutionConfig(linalg::LinalgOp linalgOp,
                                          const int64_t subgroupSize,
                                          const int64_t bestTilingFactor) {
  if (!isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp>(linalgOp)) {
    return failure();
  }
  const bool isNCHW = isa<linalg::Conv2DNchwFchwOp>(*linalgOp);
  const bool isNHWC = isa<linalg::Conv2DNhwcHwcfOp>(*linalgOp);

  const int ohIndex = isNHWC ? 1 : 2;
  const int owIndex = isNHWC ? 2 : 3;
  const int ocIndex = isNHWC ? 3 : 1;

  Type inputType = linalgOp.getDpsInputOperand(0)->get().getType();
  ArrayRef<int64_t> inputShape = llvm::cast<ShapedType>(inputType).getShape();
  Type outputType = linalgOp.getDpsInitOperand(0)->get().getType();
  ArrayRef<int64_t> outputShape = llvm::cast<ShapedType>(outputType).getShape();
  if (ShapedType::isDynamic(inputShape[3]) ||
      llvm::any_of(outputShape.drop_front(), ShapedType::isDynamic)) {
    return failure();
  }
  int64_t oh = outputShape[ohIndex], ow = outputShape[owIndex],
          oc = outputShape[ocIndex];

  // The core idea is to distribute the convolution dimensions to the workgroup
  // Z/Y/X dimensions, with each thread in a workgroup handling multiple vector
  // elements. We try to 1) utilize all threads in a subgroup, and 2) handle an
  // optimal tile size along each dimension.
  int64_t residualThreads = subgroupSize;
  int64_t residualTilingFactor = bestTilingFactor;

  SmallVector<int64_t, 3> workgroupSize(3, 1); // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(4, 1);

  if (isNCHW) {
    // OW -> x, OH -> y, OC -> z
    if (!distributeToOneDim(ow, /*isInnerMostDim=*/true, residualThreads,
                            residualTilingFactor, workgroupSize[0],
                            workgroupTileSizes[3]) ||
        !distributeToOneDim(oh, /*isInnerMostDim=*/false, residualThreads,
                            residualTilingFactor, workgroupSize[1],
                            workgroupTileSizes[2]) ||
        !distributeToOneDim(oc, /*isInnerMostDim=*/false, residualThreads,
                            residualTilingFactor, workgroupSize[2],
                            workgroupTileSizes[1])) {
      return failure();
    }
  } else {
    // OC -> x
    if (!distributeToOneDim(oc, /*isInnerMostDim=*/true, residualThreads,
                            residualTilingFactor, workgroupSize[0],
                            workgroupTileSizes[3]))
      return failure();

    // Deduce the configruation for the OW and OH dimension. Try to make them
    // even if possible given we typically have images with the same height
    // and width.
    const bool tileToSquare = distributeToSquare(
        oh, ow, residualThreads, residualTilingFactor,
        llvm::MutableArrayRef(workgroupSize).drop_front(),
        llvm::MutableArrayRef(workgroupTileSizes).drop_front().drop_back());

    // Otherwise treat OW and OH separately to allow them to have different
    // number of threads and tiling size.
    if (!tileToSquare) {
      if (!distributeToOneDim(ow, /*isInnerMostDim=*/false, residualThreads,
                              residualTilingFactor, workgroupSize[1],
                              workgroupTileSizes[2]) ||
          !distributeToOneDim(oh, /*isInnerMostDim=*/false, residualThreads,
                              residualTilingFactor, workgroupSize[2],
                              workgroupTileSizes[1])) {
        return failure();
      }
    }
  }
  auto pipeline = IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorize;
  TileSizesListType tileSizes;
  // Add reduction tile sizes.
  if (isNCHW)
    workgroupTileSizes.append({4, 1, 1});
  else if (isNHWC)
    workgroupTileSizes.append({1, 1, 4});
  tileSizes.push_back(workgroupTileSizes);

  // Tile along OH by size 1 to enable downsizing 2-D convolution to 1-D.
  SmallVector<int64_t> windowTileSizes(4, 0);
  windowTileSizes[ohIndex] = 1;
  tileSizes.push_back(windowTileSizes);
  auto funcOp = linalgOp->getParentOfType<func::FuncOp>();
  return setOpConfigAndEntryPointFnTranslation(funcOp, linalgOp, tileSizes,
                                               pipeline, workgroupSize);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   Operation *computeOp) {
  TargetInfo targetInfo = getTargetInfo(entryPointFn);
  // First try to see if there is a transform dialect configuration existing.
  if (succeeded(
          setTransformDialectConfig(entryPointFn, computeOp, targetInfo))) {
    return success();
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (succeeded(setContractConfig(entryPointFn, linalgOp, targetInfo))) {
      return success();
    }
    if (succeeded(setWarpReductionConfig(entryPointFn, linalgOp, targetInfo))) {
      return success();
    }
    if (succeeded(setConvolutionConfig(
            linalgOp, targetInfo.supportedSubgroupSizes.front(), 16))) {
      return success();
    }
    auto genericOp = dyn_cast<linalg::GenericOp>(computeOp);
    if (genericOp && succeeded(setTransposeConfig(entryPointFn, genericOp))) {
      return success();
    } else if (genericOp && succeeded(setArgmaxUkernelConfig(
                                entryPointFn, genericOp, targetInfo))) {
      return success();
    }
  }

  if (auto fftOp = dyn_cast<IREE::LinalgExt::FftOp>(computeOp)) {
    return setFftConfig(entryPointFn, fftOp, targetInfo);
  }
  if (auto sortOp = dyn_cast<IREE::LinalgExt::SortOp>(computeOp)) {
    return setSortConfig(entryPointFn, sortOp, targetInfo);
  }
  if (auto packOp = dyn_cast<tensor::PackOp>(computeOp)) {
    return setPackConfig(entryPointFn, packOp, targetInfo);
  }
  if (auto ukernelOp = dyn_cast<IREE::Codegen::UKernelOpInterface>(computeOp)) {
    return setUKernelConfig(entryPointFn, ukernelOp);
  }

  return setRootDefaultConfig(entryPointFn, computeOp, targetInfo);
}

// Propogate the configuration to the other ops.
// TODO(ravishankarm, thomasraoux): This is a very specific use (and
// fragile). In general, this should not be needed. Things are already tiled
// and distributed. The rest of the compilation must be structured to either
// use `TileAndFuse` or they are independent configurations that are
// determined based on the op.
static void propagateLoweringConfig(Operation *rootOperation,
                                    SmallVector<Operation *> computeOps) {
  if (IREE::Codegen::LoweringConfigAttr config =
          getLoweringConfig(rootOperation)) {
    for (auto op : computeOps) {
      if (op == rootOperation)
        continue;
      setLoweringConfig(op, config);
    }
  }
}

LogicalResult initGPULaunchConfig(ModuleOp moduleOp) {
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp)
      continue;
    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    if (getTranslationInfo(exportOp)) {
      // Currently LLVMGPU requires propagation of user lowering configs.
      for (auto op : computeOps) {
        if (getLoweringConfig(op)) {
          propagateLoweringConfig(op, computeOps);
          break;
        }
      }
      continue;
    }

    Operation *rootOperation = nullptr;

    // Find the root operation. linalg.generic and linalg.fill are not root
    // operations if there are other compute operations present.
    for (Operation *op : llvm::reverse(computeOps)) {
      if (!isa<linalg::GenericOp, linalg::FillOp>(op)) {
        rootOperation = op;
        break;
      }
      if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        // linalg.generic with `reduction` iterator types are roots as well.
        if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
          rootOperation = op;
          break;
        }
      }
    }

    if (!rootOperation) {
      for (Operation *op : llvm::reverse(computeOps)) {
        if (isa<linalg::GenericOp, linalg::FillOp>(op)) {
          rootOperation = op;
          break;
        }
      }
    }

    if (!rootOperation) {
      // No root operation found, set it to none.
      auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
          funcOp.getContext(),
          IREE::Codegen::DispatchLoweringPassPipeline::None);
      if (failed(setTranslationInfo(funcOp, translationInfo))) {
        return failure();
      }
      continue;
    }

    if (failed(setRootConfig(funcOp, rootOperation)))
      continue;

    propagateLoweringConfig(rootOperation, computeOps);
  }
  return success();
}

} // namespace mlir::iree_compiler
