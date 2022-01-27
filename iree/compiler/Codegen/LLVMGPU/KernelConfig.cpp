// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"

#include <numeric>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static constexpr unsigned cudaWarpSize = 32;

namespace {
struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  std::array<int64_t, 3> workgroupSize;
};
}  // namespace

/// Return the best combination of tile size and wg size. It will then used to
/// pick the best size aligned with the shape dimension.
static void getMatmulConfig(SmallVectorImpl<TileWorkgroupSizePair> &tileSizes) {
  // Pick tile size so that M*K and K*N dividible by wgSize * \*vecSize=*\4.
  // This way workgroup memory copy don't need to be masked. Once we support
  // masked load we can get performance out of more configuration.
  tileSizes.push_back(TileWorkgroupSizePair({{32, 128, 32}, {32, 8, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{128, 64, 8}, {16, 8, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{16, 256, 32}, {64, 2, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{8, 32, 32}, {8, 8, 1}}));

  tileSizes.push_back(TileWorkgroupSizePair({{8, 128, 4}, {32, 1, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{16, 64, 4}, {16, 2, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{1, 128, 8}, {32, 1, 1}}));
}

/// Return the best combination of tile size and wg size when using tensorcore
/// operations.
static void getTensorCoreConfig(
    SmallVectorImpl<TileWorkgroupSizePair> &tileSizes) {
  // Tile sizes are skewed towards small matmul for now. Long term the plan is
  // to not rely on hardcoded configurations.
  tileSizes.push_back(TileWorkgroupSizePair({{32, 32, 16}, {64, 2, 1}}));
}

static std::string getTargetArch(FuncOp entryPoint) {
  if (auto variantOp =
          entryPoint->getParentOfType<IREE::HAL::ExecutableVariantOp>()) {
    IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.target();
    if (auto config = targetAttr.getConfiguration()) {
      if (auto attr = config.getAs<StringAttr>("target_arch")) {
        return attr.getValue().str();
      }
    }
  }
  return "";
}

static bool supportsTensorCore(FuncOp entryPoint, linalg::LinalgOp op) {
  // Limit tensor core pipeline to matmul as not all combinations of transpose
  // are supported upstream.
  // TODO(thomasraoux): Enable batchMatmul and generic contraction.
  if (getTargetArch(entryPoint) != "sm_80" || !isa<linalg::MatmulOp>(op)) {
    return false;
  }
  // Check that we support converting any fused operation. When using the
  // tensorcore pipeline we need to be sure we can generate MMA ops otherwise
  // the code will be highly inneficent.
  bool fusedOpSupported = true;
  entryPoint.walk([&fusedOpSupported](linalg::GenericOp linalgOp) {
    for (Operation &fusedOp : linalgOp.getOps()) {
      if (!isa<arith::AddFOp, arith::MulFOp, arith::MaxFOp, arith::MinFOp,
               linalg::YieldOp, arith::DivFOp>(fusedOp)) {
        fusedOpSupported = false;
        break;
      }
    }
  });
  if (!fusedOpSupported) return false;
  return true;
}

static LogicalResult setContractConfig(FuncOp entryPoint, linalg::LinalgOp op) {
  auto setMatmulConfig =
      [&entryPoint, &op](int64_t tileX, int64_t tileY, int64_t tileK,
                         llvm::ArrayRef<int64_t> workgroupSize,
                         IREE::Codegen::DispatchLoweringPassPipeline pipeline) {
        TileSizesListType tileSizes;
        SmallVector<int64_t> ts;
        SmallVector<unsigned> partitionedLoops =
            cast<IREE::Flow::PartitionableLoopsInterface>(op.getOperation())
                .getPartitionableLoops(kNumMaxParallelDims);
        unsigned index = 0;
        // Tile all the higher parallel dimension with a size of 1 and the 2
        // most inner dimension with the tileX/tileY size.
        for (auto loopNum :
             llvm::seq<unsigned>(0, op.getNumParallelLoops() - 2)) {
          int64_t tileSize = 0;
          if (index < partitionedLoops.size() &&
              partitionedLoops[index] == loopNum) {
            tileSize = 1;
            index++;
          }
          ts.push_back(tileSize);
        }

        // Check for M loop being partitioned.
        if (index < partitionedLoops.size() &&
            partitionedLoops[index] == op.getNumParallelLoops() - 2) {
          index++;
        } else {
          // M dim isnt partitioned.
          tileX = 0;
        }

        // Check for N loop being partitioned.
        if (index < partitionedLoops.size() &&
            partitionedLoops[index] == op.getNumParallelLoops() - 1) {
          index++;
        } else {
          // N dim isnt partitioned.
          tileY = 0;
        }

        ts.append({tileX, tileY});
        // Tile all the reduction dimensions.
        ts.append(op.getNumReductionLoops(), tileK);
        tileSizes.push_back(ts);  // Workgroup level.
        return setOpConfigAndEntryPointFnTranslation(
            entryPoint, op, tileSizes,
            /*nativeVectorSizes=*/ArrayRef<int64_t>{}, pipeline, workgroupSize);
      };
  // Infer the MxN size of the matmul based on operands and indexing maps.
  auto lhsShape = getUntiledShape(op.getInputOperand(0)->get());
  auto rhsShape = getUntiledShape(op.getInputOperand(1)->get());
  int64_t sizeM = ShapedType::kDynamicSize;
  int64_t sizeN = ShapedType::kDynamicSize;
  int64_t sizeK = ShapedType::kDynamicSize;
  auto outputMap = op.getTiedIndexingMap(op.getOutputOperand(0));
  for (unsigned i = 0; i < lhsShape.size(); i++) {
    if (op.getTiedIndexingMap(op.getInputOperand(0)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 2)) {
      sizeM = lhsShape[i];
      break;
    }
  }
  for (unsigned i = 0; i < rhsShape.size(); i++) {
    if (op.getTiedIndexingMap(op.getInputOperand(1)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 1)) {
      sizeN = rhsShape[i];
      break;
    }
  }
  SmallVector<unsigned> exprs;
  op.getReductionDims(exprs);
  if (exprs.size() == 1) {
    for (unsigned i = 0; i < lhsShape.size(); i++) {
      if (op.getTiedIndexingMap(op.getInputOperand(0)).getDimPosition(i) ==
          exprs[0]) {
        sizeK = lhsShape[i];
        break;
      }
    }
  }
  bool isStaticSize = sizeM != ShapedType::kDynamicSize &&
                      sizeN != ShapedType::kDynamicSize &&
                      sizeK != ShapedType::kDynamicSize;
  if (isStaticSize) {
    /// Try tensorcore config first.
    if (supportsTensorCore(entryPoint, op)) {
      SmallVector<TileWorkgroupSizePair> TCtileSizeConfig;
      getTensorCoreConfig(TCtileSizeConfig);
      // Pick the best configuration where the original shape is aligned on the
      // tile size.
      for (TileWorkgroupSizePair &config : TCtileSizeConfig) {
        if (sizeK % config.tileSize[2] == 0 &&
            sizeN % config.tileSize[1] == 0 &&
            sizeM % config.tileSize[0] == 0) {
          return setMatmulConfig(config.tileSize[0], config.tileSize[1],
                                 config.tileSize[2], config.workgroupSize,
                                 IREE::Codegen::DispatchLoweringPassPipeline::
                                     LLVMGPUMatmulTensorCore);
        }
      }
    }
    // Special case for very small matrices.
    if (sizeM * sizeN <= cudaWarpSize) {
      return setMatmulConfig(
          sizeN, sizeM, 4, {sizeM, sizeN, 1},
          IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt);
    }
    // simt matmul case
    SmallVector<TileWorkgroupSizePair> tileSizeConfig;
    // Query the best configuration.
    getMatmulConfig(tileSizeConfig);
    // Pick the best configuration where the original shape is aligned on the
    // tile size.
    for (TileWorkgroupSizePair &config : tileSizeConfig) {
      if (sizeN % config.tileSize[1] == 0 && sizeM % config.tileSize[0] == 0) {
        return setMatmulConfig(
            config.tileSize[0], config.tileSize[1], config.tileSize[2],
            config.workgroupSize,
            IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt);
      }
    }
  }
  // If we haven't found any config, fall back to default config.
  int64_t tileX = 2;
  int64_t tileY = 256;
  int64_t tileK = 4;
  SmallVector<int64_t, 3> workgroupSize = {2 * cudaWarpSize, 1, 1};
  return setMatmulConfig(
      tileX, tileY, tileK, workgroupSize,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt);
}

static LogicalResult setFftConfig(FuncOp entryPoint,
                                  IREE::LinalgExt::FftOp op) {
  auto interfaceOp = cast<IREE::Flow::PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  unsigned loopDepth = partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSize(loopDepth, 0);
  SmallVector<int64_t, 3> workgroupSize = {cudaWarpSize, 1, 1};

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
      entryPoint, op, tileSizes, /*nativeVectorSizes=*/ArrayRef<int64_t>{},
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
      workgroupSize);
}

static LogicalResult setSortConfig(FuncOp entryPoint, Operation *op) {
  TileSizesListType tileSizes;
  auto interfaceOp = cast<IREE::Flow::PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, tileSizes, /*nativeVectorSizes=*/ArrayRef<int64_t>{},
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
        {1, 1, 1});
  }
  size_t numLoops = partitionedLoops.back() + 1;
  // To get peak occupancy we need a workgroup size of at least two warps
  std::array<int64_t, 3> workgroupSize = {2 * cudaWarpSize, 1, 1};
  SmallVector<int64_t, 4> workgroupTileSizes(numLoops, 1);
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
  tileSizes.emplace_back(std::move(workgroupTileSizes));  // Workgroup level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, /*nativeVectorSizes=*/ArrayRef<int64_t>{},
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
      workgroupSize);
}

// Basic default properties for linalg ops that haven't been tuned.
static LogicalResult setRootDefaultConfig(FuncOp entryPoint, Operation *op) {
  IREE::Codegen::DispatchLoweringPassPipeline passPipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute;
  TileSizesListType tileSizes;
  auto interfaceOp = cast<IREE::Flow::PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, tileSizes, /*nativeVectorSizes=*/ArrayRef<int64_t>{},
        passPipeline, {1, 1, 1});
  }

  size_t numLoops = partitionedLoops.back() + 1;
  // To get peak occupancy we need a workgroup size of at least two warps
  std::array<int64_t, 3> workgroupSize = {2 * cudaWarpSize, 1, 1};
  unsigned vectorSize = 4;
  SmallVector<int64_t, 4> workgroupTileSizes(numLoops, 1);
  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
    }
  }

  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    for (auto outputOperand : enumerate(genericOp.getOutputOperands())) {
      if (!genericOp.getTiedIndexingMap(outputOperand.value())
               .isProjectedPermutation()) {
        vectorSize = 1;
        break;
      }
      SmallVector<int64_t> shape = getUntiledResultShape(
          cast<linalg::LinalgOp>(op), outputOperand.index());
      if (llvm::any_of(shape, ShapedType::isDynamic)) {
        vectorSize = 1;
        break;
      }
      int64_t problemSize = std::accumulate(
          shape.begin(), shape.end(), 1,
          [](const int64_t &a, const int64_t &b) { return a * b; });
      if ((problemSize / (cudaWarpSize * vectorSize)) < 64) {
        vectorSize = 1;
        break;
      }
    }
  }
  // Pick a vectorSize of 1 for op that we know won't get vectorizedd.
  // TODO(thomasraoux): This could be improved by checking if the linalg op
  // would fail vectorization.
  if (!isa<linalg::LinalgOp>(op)) vectorSize = 1;

  // Set the inner most parallel loop to `lowerTs`.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.count(depth - 1)) {
      workgroupTileSizes[depth - 1] = workgroupSize[0] * vectorSize;
      break;
    }
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    // Tile reduction dimension to 4 to allow doing load4 if the reduction size
    // is the most inner dimension.
    workgroupTileSizes.append(linalgOp.getNumReductionLoops(), 4);
  }
  tileSizes.emplace_back(std::move(workgroupTileSizes));  // Workgroup level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, /*nativeVectorSizes=*/ArrayRef<int64_t>{},
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorize,
      workgroupSize);
}

/// Propagate the configuration annotated in the incoming IR.
static LogicalResult setUserConfig(
    FuncOp entryPointFn, Operation *computeOp,
    IREE::Codegen::CompilationInfoAttr compilationInfo) {
  if (auto translationInfo = getTranslationInfo(entryPointFn)) {
    return computeOp->emitOpError(
        "multiple ops within dispatch trying to set the translation "
        "info");
  }

  SmallVector<int64_t> workgroupSize = compilationInfo.getWorkgroupSizeVals();
  setTranslationInfo(entryPointFn, compilationInfo.getTranslationInfo(),
                     workgroupSize);
  setLoweringConfig(computeOp, compilationInfo.getLoweringConfig());
  eraseCompilationInfo(computeOp);
  return success();
}

static LogicalResult setRootConfig(FuncOp entryPointFn, Operation *computeOp) {
  if (IREE::Codegen::CompilationInfoAttr compilationInfo =
          getCompilationInfo(computeOp)) {
    // If the op already has a lowering config coming from the IR use this and
    // bypass the heuristic.
    return setUserConfig(entryPointFn, computeOp, compilationInfo);
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (linalg::isaContractionOpInterface(linalgOp) &&
        linalgOp.getNumParallelLoops() >= 2) {
      return setContractConfig(entryPointFn, linalgOp);
    }
  }
  if (auto fftOp = dyn_cast<IREE::LinalgExt::FftOp>(computeOp)) {
    return setFftConfig(entryPointFn, fftOp);
  }
  if (auto sortOp = dyn_cast<IREE::LinalgExt::SortOp>(computeOp)) {
    return setSortConfig(entryPointFn, sortOp);
  }
  return setRootDefaultConfig(entryPointFn, computeOp);
}

namespace mlir {
namespace iree_compiler {

LogicalResult initGPULaunchConfig(ModuleOp moduleOp) {
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps =
      getAllEntryPoints(moduleOp);

  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    auto entryPointOp = entryPointOps.lookup(funcOp.getName());
    if (!entryPointOp) continue;
    if (getTranslationInfo(entryPointOp)) continue;
    SmallVector<Operation *> computeOps;
    SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      return funcOp.emitOpError("failed to get compute ops");
    }

    if (computeOps.empty()) {
      std::array<int64_t, 3> workgroupSize = {1, 1, 1};
      SmallVector<int64_t> workloadPerWorkgroup;
      if (!tiledLoops.empty()) {
        // If the tiled loops are not empty then this could be a corner case of
        // tensor.insert_slice being tiled and distributed, that just shows up
        // as a `flow.dispatch.tensor.load` and a `flow.dispatch.tensor.store`.
        // For now just treat the tiled loops not being empty as an indicator of
        // that. Need a better way of information flow from flow dialect to hal.
        workgroupSize[0] = cudaWarpSize;
        workloadPerWorkgroup.resize(tiledLoops.size(), 1);
        workloadPerWorkgroup.front() = cudaWarpSize * 4;
      }
      // TODO(ravishankarm): Maybe this should just return without setting
      // anything. Without any compute ops, this shouldnt be using tile and
      // distribute.
      setTranslationInfo(
          funcOp,
          IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
          workloadPerWorkgroup, workgroupSize);
      continue;
    }

    Operation *rootOperation = nullptr;
    // Find the root operation. linalg.generic, linalg.fill and linalg.copy are
    // not root operations if there are other compute operations present.
    for (Operation *op : llvm::reverse(computeOps)) {
      if (!isa<linalg::GenericOp, linalg::FillOp, linalg::CopyOp>(op)) {
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
        if (isa<linalg::GenericOp, linalg::FillOp, linalg::CopyOp>(op)) {
          rootOperation = op;
          break;
        }
      }
    }

    if (!rootOperation) {
      // TODO(ravishankarm): Maybe this should just return without setting
      // anything. Without any compute ops, this shouldnt be using tile and
      // distribute.
      setTranslationInfo(
          funcOp,
          IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
          /*workloadPerWorkgroup=*/{}, {1, 1, 1});
      continue;
    }
    if (failed(setRootConfig(funcOp, rootOperation))) continue;

    // Propogate the configuration to the other ops.
    // TODO(ravishankarm, thomasraoux): This is a very specific use (and
    // fragile). In general, this should not be needed. Things are already tiled
    // and distributed. The rest of the compilation must be structured to either
    // use `TileAndFuse` or they are independent configurations that are
    // determined based on the op.
    IREE::Codegen::LoweringConfigAttr config = getLoweringConfig(rootOperation);
    for (auto op : computeOps) {
      if (op == rootOperation) continue;
      setLoweringConfig(op, config);
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
