// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
  tileSizes.push_back(TileWorkgroupSizePair({{64, 128, 8}, {32, 4, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{128, 64, 8}, {16, 8, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{16, 256, 16}, {64, 2, 1}}));

  tileSizes.push_back(TileWorkgroupSizePair({{8, 128, 4}, {32, 1, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{16, 64, 4}, {16, 2, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{1, 128, 8}, {32, 1, 1}}));
}

static LogicalResult setContractConfig(FuncOp entryPoint, linalg::LinalgOp op) {
  TileSizesListType tileSizes;
  // Infer the MxN size of the matmul based on operands and indexing maps.
  auto lhsShape = getUntiledShape(op.getInputOperand(0)->get());
  auto rhsShape = getUntiledShape(op.getInputOperand(1)->get());
  int64_t sizeM = -1;
  int64_t sizeN = -1;
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
  // Default tile size and workgroup size.
  int64_t tileX = 2;
  int64_t tileY = 256;
  int64_t tileK = 4;
  SmallVector<int64_t, 3> workgroupSize = {2 * cudaWarpSize, 1, 1};
  // Special case for very small matrices.
  if (sizeM * sizeN <= cudaWarpSize) {
    tileX = sizeN;
    tileY = sizeM;
    workgroupSize = {sizeM, sizeN, 1};
  }
  SmallVector<TileWorkgroupSizePair> tileSizeConfig;
  // Query the best configuration.
  getMatmulConfig(tileSizeConfig);
  // Pick the best configuration where the original shape is aligned on the tile
  // size.
  for (TileWorkgroupSizePair &config : tileSizeConfig) {
    if (sizeN % config.tileSize[1] == 0 && sizeM % config.tileSize[0] == 0) {
      tileX = config.tileSize[0];
      tileY = config.tileSize[1];
      tileK = config.tileSize[2];
      workgroupSize.assign(config.workgroupSize.begin(),
                           config.workgroupSize.end());
      break;
    }
  }
  // Currently just a basic tile size to enable tiling and vectorization.
  // TODO: pick a more efficient tile size and tile at subgroup level.
  SmallVector<int64_t, 4> ts;
  // Tile all the higher parallel dimension with a size of 1 and the 2 most
  // inner dimension with the tileX/tileY size.
  ts.append(op.getNumParallelLoops() - 2, 1);
  ts.append({tileX, tileY});
  // Tile all the reduction dimensions.
  ts.append(op.getNumReductionLoops(), tileK);
  tileSizes.push_back(ts);  // Workgroup level.
  tileSizes.push_back({});  // Subgroup level.
  // At the thread level only tile parallel loops.
  SmallVector<int64_t, 4> invocationLevelTs(op.getNumParallelLoops() - 2, 1);
  invocationLevelTs.append(
      {tileX / workgroupSize[1], tileY / workgroupSize[0]});
  tileSizes.push_back(invocationLevelTs);  // Thread level.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, /*nativeVectorSizes=*/ArrayRef<int64_t>{},
      IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt,
      workgroupSize);
}

// Basic default properties for linalg ops that haven't been tuned.
static LogicalResult setRootDefaultConfig(FuncOp entryPoint, Operation *op) {
  IREE::HAL::DispatchLoweringPassPipeline passPipeline =
      IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUDistribute;
  TileSizesListType tileSizes;
  SmallVector<unsigned> partitionedLoops = getPartitionedLoops(op);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, tileSizes, /*nativeVectorSizes=*/ArrayRef<int64_t>{},
        passPipeline, {1, 1, 1});
  }

  size_t numLoops = partitionedLoops.back() + 1;

  std::array<int64_t, 3> workgroupSize = {cudaWarpSize, 1, 1};
  unsigned vectorSize = 4;
  SmallVector<int64_t, 4> workgroupTileSizes(numLoops, 1),
      threadTileSizes(numLoops, 1);
  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
      threadTileSizes[depth] = 0;
    }
  }

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  bool outputSizeIsProblemSize =
      genericOp &&
      llvm::all_of(genericOp.getOutputOperands(),
                   [&genericOp](OpOperand *outputOperand) {
                     return genericOp.getTiedIndexingMap(outputOperand)
                         .isProjectedPermutation();
                   });
  if (outputSizeIsProblemSize) {
    // Calculate the problem size to adjust the tile size.
    int64_t problemSize = 1;
    entryPoint.walk([&problemSize](IREE::Flow::DispatchTensorStoreOp storeOp) {
      ArrayRef<int64_t> shape = storeOp.target()
                                    .getType()
                                    .cast<IREE::Flow::DispatchTensorType>()
                                    .getShape();
      int64_t prod = 1;
      for (int64_t dim : shape) prod *= dim;
      problemSize = std::max(prod, problemSize);
    });
    // If the problem size is too small or if the op cannot be vectorized,
    // reduce the vector size to prevent bad memory access patterns.
    if ((problemSize / (cudaWarpSize * vectorSize)) < 64) vectorSize = 1;
  }
  // Pick a vectorSize of 1 for op that we know won't get vectorizedd.
  // TODO(thomasraoux): This could be improved by checking if the linalg op
  // would fail vectorization.
  if (!isa<linalg::LinalgOp>(op)) vectorSize = 1;

  // Set the inner most parallel loop to `lowerTs`.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.count(depth - 1)) {
      workgroupTileSizes[depth - 1] = cudaWarpSize * vectorSize;
      threadTileSizes[depth - 1] = vectorSize;
      break;
    }
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    // Tile reduction dimension to 1. Using a large tile size may allow better
    // scheduling and could help in case one of the input has transpose.
    // TODO(thomasraoux): improve the heuristic.
    workgroupTileSizes.append(linalgOp.getNumReductionLoops(), 1);
  }
  tileSizes.emplace_back(std::move(workgroupTileSizes));  // Workgroup level
  tileSizes.push_back({});                                // Subgroup level.
  tileSizes.emplace_back(std::move(threadTileSizes));     // Thread level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, /*nativeVectorSizes=*/ArrayRef<int64_t>{},
      IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUVectorize, workgroupSize);
}

static LogicalResult setRootConfig(FuncOp entryPointFn, Operation *computeOp) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (linalg::isaContractionOpInterface(linalgOp) &&
        linalgOp.getNumParallelLoops() >= 2) {
      return setContractConfig(entryPointFn, linalgOp);
    }
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
    SmallVector<Operation *, 4> computeOps;
    SmallVector<Operation *, 4> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      return funcOp.emitOpError("failed to get compute ops");
    }

    if (computeOps.empty()) {
      // TODO(ravishankarm): Maybe this should just return without setting
      // anything. Without any compute ops, this shouldnt be using tile and
      // distribute.
      setTranslationInfo(
          funcOp, IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUDistribute,
          {1, 1, 1}, /*workloadPerWorkgroup=*/{});
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
          funcOp, IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUDistribute,
          {1, 1, 1}, /*workloadPerWorkgroup=*/{});
      continue;
    }
    if (failed(setRootConfig(funcOp, rootOperation))) continue;

    // Propogate the configuration to the other ops.
    // TODO(ravishankarm, thomasraoux): This is a very specific use (and
    // fragile). In general, this should not be needed. Things are already tiled
    // and distributed. The rest of the compilation must be structured to either
    // use `TileAndFuse` or they are independent configurations that are
    // determined based on the op.
    IREE::HAL::LoweringConfig config = getLoweringConfig(rootOperation);
    for (auto op : computeOps) {
      if (op == rootOperation) continue;
      setLoweringConfig(op, config);
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
