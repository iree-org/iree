// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

static const unsigned kNumMaxParallelDims = 3;

namespace mlir {
namespace iree_compiler {

// TODO(ravishankarm): This needs to be put in a common place for the CPU and
// GPU backends to use.
static llvm::cl::list<unsigned> clLLVMTileSizes(
    "iree-llvm-tile-size",
    llvm::cl::desc("Set tile sizes to use for tiling Linalg operations in "
                   "LLVM code generation"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

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
static llvm::cl::opt<int> batchMatmulL2TileSize(
    "iree-codegen-llvm-batch-matmul-vector-size",
    llvm::cl::desc("linalg.batch_matmul vector tile size"), llvm::cl::init(4));

static llvm::cl::opt<int> genericOpsWorkgroupTileSize(
    "iree-codegen-llvm-generic-ops-workgroup-size",
    llvm::cl::desc(
        "linalg.generic and linalg.indexed_generic workgroup tile size"),
    llvm::cl::init(128));

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static Optional<IREE::HAL::DispatchLoweringPassPipeline> setRootConfig(
    linalg::ContractionOpInterface contractionOp) {
  assert(!hasLoweringConfig(contractionOp) &&
         "illegal to update configuration of root");
  if (contractionOp.isRowMajorMatmul()) {
    int mWorkgroupSize = matmulWorkgroupTileSize;
    int nWorkgroupSize = matmulWorkgroupTileSize;
    int mL1TileSize = matmulL1TileSize;
    int nL1TileSize = matmulL1TileSize;
    int kL1TileSize = matmulL1TileSize;
    auto lhsShape = getUntiledShape(contractionOp.lhs());
    auto rhsShape = getUntiledShape(contractionOp.rhs());
    if (!lhsShape.empty() && !rhsShape.empty()) {
      // Find largest tile size that is a multiple of the vector size.
      auto getTileSize = [](int dim, int maxSize) {
        if (dim == ShapedType::kDynamicSize) return maxSize;
        if (dim < matmulVectorSize) return matmulVectorSize.getValue();
        for (int i = std::min(maxSize, dim); i > 0; --i) {
          if (dim % i == 0 && i % matmulVectorSize == 0) {
            return i;
          }
        }
        return maxSize;
      };
      mWorkgroupSize = getTileSize(lhsShape[0], mWorkgroupSize);
      nWorkgroupSize = getTileSize(rhsShape[1], nWorkgroupSize);
      mL1TileSize = getTileSize(mWorkgroupSize, mL1TileSize);
      nL1TileSize = getTileSize(nWorkgroupSize, nL1TileSize);
      kL1TileSize = getTileSize(rhsShape[0], kL1TileSize);
    }
    TileSizesListType tileSizes = {
        {mWorkgroupSize, nWorkgroupSize},
        {mL1TileSize, nL1TileSize, kL1TileSize},
        {matmulVectorSize, matmulVectorSize, matmulVectorSize}};
    SmallVector<int64_t, 4> nativeVectorSize = {
        matmulVectorSize, matmulVectorSize, matmulVectorSize};
    IREE::HAL::LoweringConfig config =
        getConfigAttr(tileSizes, nativeVectorSize, contractionOp->getContext());
    setLoweringConfig(contractionOp, config);
    return IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization;
  }
  if (contractionOp.isRowMajorBatchMatmul()) {
    // TODO(ataei, ravishankarm): This should just use the configuration for
    // matmul above. setting the tile size to 1 for all the batch dimensions.
    TileSizesListType tileSizes = {
        {1, batchMatmulWorkgroupTileSize, batchMatmulWorkgroupTileSize},
        {1, batchMatmulL1TileSize, batchMatmulL1TileSize,
         batchMatmulL1TileSize},
        {1, batchMatmulL2TileSize, batchMatmulL2TileSize,
         batchMatmulL2TileSize}};
    SmallVector<int64_t, 4> nativeVectorSize = {
        1, batchMatmulL2TileSize, batchMatmulL2TileSize, batchMatmulL2TileSize};
    IREE::HAL::LoweringConfig config =
        getConfigAttr(tileSizes, nativeVectorSize, contractionOp->getContext());
    setLoweringConfig(contractionOp, config);
    return IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization;
  }
  return llvm::None;
}

/// Legalized the tile sizes for the first-level of tiling
/// (i.e. workgroup-level) to stay consistent with the distribution done at the
/// Flow dialect level, where the last `kNumMaxParallelDims` of the outer
/// parallel loops are distributed.
SmallVector<int64_t, 4> getDistributedWorkgroupTileSizes(
    int64_t numOuterParallelLoops, ArrayRef<int64_t> workgroupTileSizes) {
  SmallVector<int64_t, 4> distributedTileSizes =
      llvm::to_vector<4>(workgroupTileSizes);
  for (int64_t i = 0; i < numOuterParallelLoops - kNumMaxParallelDims; i++) {
    distributedTileSizes[i] = 0;
  }
  return distributedTileSizes;
}

/// Sets the lowering configuration for dispatch region with root op being a
/// generic op.
static Optional<IREE::HAL::DispatchLoweringPassPipeline> setRootConfig(
    linalg::GenericOp genericOp) {
  int64_t numOuterParallelLoops = getNumOuterParallelLoops(genericOp);
  SmallVector<int64_t, 4> workgroupTileSizes(numOuterParallelLoops,
                                             genericOpsWorkgroupTileSize);
  workgroupTileSizes = getDistributedWorkgroupTileSizes(numOuterParallelLoops,
                                                        workgroupTileSizes);
  TileSizesListType tileSizes = {workgroupTileSizes};
  IREE::HAL::LoweringConfig config =
      getConfigAttr(tileSizes, ArrayRef<int64_t>{}, genericOp->getContext());
  setLoweringConfig(genericOp, config);
  return IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization;
}

/// Sets the configuration for a linalg op that is not the root of the
/// dispatch. The configuration should use the tile sizes of the first level of
/// tiling passed in through `firstLevelTileSizes` for correctness.
LogicalResult setNonRootConfig(linalg::LinalgOp linalgOp,
                               ArrayRef<int64_t> parallelLoopTileSizes) {
  int64_t numOuterParallelLoops = getNumOuterParallelLoops(linalgOp);
  if (parallelLoopTileSizes.size() != numOuterParallelLoops) {
    return linalgOp.emitError(
        "expected non root ops to have same number of outer-parallel loops as "
        "root op");
  }
  // TODO(ravishankarm): For now just set the first level of tile-size, but need
  // to extend this to make op-specific decision.
  auto vec = llvm::to_vector<4>(parallelLoopTileSizes);
  TileSizesListType tileSizes = {vec};
  IREE::HAL::LoweringConfig config =
      getConfigAttr(tileSizes, ArrayRef<int64_t>{}, linalgOp->getContext());
  setLoweringConfig(linalgOp, config);
  return success();
}

/// Finds the root operation in the given list of linalg operations and sets its
/// configuration. Returns the root operation.
static LogicalResult setRootConfig(
    ArrayRef<linalg::LinalgOp> linalgOps,
    Optional<IREE::HAL::DispatchLoweringPassPipeline> &passPipeline,
    SmallVectorImpl<int64_t> &parallelLoopTileSizes) {
  // First iterate over all operations to find the root operations and set its
  // lowering configuration (that are not linalg.generic).
  linalg::LinalgOp rootOp = nullptr;

  auto checkOrUpdatePassPipeline =
      [&](linalg::LinalgOp linalgOp,
          Optional<IREE::HAL::DispatchLoweringPassPipeline> opPassPipeline)
      -> LogicalResult {
    if (!opPassPipeline) return success();
    if (passPipeline && passPipeline.getValue() != opPassPipeline.getValue()) {
      return linalgOp.emitError(
          "mismatch in pass-pipeline chosen for ops in dispatch region");
    }
    if (!passPipeline) {
      passPipeline = opPassPipeline.getValue();
      rootOp = linalgOp;
    }
    return success();
  };

  for (auto linalgOp : linalgOps) {
    if (!hasMarker(linalgOp, getWorkgroupMarker())) continue;
    auto opPassPipeline =
        TypeSwitch<Operation *,
                   Optional<IREE::HAL::DispatchLoweringPassPipeline>>(
            linalgOp.getOperation())
            .Case<linalg::ContractionOpInterface>(
                [&](auto op) { return setRootConfig(op); })
            .Default([](Operation *)
                         -> Optional<IREE::HAL::DispatchLoweringPassPipeline> {
              return llvm::None;
            });
    auto status = checkOrUpdatePassPipeline(linalgOp, opPassPipeline);
    if (failed(status)) {
      return status;
    }
  }

  // If no root operation found, check if the dispatch region contains a single
  // generic op and chose pipeline based on that.
  if (!passPipeline) {
    for (auto linalgOp : linalgOps) {
      if (!hasMarker(linalgOp, getWorkgroupMarker())) continue;
      auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
      if (!genericOp) continue;
      auto opPassPipeline = setRootConfig(genericOp);
      auto status = checkOrUpdatePassPipeline(linalgOp, opPassPipeline);
      if (failed(status)) {
        return status;
      }
    }
  }

  // If still no root operation, use default.
  if (!passPipeline) return success();

  parallelLoopTileSizes =
      getTileSizes(rootOp, static_cast<unsigned>(TilingLevel::WorkGroupTiles));

  // Some consistency checks.
  int64_t numOuterParallelLoops = getNumOuterParallelLoops(rootOp);
  if (parallelLoopTileSizes.size() != numOuterParallelLoops) {
    return rootOp.emitError(
        "expected as many tiles sizes as the parallel loops of the operation");
  }
  auto distributedStart =
      std::max<int64_t>(0, numOuterParallelLoops - kNumMaxParallelDims);
  ArrayRef<int64_t> parallelLoopTileSizesRef(parallelLoopTileSizes);
  // THe outer non-distributed paralle loops must be zero.
  if (distributedStart &&
      llvm::any_of(parallelLoopTileSizesRef.take_front(distributedStart),
                   [](int64_t v) -> bool { return v; })) {
    return rootOp.emitError(
        "expected non-distributed parallel loop tile size to be 0");
  }
  if (llvm::any_of(parallelLoopTileSizesRef.take_back(numOuterParallelLoops -
                                                      distributedStart),
                   [](int64_t v) -> bool { return !v; })) {
    return rootOp.emitError(
        "expected distributed parallel loop tile size to be non-zero");
  }
  return success();
}

FailureOr<IREE::HAL::DispatchLoweringPassPipeline> initCPULaunchConfig(
    ModuleOp moduleOp) {
  // The current linalg based lowering only tested for a single function case.
  auto funcOps = moduleOp.getOps<FuncOp>();
  if (!llvm::hasSingleElement(funcOps)) {
    return IREE::HAL::DispatchLoweringPassPipeline::CPUDefault;
  }
  FuncOp funcOp = *funcOps.begin();
  SmallVector<linalg::LinalgOp, 4> linalgOps;
  SmallVector<Operation *, 4> tiledLoops;
  // If there are no linalg ops, not using Linalg based lowering.
  if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops)) ||
      linalgOps.empty()) {
    return IREE::HAL::DispatchLoweringPassPipeline::CPUDefault;
  }

  Optional<IREE::HAL::DispatchLoweringPassPipeline> passPipelineOpt;
  SmallVector<int64_t> parallelLoopTileSizes;
  if (failed(
          setRootConfig(linalgOps, passPipelineOpt, parallelLoopTileSizes)) ||
      !passPipelineOpt) {
    return IREE::HAL::DispatchLoweringPassPipeline::CPUDefault;
  }
  auto passPipeline = passPipelineOpt.getValue();

  // Set the configuration of all other linalg operations that are not the root
  // operation.
  LogicalResult status = success();
  for (auto linalgOp : linalgOps) {
    if (hasLoweringConfig(linalgOp)) continue;
    status = setNonRootConfig(linalgOp, parallelLoopTileSizes);
    if (failed(status)) break;
  }
  if (failed(status)) {
    for (auto linalgOp : linalgOps) {
      eraseLoweringConfig(linalgOp);
    }
    return IREE::HAL::DispatchLoweringPassPipeline::CPUDefault;
  }

  return passPipeline;
}

}  // namespace iree_compiler
}  // namespace mlir
