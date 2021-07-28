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

static llvm::cl::opt<int> defaultWorkgroupTileSize(
    "iree-codegen-llvm-generic-ops-workgroup-size",
    llvm::cl::desc(
        "linalg.generic and linalg.indexed_generic workgroup tile size"),
    llvm::cl::init(128));

/// Usually the tile sizes for the first level of tiling decides the workgroup
/// size for the dispatch on the CPU backend. This is a general helper that
/// converts tile sizes of the first level into workgroup sizes.
static SmallVector<int64_t, 3> getWorkloadPerWorkgroup(
    ArrayRef<int64_t> distributedTileSizes) {
  if (distributedTileSizes.size() > kNumMaxParallelDims) {
    distributedTileSizes = distributedTileSizes.take_back(kNumMaxParallelDims);
  }
  return llvm::to_vector<3>(llvm::reverse(distributedTileSizes));
}

/// Sets the translation info on the `hal.executable.entry_point` op
/// corresponding to the `entryPointFn`. Returns failure if a translation info
/// is already set on the entry point op and is incompatible with what is being
/// set.
static LogicalResult setTranslationInfo(
    FuncOp entryPointFn, IREE::HAL::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workloadPerWorkgroup) {
  auto entryPointOp = getEntryPoint(entryPointFn);
  auto translationInfo = buildTranslationInfo(
      passPipeline, workloadPerWorkgroup, entryPointFn.getContext());
  return setTranslationInfo(entryPointOp, translationInfo);
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, linalg::ContractionOpInterface contractionOp) {
  if (hasLoweringConfig(entryPointFn)) return success();
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
    IREE::HAL::LoweringConfig config = buildConfigAttr(
        tileSizes, nativeVectorSize, contractionOp->getContext());
    setLoweringConfig(contractionOp, config);
    return setTranslationInfo(
        entryPointFn, IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization,
        getWorkloadPerWorkgroup(tileSizes[0]));
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
    IREE::HAL::LoweringConfig config = buildConfigAttr(
        tileSizes, nativeVectorSize, contractionOp->getContext());
    setLoweringConfig(contractionOp, config);
    return setTranslationInfo(
        entryPointFn, IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization,
        getWorkloadPerWorkgroup(tileSizes[0]));
  }
  return success();
}

/// Returns the loops that are partitioned during dispatch region formations, in
/// order, i.e. starting from the outer-most to innermost.
/// Note that this is the same method that is used at the Flow dispatch region
/// formation to tile and distribute the ops.
static SmallVector<unsigned> getPartitionedLoops(Operation *op) {
  SmallVector<unsigned> partitionedLoops;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    size_t numOuterParallelLoops = getNumOuterParallelLoops(linalgOp);
    partitionedLoops =
        llvm::to_vector<4>(llvm::seq<unsigned>(0, numOuterParallelLoops));
    if (partitionedLoops.size() > kNumMaxParallelDims) {
      partitionedLoops.erase(
          partitionedLoops.begin(),
          std::next(partitionedLoops.begin(),
                    numOuterParallelLoops - kNumMaxParallelDims));
    }
    return partitionedLoops;
  }
  if (auto tilableOp = dyn_cast<linalg_ext::TiledOpInterface>(op)) {
    return tilableOp.getPartitionableLoops(kNumMaxParallelDims);
  }
  return {};
}

/// Sets the lowering configuration for dispatch region with root op being a
/// generic op.
static LogicalResult setDefaultRootConfig(FuncOp entryPointFn, Operation *op) {
  if (hasLoweringConfig(op)) return success();
  auto partitionedLoops = getPartitionedLoops(op);
  if (partitionedLoops.empty()) {
    // Return success without doing anything. Eventually default will be used.
    return success();
  }
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
  TileSizesListType tileSizes = {workgroupTileSizes};
  IREE::HAL::LoweringConfig config =
      buildConfigAttr(tileSizes, ArrayRef<int64_t>{}, op->getContext());
  setLoweringConfig(op, config);
  return setTranslationInfo(
      entryPointFn, IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization,
      getWorkloadPerWorkgroup(tileSizes[0]));
}

/// Finds the root operation in the given list of linalg operations and sets its
/// configuration. Returns the root operation.
static LogicalResult setRootConfig(FuncOp entryPointFn,
                                   ArrayRef<Operation *> computeOps) {
  Operation *rootOp = nullptr;
  for (auto computeOp : computeOps) {
    if (!hasMarker(computeOp, getWorkgroupMarker())) continue;
    auto status = TypeSwitch<Operation *, LogicalResult>(computeOp)
                      .Case<linalg::ContractionOpInterface>([&](auto op) {
                        return setRootConfig(entryPointFn, op);
                      })
                      .Default([](Operation *) { return success(); });
    if (failed(status)) {
      return status;
    }
    if (hasLoweringConfig(computeOp)) {
      if (rootOp) {
        return computeOp->emitError(
            "unhandled multiple roots in dispatch region");
      }
      rootOp = computeOp;
      continue;
    }
  }

  // If no root operation found, check if the dispatch region contains a single
  // generic op and chose pipeline based on that.
  if (!rootOp) {
    for (auto computeOp : computeOps) {
      if (!hasMarker(computeOp, getWorkgroupMarker())) continue;
      // Ignore fill ops. They never end up in their own dispatch, so are never
      // root ops.
      if (isa<linalg::FillOp>(computeOp)) continue;
      if (failed(setDefaultRootConfig(entryPointFn, computeOp))) {
        return failure();
      }
      if (hasLoweringConfig(computeOp)) {
        if (rootOp) {
          return computeOp->emitError(
              "unhandled multiple roots in dispatch region");
        }
        rootOp = computeOp;
        continue;
      }
    }
  }
  return success();
}

LogicalResult initCPULaunchConfig(ModuleOp moduleOp) {
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps =
      getAllEntryPoints(moduleOp);
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    auto entryPointOp = entryPointOps.lookup(funcOp.getName());
    if (!entryPointOp) continue;
    SmallVector<Operation *, 4> computeOps;
    SmallVector<Operation *, 4> tiledLoops;
    // If there are no linalg ops, not using Linalg based lowering.
    if (succeeded(getComputeOps(funcOp, computeOps, tiledLoops)) &&
        !computeOps.empty()) {
      if (failed(setRootConfig(funcOp, computeOps))) {
        return failure();
      }
    }

    // If the function entry point already doesnt have a lowering info attribute
    // on it, just add the default.
    if (!getTranslationInfo(entryPointOp)) {
      if (failed(setTranslationInfo(
              funcOp, IREE::HAL::DispatchLoweringPassPipeline::CPUDefault,
              {}))) {
        return failure();
      }
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
