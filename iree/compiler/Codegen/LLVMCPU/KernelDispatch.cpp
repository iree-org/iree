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

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(
    FuncOp entryPointFn, linalg::ContractionOpInterface contractionOp) {
  if (getLoweringConfig(contractionOp)) return success();
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
    return setOpConfigAndEntryPointFnTranslation(
        entryPointFn, contractionOp, tileSizes, nativeVectorSize,
        IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization);
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
    return setOpConfigAndEntryPointFnTranslation(
        entryPointFn, contractionOp, tileSizes, nativeVectorSize,
        IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization);
  }
  return success();
}

/// Sets the lowering configuration for dispatch region for linalg.mmt4d root op
static LogicalResult setRootConfig(FuncOp entryPointFn,
                                   linalg::Mmt4DOp mmt4dOp) {
  // TODO(ataei): These are hand tuned for some performance benchmarks for now,
  // we want to adapt the same strategy as matmul that dynamically sets tile
  // size.
  auto getWorkgroupTileSizes = [&]() -> SmallVector<int64_t> {
    if (!mmt4dWorkgroupTileSizes.empty()) {
      return SmallVector<int64_t>(mmt4dWorkgroupTileSizes.begin(),
                                  mmt4dWorkgroupTileSizes.end());
    }
    return {64, 32};
  };

  auto getL1TileSizes = [&]() -> SmallVector<int64_t> {
    if (!mmt4dL1TileSizes.empty()) {
      return SmallVector<int64_t>(mmt4dL1TileSizes.begin(),
                                  mmt4dL1TileSizes.end());
    }
    return {1, 1, 4, 4, 1, 4};
  };

  auto getVectorSizes = [&]() -> SmallVector<int64_t> {
    if (!mmt4dVectorSizes.empty()) {
      return SmallVector<int64_t>(mmt4dVectorSizes.begin(),
                                  mmt4dVectorSizes.end());
    }
    return {1, 1, 4, 4, 1, 4};
  };

  SmallVector<int64_t, 4> nativeVectorSize = getVectorSizes();

  TileSizesListType tileSizes = {getWorkgroupTileSizes(), getL1TileSizes(),
                                 nativeVectorSize};

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, mmt4dOp, tileSizes, nativeVectorSize,
      IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization);
}

/// Sets the lowering configuration for dispatch region with root op being a
/// generic op.
static LogicalResult setDefaultRootConfig(FuncOp entryPointFn, Operation *op) {
  if (getLoweringConfig(op)) return success();
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
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes, /*nativeVectorSizes=*/ArrayRef<int64_t>{},
      IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization);
}

/// Finds the root operation in the given list of linalg operations and sets its
/// configuration. Returns the root operation.
static LogicalResult setRootConfig(FuncOp entryPointFn,
                                   ArrayRef<Operation *> computeOps) {
  Operation *rootOp = nullptr;
  for (auto computeOp : computeOps) {
    if (!hasMarker(computeOp, getWorkgroupMarker())) continue;

    auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
      return TypeSwitch<Operation *, LogicalResult>(op)
          .Case<linalg::Mmt4DOp, linalg::ContractionOpInterface>(
              [&](auto op) { return setRootConfig(entryPointFn, op); })
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
      if (getLoweringConfig(computeOp)) {
        if (rootOp) {
          return computeOp->emitError(
              "unhandled multiple roots in dispatch region");
        }
        rootOp = computeOp;
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
    if (getTranslationInfo(entryPointOp)) continue;
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
      setTranslationInfo(funcOp,
                         IREE::HAL::DispatchLoweringPassPipeline::CPUDefault);
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
