
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"

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
static LogicalResult setRootConfig(
    linalg::ContractionOpInterface contractionOp) {
  if (hasLoweringConfig(contractionOp))
    return contractionOp.emitError("overriding previously set configuration");
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
    return success();
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
    return success();
  }
  return failure();
}

/// Sets the lowering configuration for dispatch region with root op being a
/// generic op.
LogicalResult setRootConfig(linalg::GenericOp genericOp) {
  int64_t numOuterParallelLoops = getNumOuterParallelLoops(genericOp);
  SmallVector<int64_t, 4> workgroupTileSizes(numOuterParallelLoops,
                                             genericOpsWorkgroupTileSize);
  SmallVector<int64_t, 4> innerTileSizes(numOuterParallelLoops, 1);
  // At the HAL level only the inner `kNumMaxParallelDims` parallel loops are
  // tiled. Set the tile size as 0 for all the parallel loops that are not tiled
  // at HAL level
  for (int64_t i = 0; i < numOuterParallelLoops - kNumMaxParallelDims; i++) {
    workgroupTileSizes[i] = 0;
  }
  TileSizesListType tileSizes = {workgroupTileSizes};
  IREE::HAL::LoweringConfig config =
      getConfigAttr(tileSizes, ArrayRef<int64_t>{}, genericOp->getContext());
  setLoweringConfig(genericOp, config);
  return success();
}

/// Sets the configuration for a linalg op that is not the root of the
/// dispatch. The configuration should use the tile sizes of the first level of
/// tiling passed in through `firstLevelTileSizes` for correctness.
LogicalResult setNonRootConfig(linalg::LinalgOp linalgOp,
                               ArrayRef<int64_t> firstLevelTileSizes) {
  auto vec = llvm::to_vector<4>(firstLevelTileSizes);
  int64_t numOuterParallelLoops = getNumOuterParallelLoops(linalgOp);
  vec.resize(numOuterParallelLoops, 0);
  // TODO(ravishankarm): For now just set the first level of tile-size, but need
  // to extend this to make op-specific decision.
  TileSizesListType tileSizes = {vec};
  IREE::HAL::LoweringConfig config =
      getConfigAttr(tileSizes, ArrayRef<int64_t>{}, linalgOp->getContext());
  setLoweringConfig(linalgOp, config);
  return success();
}

/// Finds the root operation in the given list of linalg operations and sets its
/// configuration. Returns the root operation.
static FailureOr<linalg::LinalgOp> setRootConfig(
    ArrayRef<linalg::LinalgOp> linalgOps) {
  linalg::LinalgOp rootOperation = nullptr;
  // First iterate over all operations to find the root operations and set its
  // lowering configuration (that are not linalg.generic).
  for (auto linalgOp : linalgOps) {
    if (!hasMarker(linalgOp, getWorkgroupMarker())) continue;
    auto status =
        TypeSwitch<Operation *, LogicalResult>(linalgOp.getOperation())
            .Case<linalg::ContractionOpInterface>(
                [&](auto op) { return setRootConfig(op); })
            .Default([](Operation *) { return failure(); });
    if (succeeded(status)) {
      if (rootOperation) {
        return static_cast<LogicalResult>(linalgOp.emitError(
            "unhandled multiple root operations in dispatch region"));
      }
      rootOperation = linalgOp;
    }
  }

  // If no root operation found, check if the dispatch region contains a single
  // generic op and set its configuration.
  if (!rootOperation) {
    for (auto linalgOp : linalgOps) {
      if (!hasMarker(linalgOp, getWorkgroupMarker())) continue;
      auto status =
          TypeSwitch<Operation *, LogicalResult>(linalgOp.getOperation())
              .Case<linalg::GenericOp>(
                  [&](auto op) { return setRootConfig(op); })
              .Default([](Operation *) { return failure(); });
      if (succeeded(status)) {
        if (rootOperation) {
          return static_cast<LogicalResult>(linalgOp.emitError(
              "unhandled multiple root operations in dispatch region"));
        }
        rootOperation = linalgOp;
      }
    }
  }
  return rootOperation;
}

LogicalResult initCPULaunchConfig(ArrayRef<linalg::LinalgOp> linalgOps) {
  if (linalgOps.empty()) return success();

  SmallVector<int64_t, 4> firstLevelTileSizes;
  if (!clLLVMTileSizes.empty()) {
    firstLevelTileSizes.assign(clLLVMTileSizes.begin(), clLLVMTileSizes.end());
  } else {
    auto rootOperation = setRootConfig(linalgOps);
    if (failed(rootOperation)) return failure();
    // If root operation is null. Nothing to do.
    if (!rootOperation.getValue()) return success();
    firstLevelTileSizes = getTileSizes(
        *rootOperation, static_cast<unsigned>(TilingLevel::WorkGroupTiles));
  }

  // Set the configuration of all other linalg operations that are not the root
  // operation.
  for (auto linalgOp : linalgOps) {
    if (hasLoweringConfig(linalgOp)) continue;
    auto status = setNonRootConfig(linalgOp, firstLevelTileSizes);
    if (failed(status)) {
      return failure();
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
