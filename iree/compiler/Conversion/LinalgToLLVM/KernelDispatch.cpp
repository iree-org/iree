
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

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"

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
    "iree-codegen-linalg-to-llvm-kernel-dispatch-matmul-workgroup-tile-size",
    llvm::cl::desc(
        "linalg.matmul tile size for workgroups spliting of M, N dimension"),
    llvm::cl::init(64));

static llvm::cl::opt<int> matmulL1TileSize(
    "iree-codegen-linalg-to-llvm-kernel-dispatch-matmul-l1-tile-size",
    llvm::cl::desc(
        "linalg.matmul tile size for workgroups spliting of M, N dimension"),
    llvm::cl::init(32));

static llvm::cl::opt<int> matmulL2TileSize(
    "iree-codegen-linalg-to-llvm-kernel-dispatch-matmul-l2-tile-size",
    llvm::cl::desc(
        "linalg.matmul tile size for workgroups spliting of M, N dimension"),
    llvm::cl::init(4));

static llvm::cl::opt<int> batchMatmulWorkgroupTileSize(
    "iree-codegen-linalg-to-llvm-kernel-dispatch-batch-matmul-workgroup-tile-"
    "size",
    llvm::cl::desc(
        "linalg.matmul tile size for workgroups spliting of M, N dimension"),
    llvm::cl::init(32));

static llvm::cl::opt<int> batchMatmulL1TileSize(
    "iree-codegen-linalg-to-llvm-kernel-dispatch-batch-matmul-l1-tile-size",
    llvm::cl::desc(
        "linalg.matmul tile size for workgroups spliting of M, N dimension"),
    llvm::cl::init(16));

static llvm::cl::opt<int> batchMatmulL2TileSize(
    "iree-codegen-linalg-to-llvm-kernel-dispatch-batch-matmul-l2-tile-size",
    llvm::cl::desc(
        "linalg.matmul tile size for workgroups spliting of M, N dimension"),
    llvm::cl::init(4));

namespace {
template <TilingLevel tilingLevel>
llvm::SmallVector<int64_t, 4> getTileSizes(Operation *op) {
  if (auto contractionOp = dyn_cast<linalg::ContractionOpInterface>(op)) {
    if (contractionOp.isRowMajorMatmul()) {
      switch (tilingLevel) {
        case TilingLevel::WorkGroupTiles: {
          return {matmulWorkgroupTileSize, matmulWorkgroupTileSize};
        }
        case TilingLevel::Level1Tiles: {
          return {matmulL1TileSize, matmulL1TileSize, matmulL1TileSize};
        }
        case TilingLevel::Level2Tiles: {
          return {matmulL2TileSize, matmulL2TileSize, matmulL2TileSize};
        }
      }
    }
    if (contractionOp.isRowMajorBatchMatmul()) {
      switch (tilingLevel) {
        case TilingLevel::WorkGroupTiles: {
          return {1, batchMatmulWorkgroupTileSize,
                  batchMatmulWorkgroupTileSize};
        }
        case TilingLevel::Level1Tiles: {
          return {1, batchMatmulL1TileSize, batchMatmulL1TileSize,
                  batchMatmulL1TileSize};
        }
        case TilingLevel::Level2Tiles: {
          return {1, batchMatmulL2TileSize, batchMatmulL2TileSize,
                  batchMatmulL2TileSize};
        }
      }
    }
  }

  return {1, 1, 1};
}
}  // namespace

#define DEFINE_TILE_SIZE_FN(tilingLevel)                                      \
  template <>                                                                 \
  SmallVector<Value, 4> TileSizeFn::get<tilingLevel>(OpBuilder & builder,     \
                                                     Operation * operation) { \
    auto tileSizes = getTileSizes<tilingLevel>(operation);                    \
    if (tileSizes.empty()) return {};                                         \
    SmallVector<Value, 4> tileSizesVal;                                       \
    tileSizesVal.reserve(tileSizes.size());                                   \
    for (auto val : tileSizes) {                                              \
      tileSizesVal.push_back(                                                 \
          builder.create<ConstantIndexOp>(operation->getLoc(), val));         \
    }                                                                         \
    return tileSizesVal;                                                      \
  }

DEFINE_TILE_SIZE_FN(TilingLevel::WorkGroupTiles)
DEFINE_TILE_SIZE_FN(TilingLevel::Level1Tiles)
DEFINE_TILE_SIZE_FN(TilingLevel::Level2Tiles)

#undef DEFINE_TILE_SIZE_FN

Optional<LaunchConfig> initCPULaunchConfig(
    MLIRContext *context, const linalg::LinalgDependenceGraph &dependenceGraph,
    ArrayRef<linalg::LinalgOp> linalgOps) {
  LaunchConfig config;

  Optional<linalg::LinalgOp> rootOperation = llvm::None;
  for (auto linalgOp : linalgOps) {
    if (auto contractionOp =
            dyn_cast<linalg::ContractionOpInterface>(linalgOp.getOperation())) {
      if (!contractionOp.isRowMajorMatmul() &&
          !contractionOp.isRowMajorBatchMatmul()) {
        continue;
      }
      if (rootOperation) {
        contractionOp.emitError(
            "unhandled multiple root operations in dispatch region");
        return llvm::None;
      }
      rootOperation = linalgOp;
      SmallVector<int64_t, 4> opTileSizes;
      if (!clLLVMTileSizes.empty()) {
        opTileSizes.assign(clLLVMTileSizes.begin(), clLLVMTileSizes.end());
      } else {
        opTileSizes = getTileSizes<TilingLevel::WorkGroupTiles>(contractionOp);
      }
      config.setTileSizes(contractionOp, opTileSizes, 0);
      config.setRootOperation(contractionOp);
      continue;
    }
  }
  if (!rootOperation) {
    return config;
  }
  if (failed(propogateRootOperationLaunchConfig(config, *rootOperation,
                                                dependenceGraph)))
    return llvm::None;
  return config;
}

}  // namespace iree_compiler
}  // namespace mlir
