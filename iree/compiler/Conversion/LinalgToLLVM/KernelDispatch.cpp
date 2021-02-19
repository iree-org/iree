
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

struct TileOpParameters {
  template <typename OpT, TilingLevel tilingLevel>
  static llvm::SmallVector<int64_t, 4> getSizes(OpT op);
};

template <>
llvm::SmallVector<int64_t, 4>
TileOpParameters::getSizes<linalg::MatmulOp, TilingLevel::WorkGroupTiles>(
    linalg::MatmulOp op) {
  return {matmulWorkgroupTileSize, matmulWorkgroupTileSize};
}

template <>
llvm::SmallVector<int64_t, 4>
TileOpParameters::getSizes<linalg::MatmulOp, TilingLevel::Level1Tiles>(
    linalg::MatmulOp op) {
  return {matmulL1TileSize, matmulL1TileSize, matmulL1TileSize};
}

template <>
llvm::SmallVector<int64_t, 4>
TileOpParameters::getSizes<linalg::MatmulOp, TilingLevel::Level2Tiles>(
    linalg::MatmulOp op) {
  return {matmulL2TileSize, matmulL2TileSize, matmulL2TileSize};
}

template <>
llvm::SmallVector<int64_t, 4>
TileOpParameters::getSizes<linalg::BatchMatmulOp, TilingLevel::WorkGroupTiles>(
    linalg::BatchMatmulOp op) {
  return {1, batchMatmulWorkgroupTileSize, batchMatmulWorkgroupTileSize};
}

template <>
llvm::SmallVector<int64_t, 4>
TileOpParameters::getSizes<linalg::BatchMatmulOp, TilingLevel::Level1Tiles>(
    linalg::BatchMatmulOp op) {
  return {1, batchMatmulL1TileSize, batchMatmulL1TileSize,
          batchMatmulL1TileSize};
}

template <>
llvm::SmallVector<int64_t, 4>
TileOpParameters::getSizes<linalg::BatchMatmulOp, TilingLevel::Level2Tiles>(
    linalg::BatchMatmulOp op) {
  return {1, batchMatmulL2TileSize, batchMatmulL2TileSize,
          batchMatmulL2TileSize};
}

#define DEFINE_TILE_OP_GET_SIZES(TileLevel)                                 \
  template <>                                                               \
  llvm::SmallVector<int64_t, 4> CPUKernelDispatch::getTileSizes<TileLevel>( \
      Operation * op) const {                                               \
    if (isa<linalg::MatmulOp>(op)) {                                        \
      return TileOpParameters::getSizes<linalg::MatmulOp, TileLevel>(       \
          dyn_cast<linalg::MatmulOp>(op));                                  \
    }                                                                       \
    if (isa<linalg::BatchMatmulOp>(op)) {                                   \
      return TileOpParameters::getSizes<linalg::BatchMatmulOp, TileLevel>(  \
          dyn_cast<linalg::BatchMatmulOp>(op));                             \
    }                                                                       \
    return {1, 1, 1};                                                       \
  }

DEFINE_TILE_OP_GET_SIZES(TilingLevel::WorkGroupTiles)
DEFINE_TILE_OP_GET_SIZES(TilingLevel::Level1Tiles)
DEFINE_TILE_OP_GET_SIZES(TilingLevel::Level2Tiles)
#undef DEFINE_TILE_OP_GET_SIZES

#define DEFINE_TILE_SIZE_FN(TileLevel)                                     \
  template <>                                                              \
  SmallVector<Value, 4> TileSizeFn::get<TileLevel>(                        \
      CPUKernelDispatch cpuKernelDispatch, OpBuilder & builder,            \
      Operation * operation) {                                             \
    auto tileSizes = cpuKernelDispatch.getTileSizes<TileLevel>(operation); \
    if (tileSizes.empty()) return {};                                      \
    SmallVector<Value, 4> tileSizesVal;                                    \
    tileSizesVal.reserve(tileSizes.size());                                \
    for (auto val : tileSizes) {                                           \
      tileSizesVal.push_back(                                              \
          builder.create<ConstantIndexOp>(operation->getLoc(), val));      \
    }                                                                      \
    return tileSizesVal;                                                   \
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
#define DISPATCH(opType)                                                       \
  if (opType op = dyn_cast<opType>(linalgOp.getOperation())) {                 \
    if (rootOperation) {                                                       \
      op.emitError("unhandled multiple root operations in dispatch region");   \
      return llvm::None;                                                       \
    }                                                                          \
    rootOperation = linalgOp;                                                  \
    SmallVector<int64_t, 4> opTileSizes;                                       \
    if (!clLLVMTileSizes.empty()) {                                            \
      opTileSizes.assign(clLLVMTileSizes.begin(), clLLVMTileSizes.end());      \
    } else {                                                                   \
      opTileSizes =                                                            \
          TileOpParameters::getSizes<opType, TilingLevel::WorkGroupTiles>(op); \
    }                                                                          \
    config.setTileSizes(op, opTileSizes, 0);                                   \
    config.setRootOperation(op);                                               \
    continue;                                                                  \
  }

    DISPATCH(linalg::MatmulOp)
    DISPATCH(linalg::BatchMatmulOp)

#undef DISPATCH
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
