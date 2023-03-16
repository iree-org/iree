// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

enum class TilingLevel : unsigned {
  // Tile linalg operations to threads.
  WorkGroupTiles = 0,
  // Tile linalg operation on workgroup thread into L1 block tiles.
  L1Tiles = 1,
  // Tile linalg operations on L1 block tiles into vector tiles.
  VectorTiles = 2,
  NumTileLevels = 3
};

// TODO(hanchung): Create a pass to handle detailed logic about splitting tiling
// sizes for parallel dims and reduction dims.
// We have to fuse the fill + named_op + generic ops along parallel dims
// firstly. At this stage, we do not apply vectorization. The reduction dim
// won't get tiled if the case is matmul + generic op. In this case, we have to
// tile along reduction dim again, which needs them to be Linalg ops form.
enum class StrategyTilingLevel : unsigned {
  // Tile linalg operations to threads.
  WorkGroupTiles = 0,
  // Tile linalg operation on workgroup thread for parallel dims.
  ParallelTiles = 1,
  // Tile linalg operations on workgroup thread for reduction dims.
  ReductionTiles = 2,
  NumStrategyTileLevels = 3
};

LogicalResult initCPULaunchConfig(ModuleOp moduleOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_
