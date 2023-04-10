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

// TODO(hanchung): Create a pass to handle detailed logic about splitting tiling
// sizes for parallel dims and reduction dims.
// We have to fuse the fill + named_op + generic ops along parallel dims
// firstly. At this stage, we do not apply vectorization. The reduction dim
// won't get tiled if the case is matmul + generic op. In this case, we have to
// tile along reduction dim again, which needs them to be TilingInterface ops.
enum class TilingLevel : unsigned {
  // Tile TilingInterface operations to threads.
  WorkGroupTiles = 0,
  // Tile TilingInterface operation on workgroup thread for parallel dims.
  ParallelTiles = 1,
  // Tile TilingInterface operations on workgroup thread for reduction dims.
  ReductionTiles = 2,
  NumTileLevels = 3
};

LogicalResult initCPULaunchConfig(ModuleOp moduleOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_
