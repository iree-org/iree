// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

enum class TilingLevel : unsigned {
  // Tile linalg operations to workgroup threads.
  WorkGroupTiles = 0,
  // Tile linalg operation on workgroup thread into L1 block tiles.
  Level1Tiles = 1,
  // Tile linalg operations on L1 block tiles into vector tiles.
  Level2Tiles = 2,
  NumTileLevels = 3
};

IREE::HAL::TranslateExecutableInfo initCPULaunchConfig(ModuleOp moduleOp);

}  // namespace iree_compiler
}  // namespace mlir
