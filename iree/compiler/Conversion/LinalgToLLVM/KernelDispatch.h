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

FailureOr<IREE::HAL::DispatchLoweringPassPipeline> initCPULaunchConfig(
    ModuleOp moduleOp);

}  // namespace iree_compiler
}  // namespace mlir
