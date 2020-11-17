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

//===- KernelDispatchUtils.h - Utilities for generating dispatch info -----===//
//
// This file declares utility functions that can be used to create information
// the dispatch on the host side needs to execute an entry point function, like
// the number of workgroups to use for launch, etc.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CONVERSION_COMMON_LAUNCHCONFIG_H_
#define IREE_COMPILER_CONVERSION_COMMON_LAUNCHCONFIG_H_
#include <array>

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class FuncOp;
class LogicalResult;
class Operation;
class PatternRewriter;
class ShapedType;
class Value;

namespace linalg {
class LinalgDependenceGraph;
class LinalgOp;
}  // namespace linalg

namespace iree_compiler {

/// Store the tile sizes to use at different levels of tiling as a vector of
/// vectors.
/// - First level tiling maps to workgroups.
/// - Second level tiling maps to subgroups.
using TileSizesListType = SmallVector<SmallVector<int64_t, 4>, 1>;
using TileSizesListTypeRef = ArrayRef<SmallVector<int64_t, 4>>;

/// Based on the linalg operations in a dispatch region, the number of levels of
/// tiling, the tile sizes needed, the workgroup size, etc. need to be
/// decided. These parameters are called `LaunchConfig`. This class implements
/// one heuristic to compute these for the different linalg operations on
/// buffers. This can be adapted later to support multiple configurations that
/// can be picked based on device information/problem size information. It
/// exposes the information needed by the codegenerators, and hides the
/// implementation from the rest of the pipeline.
class LaunchConfig {
 public:
  LaunchConfig() : workgroupSize({1, 1, 1}), numSubgroups({1, 1, 1}) {}

  /// Remove attributed added to operations for retrieving tile size
  /// information.
  void finalize(FuncOp funcOp);

  /// Gets the tile size computed for an operation at all levels.
  TileSizesListTypeRef getTileSizes(Operation *op) const;

  /// Gets the tile size computed for an operation for an level.
  ArrayRef<int64_t> getTileSizes(Operation *op, size_t level) const;

  /// Returns the workgroup size to use based on the tile sizes.
  ArrayRef<int64_t> getWorkgroupSize() const { return workgroupSize; }

  /// Returns the number of subgroups to use.
  ArrayRef<int64_t> getNumSubgroups() const { return numSubgroups; }

  /// Returns true if tile sizes have been computed for the operation. If tile
  /// sizes arent set, it implies operation is not to be tiled.
  bool hasTileSizes(Operation *op, size_t level = 0) const {
    return !getTileSizes(op, level).empty();
  }

  void setTileSizes(Operation *op, TileSizesListType vTileSizes);

  void setTileSizes(Operation *op, ArrayRef<int64_t> vTileSizes, size_t level);

  void setWorkgroupSize(ArrayRef<int64_t> vWorkgroupSize);

  void setNumSubgroups(ArrayRef<int64_t> vNumSubgroups);

  void setSameConfig(Operation *sourceOp, Operation *targetOp);

 protected:
  /// Current tile size configuration per operation. They key used here to
  /// retrieve the tile size information per operation is the value of a StrAttr
  /// added to operations during `init`. When tiled this attribute is copied
  /// over to the tiled operation, thereby the same key can be used to retrieve
  /// the tile sizes for the next level of tiling. The `finalize` method removes
  /// these attributes.
  llvm::StringMap<TileSizesListType> tileSizes;

  /// Workgroup size to use.
  std::array<int64_t, 3> workgroupSize = {1, 1, 1};

  /// Number of subgroups that are logically distributed along x, y & z.
  std::array<int64_t, 3> numSubgroups = {1, 1, 1};
};

/// Propogates tile sizes from `rootOperation` to other linalg operations in the
/// dispatch region. This assumes that each dispatch region has a single root
/// operation (like matmul, conv, etc.) that determines the tile sizes to use
/// for tile+fuse+distribute. These are then propogated to the other operations.
/// Note: This is a temporary solution and might be defunct when the codegen
/// becomes more sophisticated.
LogicalResult propogateRootOperationLaunchConfig(
    LaunchConfig &launchConfig, linalg::LinalgOp rootOperation,
    const linalg::LinalgDependenceGraph &dependenceGraph);

}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_CONVERSION_COMMON_LAUNCHCONFIG_H_
