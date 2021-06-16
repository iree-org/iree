// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LaunchConfig.h - Configuration used to drive arch specific codegen -===//
//
// This file declares the data structure that is used by the codegeneration to
// lower to target specific IR. The values of the parameters are archtecture
// specific. Once set the same transformations can be used to generate the
// desired code. This allows sharing codegen infra between different backends.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CONVERSION_COMMON_LAUNCHCONFIG_H_
#define IREE_COMPILER_CONVERSION_COMMON_LAUNCHCONFIG_H_
#include <array>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

/// Stores the tile sizes to use at different levels of tiling as a vector of
/// vectors.
/// - First level tiling maps to workgroups.
/// - Second level tiling maps to subgroups.
/// - Third level tiling maps to invocations.
using TileSizesListType = SmallVector<SmallVector<int64_t, 4>, 1>;
using TileSizesListTypeRef = ArrayRef<SmallVector<int64_t, 4>>;

/// Configurations for mapping Linalg ops to CPU/GPU parallel hiearchies.
///
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

  /// Removes attributes added to operations for retrieving tile size
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

  /// Of the given operations return the operation that has been marked as the
  /// root operation. Within a dispatch region a single root operation (like
  /// matmul, conv, etc.) decides the launch configuration to be used. The rest
  /// of the ops that are fused with it obey this configuration. Returns nullptr
  /// if unable to find an operation that is set as root in the list.
  Operation *getRootOperation(ArrayRef<Operation *> ops);

  /// Returns true if tile sizes have been computed for the operation. If tile
  /// sizes arent set, it implies operation is not to be tiled.
  bool hasTileSizes(Operation *op, size_t level = 0) const {
    return !getTileSizes(op, level).empty();
  }

  /// Use vectorize transformations.
  bool useVectorize() const { return vectorize; }

  /// Sets the tile sizes to use for all levels of tiling of `op`.
  void setTileSizes(Operation *op, TileSizesListType vTileSizes);

  /// Sets the tile sizes to use for a given `level` of tiling of `op`.
  void setTileSizes(Operation *op, ArrayRef<int64_t> vTileSizes, size_t level);

  /// Sets the workgroup size to use for the function.
  void setWorkgroupSize(ArrayRef<int64_t> vWorkgroupSize);

  /// Sets number of subgroups to use.
  void setNumSubgroups(ArrayRef<int64_t> vNumSubgroups);

  /// Sets the root operation. Within a dispatch region a single root operation
  /// (like matmul, conv, etc.) decides the launch configuration to be used. The
  /// rest of the ops that are fused with it obey this configuration.
  void setRootOperation(Operation *root);

  /// Sets the configuration of the `targetOp` to be same as the configuration
  /// of the `sourceOp`.
  void setSameConfig(Operation *sourceOp, Operation *targetOp);

  /// Sets flag to enable vectorization.
  void setVectorize(bool enableVectorize);

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

  /// Use vectorization.
  bool vectorize = false;
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
