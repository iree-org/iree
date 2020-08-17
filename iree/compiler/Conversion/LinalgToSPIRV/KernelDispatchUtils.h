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
#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_KERNELDISPATCHUTILS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_KERNELDISPATCHUTILS_H_

#include <array>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
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
class LinalgOp;
}
namespace iree_compiler {
struct SPIRVCodegenOptions;
}

namespace iree_compiler {

/// Generates a function that computes the number of workgroups as
///  [ceil(`parallelLoopRange`[2] / `tileSizes`[2]),
///   ceil(`parallelLoopRange`[1] / `tileSizes`[1]),
///   ceil(`parallelLoopRange`[0] / `tileSizes`[0])]
/// where `parallelLoopRange` is the ranges of the parallel loops of `linalgOp`
/// distributed across workgroups.
LogicalResult createNumWorkgroupsFromResultShape(PatternRewriter &rewriter,
                                                 linalg::LinalgOp linalgOp,
                                                 FuncOp entryPointFn,
                                                 ArrayRef<int64_t> tileSizes);

/// Generates a function that computes the number of workgroups as
///  ceil(`parallelLoopRange`[0] * `parallelLoopRange`[1] * ... *
///       `parallelLoopRange`[n-1]  /  `workgroupSizeX`)
/// where `parallelLoopRange` is the ranges of the parallel loops of `linalgOp`
/// distributed across workgroups.
LogicalResult createNumWorkgroupsFromLinearizedResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    int64_t workgroupSizeX);

/// For a given `entryPointFn` return the function that computes the number of
/// workgroups to use at launch time.
FuncOp getNumWorkgroupsFn(FuncOp entryPointFn);

/// Store the tile sizes to use at different levels of tiling as a vector of
/// vectors.
/// - First level tiling maps to workgroups.
/// - Second level tiling maps to subgroups.
using TileSizesListType = SmallVector<SmallVector<int64_t, 4>, 1>;

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

  /// Given the sequence of `linalgOps` (and `options`), decide the launch
  /// configuration by deciding
  /// - the number of levels of tiling,
  /// - tile sizes for each level,
  /// - the workgroup size, and
  /// - number of subgroups to use.
  LogicalResult init(const SPIRVCodegenOptions &options,
                     ArrayRef<linalg::LinalgOp> linalgOps);

  /// Gets the tile size computed for an operation at all levels.
  TileSizesListType getTileSizes(Operation *op) const {
    return tileSizes.lookup(op->getName().getStringRef());
  }

  /// Gets the tile size computed for an operation for an level.
  ArrayRef<int64_t> getTileSizes(Operation *op, size_t level) const {
    auto it = tileSizes.find(op->getName().getStringRef());
    if (it == tileSizes.end() || level >= it->second.size()) return {};
    return it->second[level];
  }

  /// Returns the workgroup size to use based on the tile sizes.
  ArrayRef<int64_t> getWorkgroupSize() const { return workgroupSize; }

  /// Returns the number of subgroups to use.
  ArrayRef<int64_t> getNumSubgroups() const { return numSubgroups; }

 protected:
  /// Current tile size configuration per operation.

  // TODO: For now just use the operation name for the mapping. The tile sizes
  // will be selected only for operations like matmul, conv, pool, etc. and
  // assume that there is only one such operation per dispatch
  // region. Eventually this might need to be relaxed, and some name-marker
  // based mechanism might be needed.
  llvm::StringMap<TileSizesListType> tileSizes;

  /// Workgroup size to use.
  std::array<int64_t, 3> workgroupSize;

  /// Number of subgroups that are logically distributed along x, y & z.
  std::array<int64_t, 3> numSubgroups;
};

}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_CONVERSION_LINALGTOSPIRV_DISPATCHUTILS_H_
