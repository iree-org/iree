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

//===- Transforms.h - Transformations common to all backends --------------===//
//
// Defines transformations that are common to backends
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CONVERSION_COMMON_TRANSFORMS_H_
#define IREE_COMPILER_CONVERSION_COMMON_TRANSFORMS_H_

#include "iree/compiler/Conversion/Common/LaunchConfig.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {

/// Apply canonicalizations related to tiling to make promotion/vectorization
/// easier.
void applyCanonicalizationPatternsForTiling(MLIRContext *context,
                                            Operation *op);

struct TileAndFuseOptions {
  linalg::LinalgLoopDistributionOptions distributionOptions;
  linalg::AllocBufferCallbackFn allocationFn = nullptr;
};
/// Method to tile and fuse sequence of Linalg operations in `linalgOps`. Uses
/// the tile sizes for the first level of tiling specified in
/// `launchConfig`. Proceeds by
/// 1) Find the common loops around `linalgOps` that can be fused.
/// 2) Tile the fusable loops in the last operation in the sequence.
/// 3) Creates tiled version of the other ops within the inter-tile loops
///    generated in step 2.
/// 4) For all the tiled+fused ops, tile the unfused loops as specified by
///    launchconfig.
LogicalResult tileAndFuseLinalgBufferOps(
    FuncOp funcOp, ArrayRef<linalg::LinalgOp> linalgOps,
    const linalg::LinalgDependenceGraph &dependenceGraph,
    const LaunchConfig &launchConfig, const TileAndFuseOptions &options);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_COMMON_TRANSFORMS_H_
