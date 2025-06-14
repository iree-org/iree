// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TILEANDFUSEUTILS_H_
#define IREE_COMPILER_CODEGEN_COMMON_TILEANDFUSEUTILS_H_

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include <queue>

namespace mlir::iree_compiler {

/// Tile and fuse producers of extract slice operations from the worklist into
/// the given loops, adding any new fusion opportunities back to the worklist,
/// proceeding recursively until fixed point is reached.
void fuseProducersOfSlices(RewriterBase &rewriter,
                           std::queue<Operation *> &worklist,
                           scf::SCFTileAndFuseOptions &options,
                           MutableArrayRef<LoopLikeOpInterface> loops);

/// Consider the following case
///
/// ```mlir
/// %0:2 = linalg.generic {
///     indexing_maps = [....,
///                      affine_map<(d0, d1, d2) -> (d0, d1),
///                      affine_map<(d0, d1, d2) -> (d0, d1)>]}
/// %1 = linalg.generic ins(%0#0, %0#1) {
///     indexing_maps = [affine_map<(d0, d1) -> (d0, d1),
///                      affine_map<(d0, d1) -> (d0, d1)]}
/// ```
///
/// After tiling the first op we get
///
/// ```
/// %0:2 = scf.forall ... {
///   %1:2 = linalg.generic {
///       indexing_maps = [....,
///                        affine_map<(d0, d1, d2) -> (d0, d1),
///                        affine_map<(d0, d1, d2) -> (d0, d1)>]}
///   }
/// }
/// %2 = linalg.generic ins(%0#0, %0#1) {
///     indexing_maps = [affine_map<(d0, d1) -> (d0, d1),
///                      affine_map<(d0, d1) -> (d0, d1)]}
/// ```
///
/// Due to a quirk of the fusion of consumers, fusing this consumer into the
/// loop results in
///
/// ```
/// %0:2 = scf.forall ... {
///   %1:2 = linalg.generic {
///       indexing_maps = [....,
///                        affine_map<(d0, d1, d2) -> (d0, d1),
///                        affine_map<(d0, d1, d2) -> (d0, d1)>]}
///   %2 = tensor.extract_slice %0#1 [...]
///   %3 = linalg.generic ins(%1#0, %2) {
///       indexing_maps = [affine_map<(d0, d1) -> (d0, d1),
///                        affine_map<(d0, d1) -> (d0, d1)]}
///   }
/// }
/// ```
///
/// This is an SSA violation because of `%0#1` being used in the loop. This
/// needs to be fixed upstream, but for cases where
/// 1. The root operation produces results using an identity indexing map (when
/// ignoring the iteration space dimensions corresponding to the reduction
/// loops)
/// 2. For all consumers of the results of the root operation, access the data
/// using identity indexing map then for each consumer fusion step it is valid
/// to replace all uses of slices of the outer loop that occur within the loop
/// with the correponding tiled result value.
/// This is a workaround till upstream transformation can fix this issue. The
/// following method is testing if such a case exists to implement the
/// work-around.
bool warForConsumerFusionSSAViolation(
    Operation *rootOp,
    const llvm::SmallDenseSet<Operation *> &tiledAndFusedOps);

/// Starting from `op` walk all operands backwards to find all
/// potentially fusible operations, i.e. operations that implement
/// the `TilingInterface`.
void collectTiledAndFusedOps(Operation *rootOp,
                             llvm::SmallDenseSet<Operation *> &result);

// Fuse all consumers of the given `tiledOp` into the surrounding `scf.forall`.
// Returns a list of new `tensor.extract_slice` ops with new fusion
// opportunities, as well as the new surrounding `scf.forall` (because consumer
// fusion replaces the loop).
FailureOr<std::queue<Operation *>>
fuseConsumersIntoForall(RewriterBase &rewriter, Operation *tiledOp,
                        MutableArrayRef<LoopLikeOpInterface> loops,
                        bool useWARForConsumerFusionSSAViolation);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_TILEANDFUSEUTILS_H_
