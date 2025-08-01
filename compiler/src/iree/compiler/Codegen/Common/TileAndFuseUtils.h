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

/// Starting from `op` walk all operands backwards to find all
/// potentially fusible operations, i.e. operations that implement
/// the `TilingInterface`.
void collectTiledAndFusedOps(Operation *rootOp,
                             llvm::SmallDenseSet<Operation *> &result);

/// Fuse all consumers of the given `tiledOps` into the surrounding `scf.forall`
/// unless specified otherwise by `filterFn`. Returns a list of new
/// `tensor.extract_slice` ops with new fusion opportunities.
FailureOr<std::queue<Operation *>> fuseConsumersIntoForall(
    RewriterBase &rewriter, ArrayRef<Operation *> tiledOps,
    MutableArrayRef<LoopLikeOpInterface> loops,
    std::function<bool(Operation *)> filterFn = [](Operation *) {
      return true;
    });

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_TILEANDFUSEUTILS_H_
