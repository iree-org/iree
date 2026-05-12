// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// Populate functions.
//===----------------------------------------------------------------------===//

/// Populate patterns to propagate reshapes by expansion. This folds
/// tensor.expand_shape and tensor.collapse_shape ops with their producer
/// and consumer operations respectively.
void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFoldingReshapes);

/// Populate patterns to lower an `iree_codegen.inner_tiled` op with empty
/// iteration bounds into the per-intrinsic ops emitted by its kind's
/// `buildUnderlyingOperations` (e.g. `llvm.call_intrinsic` on CPU,
/// `amdgpu.mfma` / `nvgpu.mma.sync` / etc. on GPU). Inserts shape_casts and
/// hoistable conversion pairs around the accumulator as needed.
void populateLowerInnerTiledPatterns(RewritePatternSet &patterns);

/// Populate patterns to drop unit iteration bounds from an
/// `iree_codegen.inner_tiled` op, preparing it for
/// `populateLowerInnerTiledPatterns` which expects empty bounds. Inserts
/// hoistable extract/broadcast pairs so the accumulator's intrinsic-register
/// type becomes loop-carried.
void populateDropInnerTiledUnitDimsPatterns(RewritePatternSet &patterns);

/// Populate patterns to unroll an `iree_codegen.inner_tiled` op along its
/// iteration dimensions, with the unroll shape and traversal order specified
/// by `options`. Wraps the per-intrinsic ACC distribute and reassemble in a
/// hoistable conversion pair so the loop-carried accumulator type is the
/// per-intrinsic one.
void populateUnrollInnerTiledPatterns(
    RewritePatternSet &patterns, const vector::UnrollVectorOptions &options);

/// Convenience overload: unroll to a unit iteration shape, with a matmul-like
/// traversal order that reuses the LHS register (assumes the LHS is the first
/// input).
void populateUnrollInnerTiledPatterns(RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::IREE::Codegen

#endif // IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_TRANSFORMS_TRANSFORMS_H_
