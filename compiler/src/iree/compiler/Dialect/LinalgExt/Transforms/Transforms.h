// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Fold expand_shape ops with their producers (only `AttentionOp` supported)
void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFoldingReshapes);

/// Fuse transpose-like ops into LinalgExt ops (only `AttentionOp` supported).
void populateFuseLinalgExtOpsWithTransposes(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFusionFn);

/// Helper struct to hold the results of collapsing an operation.
struct CollapseResult {
  SmallVector<Value> results;
  Operation *collapsedOp;
};

/// Collapse the iteration dimension of `op` as described by
/// `foldedIterationDims`. Returns failure when the op cannot be collapsed or it
/// is a no-op.
FailureOr<CollapseResult>
collapseOpIterationDims(AttentionOp op,
                        ArrayRef<ReassociationIndices> foldedIterationDims,
                        RewriterBase &rewriter);

}; // namespace mlir::iree_compiler::IREE::LinalgExt
