// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Fold expand_shape ops with their producers (only `AttentionOp` supported)
void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFoldingReshapes);

/// Fuse transpose-like ops into LinalgExt ops (only `AttentionOp` supported).
void populateFuseLinalgExtOpsWithTransposes(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFusionFn);

/// Bubble up transpose-like ops from LinalgExt ops (only `AttentionOp`
/// supported).
void populateBubbleTransposeFromLinalgExtOps(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFusionFn);

/// Default function to drop unit dims for for linalgext ops.
SmallVector<unsigned> defaultControlDropUnitDims(Operation *op);

/// Drop unit extent dims from linalg ext ops
void populateFoldUnitExtentDimsPatterns(
    RewritePatternSet &patterns, const linalg::ControlDropUnitDims &options);

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

/// Rewrite an input rfft op (from an upstream dialect) into linalg_ext.fft.
/// Only applies to power of 2 FFT sizes. Return real and imaginary tensor
/// values or failure.
///
/// op: merely used for getting the Location and for reporting match failures.
/// operand: input operand to FFT, must be of type RankedTensorType and with
///   static shapes for the pattern to apply.
/// fftLength: size of input dimension
///   along which FFT is computed, must be power of 2 for the pattern to apply
///
/// Using Torch dialect as an example, see snippet below.
/// `%arg0` is the |operand|, `torch.aten.fft_rfft` is passed as |op|, and
/// |fftLength| (= 16) is statically computed accessing the shape of `%arg0`
/// ([3,8,16]) at the FFT dimension index (%int-1).
///
/// %int-1 = torch.constant.int -1
/// %out = torch.aten.fft_rfft  %arg0, %none, %int-1, %none :
///           !torch.vtensor<[3,8,16],f32>, !torch.none, !torch.int, !torch.none
///           -> !torch.vtensor<[3,8,9],complex<f32>>
FailureOr<std::pair<Value, Value>> rewriteFft(Operation *op, Value operand,
                                              int64_t fftLength,
                                              PatternRewriter &rewriter);

}; // namespace mlir::iree_compiler::IREE::LinalgExt
