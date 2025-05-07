// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

//===----------------------------------------------------------------------===//
// Populate functions.
//===----------------------------------------------------------------------===//

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

/// Drop unit extent dims from linalg ext ops
void populateFoldUnitExtentDimsPatterns(
    RewritePatternSet &patterns, const linalg::ControlDropUnitDims &options);

/// Patterns to convert linalg convolution ops into a gemm with an im2col
/// op and reshapes on the inputs.
/// TODO(Max191): Maybe move to transforms and use a funcOp walk instead of a
///               rewrite pattern for this.
void populateConvToIm2colOpPatterns(
    RewritePatternSet &patterns,
    std::optional<std::function<bool(Operation *)>> controlFn = std::nullopt);

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

/// Default function to drop unit dims for for linalgext ops.
SmallVector<unsigned> defaultControlDropUnitDims(Operation *op);

//===----------------------------------------------------------------------===//
// Transformations exposed as functional-style API calls.
//===----------------------------------------------------------------------===//

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

void convertToOnlineAttention(IREE::LinalgExt::AttentionOp attnOp,
                              SmallVectorImpl<Operation *> &ops,
                              RewriterBase &rewriter);

/// Function signature to control reduction splitting. This returns the split
/// reduction ratio used to split the reduction dimension. The ratio is applied
/// to the reduction dimension of TopK. If the ratio value is less or equal to 1
/// then nothing will be done. Input is the current depth of recursive split
/// reduction, starting from 0 (first level).
using TopkSplitReductionControlFn =
    std::function<int64_t(int64_t splitReductionDepth)>;

/// Transforms an applicable standard single reduction TopkOp into a parallel
/// reduction TopkOp with a reduce step following.
///
/// Handles parallel reductions in 2 phases: A "map" parallel phase and the a
/// single "reduce" reduction phase. The first phase expands the input tensor
/// shape by breaking the reduction dimension into multiple parallel reductions
/// (upping the rank of the input). Topk is run on these dimensions in parallel
/// The second phase collapses the parallel results into a single final reduce.
/// Topk is run again on the combined output to produce a final output.
///
/// Currently only topk operations without input indices are supported.
LogicalResult
splitReduction(RewriterBase &rewriter, LinalgExt::TopkOp topkOp,
               const TopkSplitReductionControlFn &splitReductionFn);

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
