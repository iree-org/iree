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

/// Fold expand_shape ops with their producers and collapse_shape ops with
/// consumers.
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

/// Drop unit extent dims from linalg ext ops. Uses reshapes to fold scatter
/// and slices to fold gather operations.
/// TODO: Respect the rank reduction strategy specified in ControlDropUnitDims
/// options.
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

/// Apply transformation to split a linalg.generic argmax reduction
/// into a two-stage reduction using an additional parallel dimension.
/// The transformation first computes a partial argmax over tiles (parallel),
/// then reduces those results into a final result (reduction).
///
/// This pattern is specialized for reductions that yield both the maximum
/// value and its index, using the combination of `arith.maximumf`,
/// `arith.cmpf`, and `arith.select` ops. It assumes a known structure of the
/// region and injects index computations to track global indices.
///
/// The transformation proceeds in two steps:
/// 1. Emit a strict argmax op that computes the maximum value and local index
///    within each tile (i.e., the inner dimension after splitting the
///    reduction).
/// 2. Emit a final argmax-style reduction that directly computes the global
/// index
///    on-the-fly as `globalIndex = outer × tileSize + local`, using the
///    original combiner structure (`maximumf`, `cmpf`, `select`).
///
/// Returns the resulting partial and final linalg.generic ops, or failure
/// if the pattern does not match or cannot be split.
///
/// Example: original argmax op reducing over dim=512
/// %4:2 = linalg.generic {
///   indexing_maps = [...],
///   iterator_types = ["parallel", "reduction"]
/// } ins(%arg0 : tensor<?x512xbf16>)
///   outs(%out_val, %out_idx : tensor<?xbf16>, tensor<?xi64>) {
/// ^bb0(%in: bf16, %out: bf16, %out_0: i64):
///   %idx = linalg.index 1 : index
///   %cast = arith.index_cast %idx : index to i64
///   %max = arith.maximumf %in, %out : bf16
///   %cmp = arith.cmpf ogt, %in, %out : bf16
///   %sel = arith.select %cmp, %cast, %out_0 : i64
///   linalg.yield %max, %sel : bf16, i64
/// } -> (tensor<?xbf16>, tensor<?xi64>)
///
/// To: splitting K=512 into 4 x 128 + final argmax over the tile dimension
///     (dim=1 of ?x4)
///
/// %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<?x512xbf16>
///     into tensor<?x4x128xbf16>
///
/// %init_val = linalg.fill ... : tensor<?x4xbf16>
/// %init_idx = linalg.fill ... : tensor<?x4xi64>
///
/// %partial:2 = linalg.generic {
///   indexing_maps = [...],
///   iterator_types = ["parallel", "reduction"]
/// } ins(%expanded)
///   outs(%init_val, %init_idx) {
///   // strict argmax over local tile
/// }
///
/// %final:2 = linalg.generic {
///   indexing_maps = [...],
///   iterator_types = ["parallel", "reduction"]
/// } ins(%partial#0, %partial#1)
///   outs(%out_val, %out_idx) {
///   // compute global index = outer × 128 + local inside region
///   // then apply argmax combiner using (value, global index) pairs
/// }
FailureOr<linalg::SplitReductionResult>
splitArgmaxReduction(RewriterBase &rewriter, linalg::GenericOp genericOp,
                     linalg::ControlSplitReductionFn controlSplitReductionFn);

}; // namespace mlir::iree_compiler::IREE::LinalgExt
