// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_TRANSPOSEPROPAGATIONPATTERNS_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_TRANSPOSEPROPAGATIONPATTERNS_H_

#include <functional>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::GlobalOptimization {

// Decides whether a transpose may be moved across the use edge |operand|.
// Patterns call it once, after the structural match, so the edge always
// connects the matched ops: the owner of |operand| consumes the value produced
// by the op on the other side of the edge. When a transpose is sunk, the
// transpose defines the operand; when a transpose is bubbled, the transpose
// owns the operand.
using ControlTransposePropagationFn = std::function<bool(OpOperand *operand)>;

// Rewrites |genericOp| into a linalg.transpose if it only transposes its single
// input. Returns failure without modifying the IR otherwise.
LogicalResult specializeGenericTransposeOp(RewriterBase &rewriter,
                                           linalg::GenericOp genericOp);

// Specializes linalg.generic transposes into linalg.transpose. |controlFn|
// receives the input operand of the generic transpose.
void populateSpecializeGenericTransposePattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    PatternBenefit benefit = 1);

//===----------------------------------------------------------------------===//
// Transpose bubbling patterns
//===----------------------------------------------------------------------===//

// Fuses a transpose into the init of its producer linalg.generic. Named
// contraction producers are generalized first when |allowGeneralizing| is set
// and convolution producers only when |allowConvolution| is set.
void populateFuseTransposeWithProducerLinalgOpPattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    bool allowGeneralizing, bool allowConvolution, PatternBenefit benefit = 1);

// Bubbles a transpose above its single-use tensor.collapse_shape producer.
void populateBubbleTransposeThroughCollapseShapePattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    PatternBenefit benefit = 1);

// Bubbles a transpose above an elementwise producer whose iteration-space
// permutation only affects a single input.
void populateBubbleTransposeThroughUnaryElementwiseDpsInitPattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    PatternBenefit benefit = 1);

// Fuses a transpose into its single-use iree_linalg_ext.dequantize_affine
// producer by permuting the dequantize's output map.
void populateFuseTransposeIntoDequantizePattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    PatternBenefit benefit = 1);

//===----------------------------------------------------------------------===//
// Transpose sinking patterns
//===----------------------------------------------------------------------===//

// Composes a transpose of a transpose into a single transpose, or removes both
// when they cancel.
void populateComposeTransposesPattern(RewritePatternSet &patterns,
                                      ControlTransposePropagationFn controlFn,
                                      PatternBenefit benefit = 1);

// Sinks a transpose below a tensor.extract_slice consumer.
void populateSinkTransposeThroughExtractSlicePattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    PatternBenefit benefit = 1);

// Sinks a single-use transpose below a tensor.expand_shape consumer.
void populateSinkTransposeThroughExpandShapePattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    PatternBenefit benefit = 1);

// Sinks a transpose below a tensor.pad consumer whose padding value does not
// depend on the padded index.
void populateSinkTransposeThroughPadPattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    PatternBenefit benefit = 1);

// Fuses a single-use transpose into the indexing map of a consumer
// linalg.generic. Named contraction consumers are generalized first when
// |allowGeneralizing| is set and convolution consumers only when
// |allowConvolution| is set.
void populateFuseTransposeWithLinalgOpConsumerPattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    bool allowGeneralizing, bool allowConvolution, PatternBenefit benefit = 1);

// Sinks a transpose below an elementwise consumer whose iteration-space
// permutation only affects the transposed input.
void populateSinkTransposeThroughUnaryElementwiseInputPattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    PatternBenefit benefit = 1);

// Folds a transpose of a matmul or batch_matmul operand into the transposed
// named op variant, and a transposed named op input back into the plain op.
void populateFoldTransposeIntoNamedMatmulPatterns(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    PatternBenefit benefit = 1);

// Fuses a transpose into the indexing map of a reduction linalg.generic when
// the resulting access order stays increasing.
void populateFuseTransposeThroughGenericReductionPattern(
    RewritePatternSet &patterns, ControlTransposePropagationFn controlFn,
    PatternBenefit benefit = 1);

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_TRANSPOSEPROPAGATIONPATTERNS_H_
