// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/TransposePropagationPatterns.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_PROPAGATEAFFINEQUANTIZATIONPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using IREE::LinalgExt::DequantizeAffineOp;
using IREE::LinalgExt::QuantizeAffineOp;

namespace {

/// Bubbles a pad of a dequantized value onto the quantized side.
///
/// The quantized pad uses the dequantize's zero point, which represents real
/// zero through `(zp - zp) * scale`. Moving the pad above the dequantize makes
/// the dequantize the immediate producer of whatever consumed the pad, and
/// pads narrower data on the way.
struct BubblePadThroughDequantize : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto dequantizeOp = padOp.getSource().getDefiningOp<DequantizeAffineOp>();
    if (!dequantizeOp) {
      return rewriter.notifyMatchFailure(padOp, "source is not a dequantize");
    }
    if (!dequantizeOp->getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(padOp, "dequantize has other users");
    }
    // Only a pad with real zero can move to the quantized side, where the zero
    // point represents it.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue || !matchPattern(padValue, m_AnyZeroFloat())) {
      return rewriter.notifyMatchFailure(padOp, "pad value is not a zero "
                                                "constant");
    }

    // The quantized pad has to yield a single value: a zero point that varies
    // per channel, even along an unpadded dimension, would need a pad whose
    // value depends on the padded index, which downstream pad handling does
    // not support.
    Value zeroPoint = dequantizeOp.getZeroPoint();
    if (zeroPoint && dequantizeOp.getZeroPointMap().getNumResults() != 0) {
      return rewriter.notifyMatchFailure(padOp, "zero point is not per-tensor");
    }
    if (zeroPoint && isa<ShapedType>(zeroPoint.getType()) &&
        !isa<RankedTensorType>(zeroPoint.getType())) {
      return rewriter.notifyMatchFailure(
          padOp, "zero point is neither a scalar nor a 0-d tensor");
    }

    // After the rewrite the dequantize also computes the padded positions, so
    // a scale indexed by a padded dimension would be read past its end. The
    // zero point is per-tensor at this point and cannot be indexed that way.
    AffineMap outputToIteration =
        inversePermutation(dequantizeOp.getOutputMap());
    AffineMap scaleMap = dequantizeOp.getScaleMap().compose(outputToIteration);
    llvm::SmallBitVector paddedDims = padOp.getPaddedDims();
    if (llvm::any_of(paddedDims.set_bits(), [&](unsigned dim) {
          return scaleMap.isFunctionOfDim(dim);
        })) {
      return rewriter.notifyMatchFailure(
          padOp, "a padded dimension indexes the quantization parameters");
    }

    Location loc = padOp.getLoc();
    Type storageType = dequantizeOp.getInputType().getElementType();
    // A symmetric dequantize has an implicit zero point of zero.
    Value quantizedPadValue;
    if (zeroPoint) {
      if (isa<RankedTensorType>(zeroPoint.getType())) {
        zeroPoint = tensor::ExtractOp::create(rewriter, loc, zeroPoint,
                                              /*indices=*/ValueRange{});
      }
      quantizedPadValue =
          convertScalarToDtype(rewriter, loc, zeroPoint, storageType,
                               /*isUnsignedCast=*/dequantizeOp.getZpUnsigned());
    } else {
      quantizedPadValue = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(storageType));
    }

    AffineMap outputToInput =
        dequantizeOp.getInputMap().compose(outputToIteration);
    auto paddedInput = tensor::PadOp::create(
        rewriter, loc, /*resultType=*/Type(), dequantizeOp.getInput(),
        applyPermutationMap<OpFoldResult>(outputToInput,
                                          padOp.getMixedLowPad()),
        applyPermutationMap<OpFoldResult>(outputToInput,
                                          padOp.getMixedHighPad()),
        quantizedPadValue, padOp.getNofold());

    Value init = tensor::EmptyOp::create(
        rewriter, loc,
        applyPermutationMap<OpFoldResult>(
            inversePermutation(outputToInput),
            tensor::getMixedSizes(rewriter, loc, paddedInput)),
        padOp.getResultType().getElementType());
    SmallVector<Value> operands = dequantizeOp->getOperands();
    operands.front() = paddedInput.getResult();
    operands.back() = init;
    Operation *dequantizedPad =
        mlir::clone(rewriter, dequantizeOp, padOp->getResultTypes(), operands);
    rewriter.replaceOp(padOp, dequantizedPad->getResults());
    return success();
  }
};

/// Allows a collapse_shape feeding quantize or an expand_shape consuming
/// dequantize. Both require a single-use edge so propagation removes a reshape
/// from the real-valued side without adding one for another consumer. Only the
/// quantize's value operand counts: a collapse_shape producing a quantization
/// parameter does not separate the quantize from its real-valued producer.
static bool canPropagateAffineQuantizationReshape(OpOperand *operand) {
  if (auto quantize = dyn_cast<QuantizeAffineOp>(operand->getOwner())) {
    return operand == &quantize.getInputMutable() &&
           operand->get().getDefiningOp<tensor::CollapseShapeOp>() &&
           operand->get().hasOneUse();
  }
  return isa<tensor::ExpandShapeOp>(operand->getOwner()) &&
         operand->get().hasOneUse();
}

/// Restricts transpose rewrites to transposes of dequantized values. The
/// general transpose propagation pass owns every other transpose.
static bool isTransposeOfDequantize(OpOperand *operand) {
  return isa_and_nonnull<DequantizeAffineOp>(operand->get().getDefiningOp());
}

struct PropagateAffineQuantizationPass
    : public impl::PropagateAffineQuantizationPassBase<
          PropagateAffineQuantizationPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    IREE::LinalgExt::populatePropagateAffineQuantizationReshapesPatterns(
        patterns, canPropagateAffineQuantizationReshape);
    populateSpecializeGenericTransposePattern(patterns,
                                              isTransposeOfDequantize);
    populateFuseTransposeIntoDequantizePattern(patterns,
                                               isTransposeOfDequantize);
    patterns.add<BubblePadThroughDequantize>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
