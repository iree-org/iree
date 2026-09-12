// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Moves affine quantization toward the contractions that produce or consume
// the corresponding real-valued tensors.
//
// This exists to establish one invariant for the QDQ-to-integer-math rewrite
// that follows it: a dequantize feeding a contraction is that contraction's
// immediate producer. The rewrite needs that to match, and leaving it to
// whichever general purpose pass happens to run first makes the rewrite's
// success depend on pipeline order rather than on anything local. Exported
// models routinely break the adjacency:
//
//   * `aten.linear` dequantizes the weights and then transposes them.
//   * a padded convolution pads the dequantized image.
//   * a classifier reshapes a pooled activation before the matmul.
//
// Quantize moves toward its real-valued producer through a collapse_shape.
// Dequantize moves toward its real-valued consumer through expand_shape,
// transpose, and legal zero padding. Padding the real-valued side with zero is
// the same as padding the quantized side with the zero point.
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_PROPAGATEQUANTIZATIONTOWARDCONTRACTIONSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using IREE::LinalgExt::DequantizeAffineOp;
using IREE::LinalgExt::QuantizeAffineOp;

namespace {

/// True when both value maps of `op` are the identity, which is the form the
/// quantization ops are built in and the form these patterns produce. A
/// permuted value map would require remapping the transpose or pad dimensions.
template <typename OpTy>
static bool hasIdentityValueMaps(OpTy op) {
  return op.getInputMap().isIdentity() && op.getOutputMap().isIdentity();
}

/// Bubbles a transpose that consumes a dequantize op above it, so that the
/// transpose lands on the quantized side. That is both cheaper, since it moves
/// fewer bits, and more likely to disappear entirely, since quantized weights
/// are usually constants that the transpose folds into.
///
/// For an elementwise op, `transpose(f(x)) == f(transpose(x))` once the
/// quantization parameters follow the permutation: an output element that used
/// to sit at `pi(j)` now sits at `j`, so each parameter map is composed with
/// `pi`, the map of the inverse permutation.
struct SinkTransposeThroughDequantize
    : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto dequantizeOp =
        transposeOp.getInput().getDefiningOp<DequantizeAffineOp>();
    if (!dequantizeOp) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "input is not a dequantize");
    }
    // With other readers the quantization op has to stay, so moving the
    // transpose would duplicate it rather than simplify anything.
    if (!dequantizeOp->getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "dequantize has other users");
    }
    if (!hasIdentityValueMaps(dequantizeOp)) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "value maps are not the identity");
    }

    ArrayRef<int64_t> permutation = transposeOp.getPermutation();
    AffineMap inverseMap = AffineMap::getPermutationMap(
        invertPermutationVector(permutation), rewriter.getContext());

    Location loc = transposeOp.getLoc();
    // The transpose moves onto the quantized operand, and its init has the
    // right shape for that operand once its element type is the quantized one.
    Value quantizedInit = tensor::EmptyOp::create(
        rewriter, loc,
        applyPermutation(
            tensor::getMixedSizes(rewriter, loc, dequantizeOp.getInput()),
            permutation),
        dequantizeOp.getInputType().getElementType());
    Value transposedInput =
        linalg::TransposeOp::create(rewriter, loc, dequantizeOp.getInput(),
                                    quantizedInit, permutation)
            ->getResult(0);

    SmallVector<AffineMap> maps = dequantizeOp.getIndexingMapsArray();
    // The value maps stay the identity; only the parameters are reindexed.
    for (AffineMap &map :
         MutableArrayRef<AffineMap>(maps).drop_front().drop_back()) {
      map = map.compose(inverseMap);
    }

    SmallVector<Value> operands = dequantizeOp->getOperands();
    operands.front() = transposedInput;
    operands.back() = transposeOp.getDpsInits()[0];
    auto newOp = cast<DequantizeAffineOp>(mlir::clone(
        rewriter, dequantizeOp, transposeOp->getResultTypes(), operands));
    newOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    rewriter.replaceOp(transposeOp, newOp->getResults());
    rewriter.eraseOp(dequantizeOp);
    return success();
  }
};

/// Bubbles a pad of a dequantized value onto the quantized side.
///
/// Padding the real valued side with zero and padding the quantized side with
/// the zero point describe the same tensor, because dequantizing the zero point
/// yields `(zp - zp) * scale`, which is exactly zero. Moving the pad below the
/// dequantize makes the dequantize the immediate producer of whatever consumed
/// the pad, and pads narrower data on the way.
struct SinkPadThroughDequantize : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto dequantizeOp = padOp.getSource().getDefiningOp<DequantizeAffineOp>();
    if (!dequantizeOp) {
      return rewriter.notifyMatchFailure(padOp, "source is not a dequantize");
    }
    if (!dequantizeOp->getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(dequantizeOp,
                                         "dequantize has other users");
    }
    if (!hasIdentityValueMaps(dequantizeOp)) {
      return rewriter.notifyMatchFailure(padOp,
                                         "value maps are not the identity");
    }

    // Only a constant zero pad is the zero point in disguise.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue || !matchPattern(padValue, m_AnyZeroFloat())) {
      return rewriter.notifyMatchFailure(padOp, "pad value is not a zero "
                                                "constant");
    }

    // The whole padded region takes one value, so the zero point has to be a
    // single value rather than one per channel.
    Value zeroPoint = dequantizeOp.getZeroPoint();
    if (zeroPoint && isa<ShapedType>(zeroPoint.getType())) {
      return rewriter.notifyMatchFailure(padOp, "zero point is not a scalar");
    }

    // A padded dimension that indexes the quantization parameters would leave
    // the parameters too short for the padded value.
    AffineMap scaleMap = dequantizeOp.getScaleMap();
    for (auto [dim, low, high] :
         llvm::zip_equal(llvm::seq<unsigned>(padOp.getSourceType().getRank()),
                         padOp.getMixedLowPad(), padOp.getMixedHighPad())) {
      if (isConstantIntValue(low, 0) && isConstantIntValue(high, 0)) {
        continue;
      }
      if (scaleMap.isFunctionOfDim(dim)) {
        return rewriter.notifyMatchFailure(
            padOp, "a padded dimension indexes the quantization parameters");
      }
    }

    Location loc = padOp.getLoc();
    Type storageType = dequantizeOp.getInputType().getElementType();
    // Dequantizing the zero point gives zero, so it is what pads the quantized
    // side. A symmetric dequantize has an implicit zero point of zero.
    Value quantizedPadValue;
    if (zeroPoint) {
      quantizedPadValue =
          convertScalarToDtype(rewriter, loc, zeroPoint, storageType,
                               /*isUnsignedCast=*/dequantizeOp.getZpUnsigned());
    } else {
      quantizedPadValue = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(storageType));
    }

    auto paddedInput = tensor::PadOp::create(
        rewriter, loc, /*resultType=*/Type(), dequantizeOp.getInput(),
        padOp.getMixedLowPad(), padOp.getMixedHighPad(), quantizedPadValue,
        padOp.getNofold());

    Value init = tensor::EmptyOp::create(
        rewriter, loc, tensor::getMixedSizes(rewriter, loc, paddedInput),
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

/// Allows only the two directions that move quantization toward a contraction:
/// a collapse_shape feeding the real-valued input of quantize, or an
/// expand_shape consuming dequantize. Both require a single-use edge so the
/// rewrite removes a reshape rather than moving it onto another reader.
static bool movesQuantizationTowardContraction(OpOperand *operand) {
  if (auto quantize = dyn_cast<QuantizeAffineOp>(operand->getOwner())) {
    return operand == &quantize.getInputMutable() &&
           operand->get().getDefiningOp<tensor::CollapseShapeOp>() &&
           operand->get().hasOneUse();
  }
  return isa<tensor::ExpandShapeOp>(operand->getOwner()) &&
         operand->get().hasOneUse();
}

struct PropagateQuantizationTowardContractionsPass
    : public impl::PropagateQuantizationTowardContractionsPassBase<
          PropagateQuantizationTowardContractionsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // Reuse the generic reshape mechanics with only the directions needed by
    // the QDQ rewrite pipeline.
    IREE::LinalgExt::populatePropagateAffineQuantizationReshapesPatterns(
        patterns, movesQuantizationTowardContraction);
    patterns.add<SinkTransposeThroughDequantize, SinkPadThroughDequantize>(
        context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
