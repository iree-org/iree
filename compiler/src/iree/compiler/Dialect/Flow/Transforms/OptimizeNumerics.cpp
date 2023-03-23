// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

int getNextPotBitWidth(int bitWidth, int minBitWidth = 8) {
  for (int i = minBitWidth;; i *= 2) {
    if (i >= bitWidth) return i;
  }
}

Type withNewElementType(Type origType, Type elementType) {
  if (auto st = origType.dyn_cast<ShapedType>()) {
    return st.clone(elementType);
  } else {
    return elementType;
  }
}

Type makeLowPType(Type origType, int bitWidth) {
  auto *context = origType.getContext();
  auto elementType = IntegerType::get(context, bitWidth);
  return withNewElementType(origType, elementType);
}

Value castNumeric(Value origValue, Type toType, bool isSigned,
                  OpBuilder &builder) {
  Location loc = origValue.getLoc();
  Type origElementType = getElementTypeOrSelf(origValue.getType());
  Type toElementType = getElementTypeOrSelf(toType);

  if (origElementType.isa<FloatType>() && toElementType.isa<IntegerType>()) {
    if (isSigned) {
      return builder.create<arith::FPToSIOp>(loc, toType, origValue);
    } else {
      return builder.create<arith::FPToUIOp>(loc, toType, origValue);
    }
  } else if (origElementType.isa<IntegerType>() &&
             toElementType.isa<FloatType>()) {
    if (isSigned) {
      return builder.create<arith::SIToFPOp>(loc, toType, origValue);
    } else {
      return builder.create<arith::UIToFPOp>(loc, toType, origValue);
    }
  } else {
    // If we need int<->int and float<->float, implement those cases. Since
    // this is just needed for things in this file, it is ok to leave it
    // under implemented.
    assert(false && "unsupported numeric cast");
    return Value();
  }
}

struct NarrowParams {
  static std::optional<NarrowParams> forValue(Value value) {
    if (auto narrowOp =
            llvm::dyn_cast_or_null<IREE::Util::NumericOptionalNarrowOp>(
                value.getDefiningOp())) {
      NarrowParams params;
      params.producer = narrowOp.getOperand();
      params.fromType = value.getType();
      params.toElementType = narrowOp.getSemanticType();
      params.range = narrowOp.getIntegerRange();

      return params;
    }
    return {};
  }

  bool isFromFloat() { return getElementTypeOrSelf(fromType).isa<FloatType>(); }

  bool isToInteger() { return toElementType.isa<IntegerType>(); }

  bool isToSigned() { return toElementType.cast<IntegerType>().isSigned(); }

  int getToBitWidth() { return toElementType.cast<IntegerType>().getWidth(); }

  Value producer;
  Type fromType;
  Type toElementType;
  std::optional<std::pair<int64_t, int64_t>> range;
};

// Eliminates a cast produced by an empty by just initializing to that
// type directly.
struct TensorEmptyCast
    : OpInterfaceRewritePattern<IREE::Util::NumericCastOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(IREE::Util::NumericCastOpInterface castOp,
                                PatternRewriter &rewriter) const override {
    auto emptyOp = castOp.getInput().getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp) return failure();
    Type resultType = castOp.getCasted().getType();

    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(castOp, resultType,
                                                 emptyOp.getDynamicSizes());
    return success();
  }
};

// For a cast produced by a fill, rewrites the cast to be on the fill operands.
struct LinalgFillCast
    : public OpInterfaceRewritePattern<IREE::Util::NumericCastOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(IREE::Util::NumericCastOpInterface castOp,
                                PatternRewriter &rewriter) const override {
    auto loc = castOp.getLoc();
    auto fillOp = castOp.getInput().getDefiningOp<linalg::FillOp>();
    if (!fillOp) return failure();
    Type toElementType = getElementTypeOrSelf(castOp.getCastedType());

    Value fillInput = fillOp.value();
    Value fillInit = fillOp.output();
    fillInput = castOp
                    .cloneWithInput(
                        rewriter,
                        withNewElementType(fillInput.getType(), toElementType),
                        fillInput)
                    .getCasted();
    fillInit =
        castOp
            .cloneWithInput(
                rewriter, withNewElementType(fillInit.getType(), toElementType),
                fillInit)
            .getCasted();
    Value fillResult =
        rewriter.create<linalg::FillOp>(loc, fillInput, fillInit).result();
    rewriter.replaceOp(castOp, fillResult);
    return success();
  }
};

// For narrowable inputs, selects
struct LinalgFpMatmulToLowP : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();
    Type origResultType = matmulOp.getResult(0).getType();
    auto lhsParams = NarrowParams::forValue(matmulOp.getInputs()[0]);
    auto rhsParams = NarrowParams::forValue(matmulOp.getInputs()[1]);
    auto accumParams = NarrowParams::forValue(matmulOp.getOutputs()[0]);
    if (!lhsParams || !rhsParams || !accumParams) {
      return rewriter.notifyMatchFailure(matmulOp, "no narrowing annotations");
    }

    // TODO(#7987): This could be more flexible, allowing mix and match
    // integer/float types.
    if (!lhsParams->isFromFloat() || !rhsParams->isFromFloat()) {
      return rewriter.notifyMatchFailure(matmulOp, "not from floating point");
    }

    // TODO(#7987): Could support partial conversion to integer.
    if (!lhsParams->isToInteger() || !rhsParams->isToInteger() ||
        !accumParams->isToInteger()) {
      return rewriter.notifyMatchFailure(matmulOp, "not to an integer type");
    }

    int lhsBitWidth = lhsParams->getToBitWidth();
    int rhsBitWidth = rhsParams->getToBitWidth();

    // Handle signed/unsigned mismatch.
    // TODO(#7987): Implement a proper unsigned->signed widening.
    bool isSigned;
    if (lhsParams->isToSigned() != rhsParams->isToSigned()) {
      // Mixed signed/unsigned. Promote to signed.
      isSigned = true;
      if (!lhsParams->isToSigned()) {
        lhsBitWidth += 1;
      }
      if (!rhsParams->isToSigned()) {
        rhsBitWidth += 1;
      }
    } else {
      // Uniform signed/unsigned.
      isSigned = lhsParams->isToSigned();
    }

    // Round up to a suitable POT width.
    lhsBitWidth = getNextPotBitWidth(lhsBitWidth);
    rhsBitWidth = getNextPotBitWidth(rhsBitWidth);

    // Promote accumulator to match signedness.
    int accumBitWidth = accumParams->getToBitWidth();
    if (isSigned && !accumParams->isToSigned()) {
      // TODO(#7987): A proper unsigned widening based on range.
      accumBitWidth += 1;
    }

    // Determine an appropriate accumulator size.
    // TODO(#7987): Apply the clamp of:
    // lhsBitWidth + rhsBitWidth + log2_ceil(contraction_dim + 1) to determine
    // the accumulator size. Note: Can drop the +1 if one of lhs/rhs is signed
    // and symmetric (i.e. does not use the asymmetric lower bound).
    if (lhsBitWidth > 8 || rhsBitWidth > 8) {
      return rewriter.notifyMatchFailure(matmulOp, "outside of low-p range");
    }
    accumBitWidth = getNextPotBitWidth(accumBitWidth, 32);
    if (accumBitWidth > 32) {
      return rewriter.notifyMatchFailure(matmulOp, "accumulator > 32 bits");
    }

    Type lhsLowPType = makeLowPType(lhsParams->fromType, lhsBitWidth);
    Type rhsLowPType = makeLowPType(rhsParams->fromType, rhsBitWidth);
    Type accumLowPType = makeLowPType(accumParams->fromType, accumBitWidth);

    // Replace the matmul op.
    Value newLhs =
        castNumeric(lhsParams->producer, lhsLowPType, isSigned, rewriter);
    Value newRhs =
        castNumeric(rhsParams->producer, rhsLowPType, isSigned, rewriter);
    Value newAccum =
        castNumeric(accumParams->producer, accumLowPType, isSigned, rewriter);
    Value newResult;

    if (isSigned) {
      newResult = rewriter
                      .create<linalg::MatmulOp>(loc, ValueRange{newLhs, newRhs},
                                                ValueRange{newAccum})
                      .getResult(0);
    } else {
      newResult = rewriter
                      .create<linalg::MatmulUnsignedOp>(
                          loc, ValueRange{newLhs, newRhs}, ValueRange{newAccum})
                      .getResult(0);
    }

    // Cast back.
    newResult = castNumeric(newResult, origResultType, isSigned, rewriter);
    rewriter.replaceOp(matmulOp, ValueRange{newResult});

    return success();
  }
};

class OptimizeNumericsPass : public OptimizeNumericsBase<OptimizeNumericsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Precision reduction.
    patterns.insert<LinalgFpMatmulToLowP>(context);

    // Cast propagation.
    patterns.insert<TensorEmptyCast>(context);
    patterns.insert<LinalgFillCast>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createOptimizeNumericsPass() {
  return std::make_unique<OptimizeNumericsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
