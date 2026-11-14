// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/WidenNarrowIntegers.h"

#include <type_traits>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler {

namespace {

using IsNarrowFn = std::function<bool(Type)>;

static Type getWideIntegerType(Type type, unsigned wideWidthInBits) {
  auto elementType = dyn_cast<IntegerType>(getElementTypeOrSelf(type));
  if (!elementType || elementType.getWidth() >= wideWidthInBits ||
      !isa<IntegerType, VectorType>(type)) {
    return {};
  }
  auto wide = IntegerType::get(type.getContext(), wideWidthInBits);
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return vectorType.clone(wide);
  }
  return wide;
}

// Copies the discardable attributes of `from` to `to`.
static void copyDiscardableAttrs(PatternRewriter &rewriter, Operation *from,
                                 Operation *to) {
  rewriter.modifyOpInPlace(to, [from, to] {
    to->setDiscardableAttrs(from->getDiscardableAttrDictionary());
  });
}

// The state shared by the widening patterns: the wide width in bits and the
// predicate selecting the narrow types to widen.
template <typename OpTy>
struct NarrowIntegerPattern : OpRewritePattern<OpTy> {
  NarrowIntegerPattern(MLIRContext *context, unsigned wideWidthInBits,
                       IsNarrowFn isNarrow)
      : OpRewritePattern<OpTy>(context), wideWidthInBits(wideWidthInBits),
        isNarrow(std::move(isNarrow)) {}

  // Returns the wide integer or vector type for `type` if it is a selected
  // narrow integer type narrower than the wide width, otherwise null.
  Type getSelectedWideType(Type type) const {
    return isNarrow(type) ? getWideIntegerType(type, wideWidthInBits) : Type();
  }

  unsigned wideWidthInBits;
  IsNarrowFn isNarrow;
};

template <typename OpTy, typename ExtOpTy, bool preserveProperties = true>
struct WidenNarrowIntegerOp final : NarrowIntegerPattern<OpTy> {
  using NarrowIntegerPattern<OpTy>::NarrowIntegerPattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    Type wide = this->getSelectedWideType(type);
    if (!wide) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Location loc = op.getLoc();
    Value lhs = ExtOpTy::create(rewriter, loc, wide, op.getLhs());
    Value rhs = ExtOpTy::create(rewriter, loc, wide, op.getRhs());
    auto widened = OpTy::create(rewriter, loc, lhs, rhs);
    rewriter.modifyOpInPlace(widened, [widened, op]() mutable {
      // Zero extension changes signed values, so narrow overflow flags do
      // not apply to a widened low-bit computation (e.g. i4 -1 * -1).
      if constexpr (preserveProperties) {
        widened.getProperties() = op.getProperties();
      }
      widened->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    });
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, widened);
    return success();
  }
};

struct WidenNarrowCmpI final : NarrowIntegerPattern<arith::CmpIOp> {
  using NarrowIntegerPattern::NarrowIntegerPattern;

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    Type wide = getSelectedWideType(op.getLhs().getType());
    if (!wide) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    bool isSigned = false;
    switch (op.getPredicate()) {
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::sge:
      isSigned = true;
      break;
    default:
      break;
    }
    Location loc = op.getLoc();
    auto extend = [&rewriter, loc, wide, isSigned](Value value) -> Value {
      if (isSigned) {
        return arith::ExtSIOp::create(rewriter, loc, wide, value);
      }
      return arith::ExtUIOp::create(rewriter, loc, wide, value);
    };
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(
        op, op.getPredicate(), extend(op.getLhs()), extend(op.getRhs()));
    return success();
  }
};

// Extensions involving emulated integers first use the wide type to recover
// their logical source bits, including when the destination type is native.
//   %result = arith.extui %value : i4 to i8
// becomes
//   %wide = arith.extui %value : i4 to i32
//   %result = arith.trunci %wide : i32 to i8
template <typename ExtensionOp>
struct WidenNarrowExtension final : NarrowIntegerPattern<ExtensionOp> {
  using NarrowIntegerPattern<ExtensionOp>::NarrowIntegerPattern;

  LogicalResult matchAndRewrite(ExtensionOp op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    Type wide = getWideIntegerType(type, this->wideWidthInBits);
    if ((!this->isNarrow(type) && !this->isNarrow(op.getIn().getType())) ||
        !wide) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Value result = ExtensionOp::create(rewriter, op.getLoc(), wide, op.getIn());
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, result);
    return success();
  }
};

// Bit counts run on the zero-extended value; leading zeros drop the
// extension, trailing zeros stop at the logical width.
template <typename OpTy>
struct WidenNarrowBitCount final : NarrowIntegerPattern<OpTy> {
  using NarrowIntegerPattern<OpTy>::NarrowIntegerPattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    Type wide = this->getSelectedWideType(type);
    if (!wide) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Location loc = op.getLoc();
    unsigned wideWidthInBits = this->wideWidthInBits;
    int64_t widthInBits = getElementTypeOrSelf(type).getIntOrFloatBitWidth();
    Value operand =
        arith::ExtUIOp::create(rewriter, loc, wide, op.getOperand());
    if constexpr (std::is_same_v<OpTy, math::CountTrailingZerosOp>) {
      operand = arith::OrIOp::create(
          rewriter, loc, operand,
          createScalarOrSplatConstant(
              rewriter, loc, wide, APInt(wideWidthInBits, 1).shl(widthInBits)));
    }
    Value count = OpTy::create(rewriter, loc, operand);
    if constexpr (std::is_same_v<OpTy, math::CountLeadingZerosOp>) {
      count = arith::SubIOp::create(
          rewriter, loc, count,
          createScalarOrSplatConstant(rewriter, loc, wide,
                                      wideWidthInBits - widthInBits));
    }
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, count);
    return success();
  }
};

// Ops with one narrow operand, such as index and float conversions or an
// absolute value, read it extended; a narrow result is truncated back.
template <typename OpTy, typename ExtOpTy>
struct WidenNarrowOperand final : NarrowIntegerPattern<OpTy> {
  using NarrowIntegerPattern<OpTy>::NarrowIntegerPattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value in = op->getOperand(0);
    Type wide = this->getSelectedWideType(in.getType());
    if (!wide) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Location loc = op.getLoc();
    Value wideIn = ExtOpTy::create(rewriter, loc, wide, in);
    // A result of the operand's narrow type is computed wide and truncated.
    Type type = op.getType();
    bool truncatesResult = type == in.getType();
    auto widened =
        OpTy::create(rewriter, loc, truncatesResult ? wide : type, wideIn);
    rewriter.modifyOpInPlace(widened, [widened, op]() mutable {
      widened.getProperties() = op.getProperties();
      widened->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    });
    if (truncatesResult) {
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, widened);
    } else {
      rewriter.replaceOp(op, widened);
    }
    return success();
  }
};

struct WidenNarrowMinMaxReduction final
    : NarrowIntegerPattern<vector::ReductionOp> {
  using NarrowIntegerPattern::NarrowIntegerPattern;

  LogicalResult matchAndRewrite(vector::ReductionOp op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    if (!getSelectedWideType(type)) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    bool isSigned = false;
    switch (op.getKind()) {
    case vector::CombiningKind::MINSI:
    case vector::CombiningKind::MAXSI:
      isSigned = true;
      break;
    case vector::CombiningKind::MINUI:
    case vector::CombiningKind::MAXUI:
      break;
    default:
      return rewriter.notifyMatchFailure(op,
                                         "not an integer min/max reduction");
    }
    Location loc = op.getLoc();
    auto extend = [this, &rewriter, loc, isSigned](Value value) -> Value {
      Type wide = getWideIntegerType(value.getType(), wideWidthInBits);
      if (isSigned) {
        return arith::ExtSIOp::create(rewriter, loc, wide, value);
      }
      return arith::ExtUIOp::create(rewriter, loc, wide, value);
    };
    Value acc = op.getAcc() ? extend(op.getAcc()) : Value();
    Value result = vector::ReductionOp::create(rewriter, loc, op.getKind(),
                                               extend(op.getVector()), acc);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, result);
    return success();
  }
};

// A float conversion into a narrow type converts into the wide type, whose
// low bits hold the narrow result.
template <typename OpTy>
struct WidenNarrowFloatToInt final : NarrowIntegerPattern<OpTy> {
  using NarrowIntegerPattern<OpTy>::NarrowIntegerPattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    Type wide = this->getSelectedWideType(type);
    if (!wide) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Value result = OpTy::create(rewriter, op.getLoc(), wide, op.getIn());
    copyDiscardableAttrs(rewriter, op, result.getDefiningOp());
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, result);
    return success();
  }
};

struct WidenNarrowSelect final : NarrowIntegerPattern<arith::SelectOp> {
  using NarrowIntegerPattern::NarrowIntegerPattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    Type wide = getSelectedWideType(type);
    if (!wide) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Location loc = op.getLoc();
    Value trueValue =
        arith::ExtUIOp::create(rewriter, loc, wide, op.getTrueValue());
    Value falseValue =
        arith::ExtUIOp::create(rewriter, loc, wide, op.getFalseValue());
    Value result = arith::SelectOp::create(rewriter, loc, op.getCondition(),
                                           trueValue, falseValue);
    copyDiscardableAttrs(rewriter, op, result.getDefiningOp());
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, result);
    return success();
  }
};

} // namespace

void populateWidenNarrowIntegerHighBitOpPatterns(
    RewritePatternSet &patterns, unsigned wideWidthInBits,
    const std::function<bool(Type)> &isNarrow) {
  patterns.add<WidenNarrowIntegerOp<arith::DivUIOp, arith::ExtUIOp>,
               WidenNarrowIntegerOp<arith::RemUIOp, arith::ExtUIOp>,
               WidenNarrowIntegerOp<arith::ShRUIOp, arith::ExtUIOp>,
               WidenNarrowIntegerOp<arith::MinUIOp, arith::ExtUIOp>,
               WidenNarrowIntegerOp<arith::MaxUIOp, arith::ExtUIOp>,
               WidenNarrowIntegerOp<arith::DivSIOp, arith::ExtSIOp>,
               WidenNarrowIntegerOp<arith::RemSIOp, arith::ExtSIOp>,
               WidenNarrowIntegerOp<arith::ShRSIOp, arith::ExtSIOp>,
               WidenNarrowIntegerOp<arith::MinSIOp, arith::ExtSIOp>,
               WidenNarrowIntegerOp<arith::MaxSIOp, arith::ExtSIOp>,
               WidenNarrowOperand<arith::IndexCastOp, arith::ExtSIOp>,
               WidenNarrowOperand<arith::IndexCastUIOp, arith::ExtUIOp>,
               WidenNarrowOperand<math::AbsIOp, arith::ExtSIOp>,
               WidenNarrowOperand<arith::SIToFPOp, arith::ExtSIOp>,
               WidenNarrowOperand<arith::UIToFPOp, arith::ExtUIOp>,
               WidenNarrowMinMaxReduction, WidenNarrowCmpI,
               WidenNarrowExtension<arith::ExtSIOp>,
               WidenNarrowExtension<arith::ExtUIOp>>(patterns.getContext(),
                                                     wideWidthInBits, isNarrow);
}

void populateWidenNarrowIntegerBitCountPatterns(
    RewritePatternSet &patterns, unsigned wideWidthInBits,
    const std::function<bool(Type)> &isNarrow) {
  patterns.add<WidenNarrowBitCount<math::CountLeadingZerosOp>,
               WidenNarrowBitCount<math::CountTrailingZerosOp>,
               WidenNarrowBitCount<math::CtPopOp>>(patterns.getContext(),
                                                   wideWidthInBits, isNarrow);
}

void populateWidenNarrowIntegerLowBitOpPatterns(
    RewritePatternSet &patterns, unsigned wideWidthInBits,
    const std::function<bool(Type)> &isNarrow) {
  patterns.add<WidenNarrowIntegerOp<arith::AddIOp, arith::ExtUIOp, false>,
               WidenNarrowIntegerOp<arith::SubIOp, arith::ExtUIOp, false>,
               WidenNarrowIntegerOp<arith::MulIOp, arith::ExtUIOp, false>,
               WidenNarrowIntegerOp<arith::AndIOp, arith::ExtUIOp, false>,
               WidenNarrowIntegerOp<arith::OrIOp, arith::ExtUIOp, false>,
               WidenNarrowIntegerOp<arith::XOrIOp, arith::ExtUIOp, false>,
               WidenNarrowIntegerOp<arith::ShLIOp, arith::ExtUIOp, false>,
               WidenNarrowFloatToInt<arith::FPToSIOp>,
               WidenNarrowFloatToInt<arith::FPToUIOp>, WidenNarrowSelect>(
      patterns.getContext(), wideWidthInBits, isNarrow);
}

} // namespace mlir::iree_compiler
