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

template <typename OpTy, typename ExtOpTy, bool preserveProperties = true>
struct WidenNarrowIntegerOp final : OpRewritePattern<OpTy> {
  WidenNarrowIntegerOp(MLIRContext *context, unsigned wideWidthInBits,
                       IsNarrowFn isNarrow)
      : OpRewritePattern<OpTy>(context), wideWidthInBits(wideWidthInBits),
        isNarrow(std::move(isNarrow)) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    if (!isNarrow(type) || !getWideIntegerType(type, wideWidthInBits)) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Location loc = op.getLoc();
    Type wide = getWideIntegerType(type, wideWidthInBits);
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

  unsigned wideWidthInBits;
  IsNarrowFn isNarrow;
};

struct WidenNarrowCmpI final : OpRewritePattern<arith::CmpIOp> {
  WidenNarrowCmpI(MLIRContext *context, unsigned wideWidthInBits,
                  IsNarrowFn isNarrow)
      : OpRewritePattern(context), wideWidthInBits(wideWidthInBits),
        isNarrow(std::move(isNarrow)) {}

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getLhs().getType();
    if (!isNarrow(type) || !getWideIntegerType(type, wideWidthInBits)) {
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
    Type wide = getWideIntegerType(type, wideWidthInBits);
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

  unsigned wideWidthInBits;
  IsNarrowFn isNarrow;
};

// Extensions involving emulated integers first use the wide type to recover
// their logical source bits, including when the destination type is native.
//   %result = arith.extui %value : i4 to i8
// becomes
//   %wide = arith.extui %value : i4 to i32
//   %result = arith.trunci %wide : i32 to i8
template <typename ExtensionOp>
struct WidenNarrowExtension final : OpRewritePattern<ExtensionOp> {
  WidenNarrowExtension(MLIRContext *context, unsigned wideWidthInBits,
                       IsNarrowFn isNarrow)
      : OpRewritePattern<ExtensionOp>(context),
        wideWidthInBits(wideWidthInBits), isNarrow(std::move(isNarrow)) {}

  LogicalResult matchAndRewrite(ExtensionOp op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    if ((!isNarrow(type) && !isNarrow(op.getIn().getType())) ||
        !getWideIntegerType(type, wideWidthInBits)) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Value wide = ExtensionOp::create(rewriter, op.getLoc(),
                                     getWideIntegerType(type, wideWidthInBits),
                                     op.getIn());
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, wide);
    return success();
  }

  unsigned wideWidthInBits;
  IsNarrowFn isNarrow;
};

// Bit counts run on the zero-extended value; leading zeros drop the
// extension, trailing zeros stop at the logical width.
template <typename OpTy>
struct WidenNarrowBitCount final : OpRewritePattern<OpTy> {
  WidenNarrowBitCount(MLIRContext *context, unsigned wideWidthInBits,
                      IsNarrowFn isNarrow)
      : OpRewritePattern<OpTy>(context), wideWidthInBits(wideWidthInBits),
        isNarrow(std::move(isNarrow)) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    if (!isNarrow(type) || !getWideIntegerType(type, wideWidthInBits)) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Location loc = op.getLoc();
    Type wide = getWideIntegerType(type, wideWidthInBits);
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

  unsigned wideWidthInBits;
  IsNarrowFn isNarrow;
};

// Ops with one narrow operand, such as index and float conversions or an
// absolute value, read it extended; a narrow result is truncated back.
template <typename OpTy, typename ExtOpTy>
struct WidenNarrowOperand final : OpRewritePattern<OpTy> {
  WidenNarrowOperand(MLIRContext *context, unsigned wideWidthInBits,
                     IsNarrowFn isNarrow)
      : OpRewritePattern<OpTy>(context), wideWidthInBits(wideWidthInBits),
        isNarrow(std::move(isNarrow)) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value in = op->getOperand(0);
    if (!isNarrow(in.getType()) ||
        !getWideIntegerType(in.getType(), wideWidthInBits)) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Location loc = op.getLoc();
    Value wideIn = ExtOpTy::create(
        rewriter, loc, getWideIntegerType(in.getType(), wideWidthInBits), in);
    Type type = op.getType();
    if (type != in.getType()) {
      auto widened = OpTy::create(rewriter, loc, type, wideIn);
      rewriter.modifyOpInPlace(widened, [widened, op]() mutable {
        widened.getProperties() = op.getProperties();
        widened->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      });
      rewriter.replaceOp(op, widened);
      return success();
    }
    auto widened = OpTy::create(rewriter, loc, wideIn.getType(), wideIn);
    rewriter.modifyOpInPlace(widened, [widened, op]() mutable {
      widened.getProperties() = op.getProperties();
      widened->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    });
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, widened);
    return success();
  }

  unsigned wideWidthInBits;
  IsNarrowFn isNarrow;
};

struct WidenNarrowMinMaxReduction final
    : OpRewritePattern<vector::ReductionOp> {
  WidenNarrowMinMaxReduction(MLIRContext *context, unsigned wideWidthInBits,
                             IsNarrowFn isNarrow)
      : OpRewritePattern(context), wideWidthInBits(wideWidthInBits),
        isNarrow(std::move(isNarrow)) {}

  LogicalResult matchAndRewrite(vector::ReductionOp op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    if (!isNarrow(type) || !getWideIntegerType(type, wideWidthInBits)) {
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

  unsigned wideWidthInBits;
  IsNarrowFn isNarrow;
};

// A float conversion into a narrow type converts into the wide type, whose
// low bits hold the narrow result.
template <typename OpTy>
struct WidenNarrowFloatToInt final : OpRewritePattern<OpTy> {
  WidenNarrowFloatToInt(MLIRContext *context, unsigned wideWidthInBits,
                        IsNarrowFn isNarrow)
      : OpRewritePattern<OpTy>(context), wideWidthInBits(wideWidthInBits),
        isNarrow(std::move(isNarrow)) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    if (!isNarrow(type) || !getWideIntegerType(type, wideWidthInBits)) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Value result =
        OpTy::create(rewriter, op.getLoc(),
                     getWideIntegerType(type, wideWidthInBits), op.getIn());
    rewriter.modifyOpInPlace(result.getDefiningOp(), [result, op] {
      result.getDefiningOp()->setDiscardableAttrs(
          op->getDiscardableAttrDictionary());
    });
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, result);
    return success();
  }

  unsigned wideWidthInBits;
  IsNarrowFn isNarrow;
};

struct WidenNarrowSelect final : OpRewritePattern<arith::SelectOp> {
  WidenNarrowSelect(MLIRContext *context, unsigned wideWidthInBits,
                    IsNarrowFn isNarrow)
      : OpRewritePattern(context), wideWidthInBits(wideWidthInBits),
        isNarrow(std::move(isNarrow)) {}

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    Type type = op.getType();
    if (!isNarrow(type) || !getWideIntegerType(type, wideWidthInBits)) {
      return rewriter.notifyMatchFailure(op, "not a selected narrow integer");
    }
    Location loc = op.getLoc();
    Type wide = getWideIntegerType(type, wideWidthInBits);
    Value trueValue =
        arith::ExtUIOp::create(rewriter, loc, wide, op.getTrueValue());
    Value falseValue =
        arith::ExtUIOp::create(rewriter, loc, wide, op.getFalseValue());
    Value result = arith::SelectOp::create(rewriter, loc, op.getCondition(),
                                           trueValue, falseValue);
    rewriter.modifyOpInPlace(result.getDefiningOp(), [result, op] {
      result.getDefiningOp()->setDiscardableAttrs(
          op->getDiscardableAttrDictionary());
    });
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, type, result);
    return success();
  }

  unsigned wideWidthInBits;
  IsNarrowFn isNarrow;
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
