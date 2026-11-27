// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/StableHLOPaddingUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

struct WideInteger {
  Value low;
  Value high;
};

// Multiply in 32-bit limbs using only portable i64 operations. The partial
// sums fit u64; the returned pair is the exact unsigned 128-bit product.
static WideInteger buildWideProduct(ImplicitLocOpBuilder &b, Value lhs,
                                    Value rhs) {
  Value shift = b.create<arith::ConstantIntOp>(32, 64);
  Value mask = b.create<arith::ConstantIntOp>(0xFFFFFFFFLL, 64);
  Value lhsLow = b.createOrFold<arith::AndIOp>(lhs, mask);
  Value lhsHigh = b.createOrFold<arith::ShRUIOp>(lhs, shift);
  Value rhsLow = b.createOrFold<arith::AndIOp>(rhs, mask);
  Value rhsHigh = b.createOrFold<arith::ShRUIOp>(rhs, shift);
  Value lowProduct = b.createOrFold<arith::MulIOp>(lhsLow, rhsLow);
  Value middle = b.createOrFold<arith::AddIOp>(
      b.createOrFold<arith::MulIOp>(lhsHigh, rhsLow),
      b.createOrFold<arith::ShRUIOp>(lowProduct, shift));
  Value middleLow = b.createOrFold<arith::AddIOp>(
      b.createOrFold<arith::AndIOp>(middle, mask),
      b.createOrFold<arith::MulIOp>(lhsLow, rhsHigh));
  Value high = b.createOrFold<arith::AddIOp>(
      b.createOrFold<arith::MulIOp>(lhsHigh, rhsHigh),
      b.createOrFold<arith::ShRUIOp>(middle, shift));
  high = b.createOrFold<arith::AddIOp>(
      high, b.createOrFold<arith::ShRUIOp>(middleLow, shift));
  Value low = b.createOrFold<arith::AddIOp>(
      b.createOrFold<arith::ShLIOp>(middleLow, shift),
      b.createOrFold<arith::AndIOp>(lowProduct, mask));
  return {low, high};
}

static WideInteger addPadding(ImplicitLocOpBuilder &b, WideInteger value,
                              Value padding, bool isUnsigned) {
  Value low = b.createOrFold<arith::AddIOp>(value.low, padding);
  Value carry =
      b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::ult, low, value.low);
  carry = b.createOrFold<arith::ExtUIOp>(b.getI64Type(), carry);
  Value high = b.createOrFold<arith::AddIOp>(value.high, carry);
  if (!isUnsigned) {
    Value shift = b.create<arith::ConstantIntOp>(63, 64);
    Value sign = b.createOrFold<arith::ShRSIOp>(padding, shift);
    high = b.createOrFold<arith::AddIOp>(high, sign);
  }
  return {low, high};
}

} // namespace

PaddedDimension buildPaddedDimension(OpBuilder &builder, Location loc,
                                     Value size, Value low, Value high,
                                     Value interior, bool isUnsignedPadding) {
  ImplicitLocOpBuilder b(loc, builder);
  Value zero = b.create<arith::ConstantIntOp>(0, 64);
  Value one = b.create<arith::ConstantIntOp>(1, 64);
  Value sizeInI64 = b.createOrFold<arith::IndexCastUIOp>(b.getI64Type(), size);
  Value span = b.createOrFold<arith::SubIOp>(sizeInI64, one);
  span = b.createOrFold<arith::MaxSIOp>(span, zero);
  // With size <= INT64_MAX, even unsigned64 interior and both unsigned64
  // edges sum to at most 2^127 - 1. Signed edges can only reduce that bound,
  // and the smallest possible sum is -2^64, so the signed high limb is exact.
  WideInteger total = buildWideProduct(b, span, interior);
  total = addPadding(b, total, sizeInI64, /*isUnsigned=*/true);
  total = addPadding(b, total, low, isUnsignedPadding);
  total = addPadding(b, total, high, isUnsignedPadding);

  Value negative = b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                                 total.high, zero);
  Value bits = b.createOrFold<arith::SelectOp>(negative, zero, total.low);
  Value result = b.createOrFold<arith::IndexCastUIOp>(b.getIndexType(), bits);
  return {result, negative};
}

} // namespace mlir::iree_compiler::stablehlo
