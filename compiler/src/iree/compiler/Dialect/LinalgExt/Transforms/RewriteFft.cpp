// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

FailureOr<std::pair<Value, Value>> rewriteFft(Operation *op, Value operand,
                                              int64_t fftLength,
                                              PatternRewriter &rewriter) {

  assert(llvm::isPowerOf2_64(fftLength) &&
         "expected FFT length to be a power of two");
  Location loc = op->getLoc();

  auto operandType = llvm::dyn_cast<RankedTensorType>(operand.getType());
  if (!operandType || !operandType.hasStaticShape()) {
    return failure();
  }

  // Skip else getBitReversalOrder produces invalid dense elements attr.
  if (!operandType.getElementType().isF32())
    return rewriter.notifyMatchFailure(op, "expected F32 types");

  ImplicitLocOpBuilder b(loc, rewriter);

  auto getBitReversalOrder = [](ImplicitLocOpBuilder &b, Value real,
                                int64_t fftLength) -> SmallVector<Value> {
    ShapedType realType = llvm::cast<ShapedType>(real.getType());
    int64_t rank = realType.getRank();

    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(b, b.getLoc(), real);
    Value emptyTensor =
        b.create<tensor::EmptyOp>(mixedSizes, realType.getElementType());

    SmallVector<AffineMap> maps;
    maps.push_back(
        AffineMap::get(rank, 0, b.getAffineDimExpr(rank - 1), b.getContext()));
    maps.push_back(b.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);

    auto getBitReversalBuffer = [](ImplicitLocOpBuilder &b,
                                   int64_t fftLength) -> Value {
      SmallVector<Attribute> values;
      int64_t logn = std::log(fftLength) / std::log<int64_t>(2);
      for (int64_t i = 0; i < fftLength; ++i) {
        int64_t r = 0;
        for (int64_t j = 0; j < logn; ++j) {
          r |= ((i >> j) & 1) << (logn - j - 1);
        }
        values.push_back(b.getI64IntegerAttr(r));
      }
      auto type = RankedTensorType::get({fftLength}, b.getI64Type());
      return b.create<arith::ConstantOp>(
          type, DenseIntElementsAttr::get(type, values));
    };
    Value indices = getBitReversalBuffer(b, fftLength);
    auto genericOp = b.create<linalg::GenericOp>(
        TypeRange{realType}, indices, emptyTensor, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          SmallVector<Value> ivs;
          for (auto i : llvm::seq<uint64_t>(0, rank - 1)) {
            ivs.push_back(b.create<linalg::IndexOp>(loc, i));
          }
          ivs.push_back(
              b.create<arith::IndexCastOp>(loc, b.getIndexType(), args[0]));
          b.create<linalg::YieldOp>(
              loc, b.create<tensor::ExtractOp>(loc, real, ivs).getResult());
        });
    return {genericOp.getResult(0),
            b.create<arith::ConstantOp>(
                realType,
                DenseFPElementsAttr::get(
                    realType, llvm::cast<Attribute>(b.getF32FloatAttr(0.0))))};
  };

  SmallVector<Value> results = getBitReversalOrder(b, operand, fftLength);

  auto getCoeffConstants = [](ImplicitLocOpBuilder &b,
                              int stage) -> SmallVector<Value> {
    constexpr std::complex<double> kI(0, 1);
    int m = 1 << stage;
    int mh = m >> 1;
    SmallVector<Attribute> real, imag;
    for (auto i : llvm::seq<unsigned>(0, mh)) {
      auto v = std::exp(-2 * M_PI * i / m * kI);
      real.push_back(b.getF32FloatAttr(v.real()));
      imag.push_back(b.getF32FloatAttr(v.imag()));
    }
    auto type = RankedTensorType::get({mh}, b.getF32Type());
    return {
        b.create<arith::ConstantOp>(type, DenseFPElementsAttr::get(type, real)),
        b.create<arith::ConstantOp>(type,
                                    DenseFPElementsAttr::get(type, imag))};
  };

  int64_t lognPlus1 = std::log(fftLength) / std::log<int64_t>(2) + 1;
  for (auto s : llvm::seq<uint64_t>(1, lognPlus1)) {
    SmallVector<Value> inputs;
    inputs.push_back(b.create<arith::ConstantIndexOp>(s));
    inputs.append(getCoeffConstants(b, s));
    results = b.create<IREE::LinalgExt::FftOp>(
                   TypeRange{results[0].getType(), results[1].getType()},
                   inputs, results)
                  .getResults();
  }

  SmallVector<int64_t> shape(operandType.getShape().begin(),
                             operandType.getShape().end());
  shape.back() = fftLength / 2 + 1;
  auto ty = RankedTensorType::get(shape, operandType.getElementType());
  SmallVector<OpFoldResult> offsets(ty.getRank(), b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(ty.getRank(), b.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(b, b.getLoc(), operand);
  sizes.back() = b.getIndexAttr(shape.back());
  Value real =
      b.create<tensor::ExtractSliceOp>(ty, results[0], offsets, sizes, strides);
  Value imag =
      b.create<tensor::ExtractSliceOp>(ty, results[1], offsets, sizes, strides);

  return std::make_pair(real, imag);
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
