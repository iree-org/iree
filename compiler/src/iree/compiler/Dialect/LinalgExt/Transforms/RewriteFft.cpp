// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

// Create `arith.constant dense<...> : tensor<fftLength x i64>` of indices
// needed to shuffle FFT inputs into the expected bit-reversed order.
static Value getBitReversalBuffer(ImplicitLocOpBuilder &b, int64_t fftLength) {
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
  return arith::ConstantOp::create(b, type,
                                   DenseIntElementsAttr::get(type, values));
}

// Create the bit-reversed shuffled inputs for RFFT.
// Takes a float tensor and generates a vector of size 2:
// [0]: copy of `real` after being bit-reverse shuffled.
// [1]: tensor of zeroes for the imaginary part:
//      `arith.constant dense<0.000000e+00> : tensor<fftLength x f32>`
static SmallVector<Value> getBitReversalOrder(ImplicitLocOpBuilder &b,
                                              Value real, int64_t fftLength) {
  ShapedType realType = cast<ShapedType>(real.getType());
  int64_t rank = realType.getRank();
  auto realElementType = realType.getElementType();

  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(b, b.getLoc(), real);

  SmallVector<Value> outs;
  outs.push_back(tensor::EmptyOp::create(b, mixedSizes, realElementType));
  outs.push_back(tensor::EmptyOp::create(b, mixedSizes, realElementType));

  SmallVector<AffineMap> maps;
  maps.push_back(AffineMap::get(rank, 0, b.getAffineDimExpr(rank - 1),
                                b.getContext())); // input i64 tensor
  maps.push_back(b.getMultiDimIdentityMap(rank)); // output float
  maps.push_back(b.getMultiDimIdentityMap(rank)); // output float
  SmallVector<utils::IteratorType> iterTypes(rank,
                                             utils::IteratorType::parallel);

  Value zero =
      arith::ConstantOp::create(b, b.getLoc(), b.getZeroAttr(realElementType));

  Value indices = getBitReversalBuffer(b, fftLength);
  auto genericOp = linalg::GenericOp::create(
      b, TypeRange{outs}, indices, outs, maps, iterTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        SmallVector<Value> ivs;
        for (auto i : llvm::seq<uint64_t>(0, rank - 1)) {
          ivs.push_back(linalg::IndexOp::create(b, loc, i));
        }
        ivs.push_back(
            arith::IndexCastOp::create(b, loc, b.getIndexType(), args[0]));
        linalg::YieldOp::create(
            b, loc,
            SmallVector<Value>{
                tensor::ExtractOp::create(b, loc, real, ivs).getResult(),
                zero});
      });
  return genericOp.getResults();
}

// Create the shuffled inputs for a complex-valued FFT.
// Split complex input tensor into real and imaginary parts and shuffle
// into bit-reverse order. Generates a vector of size 2:
// [0]: `real` after being bit-reverse shuffled.
// [1]: `imag` after being bit-reverse shuffled.
// There is an optional scale factor used for normalization. This is done
// to avoid an additional scaling step at the end, since FFTs are linear,
// we can scale the input and just combine it with our existing `genericOp`
// used to shuffle the tensor and split into real and imag parts.
static SmallVector<Value>
getBitReversalOrderComplex(ImplicitLocOpBuilder &b, Value complexTensor,
                           int64_t fftLength, bool hasScale, double scale) {
  auto shapeType = cast<ShapedType>(complexTensor.getType());
  int64_t rank = shapeType.getRank();

  auto complexType = cast<ComplexType>(shapeType.getElementType());
  auto complexElemType = complexType.getElementType();

  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(b, b.getLoc(), complexTensor);

  SmallVector<Value> outs;
  outs.push_back(tensor::EmptyOp::create(b, mixedSizes, complexElemType));
  outs.push_back(tensor::EmptyOp::create(b, mixedSizes, complexElemType));

  SmallVector<AffineMap> maps;
  maps.push_back(AffineMap::get(rank, 0, b.getAffineDimExpr(rank - 1),
                                b.getContext())); // input i64 tensor
  maps.push_back(b.getMultiDimIdentityMap(rank)); // output float
  maps.push_back(b.getMultiDimIdentityMap(rank)); // output float
  SmallVector<utils::IteratorType> iterTypes(rank,
                                             utils::IteratorType::parallel);

  Value indices = getBitReversalBuffer(b, fftLength);

  Value scaleValue;
  if (hasScale) {
    scaleValue = arith::ConstantOp::create(
        b, complexElemType, b.getFloatAttr(complexElemType, scale));
  }

  auto genericOp = linalg::GenericOp::create(
      b, TypeRange{outs}, indices, outs, maps, iterTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        SmallVector<Value> ivs;
        for (auto i : llvm::seq<uint64_t>(0, rank - 1)) {
          ivs.push_back(linalg::IndexOp::create(b, loc, i));
        }
        ivs.push_back(
            arith::IndexCastOp::create(b, loc, b.getIndexType(), args[0]));
        auto elem = tensor::ExtractOp::create(b, loc, complexTensor, ivs);
        auto realPart = complex::ReOp::create(b, loc, complexElemType, elem);
        auto imagPart = complex::ImOp::create(b, loc, complexElemType, elem);
        if (hasScale) {
          linalg::YieldOp::create(
              b, loc,
              SmallVector<Value>{
                  arith::MulFOp::create(b, loc, realPart, scaleValue)
                      .getResult(),
                  arith::MulFOp::create(b, loc, imagPart, scaleValue)
                      .getResult()});
        } else {
          linalg::YieldOp::create(
              b, loc,
              SmallVector<Value>{realPart.getResult(), imagPart.getResult()});
        }
      });
  return genericOp.getResults();
}

// Get FFT coefficients for `stage` and FFT `direction`.
static SmallVector<Value> getCoeffConstants(ImplicitLocOpBuilder &b,
                                            FFTDirection direction, int stage) {
  constexpr std::complex<double> kI(0, 1);
  const double sign = FFTDirection::Forward == direction ? -1.0 : 1.0;
  int m = 1 << stage;
  int mh = m >> 1;
  SmallVector<Attribute> real, imag;
  for (auto i : llvm::seq<unsigned>(0, mh)) {
    auto v = std::exp(sign * 2.0 * M_PI * i / m * kI);
    real.push_back(b.getF32FloatAttr(v.real()));
    imag.push_back(b.getF32FloatAttr(v.imag()));
  }
  auto type = RankedTensorType::get({mh}, b.getF32Type());
  return {
      arith::ConstantOp::create(b, type, DenseFPElementsAttr::get(type, real)),
      arith::ConstantOp::create(b, type, DenseFPElementsAttr::get(type, imag))};
}

// Given appropriately shuffled and scaled real and imaginary tensors,
// run an FFT of size `fftLength`.
static SmallVector<Value> generateFFT(ImplicitLocOpBuilder &b,
                                      int64_t fftLength, FFTDirection direction,
                                      const SmallVector<Value> &inputs) {
  SmallVector<Value> results = inputs;
  int64_t lognPlus1 = std::log(fftLength) / std::log<int64_t>(2) + 1;
  for (auto s : llvm::seq<uint64_t>(1, lognPlus1)) {
    SmallVector<Value> inputs;
    inputs.push_back(arith::ConstantIndexOp::create(b, s));
    inputs.append(getCoeffConstants(b, direction, s));
    results =
        FftOp::create(b, TypeRange{results[0].getType(), results[1].getType()},
                      inputs, results)
            .getResults();
  }

  return results;
}

// Expand half-sized complex input to full size using conjugate symmetry.
// For IRFFT, the input is [X[0], X[1], ..., X[N/2]] (N/2+1 elements)
// and we need to expand it to:
// [X[0], X[1], ..., X[N/2], conj(X[N/2-1]), ..., conj(X[1])]
static Value expandConjugateSymmetry(ImplicitLocOpBuilder &b,
                                     Value halfSizedComplex,
                                     int64_t fftLength) {
  auto halfType = cast<RankedTensorType>(halfSizedComplex.getType());
  auto complexType = cast<ComplexType>(halfType.getElementType());
  int64_t rank = halfType.getRank();
  int64_t inputSize = fftLength / 2 + 1;

  // Second part: create the mirrored conjugate part (indices N/2+1 to N-1)
  // We need to extract elements from index N/2-1 down to 1 and conjugate them
  int64_t mirrorSize = fftLength - inputSize;

  SmallVector<int64_t> mirrorShape(halfType.getShape().begin(),
                                   halfType.getShape().end());
  mirrorShape.back() = mirrorSize;
  auto mirrorType = RankedTensorType::get(mirrorShape, complexType);

  SmallVector<OpFoldResult> mirrorMixedSizes;
  for (auto i : llvm::seq<uint64_t>(0, rank)) {
    mirrorMixedSizes.push_back(b.getIndexAttr(mirrorShape[i]));
  }

  Value emptyMirror = tensor::EmptyOp::create(b, mirrorMixedSizes, complexType);

  SmallVector<AffineMap> maps;
  maps.push_back(b.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iterTypes(rank,
                                             utils::IteratorType::parallel);

  auto mirrorOp = linalg::GenericOp::create(
      b, TypeRange{mirrorType}, ValueRange{}, emptyMirror, maps, iterTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Mirror index: for output index i, we want input[N/2 - 1 - i]
        // which mirrors indices N/2-1, N/2-2, ..., 1
        Value idx = linalg::IndexOp::create(b, loc, rank - 1);
        Value halfSizeMinusOne =
            arith::ConstantIndexOp::create(b, loc, fftLength / 2 - 1);
        Value mirrorIdx = arith::SubIOp::create(b, loc, halfSizeMinusOne, idx);

        SmallVector<Value> ivs;
        for (auto i : llvm::seq<uint64_t>(0, rank - 1)) {
          ivs.push_back(linalg::IndexOp::create(b, loc, i));
        }
        ivs.push_back(mirrorIdx);

        Value elem = tensor::ExtractOp::create(b, loc, halfSizedComplex, ivs);
        Value conjElem = complex::ConjOp::create(b, loc, elem);

        linalg::YieldOp::create(b, loc, conjElem);
      });

  // Concatenate first half and mirrored second half along the last dimension
  Value result = tensor::ConcatOp::create(
      b, rank - 1, SmallVector<Value>{halfSizedComplex, mirrorOp.getResult(0)});

  return result;
}

// Rewrite complex FFT, including direction and normalization.
FailureOr<std::pair<Value, Value>> rewriteFft(Operation *op, Value operand,
                                              int64_t fftLength,
                                              FFTDirection direction,
                                              FFTNormalization normalization,
                                              PatternRewriter &rewriter) {
  assert(llvm::isPowerOf2_64(fftLength) &&
         "expected FFT length to be a power of two");
  Location loc = op->getLoc();

  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  if (!operandType || !operandType.hasStaticShape()) {
    return failure();
  }

  // Require floating-point complex type.
  auto complexType = dyn_cast<ComplexType>(operandType.getElementType());
  if (!complexType || !complexType.getElementType().isFloat()) {
    return rewriter.notifyMatchFailure(op, "expected complex<float> types");
  }

  ImplicitLocOpBuilder b(loc, rewriter);

  SmallVector<Value> inputs = getBitReversalOrderComplex(
      b, operand, fftLength, normalization == FFTNormalization::Normalize,
      1.0 / (double)fftLength);

  SmallVector<Value> results = generateFFT(b, fftLength, direction, inputs);
  return std::make_pair(results[0], results[1]);
}

// Rewrite real to half-sized complex forward FFT by doing a full FFT with an
// imaginary part of `<0.0, 0.0,...>`, then truncating the result.
FailureOr<std::pair<Value, Value>> rewriteRfft(Operation *op, Value operand,
                                               int64_t fftLength,
                                               PatternRewriter &rewriter) {
  assert(llvm::isPowerOf2_64(fftLength) &&
         "expected FFT length to be a power of two");
  Location loc = op->getLoc();

  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  if (!operandType || !operandType.hasStaticShape()) {
    return failure();
  }

  // Require float type.
  if (!operandType.getElementType().isFloat()) {
    return rewriter.notifyMatchFailure(op, "expected float types");
  }

  ImplicitLocOpBuilder b(loc, rewriter);

  SmallVector<Value> inputs = getBitReversalOrder(b, operand, fftLength);
  SmallVector<Value> results =
      generateFFT(b, fftLength, FFTDirection::Forward, inputs);

  // Extract the first N/2+1 elements.
  SmallVector<int64_t> shape(operandType.getShape().begin(),
                             operandType.getShape().end());
  shape.back() = fftLength / 2 + 1;
  auto ty = RankedTensorType::get(shape, operandType.getElementType());
  SmallVector<OpFoldResult> offsets(ty.getRank(), b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(ty.getRank(), b.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(b, b.getLoc(), operand);
  sizes.back() = b.getIndexAttr(shape.back());
  Value real = tensor::ExtractSliceOp::create(b, ty, results[0], offsets, sizes,
                                              strides);
  Value imag = tensor::ExtractSliceOp::create(b, ty, results[1], offsets, sizes,
                                              strides);

  return std::make_pair(real, imag);
}

// Rewrite IRFFT (inverse real FFT).
// Takes half-sized complex input (due to conjugate symmetry) and produces
// real output. This expands the input to full size using conjugate symmetry,
// then performs a normalized backward FFT.
FailureOr<std::pair<Value, Value>> rewriteIrfft(Operation *op, Value operand,
                                                int64_t fftLength,
                                                PatternRewriter &rewriter) {
  assert(llvm::isPowerOf2_64(fftLength) &&
         "expected FFT length to be a power of two");
  Location loc = op->getLoc();

  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  if (!operandType || !operandType.hasStaticShape()) {
    return failure();
  }

  // Require floating-point complex type.
  auto complexType = dyn_cast<ComplexType>(operandType.getElementType());
  if (!complexType || !complexType.getElementType().isFloat()) {
    return rewriter.notifyMatchFailure(op, "expected complex<float> types");
  }

  // Verify input size is (fftLength / 2 + 1) on the last dimension
  int64_t expectedInputSize = fftLength / 2 + 1;
  int64_t actualInputSize = operandType.getShape().back();
  if (actualInputSize != expectedInputSize) {
    return rewriter.notifyMatchFailure(
        op, "expected input size to be fftLength/2 + 1 for IRFFT");
  }

  ImplicitLocOpBuilder b(loc, rewriter);

  // Expand half-sized input to full size using conjugate symmetry
  Value expandedInput = expandConjugateSymmetry(b, operand, fftLength);

  // Now perform a normalized backward FFT on the expanded input
  return rewriteFft(op, expandedInput, fftLength, FFTDirection::Backward,
                    FFTNormalization::Normalize, rewriter);
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
