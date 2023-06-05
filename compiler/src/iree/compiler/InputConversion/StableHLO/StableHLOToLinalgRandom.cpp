// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO random number generation to Linalg
// dialect.

#include "iree/compiler/InputConversion/StableHLO/LegalizeToLinalgUtils.h"
#include "iree/compiler/InputConversion/StableHLO/Rewriters.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {
class ArithOpBuilder {
 public:
  ArithOpBuilder(OpBuilder b, Location l, Value v)
      : builder(b), loc(l), value(v) {}

  explicit operator Value() { return value; }
  Value val() { return value; }

  ArithOpBuilder constantI(int64_t value, int64_t bits) {
    Value val = builder.create<arith::ConstantOp>(
        loc, builder.getIntegerAttr(builder.getIntegerType(bits), value));
    return ArithOpBuilder(builder, loc, val);
  }

  ArithOpBuilder extendUI(int32_t bits) {
    Value ext = builder.create<arith::ExtUIOp>(
        loc, builder.getIntegerType(bits), value);
    return ArithOpBuilder(builder, loc, ext);
  }

  ArithOpBuilder truncI(int64_t bits) {
    if (value.getType().getIntOrFloatBitWidth() == bits) return *this;
    Value trunc = builder.create<arith::TruncIOp>(
        loc, builder.getIntegerType(bits), value);
    return ArithOpBuilder(builder, loc, trunc);
  }

  ArithOpBuilder linalgIndex(int32_t index) {
    Value val = builder.create<linalg::IndexOp>(loc, index);
    return ArithOpBuilder(builder, loc, val);
  }

  ArithOpBuilder indexCast(int32_t bitwidth) {
    if (isa<IntegerType>(value.getType())) {
      Value cast = builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(), value);
      return ArithOpBuilder(builder, loc, cast);
    }

    Value cast = builder.create<arith::IndexCastOp>(
        loc, builder.getIntegerType(bitwidth), value);
    return ArithOpBuilder(builder, loc, cast);
  }

  ArithOpBuilder rotateLeft(int32_t rotation) {
    int32_t bits = value.getType().getIntOrFloatBitWidth();
    ArithOpBuilder cLeft = constantI(rotation, bits);
    ArithOpBuilder cRight = constantI(bits - rotation, bits);
    ArithOpBuilder rLeft = (*this << cLeft);
    ArithOpBuilder rRight = (*this >> cRight);
    return rLeft | rRight;
  }

  ArithOpBuilder operator+(ArithOpBuilder &rhs) {
    Value res = builder.create<arith::AddIOp>(loc, value, rhs.value);
    return ArithOpBuilder(builder, loc, res);
  }

  ArithOpBuilder operator*(ArithOpBuilder &rhs) {
    Value res = builder.create<arith::MulIOp>(loc, value, rhs.value);
    return ArithOpBuilder(builder, loc, res);
  }

  ArithOpBuilder operator|(ArithOpBuilder &rhs) {
    Value res = builder.create<arith::OrIOp>(loc, value, rhs.value);
    return ArithOpBuilder(builder, loc, res);
  }

  ArithOpBuilder operator^(ArithOpBuilder &rhs) {
    Value res = builder.create<arith::XOrIOp>(loc, value, rhs.value);
    return ArithOpBuilder(builder, loc, res);
  }

  ArithOpBuilder operator<<(ArithOpBuilder &rhs) {
    Value shl = builder.create<arith::ShLIOp>(loc, value, rhs.value);
    return ArithOpBuilder(builder, loc, shl);
  }

  ArithOpBuilder operator>>(ArithOpBuilder &rhs) {
    Value shr = builder.create<arith::ShRUIOp>(loc, value, rhs.value);
    return ArithOpBuilder(builder, loc, shr);
  }

 private:
  OpBuilder builder;
  Location loc;
  Value value;
};

std::pair<ArithOpBuilder, ArithOpBuilder> splitI64(ArithOpBuilder i64) {
  auto low = i64.truncI(32);
  auto c32 = i64.constantI(/*value=*/32, /*bits=*/64);
  auto high = (i64 >> c32).truncI(32);
  return {low, high};
}

ArithOpBuilder fuseI32s(ArithOpBuilder low, ArithOpBuilder high) {
  auto c32 = high.constantI(/*value=*/32, /*bits=*/64);
  high = high.extendUI(64) << c32;
  low = low.extendUI(64);
  return low | high;
}

// Implements the ThreeFry counter-based PRNG algorithm.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
std::pair<ArithOpBuilder, ArithOpBuilder> runThreeFry2xi32(
    ArithOpBuilder key0, ArithOpBuilder key1, ArithOpBuilder initialState) {
  ArithOpBuilder index = initialState.linalgIndex(0);
  index = index.indexCast(64);
  index = index + initialState;

  // Split into the 2xi32 used for threefry.
  std::pair<ArithOpBuilder, ArithOpBuilder> input = splitI64(index);
  ArithOpBuilder input0 = input.first;
  ArithOpBuilder input1 = input.second;

  // Magic number and rotation distances specified by the Threefry2x32
  // algorithm.
  llvm::SmallVector<int32_t, 8> rotations = {13, 15, 26, 6, 17, 29, 16, 24};
  ArithOpBuilder magic = key0.constantI(/*value=*/0x1bd11bda, /*bits=*/32);

  ArithOpBuilder key2 = magic ^ key0 ^ key1;
  std::array<ArithOpBuilder, 3> ks{key0, key1, key2};
  std::array<ArithOpBuilder, 2> x{input0 + key0, input1 + key1};

  // Performs a single round of the Threefry2x32 algorithm, with a rotation
  // amount 'rotation'.
  for (int i = 0; i < 5; ++i) {
    int32_t rot = (4 * i) % rotations.size();
    int32_t k1 = (i + 1) % ks.size();
    int32_t k2 = (i + 2) % ks.size();

    for (int j = 0; j < 4; ++j) {
      x[0] = x[0] + x[1];
      x[1] = x[1].rotateLeft(rotations[rot + j]);
      x[1] = x[0] ^ x[1];
    }

    ArithOpBuilder c = x[0].constantI(/*value=*/i + 1, /*bits=*/32);
    x[0] = x[0] + ks[k1];
    x[1] = x[1] + ks[k2];
    x[1] = x[1] + c;
  }

  return std::pair<ArithOpBuilder, ArithOpBuilder>(x[0], x[1]);
}

// Extract and potentially reconstruct the i32 key-pair as necessary.
std::pair<Value, Value> extractKey32(OpBuilder &builder, Location loc,
                                     Value store) {
  auto storeTy = cast<ShapedType>(store.getType());
  if (storeTy.getRank() != 1) return {nullptr, nullptr};

  Type storeETy = storeTy.getElementType();
  IntegerType i32Ty = builder.getIntegerType(32);
  IntegerType i64Ty = builder.getIntegerType(64);

  if (storeTy.getDimSize(0) == 4 && storeETy.isInteger(32)) {
    Value idx0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value idx1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value key0 = builder.create<tensor::ExtractOp>(loc, store, idx0);
    Value key1 = builder.create<tensor::ExtractOp>(loc, store, idx1);
    key0 = builder.create<arith::BitcastOp>(loc, i32Ty, key0);
    key1 = builder.create<arith::BitcastOp>(loc, i32Ty, key1);
    return {key0, key1};
  }

  if (storeTy.getDimSize(0) == 2 && storeETy.isInteger(64)) {
    Value idx1 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value state = builder.create<tensor::ExtractOp>(loc, store, idx1);
    Value cast = builder.create<arith::BitcastOp>(loc, i64Ty, state);
    auto pair = splitI64(ArithOpBuilder(builder, loc, cast));
    return std::pair<Value, Value>(pair.first, pair.second);
  }

  return {nullptr, nullptr};
}

// Extract and potentially reconstruct the i64 state as necessary.
Value extractState64(OpBuilder &builder, Location loc, Value store) {
  auto storeTy = cast<ShapedType>(store.getType());
  if (storeTy.getRank() != 1) return nullptr;

  Type storeETy = storeTy.getElementType();
  IntegerType i64Ty = builder.getIntegerType(64);

  if (storeTy.getDimSize(0) == 2 && storeETy.isInteger(64)) {
    Value idx1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value state = builder.create<tensor::ExtractOp>(loc, store, idx1);
    Value cast = builder.create<arith::BitcastOp>(loc, i64Ty, state);
    return cast;
  }

  if (storeTy.getDimSize(0) == 4 && storeETy.isInteger(32)) {
    Value idx2 = builder.create<arith::ConstantIndexOp>(loc, 2);
    Value idx3 = builder.create<arith::ConstantIndexOp>(loc, 3);

    Value low = builder.create<tensor::ExtractOp>(loc, store, idx2);
    Value high = builder.create<tensor::ExtractOp>(loc, store, idx3);

    ArithOpBuilder i64 = fuseI32s(ArithOpBuilder(builder, loc, high),
                                  ArithOpBuilder(builder, loc, low));
    return builder.create<arith::BitcastOp>(loc, i64Ty, i64.val());
  }

  return nullptr;
}

Value setState64(OpBuilder &b, Location loc, Value store, Value state) {
  auto storeTy = cast<ShapedType>(store.getType());
  if (storeTy.getRank() != 1) return nullptr;

  Type storeETy = storeTy.getElementType();

  if (storeTy.getDimSize(0) == 2 && storeETy.isInteger(64)) {
    state = b.create<arith::BitcastOp>(loc, storeETy, state);
    Value idx1 = b.create<arith::ConstantIndexOp>(loc, 1);
    return b.create<tensor::InsertOp>(loc, storeTy, state, store,
                                      ValueRange{idx1});
  }

  if (storeTy.getDimSize(0) == 4 && storeETy.isInteger(32)) {
    Value idx2 = b.create<arith::ConstantIndexOp>(loc, 2);
    Value idx3 = b.create<arith::ConstantIndexOp>(loc, 3);
    std::pair<ArithOpBuilder, ArithOpBuilder> states =
        splitI64(ArithOpBuilder(b, loc, state));
    Value state0 =
        b.create<arith::BitcastOp>(loc, storeETy, states.first.val());
    Value state1 =
        b.create<arith::BitcastOp>(loc, storeETy, states.second.val());
    Value insert0 = b.create<tensor::InsertOp>(loc, storeTy, state0, store,
                                               ValueRange{idx2});
    Value insert1 = b.create<tensor::InsertOp>(loc, storeTy, state1, insert0,
                                               ValueRange{idx3});
    return insert1;
  }

  return nullptr;
}

Value reshapeToTarget(OpBuilder &builder, Location loc, ShapedType destTy,
                      Value src) {
  auto srcTy = cast<ShapedType>(src.getType());
  // Expand out to the target shape.

  auto reassociationIndices =
      getReassociationIndicesForCollapse(destTy.getShape(), srcTy.getShape());
  if (reassociationIndices.has_value()) {
    src = builder.create<tensor::ExpandShapeOp>(loc, destTy, src,
                                                reassociationIndices.value());
  }

  // It is also possible our target is Rank-0, then we would
  // need to collapse.
  reassociationIndices =
      getReassociationIndicesForCollapse(srcTy.getShape(), destTy.getShape());
  if (reassociationIndices.has_value()) {
    src = builder.create<tensor::CollapseShapeOp>(loc, destTy, src,
                                                  reassociationIndices.value());
  }

  return src;
}

// Compute the shape for computing three fry.
std::pair<ShapedType, int64_t> threeFry32Shape(ShapedType resultTy) {
  if (resultTy.getRank() == 0) {
    return {resultTy, 0};
  }

  ArrayRef<int64_t> shape = resultTy.getShape();
  uint64_t halfDim =
      std::max_element(shape.begin(), shape.end()) - shape.begin();

  for (int i = 0, s = shape.size(); i < s; i++) {
    if (shape[i] & 0x1) continue;
    halfDim = i;
    break;
  }

  llvm::SmallVector<int64_t> newShape(shape);
  newShape[halfDim] = (newShape[halfDim] + 1) / 2;
  if (halfDim == (newShape.size() - 1)) {
    newShape.push_back(1);
  }

  return {RankedTensorType::get(newShape, resultTy.getElementType()), halfDim};
}

/// This implementation generates a 32-bit tensor of ThreeFry random numbers.
/// It matches the XLA implementation bit-exact and includes an inefficient
/// method of concatenating / slicing the pairs of generated numbers.
///
/// We should consider dropping the complex slicing and simply generating
/// 2x the values, then downcast to a 32-bit. It substantially simplifies
/// the computation and avoids the concat / slice behavior.
LogicalResult generateLinalgThreeFry32(OpBuilder &builder, Location loc,
                                       ShapedType resultTy, Value &store,
                                       Value &result) {
  Type resultETy = resultTy.getElementType();

  // Extract the stateful values as an i64 and increment the state ahead.
  Value initialState = extractState64(builder, loc, store);
  if (!initialState) return failure();

  std::pair<Value, Value> keys = extractKey32(builder, loc, store);
  if (!keys.first || !keys.second) return failure();

  ArithOpBuilder key0(builder, loc, keys.first);
  ArithOpBuilder key1(builder, loc, keys.second);

  // Compute the intermediate type we use to compute three fry values, including
  // the dimension that was halved.
  auto pair = threeFry32Shape(resultTy);
  ShapedType intermediateType = pair.first;
  int64_t halfDim = pair.second;
  int64_t count = intermediateType.getNumElements();

  // Compute the number of random i64s generated and increment state.
  Value countVal =
      builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(count));
  Value newState = builder.create<arith::AddIOp>(loc, initialState, countVal);

  // Generate a 1D tensor with for the random values.
  Value destLeft = builder.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>({count}), resultETy);
  Value destRight = builder.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>({count}), resultETy);

  ShapedType destTy = llvm::cast<ShapedType>(destLeft.getType());

  SmallVector<AffineMap> indexingMaps(2, builder.getMultiDimIdentityMap(1));
  SmallVector<utils::IteratorType> iterators(1, utils::IteratorType::parallel);

  linalg::GenericOp generic = builder.create<linalg::GenericOp>(
      loc, TypeRange{destTy, destTy},
      /*inputs=*/ValueRange(),
      /*outputs=*/ValueRange{destLeft, destRight},
      /*indexingMaps=*/indexingMaps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange) {
        // Grab three fry results and write to each array.
        auto split = runThreeFry2xi32(
            key0, key1, ArithOpBuilder(b, nestedLoc, initialState));
        auto first = split.first.truncI(resultETy.getIntOrFloatBitWidth());
        auto second = split.second.truncI(resultETy.getIntOrFloatBitWidth());
        b.create<linalg::YieldOp>(loc, ValueRange{first.val(), second.val()});
      });

  if (resultTy.getNumElements() == 1) {
    result = reshapeToTarget(builder, loc, resultTy, generic.getResult(0));
    store = setState64(builder, loc, store, newState);
    return success();
  }

  // Reshape to the target size and concatenate on the dimension following the
  // half dimension.
  Value random0 =
      reshapeToTarget(builder, loc, intermediateType, generic.getResult(0));
  Value random1 =
      reshapeToTarget(builder, loc, intermediateType, generic.getResult(1));
  Value concatenate = builder.create<mlir::stablehlo::ConcatenateOp>(
      loc, ValueRange{random0, random1},
      builder.getI64IntegerAttr(halfDim + 1));

  // Collapse the concat dimension back into the parent.
  llvm::SmallVector<int64_t> collapseShape(resultTy.getShape());
  collapseShape[halfDim] =
      collapseShape[halfDim] + (collapseShape[halfDim] & 1);
  Value reshape = builder.create<mlir::stablehlo::ReshapeOp>(
      loc, resultTy.clone(collapseShape), concatenate);

  // Slice to only the required results.
  llvm::SmallVector<int64_t> offset(resultTy.getRank(), 0);
  llvm::SmallVector<int64_t> stride(resultTy.getRank(), 1);
  Value slice = builder.create<mlir::stablehlo::SliceOp>(
      loc, resultTy, reshape, builder.getI64TensorAttr(offset),
      builder.getI64TensorAttr(resultTy.getShape()),
      builder.getI64TensorAttr(stride));

  // Set the new tensor values.
  store = setState64(builder, loc, store, newState);
  result = slice;

  return success();
}

LogicalResult generateLinalgThreeFry64(OpBuilder &builder, Location loc,
                                       ShapedType resultTy, Value &store,
                                       Value &result) {
  Type resultETy = resultTy.getElementType();
  int64_t count = resultTy.getNumElements();

  // Extract the stateful values as an i64 and increment the state ahead.
  Value initialState = extractState64(builder, loc, store);
  if (!initialState) return failure();

  std::pair<Value, Value> keys = extractKey32(builder, loc, store);
  if (!keys.first || !keys.second) return failure();

  ArithOpBuilder key0(builder, loc, keys.first);
  ArithOpBuilder key1(builder, loc, keys.second);

  // Compute the number of random i64s generated and increment state.
  Value countVal =
      builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(count));
  Value newState = builder.create<arith::AddIOp>(loc, initialState, countVal);

  // Generate a 1D tensor with for the random values.
  Value dest = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>({count}),
                                               resultETy);
  ShapedType destTy = llvm::cast<ShapedType>(dest.getType());

  SmallVector<AffineMap> indexingMaps(1, builder.getMultiDimIdentityMap(1));
  SmallVector<utils::IteratorType> iterators(1, utils::IteratorType::parallel);

  auto random = builder.create<linalg::GenericOp>(
      loc, destTy, /*inputs=*/ValueRange(),
      /*outputs=*/ValueRange{dest},
      /*indexingMaps=*/indexingMaps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange) {
        // Generate three fry results, fuse, and return an
        // i64.
        auto split = runThreeFry2xi32(
            key0, key1, ArithOpBuilder(b, nestedLoc, initialState));
        Value result = fuseI32s(split.first, split.second).val();
        b.create<linalg::YieldOp>(nestedLoc, result);
      });

  store = setState64(builder, loc, store, newState);
  result = reshapeToTarget(builder, loc, resultTy, random.getResult(0));
  return success();
}

using PhiloxKey = std::pair<ArithOpBuilder, ArithOpBuilder>;
using PhiloxState = std::array<ArithOpBuilder, 4>;

// Computes high and low words from multiplying 32 bit integers.
// Per the paper, mulhi and mullo of the same arguments can be computed
// Simultaneously in a single instruction on x86 architectures.
std::pair<ArithOpBuilder, ArithOpBuilder> multiplyHilo(ArithOpBuilder counter,
                                                       ArithOpBuilder key) {
  counter = counter.extendUI(64);
  key = key.extendUI(64);
  ArithOpBuilder product = counter * key;
  ArithOpBuilder ci64 = counter.constantI(/*value=*/32, /*bits=*/64);
  ArithOpBuilder hi = product >> ci64;
  hi = hi.truncI(32);
  product = product.truncI(32);
  return std::pair<ArithOpBuilder, ArithOpBuilder>{hi, product};
}

PhiloxState philoxRound(PhiloxState x, PhiloxKey key) {
  // These are philox specific constants.
  ArithOpBuilder m0 = x[0].constantI(0xD2511F53, 32);
  ArithOpBuilder m1 = x[2].constantI(0xCD9E8D57, 32);
  std::pair<ArithOpBuilder, ArithOpBuilder> p0 = multiplyHilo(x[0], m0);
  std::pair<ArithOpBuilder, ArithOpBuilder> p1 = multiplyHilo(x[2], m1);

  PhiloxState state = {p1.first ^ x[1] ^ key.first, p1.second,
                       p0.first ^ x[3] ^ key.second, p0.second};
  return state;
}

PhiloxKey raiseKey(PhiloxKey key) {
  // These are philox specific constants.
  ArithOpBuilder w0 = key.first.constantI(0x9E3779B9, 32);
  ArithOpBuilder w1 = key.first.constantI(0xBB67AE85, 32);
  return PhiloxKey{key.first + w0, key.second + w1};
}

// Implements the Philox 4x32 counter-based PRNG algorithm.
// The Philox PRNG has been proposed in:
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
std::array<ArithOpBuilder, 4> runPhilox4x32(PhiloxKey key,
                                            ArithOpBuilder state) {
  ArithOpBuilder index = state.linalgIndex(0);
  index = index.indexCast(64);
  index = index + state;

  // Split into the 2xi32 used for threefry.
  std::pair<ArithOpBuilder, ArithOpBuilder> input = splitI64(index);
  ArithOpBuilder input0 = input.first;
  ArithOpBuilder input1 = input.second;

  // We initialize the state as such to match the XLA implementation.
  PhiloxState state4 = {input0, input1, key.first, key.second};

  // We perform 10 rounds to match the XLA implementation.
  constexpr int kNumRounds = 10;
  for (int round = 0; round < kNumRounds; ++round, key = raiseKey(key)) {
    state4 = philoxRound(state4, key);
  }
  return state4;
}

// Generates an array of primitive type U32 with the given shape containing
// random bits generated by the Philox algorithm. Returns the array and the new
// state of the random number generator.
LogicalResult generateLinalgPhilox32(OpBuilder &builder, Location loc,
                                     ShapedType resultTy, Value &store,
                                     Value &result) {
  Type resultETy = resultTy.getElementType();

  Value initialState = extractState64(builder, loc, store);
  if (!initialState) return failure();

  std::pair<Value, Value> keys = extractKey32(builder, loc, store);
  if (!keys.first || !keys.second) return failure();

  int64_t numElements = resultTy.getNumElements();
  int64_t count = (numElements + 3) / 4;
  ShapedType intermediateType =
      RankedTensorType::get({count, 1}, resultTy.getElementType());
  int64_t concatDim = 1;

  // Compute the number of random i64s generated and increment state.
  Value countVal =
      builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(count));
  Value newState = builder.create<arith::AddIOp>(loc, initialState, countVal);

  // set up four outputs
  Value dest0 = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>({count}),
                                                resultETy);
  Value dest1 = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>({count}),
                                                resultETy);
  Value dest2 = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>({count}),
                                                resultETy);
  Value dest3 = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>({count}),
                                                resultETy);

  ShapedType destTy = dest0.getType().cast<ShapedType>();

  SmallVector<AffineMap> indexingMaps(4, builder.getMultiDimIdentityMap(1));
  SmallVector<utils::IteratorType> iterators(1, utils::IteratorType::parallel);

  linalg::GenericOp generic = builder.create<linalg::GenericOp>(
      loc, TypeRange{destTy, destTy, destTy, destTy},
      /*inputs=*/ValueRange(),
      /*outputs=*/ValueRange{dest0, dest1, dest2, dest3},
      /*indexingMaps=*/indexingMaps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange) {
        auto output =
            runPhilox4x32(PhiloxKey{ArithOpBuilder(b, nestedLoc, keys.first),
                                    ArithOpBuilder(b, nestedLoc, keys.second)},
                          ArithOpBuilder(b, nestedLoc, initialState));
        auto out0 = output[0].truncI(resultETy.getIntOrFloatBitWidth());
        auto out1 = output[1].truncI(resultETy.getIntOrFloatBitWidth());
        auto out2 = output[2].truncI(resultETy.getIntOrFloatBitWidth());
        auto out3 = output[3].truncI(resultETy.getIntOrFloatBitWidth());
        b.create<linalg::YieldOp>(
            loc, ValueRange{out0.val(), out1.val(), out2.val(), out3.val()});
      });

  if (resultTy.getNumElements() == 1) {
    result = reshapeToTarget(builder, loc, resultTy, generic.getResult(0));
    store = setState64(builder, loc, store, newState);
    return success();
  }

  Value r0 =
      reshapeToTarget(builder, loc, intermediateType, generic.getResult(0));
  Value r1 =
      reshapeToTarget(builder, loc, intermediateType, generic.getResult(1));
  Value r2 =
      reshapeToTarget(builder, loc, intermediateType, generic.getResult(2));
  Value r3 =
      reshapeToTarget(builder, loc, intermediateType, generic.getResult(3));

  Value concatenate = builder.create<mlir::stablehlo::ConcatenateOp>(
      loc, ValueRange{r0, r1, r2, r3}, builder.getI64IntegerAttr(concatDim));

  // Collapse the concat dimension back into the parent.
  llvm::SmallVector<int64_t> collapseShape(intermediateType.getShape());
  collapseShape[0] = collapseShape[0] * 4;
  Value reshapeIntermediate = builder.create<mlir::stablehlo::ReshapeOp>(
      loc, resultTy.clone(collapseShape), concatenate);

  // Slice to only the required results.
  collapseShape[0] = resultTy.getNumElements();

  llvm::SmallVector<int64_t> offset(resultTy.getRank(), 0);
  llvm::SmallVector<int64_t> stride(resultTy.getRank(), 1);
  Value slice = builder.create<mlir::stablehlo::SliceOp>(
      loc, intermediateType.clone(collapseShape), reshapeIntermediate,
      builder.getI64TensorAttr(offset), builder.getI64TensorAttr(collapseShape),
      builder.getI64TensorAttr(stride));
  Value reshapeResult =
      builder.create<mlir::stablehlo::ReshapeOp>(loc, resultTy, slice);

  // Set the new tensor values.
  store = setState64(builder, loc, store, newState);
  result = reshapeResult;

  return success();
}

LogicalResult generateLinalgPhilox64(OpBuilder &builder, Location loc,
                                     ShapedType resultTy, Value &store,
                                     Value &result) {
  Type resultETy = resultTy.getElementType();

  Value initialState = extractState64(builder, loc, store);
  if (!initialState) return failure();

  std::pair<Value, Value> keys = extractKey32(builder, loc, store);
  if (!keys.first || !keys.second) return failure();

  int64_t numElements = resultTy.getNumElements();
  int64_t count = (numElements + 1) / 2;
  ShapedType intermediateType =
      RankedTensorType::get({count, 1}, resultTy.getElementType());
  int64_t concatDim = 1;

  // Compute the number of random i64s generated and increment state.
  Value countVal =
      builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(count));
  Value newState = builder.create<arith::AddIOp>(loc, initialState, countVal);

  // set up four outputs
  Value dest0 = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>({count}),
                                                resultETy);
  Value dest1 = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>({count}),
                                                resultETy);
  ShapedType destTy = dest0.getType().cast<ShapedType>();

  SmallVector<AffineMap> indexingMaps(2, builder.getMultiDimIdentityMap(1));
  SmallVector<utils::IteratorType> iterators(1, utils::IteratorType::parallel);

  linalg::GenericOp generic = builder.create<linalg::GenericOp>(
      loc, TypeRange{destTy, destTy},
      /*inputs=*/ValueRange(),
      /*outputs=*/ValueRange{dest0, dest1},
      /*indexingMaps=*/indexingMaps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange) {
        auto output =
            runPhilox4x32(PhiloxKey{ArithOpBuilder(b, nestedLoc, keys.first),
                                    ArithOpBuilder(b, nestedLoc, keys.second)},
                          ArithOpBuilder(b, nestedLoc, initialState));
        auto out0 = output[0];
        auto out1 = output[1];
        auto out2 = output[2];
        auto out3 = output[3];
        Value result1 = fuseI32s(out0, out1).val();
        Value result2 = fuseI32s(out2, out3).val();
        b.create<linalg::YieldOp>(loc, ValueRange{result1, result2});
      });

  if (resultTy.getNumElements() == 1) {
    result = reshapeToTarget(builder, loc, resultTy, generic.getResult(0));
    store = setState64(builder, loc, store, newState);
    return success();
  }

  Value r0 =
      reshapeToTarget(builder, loc, intermediateType, generic.getResult(0));
  Value r1 =
      reshapeToTarget(builder, loc, intermediateType, generic.getResult(1));
  Value concatenate = builder.create<mlir::stablehlo::ConcatenateOp>(
      loc, ValueRange{r0, r1}, builder.getI64IntegerAttr(concatDim));

  // Collapse the concat dimension back into the parent.
  llvm::SmallVector<int64_t> collapseShape(intermediateType.getShape());
  collapseShape[0] = collapseShape[0] * 2;
  Value reshapeIntermediate = builder.create<mlir::stablehlo::ReshapeOp>(
      loc, resultTy.clone(collapseShape), concatenate);

  // Slice to only the required results.
  collapseShape[0] = resultTy.getNumElements();

  llvm::SmallVector<int64_t> offset(resultTy.getRank(), 0);
  llvm::SmallVector<int64_t> stride(resultTy.getRank(), 1);
  Value slice = builder.create<mlir::stablehlo::SliceOp>(
      loc, intermediateType.clone(collapseShape), reshapeIntermediate,
      builder.getI64TensorAttr(offset), builder.getI64TensorAttr(collapseShape),
      builder.getI64TensorAttr(stride));
  Value reshapeResult =
      builder.create<mlir::stablehlo::ReshapeOp>(loc, resultTy, slice);

  // Set the new tensor values.
  store = setState64(builder, loc, store, newState);
  result = reshapeResult;

  return success();
}

LogicalResult generateLinalgThreeFry(OpBuilder &builder, Location loc,
                                     ShapedType resultTy, Value &state,
                                     Value &result) {
  Type eTy = resultTy.getElementType();
  unsigned bitwidth = eTy.getIntOrFloatBitWidth();

  if (bitwidth == 64) {
    return generateLinalgThreeFry64(builder, loc, resultTy, state, result);
  }
  if (bitwidth == 32) {
    return generateLinalgThreeFry32(builder, loc, resultTy, state, result);
  }
  if (bitwidth == 16) {
    return generateLinalgThreeFry32(builder, loc, resultTy, state, result);
  }

  return failure();
}

LogicalResult generateLinalgPhilox(OpBuilder &builder, Location loc,
                                   ShapedType resultTy, Value &state,
                                   Value &result) {
  Type eTy = resultTy.getElementType();
  unsigned bitwidth = eTy.getIntOrFloatBitWidth();
  if (bitwidth == 64) {
    return generateLinalgPhilox64(builder, loc, resultTy, state, result);
  }

  // The 32 bit implementation trancates to result eTy.
  if (bitwidth == 32 || bitwidth == 16) {
    return generateLinalgPhilox32(builder, loc, resultTy, state, result);
  }

  return failure();
}

struct RngBitGeneratorConverter final
    : OpConversionPattern<mlir::stablehlo::RngBitGeneratorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::RngBitGeneratorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value state = adaptor.getInitialState();
    auto resultTy = dyn_cast_or_null<ShapedType>(
        getTypeConverter()->convertType(op.getResult(1).getType()));
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }

    if (op.getRngAlgorithm() == mlir::stablehlo::RngAlgorithm::THREE_FRY) {
      Value random;
      if (failed(
              generateLinalgThreeFry(rewriter, loc, resultTy, state, random))) {
        return failure();
      }
      rewriter.replaceOp(op, {state, random});
      return success();
    }

    if (op.getRngAlgorithm() == mlir::stablehlo::RngAlgorithm::PHILOX ||
        op.getRngAlgorithm() == mlir::stablehlo::RngAlgorithm::DEFAULT) {
      Value random;
      if (failed(
              generateLinalgPhilox(rewriter, loc, resultTy, state, random))) {
        return failure();
      }
      rewriter.replaceOp(op, {state, random});
      return success();
    }

    return failure();
  }
};

struct RngUniformConversion final
    : OpConversionPattern<mlir::stablehlo::RngOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::RngOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // We only handle uniform distributions.
    if (op.getRngDistribution() != mlir::stablehlo::RngDistribution::UNIFORM) {
      return failure();
    }
    // TODO(raikonenfnu): Handle other element types as well.
    auto minTy = dyn_cast<ShapedType>(adaptor.getA().getType());
    auto maxTy = dyn_cast<ShapedType>(adaptor.getB().getType());
    if (!isa<FloatType>(minTy.getElementType()) ||
        !isa<FloatType>(maxTy.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "expected min/max for rng op to be FloatType");
    }
    auto targetTy = dyn_cast_or_null<ShapedType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!targetTy) {
      return rewriter.notifyMatchFailure(
          op, "expected target shape of rng op to be ShapedType");
    }
    auto loc = op.getLoc();
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, targetTy, op, adaptor.getOperands());
    // Creates index map using target matrix's rank.
    auto targetRank = targetTy.getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        2, AffineMap::get(targetRank, /*symbolCount=*/0,
                          SmallVector<AffineExpr>({}), rewriter.getContext()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(targetRank));
    const int kInitialSeed = 0;

    // Generic region with LCG Algorithm that make use of element index from:
    // https://reviews.llvm.org/D101364
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensors=*/targetTy,
        /*inputs=*/
        ValueRange{adaptor.getOperands()[0], adaptor.getOperands()[1]},
        /*outputs=*/emptyTensor, indexingMaps,
        getParallelAndReductionIterators(/*nLoops=*/targetRank,
                                         /*nReduction=*/0),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          llvm::SmallVector<Value> updateVec = {b.create<arith::ConstantOp>(
              loc, b.getI32IntegerAttr(kInitialSeed))};
          Value multiplier =
              b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(1103515245));
          Value incrementStep =
              b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(12345));
          // For output matrix with rank N:
          // temp1 = (cast(I32, index(D.0)) + seed) * mult + incr
          // ...
          // tempN = (cast(I32, index(D.(N))) + tempN_1) * mult + incr
          for (int i = 0; i < targetRank; i++) {
            Value update = updateVec.back();
            Value ind = b.create<linalg::IndexOp>(loc, i);
            Value castInd =
                b.create<arith::IndexCastOp>(loc, b.getI32Type(), ind);
            Value addRes = b.create<arith::AddIOp>(loc, castInd, update);
            Value multRes = b.create<arith::MulIOp>(loc, addRes, multiplier);
            Value incRes = b.create<arith::AddIOp>(loc, multRes, incrementStep);
            updateVec.push_back(incRes);
          }
          // Scaling = (max - min) * const(F64, 2.3283064E-10)
          // which is derived from rand(min,max) = rand()/(RAND_MAX/(max-min)).
          Value epsilon = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(args[0].getType(), 2.3283064E-10));
          Value range = b.create<arith::SubFOp>(loc, args[1], args[0]);
          Value scale = b.create<arith::MulFOp>(loc, range, epsilon);
          // Res = cast(T, cast(F64, tempN) * scaling + min)
          Value updateCast = b.create<arith::UIToFPOp>(
              loc, targetTy.getElementType(), updateVec.back());
          Value scaleUpdate = b.create<arith::MulFOp>(loc, updateCast, scale);
          Value res = b.create<arith::AddFOp>(loc, scaleUpdate, args[0]);
          b.create<linalg::YieldOp>(loc, res);
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};
}  // namespace

namespace detail {
void populateStableHloRandomToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns) {
  patterns->add<RngBitGeneratorConverter, RngUniformConversion>(typeConverter,
                                                                context);
}
}  // namespace detail
}  // namespace mlir::iree_compiler::stablehlo
