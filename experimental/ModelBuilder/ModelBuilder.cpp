// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "experimental/ModelBuilder/ModelBuilder.h"

#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::ops;
using namespace mlir::edsc::intrinsics;

thread_local MLIRContext mlir::ModelBuilder::ctx;

mlir::ModelBuilder::ModelBuilder()
    : OpBuilder(&ctx),
      module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx))),
      symbolTable(*module),
      loc(module->getLoc()),
      i8(IntegerType::get(8, &ctx)),
      f32(FloatType::getF32(&ctx)),
      f64(FloatType::getF64(&ctx)) {}

Value mlir::ModelBuilder::constant_f32(float v) {
  return std_constant_float(llvm::APFloat(v),
                            FloatType::getF32(ScopedContext::getContext()));
}

Value mlir::ModelBuilder::constant_f64(double v) {
  return std_constant_float(llvm::APFloat(v),
                            FloatType::getF64(ScopedContext::getContext()));
}

FuncOp mlir::ModelBuilder::makeFunction(StringRef name, ArrayRef<Type> results,
                                        ArrayRef<Type> args,
                                        bool emitCInterface, bool declOnly) {
  auto function =
      FuncOp::create(loc, name, FunctionType::get(args, results, &ctx));
  if (!declOnly) function.addEntryBlock();
  if (emitCInterface)
    function.setAttr("llvm.emit_c_interface",
                     mlir::UnitAttr::get(getContext()));
  module->push_back(function);
  return function;
}

VectorType mlir::ModelBuilder::getVectorType(ArrayRef<int64_t> shape,
                                             Type elementalType) {
  return VectorType::get(shape, elementalType);
}

MemRefType mlir::ModelBuilder::getMemRefType(ArrayRef<int64_t> shape,
                                             Type elementType,
                                             unsigned addressSpace) {
  return MemRefType::get(shape, elementType, {}, addressSpace);
}

RankedTensorType mlir::ModelBuilder::getRankedTensorType(
    ArrayRef<int64_t> shape, Type elementType) {
  return RankedTensorType::get(shape, elementType);
}

Value mlir::ModelBuilder::fusedBiasTanh(ValueHandle x, ValueHandle bias) {
  using edsc::op::operator+;
  using edsc::op::operator*;
  using edsc::intrinsics::std_tanh;
  assert(x.getType().isF32() && bias.getType().isF32() && "f32 expected");
  ValueHandle half(constant_f32(0.5f));
  return x + half * std_tanh((x + bias) * half) + half;
}

ValueHandle mlir::ModelBuilder::FCBiasTanh(std::array<Value, 3> fcArgs,
                                           Value biasValueArg) {
  //==========================================================================//
  // Layer 1: FC
  //==========================================================================//
  ValueHandle I(fcArgs[0]), W(fcArgs[1]), O(fcArgs[2]);
  // Emit a linalg.generic op that implements matmul:
  linalg_generic_matmul(I, W, O);

  //==========================================================================//
  // Layer 2: BiasAddTanh Block
  //==========================================================================//
  // Build and capture AffineExpr i and j for building index expressions.
  AffineExpr i, j;
  bindDims(&ctx, i, j);

  // Emit a linalg.generic op that implements pointwise with `opBuilder` for:
  //   `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  //
  // This performs the (inplace) computation:
  //   `o[i, j] <- pointwise(bias[j], o[i, j])`
  //
  // in which bias is broadcast along `i`.
  StructuredIndexed o(O), bias(biasValueArg);
  linalg_generic_pointwise(fusedBiasTanh, o({i, j}), bias({j}), o({i, j}));

  return O;
}

Value ModelBuilder::FCBiasTanhTensors(RankedTensorType outputTensorType,
                                      std::array<Value, 2> fcArgs,
                                      Value biasValueArg) {
  //==========================================================================//
  // Layer 1: FC
  //==========================================================================//
  ValueHandle I(fcArgs[0]), W(fcArgs[1]);
  Value O2 = linalg_generic_matmul(I, W, outputTensorType)->getResult(0);

  //==========================================================================//
  // Layer 2: BiasAddTanh Block
  //==========================================================================//
  ValueHandle Bias(biasValueArg);
  AffineExpr i, j;
  bindDims(&ctx, i, j);
  // in-place with explicit bias broacast
  StructuredIndexed o2(O2), bias(Bias), o3Type(outputTensorType);
  return linalg_generic_pointwise(fusedBiasTanh, o2({i, j}), bias({j}),
                                  o3Type({i, j}))
      ->getResult(0);
}
