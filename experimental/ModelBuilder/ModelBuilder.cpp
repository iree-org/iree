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
      f32(FloatType::getF32(&ctx)) {}

FuncOp mlir::ModelBuilder::makeFunction(StringRef name, ArrayRef<Type> results,
                                        ArrayRef<Type> args, bool declOnly) {
  auto function =
      FuncOp::create(loc, name, FunctionType::get(args, results, &ctx));
  if (!declOnly) function.addEntryBlock();
  module->push_back(function);
  return function;
}

MemRefType mlir::ModelBuilder::getMemRefType(ArrayRef<int64_t> shape,
                                             Type elementType) {
  return MemRefType::get(shape, elementType, {});
}

ValueHandle mlir::ModelBuilder::FCBiasTanh(std::array<Value, 3> fcArgs,
                                           Value biasValueArg) {
  //==========================================================================//
  // Layer 1: FC
  //==========================================================================//
  ValueHandle I(fcArgs[0]), W(fcArgs[1]), O(fcArgs[2]);
  // Emit a linalg.generic op that implements matmul:
  linalg_matmul(I, W, O);

  //==========================================================================//
  // Layer 2: BiasAddTanh Block
  //==========================================================================//
  // Build and capture AffineExpr i and j for building index expressions.
  AffineExpr i, j;
  bindDims(&ctx, i, j);

  // Define the pointwise computation:
  //   `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  // This assumes ValueHandle captures an MLIR Value with a proper type
  // (in this case `f32`)
  auto opBuilder = [this](const ValueHandle &x,
                          const ValueHandle &bias) -> Value {
    using edsc::op::operator+;
    using edsc::op::operator*;
    using edsc::intrinsics::tanh;

    // `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
    auto half = constant_float(llvm::APFloat(0.5f), f32);
    return x + half * tanh((x + bias) * half) + half;
  };

  // Emit a linalg.generic op that implements pointwise with `opBuilder` for:
  //   `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  //
  // This performs the (inplace) computation:
  //   `SO[i, j] <- pointwise(SBias[j], SO[i, j])`
  //
  // in which SBias is broadcast along `i`.
  ValueHandle Bias(biasValueArg);
  StructuredIndexed SO(O), SBias(Bias);
  linalg_pointwise(opBuilder, SO({i, j}), SBias({j}), SO({i, j}));

  return O;
}
