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

// ModelBuilder.h
// -----------------------------------------------------------------------------
//
// MLIR Model Builders demonstrate C++ metaprogramming features that are
// available in MLIR core. At a high-level, metaprogramming can be interpreted
// as "program with a level of indirection": one writes C++ that emits MLIR.
// The MLIR is then JIT compiled into a binary that can be invoked.
//
// The ModelBuilder exposes relevant core MLIR classes and APIs that are
// sufficient to build whole models. This set of classes and APIs encompass:
//  1. mlir::FuncOp creation.
//  2. key types creation such as mlir::FloatType, mlir::IntegerType,
//     mlir::VectorType, and mlir::MemRefType.
//  3. layer creation functions such as FCBiasTanh.
//
// Usage:
// ======
//
// ```
//
//    ModelBuilder builder;
//    auto func = builder.makeFunction(...);
//    OpBuilder b(&func.getBody());
//    ScopedContext scope(b, func.getLoc());
//
//    // ... build the body of func ...
//
//    builder.getModule().print(llvm::outs()); // print MLIR
// ```

#ifndef IREE_EXPERIMENTAL_MODELBUILDER_MODELBUILDER_H_
#define IREE_EXPERIMENTAL_MODELBUILDER_MODELBUILDER_H_

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/LoopOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"

namespace mlir {
using edsc::ScopedContext;
using edsc::StructuredIndexed;
using edsc::ValueHandle;

// List of MLIR EDSC instrinsics exposed to external clients of ModelBuilder.
// All other intrinsics are abstracted away via ModelBuilder methods.
// -----------------------------------------------------------------------------
// From the Linalg Dialect.
using edsc::intrinsics::linalg_fill;
using edsc::intrinsics::linalg_yield;
using edsc::ops::linalg_matmul;
// From the Vector Dialect.
using edsc::intrinsics::vector_broadcast;
using edsc::intrinsics::vector_contract;
using edsc::intrinsics::vector_print;
using edsc::ops::vector_contraction;
using edsc::ops::vector_matmul;
// From the Std Dialect.
using edsc::intrinsics::std_alloc;
using edsc::intrinsics::std_constant_float;
using edsc::intrinsics::std_constant_index;
using edsc::intrinsics::std_dealloc;
using edsc::intrinsics::std_dim;
using edsc::intrinsics::std_ret;
using edsc::intrinsics::StdIndexedValue;
// From the Affine Dialect.
using edsc::intrinsics::AffineIndexedValue;
// -----------------------------------------------------------------------------

// Entry point class to build a whole model declaratively with C++ EDSCs.
class ModelBuilder : public OpBuilder {
 public:
  using OpBuilder::create;

  // Creates a ModelBuilder and sets up an owned MLIRContext, ModuleOp and
  // SymbolTable as well as uniqued MLIR types.
  ModelBuilder();

  OwningModuleRef &getModuleRef() { return module; }

  // Build the MLIR representation for an f32 constant.
  static Value constant_f32(float v);

  // Build the MLIR representation for an f64 constant.
  static Value constant_f64(double v);

  // Build an MLIR FuncOp that will be callable after JIT compilation occured.
  FuncOp makeFunction(StringRef name, ArrayRef<Type> results = {},
                      ArrayRef<Type> args = {}, bool declOnly = false);

  // Build an MLIR VectorType with a base `elementalType` and a `shape`.
  VectorType getVectorType(ArrayRef<int64_t> shape, Type elementalType);

  // Build an MLIR MemRefType with a base `elementType` and a `shape` that can
  // be any mix of static and dynamic values. For now this only supports a dense
  // and contiguous layout.
  // In the future, this can be extended support more advanced layouts, on a
  // per-need basis.
  MemRefType getMemRefType(ArrayRef<int64_t> shape, Type elementType,
                           unsigned addressSpace = 0);

  // Build an MLIR RankedTensorType with a base `elementType` and a `shape` that
  // can be any mix of static and dynamic values. For now this only supports a
  // dense and contiguous layout.
  // In the future, this can be extended support more advanced layouts, on a
  // per-need basis.
  RankedTensorType getRankedTensorType(ArrayRef<int64_t> shape,
                                       Type elementType);

  // Build the MLIR representation for:
  //   1. fc(I, W, O)
  //   2. pointwise(O, bias) in-place with explicit bias broadcast to compute:
  //      `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  // Returns O.
  // Version with a MemRef output argument.
  ValueHandle FCBiasTanh(std::array<Value, 3> fcArgs, Value biasValueArg);
  // Version with a RankedTensor result.
  Value FCBiasTanhTensors(RankedTensorType outputTensorType,
                          std::array<Value, 2> fcArgs, Value biasValueArg);

  // Build the MLIR representation for:
  //   `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  // This assumes `x` and `bias` capture scalar MLIR values of type f32.
  // This is used as a region builder when constructing e.g. a pointwise op.
  static Value fusedBiasTanh(ValueHandle x, ValueHandle bias);

 protected:
  static thread_local MLIRContext ctx;
  mlir::OwningModuleRef module;
  mlir::SymbolTable symbolTable;

 public:
  // The mlir::Location of the single owned Module.
  Location loc;
  // The unique mlir::IntegerType of 8 bits.
  IntegerType i8;
  // The unique mlir::FloatType of 32 bits.
  FloatType f32;
  // The unique mlir::FloatType of 64 bits.
  FloatType f64;
};

}  // namespace mlir

#endif  // IREE_EXPERIMENTAL_MODELBUILDER_MODELBUILDER_H_
