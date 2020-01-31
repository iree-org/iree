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
//  2. key types creation such as mlir::FloatType, mlir::IntegerType and
//     mlir::MemRefType.
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

#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"

namespace mlir {

using edsc::ScopedContext;
using edsc::StructuredIndexed;
using edsc::ValueHandle;

// List of MLIR EDSC instrinsics exposed to external clients of ModelBuilder.
// All other intrinsics are abstracted away via ModelBuilder methods.
// -----------------------------------------------------------------------------
// From the Linalg Dialect.
using edsc::intrinsics::linalg_fill;
// From the Std Dialect.
using edsc::intrinsics::alloc;
using edsc::intrinsics::constant_float;
using edsc::intrinsics::dealloc;
using edsc::intrinsics::dim;
using edsc::intrinsics::ret;
// -----------------------------------------------------------------------------

// Entry point class to build a whole model declaratively with C++ EDSCs.
class ModelBuilder : public OpBuilder {
 public:
  using OpBuilder::create;

  // Creates a ModelBuilder and sets up an owned MLIRContext, ModuleOp and
  // SymbolTable as well as uniqued MLIR types.
  ModelBuilder();

  OwningModuleRef &getModuleRef() { return module; }

  // Build an MLIR FuncOp that will be callable after JIT compilation occured.
  FuncOp makeFunction(StringRef name, ArrayRef<Type> results = {},
                      ArrayRef<Type> args = {}, bool declOnly = false);

  // Build an MLIR MemRefType with a base `elementType` and a `shape` that can
  // be any mix of static and dynamic values. For now this only supports a dense
  // and contiguous layout.
  // In the future, this can be extended support more advanced layouts, on a
  // per-need basis.
  MemRefType getMemRefType(ArrayRef<int64_t> shape, Type elementType);

  // FCBiasTanh implements:
  //   1. fc(I, W, O)
  //   2. pointwise(O, bias) in-place with explicit bias broadcast to compute:
  //      `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  // Returns O.
  ValueHandle FCBiasTanh(std::array<Value, 3> fcArgs, Value biasValueArg);

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
};

}  // namespace mlir

#endif  // IREE_EXPERIMENTAL_MODELBUILDER_MODELBUILDER_H_
