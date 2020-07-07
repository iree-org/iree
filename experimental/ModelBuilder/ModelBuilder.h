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
//    builder.getOperation().print(llvm::outs()); // print MLIR
// ```

#ifndef IREE_EXPERIMENTAL_MODELBUILDER_MODELBUILDER_H_
#define IREE_EXPERIMENTAL_MODELBUILDER_MODELBUILDER_H_

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/SCF/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"

namespace mlir {
using edsc::ScopedContext;
using edsc::StructuredIndexed;

// List of MLIR EDSC instrinsics exposed to external clients of ModelBuilder.
// All other intrinsics are abstracted away via ModelBuilder methods.
// -----------------------------------------------------------------------------
// From the Linalg Dialect.
using edsc::intrinsics::linalg_fill;
using edsc::intrinsics::linalg_matmul;
using edsc::intrinsics::linalg_yield;
using edsc::ops::linalg_generic_matmul;
// From the Vector Dialect.
using edsc::intrinsics::vector_broadcast;
using edsc::intrinsics::vector_contract;
using edsc::intrinsics::vector_extract;
using edsc::intrinsics::vector_matmul;
using edsc::intrinsics::vector_outerproduct;
using edsc::intrinsics::vector_print;
using edsc::intrinsics::vector_transpose;
using edsc::ops::vector_contraction;
using edsc::ops::vector_contraction_matmul;
// From the Std Dialect.
using edsc::MemRefBoundsCapture;
using edsc::VectorBoundsCapture;
using edsc::intrinsics::std_addf;
using edsc::intrinsics::std_alloc;
using edsc::intrinsics::std_call;
using edsc::intrinsics::std_constant_float;
using edsc::intrinsics::std_constant_index;
using edsc::intrinsics::std_dealloc;
using edsc::intrinsics::std_dim;
using edsc::intrinsics::std_mulf;
using edsc::intrinsics::std_ret;
using edsc::intrinsics::StdIndexedValue;
// From the Affine Dialect.
using edsc::intrinsics::affine_max;
using edsc::intrinsics::affine_min;
using edsc::intrinsics::AffineIndexedValue;
// From the Loop Dialect.
using edsc::loopNestBuilder;
// -----------------------------------------------------------------------------

// Helper class to simplify MLIR function construction by adding proper
// attributes, some of which pass through to LLVM.
struct MLIRFuncOpConfig {
  // Applies the MLIRFuncOpConfig to `f`.
  void apply(FuncOp &f);

  // Attributes that pass through to LLVM and modify the behavior of the LLVM
  // compiler.
  bool noInline = false;
  MLIRFuncOpConfig &setNoInline(bool v);

  bool preferAvx512 = false;
  MLIRFuncOpConfig &setPreferAvx512(bool v);

  std::string targetCpu = "";
  MLIRFuncOpConfig &setTargetCpu(StringRef s);

  // When true, the function remains body-less. This is good for declaring
  // external functions.
  bool declOnly = false;
  MLIRFuncOpConfig &setDeclOnly(bool v);

  // When true, an mlir_c_iface_xxx shim function is emitted with C compatible
  // strided memref ABI.
  bool emitCInterface = false;
  MLIRFuncOpConfig &setEmitCInterface(bool v);
};

// Entry point class to build a whole model declaratively with C++ EDSCs.
class ModelBuilder : public OpBuilder {
 public:
  using OpBuilder::create;

  // Create a ModelBuilder and sets up an owned MLIRContext, ModuleOp and
  // SymbolTable as well as uniqued MLIR types.
  ModelBuilder();

  // Register all the dialects used by ModelBuilder.
  static void registerAllDialects();

  // Return a reference to the underlying module.
  OwningModuleRef &getModuleRef() { return module; }

  // Build an MLIR FuncOp that will be callable after JIT compilation occured.
  // `config` is a convenience class provided to simplify the configuration of
  // the function with common attributes that are non-obvious to the newcomer.
  FuncOp makeFunction(StringRef name, ArrayRef<Type> results,
                      ArrayRef<Type> args,
                      MLIRFuncOpConfig config = MLIRFuncOpConfig());
  FuncOp makeFunction(std::function<std::string(FunctionType)> nameBuilder,
                      ArrayRef<Type> results, ArrayRef<Type> args,
                      MLIRFuncOpConfig config = MLIRFuncOpConfig());

  // Add GPU attribute to the module.
  void addGPUAttr();

  // Build a MLIR GPU module. GPUFuncOp can later be added to the module.
  gpu::GPUModuleOp makeGPUModule(StringRef name);

  // Build a MLIR GPU kernel within a GPU module.
  gpu::GPUFuncOp makeGPUKernel(StringRef name, gpu::GPUModuleOp GPUModule,
                               ArrayRef<int32_t> workgroupSize,
                               ArrayRef<Type> args = {},
                               ArrayRef<Type> results = {});

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

  // Build the MLIR representation for constants of common types.
  static Value constant_f32(float v);
  static Value constant_f64(double v);
  static Value constant_index(int64_t v);

  // Build the MLIR representation for:
  //   1. fc(I, W, O)
  //   2. pointwise(O, bias) in-place with explicit bias broadcast to compute:
  //      `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  // Returns O.
  // Version with a MemRef output argument.
  static Value FCBiasTanh(std::array<Value, 3> fcArgs, Value biasValueArg);
  // Version with a RankedTensor result.
  static Value FCBiasTanhTensors(RankedTensorType outputTensorType,
                                 std::array<Value, 2> fcArgs,
                                 Value biasValueArg);

  // Build the MLIR representation for:
  //   `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  // This assumes `x` and `bias` capture scalar MLIR values of type f32.
  // This is used as a region builder when constructing e.g. a pointwise op.
  static Value fusedBiasTanh(Value x, Value bias);

  // ---------------------------------------------------------------------------
  // Support for emitting special function calls.
  // ---------------------------------------------------------------------------
  static Value call_tanhf(Value v);
  static void call_print_memref_f32(Value v);  // needs libmlir_runner_utils.so

 protected:
  // Helper function to support calling into known functions (e.g. libmath).
  static Operation *emitCallToRegisteredSymbol(StringRef functionName,
                                               ArrayRef<Type> returnTypes,
                                               ValueRange values);

  // ---------------------------------------------------------------------------
  // Members.
  // ---------------------------------------------------------------------------
 protected:
  // Thread-safe context owned by ModelBuilder. All IR is built in this context.
  static thread_local MLIRContext ctx;
  mlir::OwningModuleRef module;
  // The symbol table for the module.
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

// -----------------------------------------------------------------------------
// EDSC extensions.
// -----------------------------------------------------------------------------
namespace edsc {
namespace extensions {

template <typename T>
SmallVector<Value, 4> std_constant_indices(ArrayRef<T> a) {
  auto makeIndex = [](int64_t v) { return mlir::std_constant_index(v).value; };
  return llvm::to_vector<4>(llvm::map_range(a, makeIndex));
}
// Build the MLIR representation for op(a, b) for each pair of elements in
// zip(`a`, `b`).
SmallVector<Value, 4> operator+(ValueRange a, ValueRange b);
SmallVector<Value, 4> operator-(ValueRange a, ValueRange b);
// Build the MLIR representation for select(a cmp b, a, b) for each pair of
// elements in zip(`a`, `b`).
SmallVector<Value, 4> std_max(ValueRange a, ValueRange b);
SmallVector<Value, 4> std_min(ValueRange a, ValueRange b);
// Build the MLIR representation for affine_cmp(a, b) for each pair of elements
// in zip(`a`, `b`).
SmallVector<Value, 4> affine_max(ValueRange a, ValueRange b);
SmallVector<Value, 4> affine_min(ValueRange a, ValueRange b);

}  // namespace extensions
}  // namespace edsc
}  // namespace mlir

#endif  // IREE_EXPERIMENTAL_MODELBUILDER_MODELBUILDER_H_
