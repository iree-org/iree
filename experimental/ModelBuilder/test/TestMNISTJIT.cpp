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

// RUN: test-mnist-jit 2>&1 | IreeFileCheck %s

#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
// RunnerUtils.h with iostream needed for printMemRef atm
#include "mlir/ExecutionEngine/RunnerUtils.h"

using namespace mlir;  // NOLINT

// Helper function to build a func "mnist" that takes a memref<?x784xf32> buffer
//    (use batch size B=3 in this example).
//
// This is a 3-layer MLP with static weights of sizes W0xW1, W1xW2 and W2xW3.
// In between each fully-connected layer we have a non-linearity fused with the
// bias addition.
//
// The fused non-linearity computes:
//   `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
//
// Most of the code below is about allocating and initializing buffers for the
// model weights and biases + cleaning up on exit. Most of this will disappear
// when starting from a tensor abstraction and are able to automatically attach
// to a preloaded model in memory.
//
// The interesting part of the model is the 3-MLP part:
// ```
//  modelBuilder.FCBiasTanh({input, h1Weights, outputBlock1}, bias1);
//  modelBuilder.FCBiasTanh({outputBlock1, h2Weights, outputBlock2}, bias2);
//  modelBuilder.FCBiasTanh({outputBlock2, h3Weights, outputBlock3}, bias3);
// ```
void buildMNIST(ModelBuilder &modelBuilder, StringLiteral funcName, unsigned B,
                unsigned W0, unsigned W1, unsigned W2, unsigned W3) {
  auto f32 = modelBuilder.f32;
  auto inputType = modelBuilder.getMemRefType({-1, W0}, f32);
  auto outputType = modelBuilder.getMemRefType({-1, W3}, f32);
  auto func =
      modelBuilder.makeFunction(funcName, {}, {inputType, outputType},
                                MLIRFuncOpConfig().setEmitCInterface(true));

  // Fill the body (3 blocks of FCBiasTanh), alloc everything manually atm.
  OpBuilder b(&func.getBody());
  ScopedContext scope(b, func.getLoc());
  Value input = func.getArgument(0);
  Value batchSize = std_dim(input, 0);
  Value h1Weights = std_alloc(modelBuilder.getMemRefType({W0, W1}, f32));
  Value h2Weights = std_alloc(modelBuilder.getMemRefType({W1, W2}, f32));
  Value h3Weights = std_alloc(modelBuilder.getMemRefType({W2, W3}, f32));
  Value bias1 = std_alloc(modelBuilder.getMemRefType({W1}, f32));
  Value bias2 = std_alloc(modelBuilder.getMemRefType({W2}, f32));
  Value bias3 = std_alloc(modelBuilder.getMemRefType({W3}, f32));
  Value outputBlock1 =
      std_alloc(modelBuilder.getMemRefType({-1, W1}, f32), batchSize);
  Value outputBlock2 =
      std_alloc(modelBuilder.getMemRefType({-1, W2}, f32), batchSize);
  Value outputBlock3 = func.getArgument(1);

  Value flt_0 = modelBuilder.constant_f32(0.0f);
  Value someVal = modelBuilder.constant_f32(0.1123f);
  linalg_fill(h1Weights, someVal);
  linalg_fill(h2Weights, someVal);
  linalg_fill(h3Weights, someVal);
  linalg_fill(bias1, someVal);
  linalg_fill(bias2, someVal);
  linalg_fill(bias3, someVal);
  linalg_fill(outputBlock1, flt_0);
  linalg_fill(outputBlock2, flt_0);

  modelBuilder.FCBiasTanh({input, h1Weights, outputBlock1}, bias1);
  modelBuilder.FCBiasTanh({outputBlock1, h2Weights, outputBlock2}, bias2);
  modelBuilder.FCBiasTanh({outputBlock2, h3Weights, outputBlock3}, bias3);

  // TODO(ntv): tensor->buffer, drop all alloc/fill/dealloc.
  // Vexing parses.
  (std_dealloc(h1Weights));
  (std_dealloc(h2Weights));
  (std_dealloc(h3Weights));
  (std_dealloc(bias1));
  (std_dealloc(bias2));
  (std_dealloc(bias3));
  (std_dealloc(outputBlock1));
  (std_dealloc(outputBlock2));

  (std_ret());
}

// Helper function to build a func `funcName` that takes a tensors for the input
// in the form of a `tensor<?x784xf32>` as well as static tensors for all the
// weights and biases.
//
// This is the counterpart of `buildMNIST` which builds a similar model on
// buffers.
void buildMNISTOnTensors(ModelBuilder &modelBuilder, StringLiteral funcName,
                         int64_t B, int64_t W0, int64_t W1, int64_t W2,
                         int64_t W3) {
  auto f32 = modelBuilder.f32;
  auto inputType = modelBuilder.getRankedTensorType({B, W0}, f32);
  auto h1WeightsType = modelBuilder.getRankedTensorType({W0, W1}, f32);
  auto h2WeightsType = modelBuilder.getRankedTensorType({W1, W2}, f32);
  auto h3WeightsType = modelBuilder.getRankedTensorType({W2, W3}, f32);
  auto bias1Type = modelBuilder.getRankedTensorType({W1}, f32);
  auto bias2Type = modelBuilder.getRankedTensorType({W2}, f32);
  auto bias3Type = modelBuilder.getRankedTensorType({W3}, f32);
  auto outputType = modelBuilder.getRankedTensorType({B, W3}, f32);
  auto func = modelBuilder.makeFunction(
      funcName, {outputType},
      {inputType, h1WeightsType, h2WeightsType, h3WeightsType, bias1Type,
       bias2Type, bias3Type});
  Value input = func.getArgument(0);
  Value h1Weights = func.getArgument(1);
  Value h2Weights = func.getArgument(2);
  Value h3Weights = func.getArgument(3);
  Value bias1 = func.getArgument(4);
  Value bias2 = func.getArgument(5);
  Value bias3 = func.getArgument(6);

  // 2. Fill the body (3 blocks of FCBiasTanh), alloc everything manually atm.
  OpBuilder b(&func.getBody());
  ScopedContext scope(b, func.getLoc());

  auto outputBlock1Type = modelBuilder.getRankedTensorType({B, W1}, f32);
  auto outputBlock1 = modelBuilder.FCBiasTanhTensors(outputBlock1Type,
                                                     {input, h1Weights}, bias1);
  auto outputBlock2Type = modelBuilder.getRankedTensorType({B, W2}, f32);
  auto outputBlock2 = modelBuilder.FCBiasTanhTensors(
      outputBlock2Type, {outputBlock1, h2Weights}, bias2);
  auto outputBlock3Type = outputType;
  auto outputBlock3 = modelBuilder.FCBiasTanhTensors(
      outputBlock3Type, {outputBlock2, h3Weights}, bias3);
  // Vexing parses.
  (std_ret(outputBlock3));
}

int main() {
  ModelBuilder::registerAllDialects();
  constexpr unsigned B = 3, W0 = 784, W1 = 256, W2 = 256, W3 = 10;

  ModelBuilder modelBuilder;
  // 1. Build a func "test_mnist_jit_tensors".
  constexpr StringLiteral kFuncTensorsName = "test_mnist_jit_tensors";
  buildMNISTOnTensors(modelBuilder, kFuncTensorsName, ShapedType::kDynamicSize,
                      W0, W1, W2, W3);
  // 1.b. Dump the function for testing and erase it: we can't compile it to
  // buffers for now.
  modelBuilder.getModuleRef()->dump();
  SymbolTable::lookupNearestSymbolFrom(
      modelBuilder.getModuleRef()->getOperation(), kFuncTensorsName)
      ->erase();

  // 2. Build a separate func "test_mnist_jit_buffers" that takes a
  // memref<?x784xf32> buffer
  //    (use batch size M=3 in this example)
  // In the future, when we can lower the function built in 1. to buffers we
  // will.
  constexpr StringLiteral kFuncBuffersName = "test_mnist_jit_buffers";
  buildMNIST(modelBuilder, kFuncBuffersName, B, W0, W1, W2, W3);

  // 3. Compile the function.
  ModelRunner runner(modelBuilder.getModuleRef());
  runner.compile(CompilationOptions());

  // 4. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
  auto inputLinearInit = [](unsigned idx, float *ptr) {
    *(ptr + idx) = 0.032460f;
  };
  // Exercise the ranked strided memref descriptor.
  auto inputBuffer = makeInitializedStridedMemRefDescriptor<float, 2>(
      {B, W0}, inputLinearInit);
  auto outputLinearInit = [](unsigned idx, float *ptr) { *(ptr + idx) = 0.0f; };
  // Exercise the unranked memref descriptor, with extra level of indirection.
  auto outputBuffer =
      makeInitializedUnrankedDescriptor<float, 2>({B, W3}, outputLinearInit);

  // 5. Call the funcOp name `kFuncBuffersName` with arguments.
  auto *inputDescriptor = inputBuffer.get();
  void *args[] = {&inputDescriptor, &outputBuffer->descriptor};
  auto error = runner.invokeIndirect(kFuncBuffersName, args);

  // 6. Dump content of output buffer for testing with FileCheck.
  if (error) {
    runner.module->dump();
    llvm::errs() << "ERROR: " << error << "\n";
    return 1;
  }
  ::impl::printMemRef(
      *static_cast<StridedMemRefType<float, 2> *>(outputBuffer->descriptor));
}

// For now, we can only dump the IR for `test_mnist_jit_tensors`.
// Once buffer allocation is implemented we will only have an execution test.
//
// CHECK: func @test_mnist_jit_tensors
//
// Matmul
// CHECK: linalg.generic
// CHECK:   tensor<?x784xf32>, tensor<784x256xf32> -> tensor<?x256xf32>
//
// Pointwise
// CHECK: linalg.generic
// CHECK:   addf
// CHECK:   mulf
// CHECK:   tanh
// CHECK:   mulf
// CHECK:   addf
// CHECK:   addf
// CHECK:   tensor<?x256xf32>, tensor<256xf32> -> tensor<?x256xf32>
//
// Matmul
// CHECK: linalg.generic
// CHECK:   tensor<?x256xf32>, tensor<256x256xf32> -> tensor<?x256xf32>
//
// Pointwise
// CHECK: linalg.generic
// CHECK:   tensor<?x256xf32>, tensor<256xf32> -> tensor<?x256xf32>
//
// Matmul
// CHECK: linalg.generic
// CHECK:   tensor<?x256xf32>, tensor<256x10xf32> -> tensor<?x10xf32>
//
// Pointwise
// CHECK: linalg.generic
// CHECK:   tensor<?x10xf32>, tensor<10xf32> -> tensor<?x10xf32>
// CHECK:   return {{.*}} : tensor<?x10xf32>

// Execution test for `test_mnist_jit_buffers`.
//
// CHECK: Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [3, 10]
// CHECK-SAME: strides = [10, 1] data =
// clang-format off
// CHECK-COUNT-3: {{.*[[:space:]].*}}[3177.93,   3177.93,   3177.93,   3177.93,   3177.93,   3177.93,   3177.93,   3177.93,   3177.93,   3177.93]
// clang-format on
