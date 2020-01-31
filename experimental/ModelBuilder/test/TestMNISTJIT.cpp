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

// RUN: test-mnist-jit | IreeFileCheck %s

#include "experimental/ModelBuilder/MemRefUtils.h"
#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"

using namespace mlir;

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
  auto func = modelBuilder.makeFunction(funcName, {}, {inputType, outputType});

  // Fill the body (3 blocks of FCBiasTanh), alloc everything manually atm.
  OpBuilder b(&func.getBody());
  ScopedContext scope(b, func.getLoc());
  Value input = func.getArgument(0);
  Value batchSize = dim(input, 0);
  Value h1Weights = alloc(modelBuilder.getMemRefType({W0, W1}, f32));
  Value h2Weights = alloc(modelBuilder.getMemRefType({W1, W2}, f32));
  Value h3Weights = alloc(modelBuilder.getMemRefType({W2, W3}, f32));
  Value bias1 = alloc(modelBuilder.getMemRefType({W1}, f32));
  Value bias2 = alloc(modelBuilder.getMemRefType({W2}, f32));
  Value bias3 = alloc(modelBuilder.getMemRefType({W3}, f32));
  Value outputBlock1 =
      alloc(modelBuilder.getMemRefType({-1, W1}, f32), batchSize);
  Value outputBlock2 =
      alloc(modelBuilder.getMemRefType({-1, W2}, f32), batchSize);
  Value outputBlock3 = func.getArgument(1);

  auto zero = constant_float(llvm::APFloat(0.0f), f32);
  auto someVal = constant_float(llvm::APFloat(0.1123f), f32);
  linalg_fill(h1Weights, someVal);
  linalg_fill(h2Weights, someVal);
  linalg_fill(h3Weights, someVal);
  linalg_fill(bias1, someVal);
  linalg_fill(bias2, someVal);
  linalg_fill(bias3, someVal);
  linalg_fill(outputBlock1, zero);
  linalg_fill(outputBlock2, zero);

  modelBuilder.FCBiasTanh({input, h1Weights, outputBlock1}, bias1);
  modelBuilder.FCBiasTanh({outputBlock1, h2Weights, outputBlock2}, bias2);
  modelBuilder.FCBiasTanh({outputBlock2, h3Weights, outputBlock3}, bias3);

  // TODO(ntv): tensor->buffer, drop all alloc/fill/dealloc.
  // Vexing parses.
  (dealloc(h1Weights));
  (dealloc(h2Weights));
  (dealloc(h3Weights));
  (dealloc(bias1));
  (dealloc(bias2));
  (dealloc(bias3));
  (dealloc(outputBlock1));
  (dealloc(outputBlock2));

  (ret());
}

int main() {
  constexpr StringLiteral kFuncName = "test_mnist_jit";
  constexpr unsigned B = 3, W0 = 784, W1 = 256, W2 = 256, W3 = 10;

  // 1. Build a func "mnist" that takes a memref<?x784xf32> buffer
  //    (use batch size M=3 in this example)
  ModelBuilder modelBuilder;
  buildMNIST(modelBuilder, kFuncName, B, W0, W1, W2, W3);

  // 2. Compile the function.
  ModelRunner runner(modelBuilder.getModuleRef());
  runner.compile(/*llvmOptLevel=*/3, /*llcOptLevel=*/3);

  // 3. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
  auto inputLinearInit = [](unsigned idx, float *ptr) { *ptr = 0.032460f; };
  ManagedUnrankedMemRefDescriptor inputBuffer =
      makeInitializedUnrankedDescriptor<float>({B, W0}, inputLinearInit);
  auto outputLinearInit = [](unsigned idx, float *ptr) { *ptr = 0.0f; };
  ManagedUnrankedMemRefDescriptor outputBuffer =
      makeInitializedUnrankedDescriptor<float>({B, W3}, outputLinearInit);

  // 4. Call the funcOp name `kFuncName` with arguments.
  void *args[2] = {&inputBuffer->descriptor, &outputBuffer->descriptor};
  auto error =
      runner.engine->invoke(kFuncName, llvm::MutableArrayRef<void *>{args});

  // 5. Dump content of output buffer for testing with FileCheck.
  if (!error)
    ::impl::printMemRef(
        *static_cast<StridedMemRefType<float, 2> *>(outputBuffer->descriptor));
}

// CHECK: Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [3, 10]
// CHECK-SAME: strides = [10, 1] data =
// clang-format off
// CHECK-COUNT-3: {{.*[[:space:]].*}}[3177.93,   3177.93,   3177.93,   3177.93,   3177.93,   3177.93,   3177.93,   3177.93,   3177.93,   3177.93]
// clang-format on
