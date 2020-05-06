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

// clang-format off

// NOLINTNEXTLINE
// TODO(thomasraoux): Set the right path to vulkan wrapper shared library. The
// test won't run until this is done.
// RUN: test-simple-jit-vulkan -vulkan-wrapper=$(dirname %s)/../../../../llvm/llvm-project/mlir/tools/libvulkan-runtime-wrappers.so -runtime-support=$(dirname %s)/../../../../llvm/llvm-project/mlir/test/mlir-cpu-runner/libmlir_runner_utils.so 2>&1 | IreeFileCheck %s

#include <string>

#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser.h"
#include "iree/base/initializer.h"

static llvm::cl::opt<std::string> vulkanWrapper(
    "vulkan-wrapper", llvm::cl::desc("Vulkan wrapper library"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> runtimeSupport(
    "runtime-support", llvm::cl::desc("Runtime support library filename"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

using namespace mlir;  // NOLINT

void testVectorAdd1d() {
  MLIRContext context;
  ModelBuilder modelBuilder;
  auto typeA = modelBuilder.getMemRefType(8, modelBuilder.f32);
  auto typeB = modelBuilder.getMemRefType(8, modelBuilder.f32);
  auto typeC = modelBuilder.getMemRefType(8, modelBuilder.f32);
  gpu::GPUFuncOp kernelFunc;
  {
    // create the GPU module.
    auto kernelModule = modelBuilder.makeGPUModule("kernels");
    // create kernel
    kernelFunc = modelBuilder.makeGPUKernel("kernel_add", kernelModule,
                                            {typeA, typeB, typeC});
    OpBuilder b(&kernelFunc.body());
    ScopedContext scope(b, kernelFunc.getLoc());

    StdIndexedValue A(kernelFunc.getArgument(0)), B(kernelFunc.getArgument(1)),
        C(kernelFunc.getArgument(2));
    auto index = b.create<gpu::BlockIdOp>(modelBuilder.loc, b.getIndexType(),
                                          b.getStringAttr("x"));
    C(index) = A(index) + B(index);
    b.create<gpu::ReturnOp>(kernelFunc.getLoc());
  }
  const std::string funcName("add_dispatch");
  {
    // Add host side code, simple dispatch:
    auto f = modelBuilder.makeFunction(funcName, {}, {typeA, typeB, typeC},
      MLIRFuncOpConfig().setEmitCInterface(true));
    OpBuilder b(&f.getBody());
    ScopedContext scope(b, f.getLoc());
    auto eight = std_constant_index(8);
    auto one = std_constant_index(1);
    b.create<gpu::LaunchFuncOp>(
        f.getLoc(), kernelFunc, gpu::KernelDim3{eight, one, one},
        gpu::KernelDim3{one, one, one},
        ValueRange({f.getArgument(0), f.getArgument(1), f.getArgument(2)}));
    modelBuilder.call_print_memref_f32(f.getArgument(2));
    std_ret();
  }

  // 2. Compile the function, pass in runtime support library
  //    to the execution engine for vector.print.
  ModelRunner runner(modelBuilder.getModuleRef(),
                     ModelRunner::Target::GPUTarget);
  runner.compile(CompilationOptions(), {vulkanWrapper, runtimeSupport});

  // 3. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
  auto oneInit = [](unsigned idx, Vector1D<8, float> *ptr) {
    (*ptr)[idx] = 1.0f;
  };
  auto incInit = [](unsigned idx, Vector1D<8, float> *ptr) {
    (*ptr)[idx] = 1.0f + idx;
  };
  auto zeroInit = [](unsigned idx, Vector1D<8, float> *ptr) {
    (*ptr)[idx] = 0.0f;
  };
  auto A = makeInitializedStridedMemRefDescriptor<Vector1D<8, float>, 1>(
      {8}, oneInit);
  auto B = makeInitializedStridedMemRefDescriptor<Vector1D<8, float>, 1>(
      {8}, incInit);
  auto C = makeInitializedStridedMemRefDescriptor<Vector1D<8, float>, 1>(
      {8}, zeroInit);

  // 4. Call the funcOp named `funcName`.
  auto err = runner.invoke(funcName, A, B, C);
  if (err) llvm_unreachable("Error running function.");
}

int main(int argc, char **argv) {
  iree::Initializer::RunInitializers();
  // Allow LLVM setup through command line and parse the
  // test specific option for a runtime support library.
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TestSimpleJITVulkan\n");

  // CHECK: [2,  3,  4,  5,  6,  7,  8,  9]
  testVectorAdd1d();
}
