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
// RUN: test-vec-to-gpu -vulkan-wrapper=$(dirname %s)/../../../../llvm/llvm-project/mlir/tools/libvulkan-runtime-wrappers.so 2>&1 | IreeFileCheck %s

// clang-format on
#include <string>
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "experimental/ModelBuilder/VulkanWrapperPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser.h"
#include "iree/base/initializer.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Pass/PassManager.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;                    // NOLINT
using namespace mlir::edsc;              // NOLINT
using namespace mlir::edsc::intrinsics;  // NOLINT

static llvm::cl::opt<std::string> vulkanWrapper(
    "vulkan-wrapper", llvm::cl::desc("Vulkan wrapper library"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static void addLoweringPasses(mlir::PassManager &pm,
                              llvm::ArrayRef<int64_t> workgroupSize,
                              llvm::ArrayRef<Type> args) {
  pm.addPass(mlir::iree_compiler::createVectorToGPUPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::iree_compiler::createConvertToSPIRVPass());

  auto &spirvModulePM = pm.nest<mlir::spirv::ModuleOp>();
  spirvModulePM.addPass(mlir::createSetSpirvABIPass());
  spirvModulePM.addPass(mlir::spirv::createLowerABIAttributesPass());
  spirvModulePM.addPass(mlir::createCanonicalizerPass());
  spirvModulePM.addPass(mlir::createCSEPass());
  spirvModulePM.addPass(
      mlir::spirv::createUpdateVersionCapabilityExtensionPass());

  pm.addPass(mlir::createAddVulkanLaunchWrapperPass(workgroupSize, args));
  mlir::LowerToLLVMOptions llvmOptions = {
      /*useBarePtrCallConv=*/false,
      /*emitCWrappers=*/true,
      /*indexBitwidth=*/mlir::kDeriveIndexBitwidthFromDataLayout};
  pm.addPass(createLowerToLLVMPass(llvmOptions));
  pm.addPass(mlir::createConvertVulkanLaunchFuncToVulkanCallsPass());
}

void testVecAdd() {
  const int warpSize = 32;
  // Simple test a single warp.
  const int width = warpSize;
  StringLiteral funcName = "kernel_vecadd";
  MLIRContext context;
  ModelBuilder modelBuilder;
  auto nVectorType = modelBuilder.getVectorType(width, modelBuilder.f32);
  auto typeA = modelBuilder.getMemRefType({width}, modelBuilder.f32);
  auto typeB = modelBuilder.getMemRefType({width}, modelBuilder.f32);
  auto typeC = modelBuilder.getMemRefType({width}, modelBuilder.f32);
  // 1. Build the kernel.
  {
    modelBuilder.addGPUAttr();
    // create kernel
    FuncOp kernelFunc = modelBuilder.makeFunction(
        funcName, {}, {typeA, typeB, typeC}, MLIRFuncOpConfig());
    // Right now we map one workgroup to one warp.
    kernelFunc.setAttr(spirv::getEntryPointABIAttrName(),
                       spirv::getEntryPointABIAttr({warpSize, 1, 1}, &context));
    OpBuilder b(&kernelFunc.getBody());
    ScopedContext scope(b, kernelFunc.getLoc());

    auto A = kernelFunc.getArgument(0);
    auto B = kernelFunc.getArgument(1);
    auto C = kernelFunc.getArgument(2);

    auto zero = modelBuilder.constant_index(0);
    Value vA = vector_transfer_read(nVectorType, A, ValueRange({zero}));
    Value vB = vector_transfer_read(nVectorType, B, ValueRange({zero}));
    auto vC = std_addf(vA, vB);
    vector_transfer_write(vC, C, ValueRange({zero}));
    std_ret();
  }
  // 2. Compile the function, pass in runtime support library
  //    to the execution engine for vector.print.
  ModelRunner runner(modelBuilder.getModuleRef(),
                     ModelRunner::Target::CPUTarget);
  CompilationOptions options;
  auto lowering = [&](mlir::PassManager &pm) {
    addLoweringPasses(pm, {1, 1, 1}, {typeA, typeB, typeC});
  };
  options.loweringPasses = lowering;
  runner.compile(options, {vulkanWrapper});

  // 3. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
  auto oneInit = [](unsigned idx, float *ptr) { ptr[idx] = 2.0f + 3 * idx; };
  auto incInit = [](unsigned idx, float *ptr) { ptr[idx] = 1.0f + idx; };
  auto zeroInit = [](unsigned idx, float *ptr) { ptr[idx] = 0.0f; };
  auto A = makeInitializedStridedMemRefDescriptor<float, 1>({width}, oneInit);
  auto B = makeInitializedStridedMemRefDescriptor<float, 1>({width}, incInit);
  auto C = makeInitializedStridedMemRefDescriptor<float, 1>({width}, zeroInit);

  // 4. Call the funcOp named `funcName`.
  auto err = runner.invoke(std::string(funcName) + "_wrapper", A, B, C);
  if (err) llvm_unreachable("Error running function.");

  // 5. Dump content of input and output buffer for testing with FileCheck.
  ::impl::printMemRef(*A);
  ::impl::printMemRef(*B);
  ::impl::printMemRef(*C);
}

int main(int argc, char **argv) {
  ModelBuilder::registerAllDialects();
  iree::Initializer::RunInitializers();
  // Allow LLVM setup through command line and parse the
  // test specific option for a runtime support library.
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TestVecAdd\n");
  // clang-format off

  // CHECK: Memref
  // CHECK: [2,  5,  8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,
  // CHECK: 44,  47,  50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,
  // CHECK: 86,  89,  92,  95]
  // CHECK: Memref
  // CHECK: [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,
  // CHECK: 16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
  // CHECK: 30,  31,  32]
  // CHECK: Memref
  // CHECK: [3,  7,  11,  15,  19,  23,  27,  31,  35,  39,  43,  47,  51,  55,
  // CHECK: 59,  63,  67,  71,  75,  79,  83,  87,  91,  95,  99,  103,  107,
  // CHECK: 111,  115,  119,  123,  127]
  testVecAdd();
}
