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
// RUN: test-matmul-vulkan -vulkan-wrapper=$(dirname %s)/../../../../llvm/llvm-project/mlir/tools/libvulkan-runtime-wrappers.so 2>&1 | IreeFileCheck %s

// NOLINTNEXTLINE
// RUN: test-matmul-vulkan -vulkan-wrapper=$(dirname %s)/../../../../llvm/llvm-project/mlir/tools/libvulkan-runtime-wrappers.so -use-workgroup-memory -workgroup-size=2,2 2>&1 | IreeFileCheck %s

// clang-format on
#include <string>
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

static llvm::cl::opt<std::string> vulkanWrapper(
    "vulkan-wrapper", llvm::cl::desc("Vulkan wrapper library"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> useWorkgroupMemory(
    "use-workgroup-memory", llvm::cl::desc("Enable use of workgroup memory"),
    llvm::cl::value_desc("boolean"), llvm::cl::init(false));

static llvm::cl::list<int> workgroupSize(
    "workgroup-size", llvm::cl::desc("Workgroup size to use"),
    llvm::cl::CommaSeparated);

static llvm::cl::list<int> tileSizes("tile-sizes",
                                     llvm::cl::desc("Tile sizes to use"),
                                     llvm::cl::CommaSeparated);

using namespace mlir;                    // NOLINT
using namespace mlir::edsc;              // NOLINT
using namespace mlir::edsc::intrinsics;  // NOLINT

void testMatMul() {
  const int height = 4;
  const int width = 4;
  StringLiteral funcName = "kernel_matmul";
  MLIRContext context;
  ModelBuilder modelBuilder;
  auto typeA = modelBuilder.getMemRefType({width, height}, modelBuilder.f32);
  auto typeB = modelBuilder.getMemRefType({width, height}, modelBuilder.f32);
  auto typeC = modelBuilder.getMemRefType({width, height}, modelBuilder.f32);
  // 1. Build the kernel.
  {
    modelBuilder.addGPUAttr();
    // create kernel
    FuncOp kernelFunc = modelBuilder.makeFunction(
        funcName, {}, {typeA, typeB, typeC}, MLIRFuncOpConfig());
    OpBuilder b(&kernelFunc.getBody());
    ScopedContext scope(b, kernelFunc.getLoc());

    Value A = kernelFunc.getArgument(0);
    Value B = kernelFunc.getArgument(1);
    Value C = kernelFunc.getArgument(2);
    (linalg_matmul(TypeRange{}, ValueRange{A, B, C}));
    std_ret();
  }
  // 2. Compile the function, pass in runtime support library
  //    to the execution engine for vector.print.
  ModelRunner runner(modelBuilder.getModuleRef(),
                     ModelRunner::Target::GPUTarget);
  CompilationOptions options;
  SmallVector<Type, 3> args = {typeA, typeB, typeC};
  SmallVector<int64_t, 4> vWorkgroupSizes(workgroupSize.begin(),
                                          workgroupSize.end());
  SmallVector<int64_t, 4> vTileSizes(tileSizes.begin(), tileSizes.end());
  auto lowering = [&](mlir::PassManager &pm) {
    pm.addPass(mlir::iree_compiler::createLinalgTileAndFusePass(
        vWorkgroupSizes, vTileSizes, useWorkgroupMemory));
    pm.addPass(mlir::iree_compiler::createConvertToGPUPass());
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

    int numWorkgroupX =
        vWorkgroupSizes.empty()
            ? 1
            : (width + vWorkgroupSizes[0] - 1) / vWorkgroupSizes[0];
    int numWorkgroupY =
        vWorkgroupSizes.size() < 2
            ? 1
            : (height + vWorkgroupSizes[1] - 1) / vWorkgroupSizes[1];
    pm.addPass(mlir::createAddVulkanLaunchWrapperPass(
        {numWorkgroupX, numWorkgroupY, 1}, args));
    mlir::LowerToLLVMOptions llvmOptions = {
        /*useBarePtrCallConv =*/false,
        /*emitCWrappers = */ true,
        /*indexBitwidth =*/mlir::kDeriveIndexBitwidthFromDataLayout};
    pm.addPass(createLowerToLLVMPass(llvmOptions));
    pm.addPass(mlir::createConvertVulkanLaunchFuncToVulkanCallsPass());
  };
  options.loweringPasses = lowering;
  runner.compile(options, {vulkanWrapper});

  // 3. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
  auto oneInit = [](unsigned idx, float *ptr) { ptr[idx] = 2.0f + 3 * idx; };
  auto incInit = [](unsigned idx, float *ptr) { ptr[idx] = 1.0f + idx; };
  auto zeroInit = [](unsigned idx, float *ptr) { ptr[idx] = 0.0f; };
  auto A = makeInitializedStridedMemRefDescriptor<float, 2>({width, height},
                                                            oneInit);
  auto B = makeInitializedStridedMemRefDescriptor<float, 2>({width, height},
                                                            incInit);
  auto C = makeInitializedStridedMemRefDescriptor<float, 2>({width, height},
                                                            zeroInit);

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
  llvm::cl::ParseCommandLineOptions(argc, argv, "TestMatMulVulkan\n");
  // clang-format off

  // CHECK: Memref
  // CHECK: [2,   5,   8,   11],
  // CHECK: [14,   17,   20,   23],
  // CHECK: [26,   29,   32,   35],
  // CHECK: [38,   41,   44,   47]
  // CHECK: Memref
  // CHECK: [1,   2,   3,   4],
  // CHECK: [5,   6,   7,   8],
  // CHECK: [9,   10,   11,   12],
  // CHECK: [13,   14,   15,   16]
  // CHECK: Memref
  // CHECK: [242,   268,   294,   320],
  // CHECK: [578,   652,   726,   800],
  // CHECK: [914,   1036,   1158,   1280],
  // CHECK: [1250,   1420,   1590,   1760]
  testMatMul();
}
