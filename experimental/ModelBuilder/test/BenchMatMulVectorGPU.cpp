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
#include <string>

#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "experimental/ModelBuilder/VulkanWrapperPass.h"
#include "iree/base/initializer.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;                    // NOLINT
using namespace mlir::edsc;              // NOLINT
using namespace mlir::edsc::intrinsics;  // NOLINT

static llvm::cl::opt<std::string> vulkanWrapper(
    "vulkan-wrapper", llvm::cl::desc("Vulkan wrapper library"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> correctness(
    "correctness",
    llvm::cl::desc(
        "Compare the result to value calculated on CPU. We will use a smaller "
        "matrix multiply in this case to avoid long runtime."),
    llvm::cl::init(false));

static void addLoweringPasses(mlir::PassManager &pm,
                              llvm::ArrayRef<int64_t> numWorkgroups,
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

  pm.addPass(mlir::createAddVulkanLaunchWrapperPass(numWorkgroups, args));
  mlir::LowerToLLVMOptions llvmOptions = {
      /*useBarePtrCallConv=*/false,
      /*emitCWrappers=*/true,
      /*indexBitwidth=*/mlir::kDeriveIndexBitwidthFromDataLayout};
  pm.addPass(createLowerToLLVMPass(llvmOptions));
  pm.addPass(mlir::createConvertVulkanLaunchFuncToVulkanCallsPass());
}

void matMul(int m, int n, int k, int tileM, int tileN, int tileK,
            bool correctness) {
  const int warpSize = 32;
  const int resRows = m;
  const int resColumns = n;
  const int reductionSize = k;
  StringLiteral funcName = "kernel_matmul";
  MLIRContext context;
  ModelBuilder modelBuilder;

  auto typeA =
      modelBuilder.getMemRefType({resRows, reductionSize}, modelBuilder.i8);
  auto typeB =
      modelBuilder.getMemRefType({reductionSize, resColumns}, modelBuilder.i8);
  auto typeC = modelBuilder.getMemRefType({resRows, resColumns},
                                          modelBuilder.getI32Type());
  // 1. Build the kernel.
  {
    modelBuilder.addGPUAttr();
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

    linalg_matmul(TypeRange{}, ValueRange{A, B, C});
    std_ret();
  }

  // 2. Compile the function, pass in runtime support library to the execution
  // engine for vector.print.
  ModelRunner runner(modelBuilder.getModuleRef(),
                     ModelRunner::Target::GPUTarget);
  CompilationOptions options;
  options.loweringPasses = [&](mlir::PassManager &pm) {
    MatmulCodegenStrategy strategy;
    // Use hardcoded value for cooperative matrix size. Those will be pulled
    // from device properties eventually.
    const int cooperativeMatrixM = 8;
    const int cooperativeMatrixN = 8;
    const int cooperativeMatrixK = 32;
    // Swap the order of the parallel loops because PLoopToGPU pattern assigns
    // dimension in reverse order of the loop.
    // TODO(thomasraooux) LICM is disabled due to limitation in SPIR-V
    strategy
        .tile<linalg::MatmulOp>(
            linalg::LinalgTilingOptions()
                .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
                .setTileSizes({tileM, tileN, tileK})
                .setInterchange({1, 0, 2}))
        .setHoistInvariantCode(false)
        .vectorize<linalg::MatmulOp>()
        .unrollVector<vector::ContractionOp>(
            {cooperativeMatrixM, cooperativeMatrixN, cooperativeMatrixK});
    modelBuilder.getModuleRef()->walk(
        [&](FuncOp fn) { strategy.transform(fn); });
    addLoweringPasses(pm, {resRows / tileM, resColumns / tileN, 1},
                      {typeA, typeB, typeC});
  };
  runner.compile(options, {vulkanWrapper});

  // 3. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
  auto oneInit = [](unsigned idx, uint8_t *ptr) { ptr[idx] = 2 * idx + 1; };
  auto incInit = [](unsigned idx, uint8_t *ptr) { ptr[idx] = idx; };
  auto zeroInit = [](unsigned idx, uint32_t *ptr) { ptr[idx] = 0; };
  auto A = makeInitializedStridedMemRefDescriptor<uint8_t, 2>(
      {resRows, reductionSize}, oneInit);
  auto B = makeInitializedStridedMemRefDescriptor<uint8_t, 2>(
      {reductionSize, resColumns}, incInit);
  auto C = makeInitializedStridedMemRefDescriptor<uint32_t, 2>(
      {resRows, resColumns}, zeroInit);
  auto CPURes = makeInitializedStridedMemRefDescriptor<uint32_t, 2>(
      {resRows, resColumns}, zeroInit);

  // Is checking corretness compare to the value computed on CPU.
  if (correctness) {
    for (int i = 0; i < resRows; i++) {
      for (int j = 0; j < resColumns; j++) {
        uint32_t acc = (*C)[i][j];
        for (int k = 0; k < reductionSize; k++) {
          uint32_t a = (*A)[i][k];
          uint32_t b = (*B)[k][j];
          acc += a * b;
        }
        (*CPURes)[i][j] = acc;
      }
    }
  }

  // 4. Call the funcOp named `funcName`.
  auto err = runner.invoke(std::string(funcName) + "_wrapper", A, B, C);
  if (err) llvm_unreachable("Error running function.");

  if (correctness) {
    bool correct = true;
    for (int i = 0; i < resRows; i++) {
      for (int j = 0; j < resColumns; j++) {
        if ((*CPURes)[i][j] != (*C)[i][j]) {
          correct = false;
          printf("mismatch at index(%i, %i) was expecting %i but got %i\n", i,
                 j, (*CPURes)[i][j], (*C)[i][j]);
        }
      }
    }
    if (correct) printf("pass\n");
  }
}

int main(int argc, char **argv) {
  ModelBuilder::registerAllDialects();
  iree::Initializer::RunInitializers();
  // Allow LLVM setup through command line and parse the
  // test specific option for a runtime support library.
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "BenchMatMulVectorGPU\n");
  int m = 4096;
  int n = 4096;
  int k = 4096;
  if (correctness) {
    m = 256;
    n = 256;
    k = 256;
  }
  printf("Matrix size: %ix%ix%i", m, n, k);
  int tileK = 32;
  for (int tileM = 8; tileM <= 128; tileM *= 2) {
    for (int tileN = 8; tileN <= 128; tileN *= 2) {
      printf("tileM=%i tileN=%i tileK=%i\n", tileM, tileN, tileK);
      // For non-power of two tile sizes, round up the matrix size to
      // be an even multiple of the tile size.
      // TODO(thomasraoux): enable non power of two tiles once affine.min
      // folding is fixed.
      auto paddedM = (m + tileM - 1) / tileM * tileM;
      auto paddedN = (n + tileN - 1) / tileN * tileN;
      auto paddedK = (k + tileK - 1) / tileK * tileK;

      matMul(paddedM, paddedN, paddedK, tileM, tileN, tileK, correctness);
    }
  }
}
