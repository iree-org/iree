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

#include "experimental/ModelBuilder/ModelRunner.h"

#include "experimental/ModelBuilder/ModelBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

static llvm::cl::opt<bool> mlirDebug(
    "mlir-debug", llvm::cl::desc("Single thread and print-ir-after-all"),
    llvm::cl::init(false));

struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
};
static LLVMInitializer initializer;

namespace llvm {
extern Pass* createLowerMatrixIntrinsicsPass();
}  // end namespace llvm

void mlir::ModelRunner::compile(CompilationOptions compilationOptions,
                                llvm::ArrayRef<const std::string> runtime) {
  if (target == Target::CPUTarget) {
    // Lower vector operations progressively into more elementary
    // vector operations before running the regular compiler passes.
    mlir::OwningRewritePatternList patterns;
    mlir::vector::populateVectorSlicesLoweringPatterns(patterns,
                                                       module->getContext());
    mlir::vector::populateVectorContractLoweringPatterns(
        patterns, module->getContext(),
        compilationOptions.vectorTransformsOptions);
    mlir::applyPatternsAndFoldGreedily(*module, patterns);
  }
  runLoweringPass(compilationOptions.loweringPasses
                      ? compilationOptions.loweringPasses
                      : getDefaultMLIRPassBuilder());

  // Make sure the execution engine runs LLVM passes for the specified
  // optimization level.
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  auto t = tmBuilderOrError->getTargetTriple().getTriple();
  assert(tmBuilderOrError);
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) llvm::errs() << tmOrError.takeError() << "\n";
  assert(tmOrError);
  targetMachine = std::move(tmOrError.get());
  SmallVector<const llvm::PassInfo*, 4> llvmPasses;
  if (target == Target::CPUTarget) {
    // TODO(ntv): Looking up the pass by name fails quite surprisingly. Just
    // build the pass to get its ID to look up the PassInfo.
    const llvm::PassInfo* lowerMatrixIntrinsics = llvm::Pass::lookupPassInfo(
        llvm::createLowerMatrixIntrinsicsPass()->getPassID());
    assert(lowerMatrixIntrinsics);
    llvmPasses.push_back(lowerMatrixIntrinsics);
  }
  auto transformer = mlir::makeLLVMPassesTransformer(
      llvmPasses, compilationOptions.llvmOptLevel, targetMachine.get(),
      /*optPassesInsertPos=*/0);

  // Pass in runtime support library when specified.
  SmallVector<StringRef, 4> libs(runtime.begin(), runtime.end());

  // Obtain the execution engine.
  auto created = mlir::ExecutionEngine::create(
      *module, transformer,
      static_cast<llvm::CodeGenOpt::Level>(compilationOptions.llcOptLevel),
      libs,
      /*enableObjectCache=*/true,
      /*enableGDBNotificationListener=*/false);
  llvm::handleAllErrors(created.takeError(), [](const llvm::ErrorInfoBase& b) {
    b.log(llvm::errs());
    assert(false);
  });
  engine = std::move(*created);
}

static void addVulkanLoweringPass(mlir::PassManager& manager) {
  manager.addPass(mlir::createGpuKernelOutliningPass());
  manager.addPass(mlir::createLegalizeStdOpsForSPIRVLoweringPass());
  manager.addPass(mlir::createConvertGPUToSPIRVPass());
  mlir::OpPassManager& modulePM = manager.nest<mlir::spirv::ModuleOp>();
  modulePM.addPass(mlir::spirv::createLowerABIAttributesPass());
  modulePM.addPass(mlir::spirv::createUpdateVersionCapabilityExtensionPass());
  manager.addPass(mlir::createConvertGpuLaunchFuncToVulkanLaunchFuncPass());
  mlir::LowerToLLVMOptions llvmOptions = {
      /*useBarePtrCallConv =*/false,
      /*emitCWrappers = */ true,
      /*indexBitwidth =*/mlir::kDeriveIndexBitwidthFromDataLayout};
  manager.addPass(createLowerToLLVMPass(llvmOptions));
  manager.addPass(mlir::createConvertVulkanLaunchFuncToVulkanCallsPass());
}

static void addCPULoweringPass(mlir::PassManager& manager) {
  // Set up compiler passes.
  manager.addPass(mlir::createConvertVectorToSCFPass());
  manager.addPass(mlir::createConvertLinalgToLoopsPass());
  manager.addPass(mlir::createConvertLinalgToLLVMPass());
  manager.addPass(mlir::createConvertVectorToLLVMPass());
  manager.addPass(mlir::createLowerToLLVMPass());
}

std::function<void(mlir::PassManager&)>
mlir::ModelRunner::getDefaultMLIRPassBuilder() {
  if (target == Target::CPUTarget) {
    return addCPULoweringPass;
  } else {
    assert(target == Target::GPUTarget);
    return addVulkanLoweringPass;
  }
}

void mlir::ModelRunner::runLoweringPass(
    std::function<void(mlir::PassManager&)> passBuilder) {
  PassManager manager(module->getContext());
  if (mlirDebug) {
    manager.getContext()->disableMultithreading();
    manager.enableIRPrinting([](Pass*, Operation*) { return true; },
                             [](Pass*, Operation*) { return true; }, true, true,
                             llvm::errs());
  }
  passBuilder(manager);
  if (failed(manager.run(*module))) {
    llvm::errs() << "conversion to the LLVM IR dialect failed\n";
    return;
  }
}
