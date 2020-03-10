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

#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
};
static LLVMInitializer initializer;

void mlir::ModelRunner::compile(int llvmOptLevel, int llcOptLevel,
                                const std::string& runtime) {
  // Lower vector operations progressively into more elementary
  // vector operations before running the regular compiler passes.
  {
    OwningRewritePatternList patterns;
    vector::populateVectorSlicesLoweringPatterns(patterns,
                                                 module->getContext());
    vector::populateVectorContractLoweringPatterns(patterns,
                                                   module->getContext());
    mlir::applyPatternsGreedily(*module, patterns);
  }

  // Set up compiler passes.
  PassManager manager(module->getContext());
  manager.addPass(mlir::createConvertLinalgToLoopsPass());
  manager.addPass(mlir::createConvertLinalgToLLVMPass());
  manager.addPass(mlir::createLowerToLLVMPass());
  if (failed(manager.run(*module))) {
    llvm::errs() << "conversion to the LLVM IR dialect failed\n";
    return;
  }

  // Make sure the execution engine runs LLVM passes for the specified
  // optimization level.
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  auto t = tmBuilderOrError->getTargetTriple().getTriple();
  assert(tmBuilderOrError);
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) llvm::errs() << tmOrError.takeError() << "\n";
  assert(tmOrError);
  targetMachine = std::move(tmOrError.get());
  auto transformer = mlir::makeLLVMPassesTransformer(
      /*llvmPasses=*/{},
      llvmOptLevel == -1 ? llvm::Optional<unsigned>() : llvmOptLevel,
      targetMachine.get(),
      /*optPassesInsertPos=*/0);

  // Pass in runtime support library when specified.
  SmallVector<StringRef, 4> libs;
  if (!runtime.empty()) libs.push_back(runtime);

  // Obtain the execution engine.
  auto created = mlir::ExecutionEngine::create(
      *module, transformer, static_cast<llvm::CodeGenOpt::Level>(llcOptLevel),
      libs,
      /*enableObjectCache=*/true,
      /*enableGDBNotificationListener=*/false);
  llvm::handleAllErrors(created.takeError(), [](const llvm::ErrorInfoBase& b) {
    b.log(llvm::errs());
    assert(false);
  });
  engine = std::move(*created);
}
