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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

std::unique_ptr<llvm::TargetMachine> createTargetMachine(
    const LLVMTargetOptions& options) {
  std::string errorMessage;
  auto target =
      llvm::TargetRegistry::lookupTarget(options.targetTriple, errorMessage);
  if (!target) return nullptr;
  // TODO(ataei): Once we have an AOT backend pass cpu and cpu-features
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      options.targetTriple, "generic" /* cpu e.g k8*/,
      "" /* cpu features e.g avx512fma*/, {}, {}));
  return machine;
}

LogicalResult runLLVMIRPasses(const LLVMTargetOptions& options,
                              std::unique_ptr<llvm::TargetMachine> machine,
                              llvm::Module* module) {
  llvm::LoopAnalysisManager loopAnalysisManager;
  llvm::FunctionAnalysisManager functionAnalysisManager;
  llvm::CGSCCAnalysisManager cGSCCAnalysisManager;
  llvm::ModuleAnalysisManager moduleAnalysisManager;

  llvm::PassInstrumentationCallbacks passInstrumentationCallbacks;
  llvm::StandardInstrumentations standardInstrumentations;
  standardInstrumentations.registerCallbacks(passInstrumentationCallbacks);

  llvm::PassBuilder passBuilder(machine.get(), options.pipelineTuningOptions,
                                {}, &passInstrumentationCallbacks);
  llvm::AAManager aa = passBuilder.buildDefaultAAPipeline();
  functionAnalysisManager.registerPass([&] { return std::move(aa); });

  passBuilder.registerModuleAnalyses(moduleAnalysisManager);
  passBuilder.registerCGSCCAnalyses(cGSCCAnalysisManager);
  passBuilder.registerFunctionAnalyses(functionAnalysisManager);
  passBuilder.registerLoopAnalyses(loopAnalysisManager);
  passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager,
                                   cGSCCAnalysisManager, moduleAnalysisManager);
  llvm::ModulePassManager modulePassManager;
  modulePassManager =
      passBuilder.buildPerModuleDefaultPipeline(options.optLevel);
  modulePassManager.run(*module, moduleAnalysisManager);

  if (llvm::verifyModule(*module)) return failure();

  return success();
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
