// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static llvm::CodeGenOpt::Level passBuilderOptLevelToCodeGenOptLevel(
    const llvm::PassBuilder::OptimizationLevel &level) {
  switch (level.getSpeedupLevel()) {
    case 0:
      return llvm::CodeGenOpt::None;
    case 1:
      return llvm::CodeGenOpt::Less;
    case 2:
    default:
      return llvm::CodeGenOpt::Default;
    case 3:
      return llvm::CodeGenOpt::Aggressive;
  }
}

std::unique_ptr<llvm::TargetMachine> createTargetMachine(
    const LLVMTargetOptions &targetOptions) {
  std::string errorMessage;
  auto target = llvm::TargetRegistry::lookupTarget(targetOptions.targetTriple,
                                                   errorMessage);
  if (!target) return nullptr;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetOptions.targetTriple, targetOptions.targetCPU /* cpu e.g k8*/,
      targetOptions.targetCPUFeatures /* cpu features e.g avx512fma*/,
      targetOptions.options, llvm::Reloc::Model::PIC_, {},
      passBuilderOptLevelToCodeGenOptLevel(targetOptions.optLevel),
      /*JIT=*/false));
  return machine;
}

LogicalResult runLLVMIRPasses(const LLVMTargetOptions &options,
                              llvm::TargetMachine *machine,
                              llvm::Module *module) {
  llvm::LoopAnalysisManager loopAnalysisManager;
  llvm::FunctionAnalysisManager functionAnalysisManager;
  llvm::CGSCCAnalysisManager cGSCCAnalysisManager;
  llvm::ModuleAnalysisManager moduleAnalysisManager;

  llvm::PassInstrumentationCallbacks passInstrumentationCallbacks;
  llvm::StandardInstrumentations standardInstrumentations(
      /*DebugLogging=*/false);
  standardInstrumentations.registerCallbacks(passInstrumentationCallbacks);

  llvm::PassBuilder passBuilder(machine, options.pipelineTuningOptions, {},
                                &passInstrumentationCallbacks);
  llvm::AAManager aa = passBuilder.buildDefaultAAPipeline();
  functionAnalysisManager.registerPass([&] { return std::move(aa); });

  passBuilder.registerModuleAnalyses(moduleAnalysisManager);
  passBuilder.registerCGSCCAnalyses(cGSCCAnalysisManager);
  passBuilder.registerFunctionAnalyses(functionAnalysisManager);
  passBuilder.registerLoopAnalyses(loopAnalysisManager);
  passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager,
                                   cGSCCAnalysisManager, moduleAnalysisManager);

  switch (options.sanitizerKind) {
    case SanitizerKind::kNone:
      break;
    case SanitizerKind::kAddress: {
      passBuilder.registerOptimizerLastEPCallback(
          [](llvm::ModulePassManager &modulePassManager,
             llvm::PassBuilder::OptimizationLevel Level) {
            bool compileKernel = false;
            bool recover = false;
            bool useAfterScope = true;
            bool moduleUseAfterScope = false;
            bool useOdrIndicator = false;
            modulePassManager.addPass(
                llvm::RequireAnalysisPass<llvm::ASanGlobalsMetadataAnalysis,
                                          llvm::Module>());
            modulePassManager.addPass(llvm::ModuleAddressSanitizerPass(
                compileKernel, recover, moduleUseAfterScope, useOdrIndicator));
            modulePassManager.addPass(
                createModuleToFunctionPassAdaptor(llvm::AddressSanitizerPass(
                    compileKernel, recover, useAfterScope)));
          });
    } break;
  }

  if (options.optLevel != llvm::PassBuilder::OptimizationLevel::O0) {
    llvm::ModulePassManager modulePassManager;
    modulePassManager =
        passBuilder.buildPerModuleDefaultPipeline(options.optLevel);
    modulePassManager.run(*module, moduleAnalysisManager);
  }

  if (llvm::verifyModule(*module)) return failure();

  return success();
}

LogicalResult runEmitObjFilePasses(llvm::TargetMachine *machine,
                                   llvm::Module *module, std::string *objData) {
  llvm::SmallVector<char, 0> stream_buffer;
  {
    // TODO(ataei): Use non legacy pass mamanger for this.
    llvm::legacy::PassManager passManager;
    passManager.add(
        new llvm::TargetLibraryInfoWrapperPass(machine->getTargetTriple()));
    llvm::raw_svector_ostream ostream(stream_buffer);
    if (machine->addPassesToEmitFile(passManager, ostream,
                                     /*DwoOut=*/nullptr,
                                     llvm::CGFT_ObjectFile)) {
      return failure();
    }
    passManager.run(*module);
  }
  // TODO(ataei): This is a work around stream truncation when directly write to
  // string.
  *objData = std::string(stream_buffer.begin(), stream_buffer.end());
  return success();
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
