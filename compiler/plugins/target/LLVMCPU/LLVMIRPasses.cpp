// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/LLVMIRPasses.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"

namespace mlir::iree_compiler::IREE::HAL {

LogicalResult runLLVMIRPasses(const LLVMTarget &target,
                              llvm::TargetMachine *machine,
                              llvm::Module *module) {
  llvm::LoopAnalysisManager loopAnalysisManager;
  llvm::FunctionAnalysisManager functionAnalysisManager;
  llvm::CGSCCAnalysisManager cGSCCAnalysisManager;
  llvm::ModuleAnalysisManager moduleAnalysisManager;

  llvm::PassInstrumentationCallbacks passInstrumentationCallbacks;
  llvm::StandardInstrumentations standardInstrumentations(
      module->getContext(),
      /*DebugLogging=*/false);
  standardInstrumentations.registerCallbacks(passInstrumentationCallbacks);

  llvm::PassBuilder passBuilder(machine, target.pipelineTuningOptions, {},
                                &passInstrumentationCallbacks);
  llvm::AAManager aa = passBuilder.buildDefaultAAPipeline();
  functionAnalysisManager.registerPass([&] { return std::move(aa); });

  passBuilder.registerModuleAnalyses(moduleAnalysisManager);
  passBuilder.registerCGSCCAnalyses(cGSCCAnalysisManager);
  passBuilder.registerFunctionAnalyses(functionAnalysisManager);
  passBuilder.registerLoopAnalyses(loopAnalysisManager);
  passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager,
                                   cGSCCAnalysisManager, moduleAnalysisManager);

  switch (target.sanitizerKind) {
  case SanitizerKind::kNone:
    break;
  case SanitizerKind::kAddress: {
    passBuilder.registerOptimizerLastEPCallback(
        [](llvm::ModulePassManager &modulePassManager,
           llvm::OptimizationLevel Level) {
          llvm::AddressSanitizerOptions opts;
          // Can use Never or Always, just not the default Runtime, which
          // introduces a reference to
          // __asan_option_detect_stack_use_after_return, causing linker
          // errors, and anyway we wouldn't really want bother to with a
          // runtime switch for that.
          opts.UseAfterReturn = llvm::AsanDetectStackUseAfterReturnMode::Always;
          bool moduleUseAfterScope = false;
          bool useOdrIndicator = false;
          modulePassManager.addPass(llvm::AddressSanitizerPass(
              opts, moduleUseAfterScope, useOdrIndicator));
        });
  } break;
  case SanitizerKind::kThread: {
    passBuilder.registerOptimizerLastEPCallback(
        [](llvm::ModulePassManager &modulePassManager,
           llvm::OptimizationLevel Level) {
          modulePassManager.addPass(llvm::ModuleThreadSanitizerPass());
          modulePassManager.addPass(llvm::createModuleToFunctionPassAdaptor(
              llvm::ThreadSanitizerPass()));
        });
  } break;
  }

  if (target.optimizerOptLevel != llvm::OptimizationLevel::O0 ||
      target.sanitizerKind != SanitizerKind::kNone) {
    llvm::ModulePassManager modulePassManager;
    modulePassManager =
        passBuilder.buildPerModuleDefaultPipeline(target.optimizerOptLevel);
    modulePassManager.run(*module, moduleAnalysisManager);
  }

  if (llvm::verifyModule(*module))
    return failure();

  return success();
}

LogicalResult runEmitObjFilePasses(llvm::TargetMachine *machine,
                                   llvm::Module *module,
                                   llvm::CodeGenFileType fileType,
                                   std::string *objData) {
  llvm::SmallVector<char, 0> stream_buffer;
  {
    llvm::raw_svector_ostream ostream(stream_buffer);
    // TODO(ataei): Use non legacy pass mamanger for this.
    llvm::legacy::PassManager passManager;
    passManager.add(
        new llvm::TargetLibraryInfoWrapperPass(machine->getTargetTriple()));
    if (machine->addPassesToEmitFile(passManager, ostream,
                                     /*DwoOut=*/nullptr, fileType)) {
      return failure();
    }
    passManager.run(*module);
  }
  // TODO(ataei): This is a work around stream truncation when directly write to
  // string.
  *objData = std::string(stream_buffer.begin(), stream_buffer.end());
  return success();
}

} // namespace mlir::iree_compiler::IREE::HAL
