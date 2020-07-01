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

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
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
    const LLVMTargetOptions& targetOptions) {
  std::string errorMessage;
  auto target = llvm::TargetRegistry::lookupTarget(targetOptions.targetTriple,
                                                   errorMessage);
  if (!target) return nullptr;
  // TODO(ataei): Once we have an AOT backend pass cpu and cpu-features
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetOptions.targetTriple, "generic" /* cpu e.g k8*/,
      "" /* cpu features e.g avx512fma*/, targetOptions.options, {}));
  return machine;
}

void createLLVMInvocationFunc(const std::string& name, llvm::Module* module) {
  // TODO(ataei): This is written as a stub in LLVM IR. It would be easier to
  // have this using MLIR and lower it to LLVM like the dispatch function
  // implementation is.

  auto& ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  auto var_func = module->getFunction(name);

  auto new_type = llvm::FunctionType::get(
      builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
      /*isVarArg=*/false);

  auto new_name = "invoke_" + name;
  auto func_cst = module->getOrInsertFunction(new_name, new_type);
  llvm::Function* interface_func =
      llvm::cast<llvm::Function>(func_cst.getCallee());

  auto bb = llvm::BasicBlock::Create(ctx);
  bb->insertInto(interface_func);
  builder.SetInsertPoint(bb);
  llvm::Value* argList = interface_func->arg_begin();
  llvm::SmallVector<llvm::Value*, 8> args;
  args.reserve(llvm::size(var_func->args()));
  for (auto& indexedArg : llvm::enumerate(var_func->args())) {
    llvm::Value* arg_index = llvm::Constant::getIntegerValue(
        builder.getInt64Ty(), llvm::APInt(64, indexedArg.index()));
    llvm::Value* arg_ptr_ptr = builder.CreateGEP(argList, arg_index);
    llvm::Value* arg_ptr = builder.CreateLoad(arg_ptr_ptr);
    arg_ptr = builder.CreateBitCast(
        arg_ptr, indexedArg.value().getType()->getPointerTo());
    llvm::Value* arg = builder.CreateLoad(arg_ptr);
    args.push_back(arg);
  }
  builder.CreateCall(var_func, args);
  builder.CreateRetVoid();
}

LogicalResult runLLVMIRPasses(const LLVMTargetOptions& options,
                              llvm::TargetMachine* machine,
                              llvm::Module* module) {
  llvm::LoopAnalysisManager loopAnalysisManager;
  llvm::FunctionAnalysisManager functionAnalysisManager;
  llvm::CGSCCAnalysisManager cGSCCAnalysisManager;
  llvm::ModuleAnalysisManager moduleAnalysisManager;

  llvm::PassInstrumentationCallbacks passInstrumentationCallbacks;
  llvm::StandardInstrumentations standardInstrumentations;
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
  llvm::ModulePassManager modulePassManager;
  modulePassManager =
      passBuilder.buildPerModuleDefaultPipeline(options.optLevel);
  modulePassManager.run(*module, moduleAnalysisManager);

  if (llvm::verifyModule(*module)) return failure();

  return success();
}

LogicalResult runEmitObjFilePasses(llvm::TargetMachine* machine,
                                   llvm::Module* module, std::string* objData) {
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
