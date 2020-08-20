// Copyright 2019 Google LLC
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

#include "iree/compiler/Translation/IREEVM.h"

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/TranslationFlags.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Translation.h"

#ifdef IREE_HAVE_EMITC_DIALECT
#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"
#endif  // IREE_HAVE_EMITC_DIALECT

namespace mlir {
namespace iree_compiler {

LogicalResult convertToFlowModule(ModuleOp moduleOp) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  IREE::Flow::buildFlowTransformPassPipeline(passManager);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run flow transformation pass pipeline";
  }
  return success();
}

LogicalResult convertToHALModule(ModuleOp moduleOp,
                                 IREE::HAL::TargetOptions executableOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run flow transformation pass pipeline";
  }
  return success();
}

LogicalResult convertToVMModule(ModuleOp moduleOp,
                                IREE::VM::TargetOptions targetOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run VM transformation pass pipeline";
  }
  return success();
}

void registerIREEVMTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-transformation-pipeline",
      "Runs the full IREE input to VM transformation pipeline",
      [](OpPassManager &passManager) {
        IREE::Flow::buildFlowTransformPassPipeline(passManager);
        IREE::HAL::buildHALTransformPassPipeline(
            passManager, IREE::HAL::getTargetOptionsFromFlags());
        IREE::VM::buildVMTransformPassPipeline(
            passManager, IREE::VM::getTargetOptionsFromFlags());
        passManager.addPass(IREE::createDropCompilerHintsPass());
      });
}

static LogicalResult translateFromMLIRToVM(
    ModuleOp moduleOp, IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions) {
  // Convert from our source to a vm.module in canonical form.
  // After this completes we have a non-bytecode-specific vm.module that we
  // could lower to other forms (LLVM IR, C, etc).
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  IREE::Flow::buildFlowTransformPassPipeline(passManager);
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
  passManager.addPass(mlir::iree_compiler::IREE::createDropCompilerHintsPass());

  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError() << "conversion from source -> vm failed";
  }
  return success();
}

LogicalResult translateFromMLIRToVMBytecodeModule(
    ModuleOp moduleOp, IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions,
    IREE::VM::BytecodeTargetOptions bytecodeOptions,
    llvm::raw_ostream &output) {
  auto result =
      translateFromMLIRToVM(moduleOp, executableOptions, targetOptions);
  if (failed(result)) {
    return result;
  }

  // Serialize to bytecode.
  return translateModuleToBytecode(moduleOp, bytecodeOptions, output);
}

static LogicalResult translateFromMLIRToVMBytecodeModuleWithFlags(
    ModuleOp moduleOp, llvm::raw_ostream &output) {
  mlir::registerPassManagerCLOptions();
  auto halTargetOptions = IREE::HAL::getTargetOptionsFromFlags();
  auto vmTargetOptions = IREE::VM::getTargetOptionsFromFlags();
  auto bytecodeTargetOptions = IREE::VM::getBytecodeTargetOptionsFromFlags();
  return translateFromMLIRToVMBytecodeModule(moduleOp, halTargetOptions,
                                             vmTargetOptions,
                                             bytecodeTargetOptions, output);
}

#ifdef IREE_HAVE_EMITC_DIALECT
LogicalResult translateFromMLIRToVMCModule(
    ModuleOp moduleOp, IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions, llvm::raw_ostream &output) {
  auto result =
      translateFromMLIRToVM(moduleOp, executableOptions, targetOptions);
  if (failed(result)) {
    return result;
  }

  // Serialize to c code.
  return mlir::iree_compiler::IREE::VM::translateModuleToC(moduleOp, output);
}

static LogicalResult translateFromMLIRToVMCModuleWithFlags(
    ModuleOp moduleOp, llvm::raw_ostream &output) {
  mlir::registerPassManagerCLOptions();
  auto halTargetOptions = IREE::HAL::getTargetOptionsFromFlags();
  auto vmTargetOptions = IREE::VM::getTargetOptionsFromFlags();
  return translateFromMLIRToVMCModule(moduleOp, halTargetOptions,
                                      vmTargetOptions, output);
}
#endif  // IREE_HAVE_EMITC_DIALECT

void registerIREEVMTranslation() {
  TranslateFromMLIRRegistration toVMBytecodeModuleWithFlags(
      "iree-mlir-to-vm-bytecode-module",
      translateFromMLIRToVMBytecodeModuleWithFlags);

#ifdef IREE_HAVE_EMITC_DIALECT
  TranslateFromMLIRRegistration toVMCModuleWithFlags(
      "iree-mlir-to-vm-c-module", translateFromMLIRToVMCModuleWithFlags);
#endif  // IREE_HAVE_EMITC_DIALECT
}

}  // namespace iree_compiler
}  // namespace mlir
