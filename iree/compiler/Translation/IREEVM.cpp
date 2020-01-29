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
#include "iree/compiler/Dialect/HAL/Target/ExecutableTarget.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/TranslationFlags.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Translation.h"

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

LogicalResult convertToHALModule(
    ModuleOp moduleOp, IREE::HAL::ExecutableTargetOptions executableOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run flow transformation pass pipeline";
  }
  return success();
}

LogicalResult convertToVMModule(ModuleOp moduleOp) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  IREE::VM::buildVMTransformPassPipeline(passManager);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run VM transformation pass pipeline";
  }
  return success();
}

static PassPipelineRegistration<> transformPassPipeline(
    "iree-transformation-pipeline",
    "Runs the full IREE input to VM transformation pipeline",
    [](OpPassManager &passManager) {
      IREE::Flow::buildFlowTransformPassPipeline(passManager);
      IREE::HAL::buildHALTransformPassPipeline(
          passManager, IREE::HAL::getExecutableTargetOptionsFromFlags());
      IREE::VM::buildVMTransformPassPipeline(passManager);
      passManager.addPass(IREE::createDropCompilerHintsPass());
    });

LogicalResult translateFromMLIRToVMBytecodeModule(
    ModuleOp moduleOp, IREE::HAL::ExecutableTargetOptions executableOptions,
    IREE::VM::BytecodeTargetOptions bytecodeOptions,
    llvm::raw_ostream &output) {
  // Convert from our source to a vm.module in canonical form.
  // After this completes we have a non-bytecode-specific vm.module that we
  // could lower to other forms (LLVM IR, C, etc).
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  IREE::Flow::buildFlowTransformPassPipeline(passManager);
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  IREE::VM::buildVMTransformPassPipeline(passManager);
  passManager.addPass(mlir::iree_compiler::IREE::createDropCompilerHintsPass());
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError() << "conversion from source -> vm failed";
  }

  // Serialize to bytecode.
  return translateModuleToBytecode(moduleOp, bytecodeOptions, output);
}

static LogicalResult translateFromMLIRToVMBytecodeModuleWithFlags(
    ModuleOp moduleOp, llvm::raw_ostream &output) {
  mlir::registerPassManagerCLOptions();
  auto executableTargetOptions =
      IREE::HAL::getExecutableTargetOptionsFromFlags();
  auto bytecodeTargetOptions = IREE::VM::getBytecodeTargetOptionsFromFlags();
  return translateFromMLIRToVMBytecodeModule(moduleOp, executableTargetOptions,
                                             bytecodeTargetOptions, output);
}

static TranslateFromMLIRRegistration toVMBytecodeModuleWithFlags(
    "iree-mlir-to-vm-bytecode-module",
    translateFromMLIRToVMBytecodeModuleWithFlags);

}  // namespace iree_compiler
}  // namespace mlir
