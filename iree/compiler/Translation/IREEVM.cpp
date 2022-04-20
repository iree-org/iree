// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Translation/IREEVM.h"

#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#ifdef IREE_HAVE_EMITC_DIALECT
#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/C/TranslationFlags.h"
#endif  // IREE_HAVE_EMITC_DIALECT

namespace mlir {
namespace iree_compiler {

namespace {

IREEVMPipelineHooks &getHooks() {
  static IREEVMPipelineHooks hooks = {
      // buildConstEvalPassPipelineCallback =
      [](OpPassManager &pm) { pm.addPass(ConstEval::createJitGlobalsPass()); }};
  return hooks;
}

}  // namespace

void buildDefaultIREEVMTransformPassPipeline(OpPassManager &passManager) {
  buildIREEVMTransformPassPipeline(
      BindingOptions::FromFlags::get(), InputDialectOptions::FromFlags::get(),
      HighLevelOptimizationOptions::FromFlags::get(),
      SchedulingOptions::FromFlags::get(),
      IREE::HAL::TargetOptions::FromFlags::get(),
      IREE::VM::TargetOptions::FromFlags::get(), getHooks(), passManager);
}

void registerIREEVMTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-transformation-pipeline",
      "Runs the full IREE input to VM transformation pipeline",
      [](OpPassManager &passManager) {
        buildDefaultIREEVMTransformPassPipeline(passManager);
      });
}

// Converts from our source to a vm.module in canonical form.
// After this completes we have a non-bytecode-specific vm.module that we
// could lower to other forms (LLVM IR, C, etc).
static LogicalResult translateFromMLIRToVM(
    ModuleOp moduleOp, BindingOptions bindingOptions,
    InputDialectOptions inputOptions,
    HighLevelOptimizationOptions highLevelOptimizationOptions,
    SchedulingOptions schedulingOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  buildIREEVMTransformPassPipeline(bindingOptions, inputOptions,
                                   highLevelOptimizationOptions,
                                   schedulingOptions, executableOptions,
                                   targetOptions, getHooks(), passManager);
  if (failed(passManager.run(moduleOp))) {
    llvm::errs() << "compilation from source to vm failed\n";
    return failure();
  }
  return success();
}

// Translates an MLIR module containing a set of supported IREE input dialects
// to an IREE VM bytecode module for loading at runtime.
//
// See iree/schemas/bytecode_module_def.fbs for the description of the
// serialized module format.
//
// Exposed via the --iree-mlir-to-vm-bytecode-module translation.
static LogicalResult translateFromMLIRToVMBytecodeModuleWithFlags(
    ModuleOp moduleOp, llvm::raw_ostream &output) {
  mlir::registerPassManagerCLOptions();
  auto bindingOptions = BindingOptions::FromFlags::get();
  auto inputOptions = InputDialectOptions::FromFlags::get();
  auto highLevelOptimizationOptions =
      HighLevelOptimizationOptions::FromFlags::get();
  auto schedulingOptions = SchedulingOptions::FromFlags::get();
  auto halTargetOptions = IREE::HAL::TargetOptions::FromFlags::get();
  auto vmTargetOptions = IREE::VM::TargetOptions::FromFlags::get();
  auto bytecodeTargetOptions =
      IREE::VM::BytecodeTargetOptions::FromFlags::get();
  auto result = translateFromMLIRToVM(
      moduleOp, bindingOptions, inputOptions, highLevelOptimizationOptions,
      schedulingOptions, halTargetOptions, vmTargetOptions);
  if (failed(result)) {
    return result;
  }
  return translateModuleToBytecode(moduleOp, bytecodeTargetOptions, output);
}

#ifdef IREE_HAVE_EMITC_DIALECT
// Translates an MLIR module containing a set of supported IREE input dialects
// to an IREE VM C module.
//
// Exposed via the --iree-mlir-to-vm-c-module translation.
static LogicalResult translateFromMLIRToVMCModuleWithFlags(
    ModuleOp moduleOp, llvm::raw_ostream &output) {
  mlir::registerPassManagerCLOptions();
  auto bindingOptions = BindingOptions::FromFlags::get();
  auto inputOptions = InputDialectOptions::FromFlags::get();
  auto highLevelOptimizationOptions =
      HighLevelOptimizationOptions::FromFlags::get();
  auto schedulingOptions = SchedulingOptions::FromFlags::get();
  auto halTargetOptions = IREE::HAL::TargetOptions::FromFlags::get();
  auto vmTargetOptions = IREE::VM::TargetOptions::FromFlags::get();
  auto cTargetOptions = IREE::VM::getCTargetOptionsFromFlags();
  auto result = translateFromMLIRToVM(
      moduleOp, bindingOptions, inputOptions, highLevelOptimizationOptions,
      schedulingOptions, halTargetOptions, vmTargetOptions);
  if (failed(result)) {
    return result;
  }
  // Serialize to c code.
  return mlir::iree_compiler::IREE::VM::translateModuleToC(
      moduleOp, cTargetOptions, output);
}
#endif  // IREE_HAVE_EMITC_DIALECT

void registerIREEVMTranslationFlags() {
  BindingOptions::FromFlags::get();
  InputDialectOptions::FromFlags::get();
  HighLevelOptimizationOptions::FromFlags::get();
  IREE::HAL::TargetOptions::FromFlags::get();
  IREE::VM::TargetOptions::FromFlags::get();
  IREE::VM::BytecodeTargetOptions::FromFlags::get();
  SchedulingOptions::FromFlags::get();
}

void registerIREEVMTranslation() {
  registerIREEVMTranslationFlags();
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
