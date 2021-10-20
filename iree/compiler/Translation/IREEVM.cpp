// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Translation/IREEVM.h"

#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/TranslationFlags.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Translation.h"

#ifdef IREE_HAVE_EMITC_DIALECT
#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/C/TranslationFlags.h"
#endif  // IREE_HAVE_EMITC_DIALECT

namespace mlir {
namespace iree_compiler {

static BindingOptions getBindingOptionsFromFlags() {
  static llvm::cl::OptionCategory bindingOptionsCategory(
      "IREE translation binding support options");

  static llvm::cl::opt<bool> *bindingsNativeFlag = new llvm::cl::opt<bool>{
      "iree-native-bindings-support",
      llvm::cl::desc(
          "Include runtime support for native IREE ABI-compatible bindings"),
      llvm::cl::init(true), llvm::cl::cat(bindingOptionsCategory)};

  static llvm::cl::opt<bool> *bindingsTFLiteFlag = new llvm::cl::opt<bool>{
      "iree-tflite-bindings-support",
      llvm::cl::desc(
          "Include runtime support for the IREE TFLite compatibility bindings"),
      llvm::cl::init(false), llvm::cl::cat(bindingOptionsCategory)};

  BindingOptions bindingOptions;
  bindingOptions.native = *bindingsNativeFlag;
  bindingOptions.tflite = *bindingsTFLiteFlag;
  return bindingOptions;
}

static InputDialectOptions getInputDialectOptionsFromFlags() {
  static llvm::cl::OptionCategory inputDialectOptions(
      "IREE options for controlling the input transformations to apply");

  static llvm::cl::opt<InputDialectOptions::Type> *typeFlag =
      new llvm::cl::opt<InputDialectOptions::Type>{
          "iree-input-type", llvm::cl::desc("IREE input type"),
          llvm::cl::values(clEnumValN(InputDialectOptions::Type::none, "none",
                                      "No input dialect transformation"),
                           clEnumValN(InputDialectOptions::Type::tosa, "tosa",
                                      "Legalize from TOSA ops"),
                           clEnumValN(InputDialectOptions::Type::mhlo, "mhlo",
                                      "Legalize from MHLO ops")),
          llvm::cl::init(InputDialectOptions::Type::none),
          llvm::cl::cat(inputDialectOptions)};

  InputDialectOptions options;
  options.type = *typeFlag;
  return options;
}

// Performs initial dialect conversion to get the canonical input lowered into
// the IREE execution/dataflow dialect.
//
// This will fail if we cannot support the input yet. The hope is that any
// error that happens after this point is either backend-specific (like
// unsupported SPIR-V lowering) or a bug.
static LogicalResult convertToFlowModule(ModuleOp moduleOp) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  IREE::Flow::TransformOptions flowOptions;
  IREE::Flow::buildFlowTransformPassPipeline(passManager, flowOptions);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run flow transformation pass pipeline";
  }
  return success();
}

// Runs the flow->HAL transform pipeline to lower a flow module and compile
// executables for the specified target backends.
static LogicalResult convertToHALModule(
    ModuleOp moduleOp, IREE::HAL::TargetOptions executableOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run HAL transformation pass pipeline";
  }
  return success();
}

// Converts the lowered module to a canonical vm.module containing only vm ops.
// This uses patterns to convert from standard ops and other dialects to their
// vm ABI form.
static LogicalResult convertToVMModule(ModuleOp moduleOp,
                                       IREE::VM::TargetOptions targetOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run VM transformation pass pipeline";
  }
  return success();
}

void buildIREEVMTransformPassPipeline(
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions, OpPassManager &passManager) {
  if (bindingOptions.native) {
    IREE::ABI::buildTransformPassPipeline(passManager);
  }
  if (bindingOptions.tflite) {
    IREE::TFLite::buildTransformPassPipeline(passManager);
  }

  switch (inputOptions.type) {
    case InputDialectOptions::Type::none:
      break;
    case InputDialectOptions::Type::tosa:
      buildTOSAInputConversionPassPipeline(passManager);
      break;
    case InputDialectOptions::Type::mhlo:
      buildMHLOInputConversionPassPipeline(passManager);
      break;
  }

  IREE::Flow::TransformOptions flowOptions;

  buildCommonInputConversionPassPipeline(passManager);
  IREE::Flow::buildFlowTransformPassPipeline(passManager, flowOptions);
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
  passManager.addPass(IREE::Util::createDropCompilerHintsPass());
}

void buildDefaultIREEVMTransformPassPipeline(OpPassManager &passManager) {
  buildIREEVMTransformPassPipeline(
      getBindingOptionsFromFlags(), getInputDialectOptionsFromFlags(),
      IREE::HAL::getTargetOptionsFromFlags(),
      IREE::VM::getTargetOptionsFromFlags(), passManager);
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
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  buildIREEVMTransformPassPipeline(bindingOptions, inputOptions,
                                   executableOptions, targetOptions,
                                   passManager);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError() << "conversion from source -> vm failed";
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
  auto bindingOptions = getBindingOptionsFromFlags();
  auto inputOptions = getInputDialectOptionsFromFlags();
  auto halTargetOptions = IREE::HAL::getTargetOptionsFromFlags();
  auto vmTargetOptions = IREE::VM::getTargetOptionsFromFlags();
  auto bytecodeTargetOptions = IREE::VM::getBytecodeTargetOptionsFromFlags();
  auto result = translateFromMLIRToVM(moduleOp, bindingOptions, inputOptions,
                                      halTargetOptions, vmTargetOptions);
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
  auto bindingOptions = getBindingOptionsFromFlags();
  auto inputOptions = getInputDialectOptionsFromFlags();
  auto halTargetOptions = IREE::HAL::getTargetOptionsFromFlags();
  auto vmTargetOptions = IREE::VM::getTargetOptionsFromFlags();
  auto cTargetOptions = IREE::VM::getCTargetOptionsFromFlags();
  auto result = translateFromMLIRToVM(moduleOp, bindingOptions, inputOptions,
                                      halTargetOptions, vmTargetOptions);
  if (failed(result)) {
    return result;
  }
  // Serialize to c code.
  return mlir::iree_compiler::IREE::VM::translateModuleToC(
      moduleOp, cTargetOptions, output);
}
#endif  // IREE_HAVE_EMITC_DIALECT

void registerIREEVMTranslationFlags() {
  getBindingOptionsFromFlags();
  getInputDialectOptionsFromFlags();
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
