// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/API/MLIRInterop.h"

#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_llvmir_translations.h"
#include "iree/compiler/Tools/init_passes.h"
#include "iree/compiler/Tools/init_targets.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::iree_compiler;

// TODO: There is a loose ::IREE namespace somewhere which means that we
// have to fully qualify from the unnamed namespace.
using HALTargetOptions = mlir::iree_compiler::IREE::HAL::TargetOptions;
using VMTargetOptions = mlir::iree_compiler::IREE::VM::TargetOptions;
using VMBytecodeTargetOptions =
    mlir::iree_compiler::IREE::VM::BytecodeTargetOptions;

namespace {
// We have one composite options struct for everything. Not all components
// are applicable to every translation.
struct CompilerOptions {
  BindingOptions bindingOptions;
  InputDialectOptions inputDialectOptions;
  PreprocessingOptions preprocessingOptions;
  HighLevelOptimizationOptions highLevelOptimizationOptions;
  SchedulingOptions schedulingOptions;
  HALTargetOptions halTargetOptions;
  VMTargetOptions vmTargetOptions;
  VMBytecodeTargetOptions vmBytecodeTargetOptions;

  OptionsBinder binder;

  CompilerOptions() : binder(OptionsBinder::local()) {
    bindingOptions.bindOptions(binder);
    inputDialectOptions.bindOptions(binder);
    preprocessingOptions.bindOptions(binder);
    highLevelOptimizationOptions.bindOptions(binder);
    schedulingOptions.bindOptions(binder);
    halTargetOptions.bindOptions(binder);
    vmTargetOptions.bindOptions(binder);
    vmBytecodeTargetOptions.bindOptions(binder);
  }
};
}  // namespace

DEFINE_C_API_PTR_METHODS(IreeCompilerOptions, CompilerOptions)

void ireeCompilerRegisterAllDialects(MlirContext context) {
  DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  mlir::iree_compiler::registerLLVMIRTranslations(registry);
  unwrap(context)->appendDialectRegistry(registry);
}

void ireeCompilerRegisterAllPasses() { registerAllPasses(); }

void ireeCompilerRegisterTargetBackends() { registerHALTargetBackends(); }

IreeCompilerOptions ireeCompilerOptionsCreate() {
  auto options = new CompilerOptions;
  // TODO: Make configurable.
  options->vmTargetOptions.f32Extension = true;
  return wrap(options);
}

MlirLogicalResult ireeCompilerOptionsSetFlags(
    IreeCompilerOptions options, int argc, const char *const *argv,
    void (*onError)(MlirStringRef, void *), void *userData) {
  CompilerOptions *optionsCpp = unwrap(options);
  auto callback = [&](llvm::StringRef message) {
    if (onError) {
      onError(wrap(message), userData);
    }
  };
  if (failed(optionsCpp->binder.parseArguments(argc, argv, callback))) {
    return mlirLogicalResultFailure();
  }
  return mlirLogicalResultSuccess();
}

void ireeCompilerOptionsGetFlags(IreeCompilerOptions options,
                                 bool nonDefaultOnly,
                                 void (*onFlag)(MlirStringRef, void *),
                                 void *userData) {
  auto flagVector = unwrap(options)->binder.printArguments(nonDefaultOnly);
  for (std::string &value : flagVector) {
    onFlag(wrap(llvm::StringRef(value)), userData);
  }
}

void ireeCompilerOptionsDestroy(IreeCompilerOptions options) {
  delete unwrap(options);
}

void ireeCompilerBuildIREEVMPassPipeline(IreeCompilerOptions options,
                                         MlirOpPassManager passManager) {
  auto *optionsCpp = unwrap(options);
  auto *passManagerCpp = unwrap(passManager);
  IREEVMPipelineHooks hooks = {
      // buildConstEvalPassPipelineCallback =
      [](OpPassManager &pm) { pm.addPass(ConstEval::createJitGlobalsPass()); }};
  buildIREEVMTransformPassPipeline(
      optionsCpp->bindingOptions, optionsCpp->inputDialectOptions,
      optionsCpp->preprocessingOptions,
      optionsCpp->highLevelOptimizationOptions, optionsCpp->schedulingOptions,
      optionsCpp->halTargetOptions, optionsCpp->vmTargetOptions, hooks,
      *passManagerCpp);
}

// Translates a module op derived from the ireeCompilerBuildIREEVMPassPipeline
// to serialized bytecode. The module op may either be an outer builtin ModuleOp
// wrapping a VM::ModuleOp or a VM::ModuleOp.
MlirLogicalResult ireeCompilerTranslateModuletoVMBytecode(
    IreeCompilerOptions options, MlirOperation moduleOp,
    MlirStringCallback dataCallback, void *dataUserObject) {
  auto *optionsCpp = unwrap(options);
  Operation *moduleOpCpp = unwrap(moduleOp);
  LogicalResult result = failure();

  mlir::detail::CallbackOstream output(dataCallback, dataUserObject);
  if (auto op = llvm::dyn_cast<mlir::ModuleOp>(moduleOpCpp)) {
    result = iree_compiler::IREE::VM::translateModuleToBytecode(
        op, optionsCpp->vmTargetOptions, optionsCpp->vmBytecodeTargetOptions,
        output);
  } else if (auto op = llvm::dyn_cast<iree_compiler::IREE::VM::ModuleOp>(
                 moduleOpCpp)) {
    result = iree_compiler::IREE::VM::translateModuleToBytecode(
        op, optionsCpp->vmTargetOptions, optionsCpp->vmBytecodeTargetOptions,
        output);
  } else {
    emitError(moduleOpCpp->getLoc()) << "expected a supported module operation";
    result = failure();
  }

  return wrap(result);
}
