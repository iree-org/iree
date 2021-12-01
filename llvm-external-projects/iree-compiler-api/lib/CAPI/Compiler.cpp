// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-compiler-c/Compiler.h"

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#include "iree/compiler/Translation/IREEVM.h"
#include "iree/tools/init_targets.h"
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
  HALTargetOptions executableOptions;
  VMTargetOptions vmTargetOptions;
  VMBytecodeTargetOptions vmBytecodeTargetOptions;
};
}  // namespace

DEFINE_C_API_PTR_METHODS(IreeCompilerOptions, CompilerOptions)

void ireeCompilerRegisterTargetBackends() { registerHALTargetBackends(); }

IreeCompilerOptions ireeCompilerOptionsCreate() {
  auto options = new CompilerOptions;
  // TODO: Make configurable.
  options->vmTargetOptions.f32Extension = true;
  return wrap(options);
}

void ireeCompilerOptionsDestroy(IreeCompilerOptions options) {
  delete unwrap(options);
}

void ireeCompilerOptionsAddTargetBackend(IreeCompilerOptions options,
                                         const char *targetBackend) {
  unwrap(options)->executableOptions.targets.push_back(
      std::string(targetBackend));
}

void ireeCompilerOptionsSetInputDialectMHLO(IreeCompilerOptions options) {
  unwrap(options)->inputDialectOptions.type = InputDialectOptions::Type::mhlo;
}

void ireeCompilerOptionsSetInputDialectTOSA(IreeCompilerOptions options) {
  unwrap(options)->inputDialectOptions.type = InputDialectOptions::Type::tosa;
}

void ireeCompilerOptionsSetInputDialectXLA(IreeCompilerOptions options) {
  unwrap(options)->inputDialectOptions.type = InputDialectOptions::Type::xla;
}

void ireeCompilerBuildXLACleanupPassPipeline(MlirOpPassManager passManager) {
  auto *passManagerCpp = unwrap(passManager);
  MHLO::buildXLACleanupPassPipeline(*passManagerCpp);
}

void ireeCompilerBuildMHLOImportPassPipeline(MlirOpPassManager passManager) {
  auto *passManagerCpp = unwrap(passManager);
  MHLO::buildMHLOInputConversionPassPipeline(*passManagerCpp);
}

void ireeCompilerBuildTOSAImportPassPipeline(MlirOpPassManager passManager) {
  auto *passManagerCpp = unwrap(passManager);
  buildTOSAInputConversionPassPipeline(*passManagerCpp);
}

void ireeCompilerBuildIREEVMPassPipeline(IreeCompilerOptions options,
                                         MlirOpPassManager passManager) {
  auto *optionsCpp = unwrap(options);
  auto *passManagerCpp = unwrap(passManager);
  buildIREEVMTransformPassPipeline(
      optionsCpp->bindingOptions, optionsCpp->inputDialectOptions,
      optionsCpp->executableOptions, optionsCpp->vmTargetOptions,
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
        op, optionsCpp->vmBytecodeTargetOptions, output);
  } else if (auto op = llvm::dyn_cast<iree_compiler::IREE::VM::ModuleOp>(
                 moduleOpCpp)) {
    result = iree_compiler::IREE::VM::translateModuleToBytecode(
        op, optionsCpp->vmBytecodeTargetOptions, output);
  } else {
    emitError(moduleOpCpp->getLoc()) << "expected a supported module operation";
    result = failure();
  }

  return wrap(result);
}
