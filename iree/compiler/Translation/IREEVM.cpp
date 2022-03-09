// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Translation/IREEVM.h"

#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"
#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#ifdef IREE_HAVE_EMITC_DIALECT
#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/C/TranslationFlags.h"
#endif  // IREE_HAVE_EMITC_DIALECT

namespace mlir {
namespace iree_compiler {

void BindingOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory bindingOptionsCategory(
      "IREE translation binding support options.");

  binder.opt<bool>(
      "iree-native-bindings-support", native,
      llvm::cl::desc(
          "Include runtime support for native IREE ABI-compatible bindings."),
      llvm::cl::cat(bindingOptionsCategory));
  binder.opt<bool>("iree-tflite-bindings-support", tflite,
                   llvm::cl::desc("Include runtime support for the IREE TFLite "
                                  "compatibility bindings."),
                   llvm::cl::cat(bindingOptionsCategory));
}

void InputDialectOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory inputDialectOptions(
      "IREE options for controlling the input transformations to apply.");

  binder.opt<InputDialectOptions::Type>(
      "iree-input-type", type,
      llvm::cl::desc("Specifies the input program representation."),
      llvm::cl::values(
          clEnumValN(InputDialectOptions::Type::none, "none",
                     "No input dialect transformation."),
          clEnumValN(InputDialectOptions::Type::tosa, "tosa",
                     "Legalize from TOSA ops."),
          clEnumValN(InputDialectOptions::Type::mhlo, "mhlo",
                     "Legalize from MHLO ops."),
          clEnumValN(
              InputDialectOptions::Type::xla, "xla",
              "Legalize from MHLO ops (with XLA cleanup preprocessing).")),
      llvm::cl::cat(inputDialectOptions));
}

void HighLevelOptimizationOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE options for controlling high level optimizations.");

  binder.opt<bool>(
      "iree-opt-const-eval", constEval,
      llvm::cl::desc("Enables eager evaluation of constants using the full "
                     "compiler and runtime."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-opt-const-expr-hoisting", constExprHoisting,
      llvm::cl::desc(
          "Hoists the results of latent constant expressions into immutable "
          "global initializers for evaluation at program load."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-opt-numeric-precision-reduction", numericPrecisionReduction,
      llvm::cl::desc(
          "Reduces numeric precision to lower bit depths where possible."),
      llvm::cl::cat(category));
  binder.opt<bool>("iree-opt-strip-assertions", stripAssertions,
                   llvm::cl::desc("Strips debug assertions after any useful "
                                  "information has been extracted."),
                   llvm::cl::cat(category));
}

void SchedulingOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE options for controlling host/device scheduling.");

  binder.opt<DumpOutputFormat>(
      "iree-scheduling-dump-statistics-format", dumpStatisticsFormat,
      llvm::cl::desc("Dumps statistics in the specified output format."),
      llvm::cl::cat(category),
      llvm::cl::values(
          clEnumValN(DumpOutputFormat::Pretty, "pretty",
                     "Human-readable pretty printed output."),
          clEnumValN(DumpOutputFormat::Verbose, "verbose",
                     "Pretty printed output with additional IR."),
          clEnumValN(DumpOutputFormat::CSV, "csv", "Comma separated values."),
          clEnumValN(DumpOutputFormat::JSON, "json",
                     "JSON output with structures for data exchange")));
  binder.opt<std::string>("iree-scheduling-dump-statistics-file",
                          dumpStatisticsFile,
                          llvm::cl::desc("File path to write statistics to; or "
                                         "`` for stderr or `-` for stdout."),
                          llvm::cl::cat(category));
}

void buildIREEVMTransformPassPipeline(
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    HighLevelOptimizationOptions highLevelOptimizationOptions,
    SchedulingOptions schedulingOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions, OpPassManager &passManager) {
  // Input pipelines can result in changes to the exported functions and types
  // and must run before generating bindings.
  // After input processing, there should only be IREE legal types in
  // signatures.
  switch (inputOptions.type) {
    case InputDialectOptions::Type::none:
      break;
    case InputDialectOptions::Type::tosa:
      buildTOSAInputConversionPassPipeline(passManager);
      break;
    case InputDialectOptions::Type::mhlo:
      MHLO::buildMHLOInputConversionPassPipeline(passManager);
      break;
    case InputDialectOptions::Type::xla:
      MHLO::buildXLACleanupPassPipeline(passManager);
      MHLO::buildMHLOInputConversionPassPipeline(passManager);
      break;
  }
  buildCommonInputConversionPassPipeline(passManager);

  // Now that inputs are legalized, generate wrapper for entry functions.
  if (bindingOptions.native) {
    IREE::ABI::buildTransformPassPipeline(passManager);
  }
  if (bindingOptions.tflite) {
    IREE::TFLite::buildTransformPassPipeline(passManager);
  }

  IREE::Flow::TransformOptions flowOptions;
  flowOptions.constExprHoisting =
      highLevelOptimizationOptions.constExprHoisting;
  if (highLevelOptimizationOptions.constEval) {
    flowOptions.buildConstEvalPassPipeline = [](OpPassManager &passManager) {
      passManager.addPass(ConstEval::createJitGlobalsPass());
    };
  }
  flowOptions.numericPrecisionReduction =
      highLevelOptimizationOptions.numericPrecisionReduction;

  if (highLevelOptimizationOptions.stripAssertions) {
    // Strip std.assert & co after we perform optimizations; prior to this we
    // may use the assertions to derive information during analysis.
    passManager.addPass(IREE::Util::createStripDebugOpsPass());
  }

  IREE::Stream::TransformOptions streamOptions;
  // TODO(benvanik): find a way to share the enums w/o circular deps.
  streamOptions.dumpStatisticsFormat =
      (IREE::Stream::DumpOutputFormat)schedulingOptions.dumpStatisticsFormat;
  streamOptions.dumpStatisticsFile = schedulingOptions.dumpStatisticsFile;

  IREE::Flow::buildFlowTransformPassPipeline(passManager, flowOptions);
  IREE::Stream::buildStreamTransformPassPipeline(passManager, streamOptions);
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
  passManager.addPass(IREE::Util::createDropCompilerHintsPass());
}

void buildDefaultIREEVMTransformPassPipeline(OpPassManager &passManager) {
  buildIREEVMTransformPassPipeline(
      BindingOptions::FromFlags::get(), InputDialectOptions::FromFlags::get(),
      HighLevelOptimizationOptions::FromFlags::get(),
      SchedulingOptions::FromFlags::get(),
      IREE::HAL::TargetOptions::FromFlags::get(),
      IREE::VM::TargetOptions::FromFlags::get(), passManager);
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
  buildIREEVMTransformPassPipeline(
      bindingOptions, inputOptions, highLevelOptimizationOptions,
      schedulingOptions, executableOptions, targetOptions, passManager);
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
