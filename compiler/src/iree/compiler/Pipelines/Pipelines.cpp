// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Pipelines/Pipelines.h"

#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/Modules/HAL/Inline/Transforms/Passes.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h"
#include "iree/compiler/Preprocessing/Passes.h"
#include "iree/compiler/Utils/TracingUtils.h"

#ifdef IREE_HAVE_STABLEHLO_INPUT
#include "iree/compiler/InputConversion/StableHLO/Passes.h"
#endif // IREE_HAVE_STABLEHLO_INPUT
#ifdef IREE_HAVE_TOSA_INPUT
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#endif // IREE_HAVE_TOSA_INPUT

namespace mlir {
namespace iree_compiler {

void buildIREEPrecompileTransformPassPipeline(
    const IREE::HAL::TargetBackendRegistry &targetRegistry,
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    PreprocessingOptions preprocessingOptions,
    GlobalOptimizationOptions globalOptimizationOptions,
    SchedulingOptions schedulingOptions,
    IREE::HAL::TargetOptions executableOptions, IREEVMPipelineHooks &hooks,
    OpPassManager &passManager, IREEVMPipelinePhase compileFrom,
    IREEVMPipelinePhase compileTo) {
  // If the user specified a set of target devices we attach them to the module
  // IR so that they are available for all passes that may want to use this
  // information. If trying to compile in a generic mode the user should omit
  // specifying targets.
  if (!executableOptions.targets.empty()) {
    passManager.addPass(IREE::HAL::createAssignTargetDevicesPass(
        targetRegistry, executableOptions.targets));
  }

  // Input pipelines can result in changes to the exported functions and types
  // and must run before generating bindings.
  // After input processing, there should only be IREE legal types in
  // signatures.
  if (compileFrom < IREEVMPipelinePhase::Input) { // late-entry
    auto inputType = inputOptions.parseInputTypeMnemonic();
    IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "Input");
    if (hooks.pipelineExtensions) {
      hooks.pipelineExtensions->extendInputConversionPreprocessingPassPipeline(
          passManager, inputType);
    }
    AutoInputConversionPipelineOptions autoOptions;
    autoOptions.demoteI64ToI32 = inputOptions.demoteI64ToI32;
    autoOptions.demoteF64ToF32 = inputOptions.demoteF64ToF32;
    autoOptions.promoteBF16ToF32 = inputOptions.promoteBF16ToF32;

#ifdef IREE_HAVE_STABLEHLO_INPUT
    stablehlo::StableHloOptions stablehloOptions;
    stablehloOptions.demoteI64ToI32 = inputOptions.demoteI64ToI32;
    stablehloOptions.demoteF64ToF32 = inputOptions.demoteF64ToF32;
    stablehloOptions.promoteBF16ToF32 = inputOptions.promoteBF16ToF32;
#endif // IREE_HAVE_STABLEHLO_INPUT

    switch (inputType) {
    case InputDialectOptions::Type::none:
      break;
    case InputDialectOptions::Type::auto_detect:
      passManager.addPass(createAutoInputConversionPipelinePass(
          autoOptions, hooks.pipelineExtensions));
      break;
    case InputDialectOptions::Type::plugin: {
      bool foundExtension = false;
      if (hooks.pipelineExtensions) {
        foundExtension =
            hooks.pipelineExtensions->extendCustomInputConversionPassPipeline(
                passManager, inputOptions.inputTypeMnemonic);
      }
      // We expect that callers properly validate supported extensions and that
      // if a plugin advertises support, it actually provides it.
      assert(foundExtension &&
             "InputDialect::type::plugin extension not found");
      if (!foundExtension) {
        llvm::errs() << "internal error: InputDialect::type::plugin extension "
                        "not found ("
                     << inputOptions.inputTypeMnemonic << ")\n";
      }
      break;
    }
#ifdef IREE_HAVE_STABLEHLO_INPUT
    case InputDialectOptions::Type::stablehlo:
      stablehlo::buildStableHLOInputConversionPassPipeline(passManager,
                                                           stablehloOptions);
      break;
    case InputDialectOptions::Type::stablehlo_xla:
      stablehlo::buildStableHLOXLAInputConversionPassPipeline(passManager,
                                                              stablehloOptions);
      break;
#endif // IREE_HAVE_STABLEHLO_INPUT
#ifdef IREE_HAVE_TOSA_INPUT
    case InputDialectOptions::Type::tosa:
      buildTOSAInputConversionPassPipeline(passManager);
      break;
#endif // IREE_HAVE_TOSA_INPUT
    }
    buildCommonInputConversionPassPipeline(passManager);
    IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Input");
  }
  if (compileTo == IREEVMPipelinePhase::Input)
    return; // early-exit

  // Now that inputs are legalized, generate wrapper for entry functions.
  if (compileFrom < IREEVMPipelinePhase::ABI) { // late-entry
    IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "ABI");
    IREE::ABI::InvocationOptions invocationOptions;
    invocationOptions.invocationModel =
        schedulingOptions.executionModel ==
                SchedulingOptions::ExecutionModel::AsyncExternal
            ? IREE::ABI::InvocationModel::CoarseFences
            : IREE::ABI::InvocationModel::Sync;
    if (bindingOptions.native) {
      IREE::ABI::buildTransformPassPipeline(passManager, invocationOptions);
    }
    if (bindingOptions.tflite) {
      IREE::TFLite::buildTransformPassPipeline(passManager);
    }
    IREE_TRACE_ADD_END_FRAME_PASS(passManager, "ABI");
  }
  if (compileTo == IREEVMPipelinePhase::ABI)
    return; // early-exit

  GlobalOptimization::TransformOptions globalTransformOptions;
  globalTransformOptions.options = globalOptimizationOptions;

  // Enable const-eval via hook. For debug builds, we assert if enabled
  // without a hook. For release, we just silently skip enabling const-eval.
  if (globalOptimizationOptions.constEval) {
    assert(hooks.buildConstEvalPassPipelineCallback &&
           "if const-eval is enabled the buildConstEvalPassPipelineCallback "
           "hook must be enabled");
  }
  if (globalOptimizationOptions.constEval &&
      hooks.buildConstEvalPassPipelineCallback) {
    globalTransformOptions.buildConstEvalPassPipeline =
        hooks.buildConstEvalPassPipelineCallback;
  }

  switch (schedulingOptions.executionModel) {
  case SchedulingOptions::ExecutionModel::HostOnly:
    // No flow/stream processing (implies no tensors).
    break;
  default:
    if (compileFrom < IREEVMPipelinePhase::Preprocessing) { // late-entry.
      // Not a large enough phase for IREE_TRACE_ADD_[BEGIN,END]_FRAME_PASS.
      IREE::buildPreprocessingPassPipeline(passManager, preprocessingOptions,
                                           hooks.pipelineExtensions);
    }
    if (compileTo == IREEVMPipelinePhase::Preprocessing)
      return; // early-exit

    if (compileFrom < IREEVMPipelinePhase::GlobalOptimization) { // late-entry
      // This pass pipeline recursively invokes the compiler if constEval is
      // enabled. In that case, we have to be careful to not emit unbalanced
      // trace frames:
      //   begin 'GlobalOptimization'
      //   begin 'Input'
      //   end   'Input'
      //   begin 'GlobalOptimization' <-- unbalanced! Use a different name.
      //   end   'GlobalOptimization'
      //   ...
      //   end   'GlobalOptimization'
      if (globalOptimizationOptions.constEval) {
        IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "GlobalOptimizationConst");
      } else {
        IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "GlobalOptimization");
      }
      GlobalOptimization::buildGlobalOptimizationPassPipeline(
          passManager, globalTransformOptions);
      if (globalOptimizationOptions.constEval) {
        IREE_TRACE_ADD_END_FRAME_PASS(passManager, "GlobalOptimizationConst");
      } else {
        IREE_TRACE_ADD_END_FRAME_PASS(passManager, "GlobalOptimization");
      }
    }
    if (compileTo == IREEVMPipelinePhase::GlobalOptimization)
      return; // early-exit

    break;
  }
}

void buildIREEVMTransformPassPipeline(
    const IREE::HAL::TargetBackendRegistry &targetRegistry,
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    PreprocessingOptions preprocessingOptions,
    GlobalOptimizationOptions globalOptimizationOptions,
    SchedulingOptions schedulingOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions, IREEVMPipelineHooks &hooks,
    OpPassManager &passManager, IREEVMPipelinePhase compileFrom,
    IREEVMPipelinePhase compileTo) {

  buildIREEPrecompileTransformPassPipeline(
      targetRegistry, bindingOptions, inputOptions, preprocessingOptions,
      globalOptimizationOptions, schedulingOptions, executableOptions, hooks,
      passManager, compileFrom, compileTo);

  if (compileTo <= IREEVMPipelinePhase::GlobalOptimization)
    return; // early-exit

  IREE::Stream::TransformOptions streamOptions;
  // TODO(benvanik): find a way to share the enums w/o circular deps.
  streamOptions.dumpStatisticsFormat =
      (IREE::Stream::DumpOutputFormat)schedulingOptions.dumpStatisticsFormat;
  streamOptions.dumpStatisticsFile = schedulingOptions.dumpStatisticsFile;
  streamOptions.optimizeBindings = schedulingOptions.optimizeBindings;

  switch (schedulingOptions.executionModel) {
  case SchedulingOptions::ExecutionModel::HostOnly:
    // No flow/stream processing (implies no tensors).
    break;
  default:
    IREE::Flow::TransformOptions flowOptions;
    if (compileFrom < IREEVMPipelinePhase::Flow) { // late-entry
      IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "Flow");
      IREE::Flow::buildFlowTransformPassPipeline(passManager, flowOptions);
      IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Flow");
    }
    if (compileTo == IREEVMPipelinePhase::Flow)
      return; // early-exit

    if (compileFrom < IREEVMPipelinePhase::Stream) { // late-entry
      IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "Stream");
      IREE::Stream::buildStreamTransformPassPipeline(passManager,
                                                     streamOptions);
      IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Stream");
    }
    if (compileTo == IREEVMPipelinePhase::Stream)
      return; // early-exit
    break;
  }

  IREE::HAL::PipelinePhase halCompileFrom;
  switch (compileFrom) {
  default:
    halCompileFrom = IREE::HAL::PipelinePhase::Start;
    break;
  case IREEVMPipelinePhase::ExecutableSources:
    halCompileFrom = IREE::HAL::PipelinePhase::ExecutableSources;
    break;
  case IREEVMPipelinePhase::ExecutableTargets:
    halCompileFrom = IREE::HAL::PipelinePhase::ExecutableTargets;
    break;
  }

  IREE::HAL::PipelinePhase halCompileTo;
  switch (compileTo) {
  default:
    halCompileTo = IREE::HAL::PipelinePhase::End;
    break;
  case IREEVMPipelinePhase::ExecutableSources:
    halCompileTo = IREE::HAL::PipelinePhase::ExecutableSources;
    break;
  case IREEVMPipelinePhase::ExecutableTargets:
    halCompileTo = IREE::HAL::PipelinePhase::ExecutableTargets;
    break;
  }

  if (compileFrom < IREEVMPipelinePhase::HAL) { // late-entry
    IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "HAL");
    switch (schedulingOptions.executionModel) {
    case SchedulingOptions::ExecutionModel::HostOnly:
      // No HAL required.
      break;
    default:
    case SchedulingOptions::ExecutionModel::AsyncInternal:
    case SchedulingOptions::ExecutionModel::AsyncExternal:
      IREE::HAL::buildHALTransformPassPipeline(passManager, targetRegistry,
                                               executableOptions,
                                               halCompileFrom, halCompileTo);
      break;
    case SchedulingOptions::ExecutionModel::InlineStatic:
      IREE::HAL::Inline::buildHALInlineStaticTransformPassPipeline(
          passManager, targetRegistry, executableOptions);
      break;
    case SchedulingOptions::ExecutionModel::InlineDynamic:
      IREE::HAL::Loader::buildHALInlineDynamicTransformPassPipeline(
          passManager, targetRegistry, executableOptions);
      break;
    }
    IREE_TRACE_ADD_END_FRAME_PASS(passManager, "HAL");
  }
  if (compileTo == IREEVMPipelinePhase::HAL ||
      halCompileTo != IREE::HAL::PipelinePhase::End) {
    return; // early-exit
  }

  if (compileFrom < IREEVMPipelinePhase::VM) { // late-entry
    IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "VM");
    IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
    passManager.addPass(IREE::Util::createDropCompilerHintsPass());
    IREE_TRACE_ADD_END_FRAME_PASS(passManager, "VM");
  }
  if (compileTo == IREEVMPipelinePhase::VM)
    return; // early-exit
}

void buildDefaultIREEVMTransformPassPipeline(OpPassManager &passManager) {
  // Note that the production compiler will provide hooks here that enable
  // additional, whole-program related features, whereas this pipeline will
  // only use the defaults. In practice, this means that things like const
  // jitting are not supported by this pipeline (it would create a circular
  // dependency at this point in the compiler to also depend on the compiler).
  static IREEVMPipelineHooks defaultHooks;

  // Since a JIT hook cannot be provided in such a default pipeline, we
  // force disable const eval, which relies on the JIT.
  auto highLevelOptimizations = GlobalOptimizationOptions::FromFlags::get();
  highLevelOptimizations.constEval = false;

  buildIREEVMTransformPassPipeline(
      IREE::HAL::TargetBackendRegistry::getGlobal(),
      BindingOptions::FromFlags::get(), InputDialectOptions::FromFlags::get(),
      PreprocessingOptions::FromFlags::get(), highLevelOptimizations,
      SchedulingOptions::FromFlags::get(),
      IREE::HAL::TargetOptions::FromFlags::get(),
      IREE::VM::TargetOptions::FromFlags::get(), defaultHooks, passManager);
}

void registerIREEVMTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-transformation-pipeline",
      "Runs the full IREE input to VM transformation pipeline",
      [](OpPassManager &passManager) {
        buildDefaultIREEVMTransformPassPipeline(passManager);
      });
}

} // namespace iree_compiler
} // namespace mlir
