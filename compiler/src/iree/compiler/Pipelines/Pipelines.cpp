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

namespace mlir::iree_compiler {

static IREE::HAL::PipelinePhase
getHALPipelinePhase(IREEVMPipelinePhase phase,
                    IREE::HAL::PipelinePhase defaultPhase) {
  switch (phase) {
  default:
    return defaultPhase;
  case IREEVMPipelinePhase::ExecutableSources:
    return IREE::HAL::PipelinePhase::ExecutableSources;
  case IREEVMPipelinePhase::ExecutableConfigurations:
    return IREE::HAL::PipelinePhase::ExecutableConfigurations;
  case IREEVMPipelinePhase::ExecutableTargets:
    return IREE::HAL::PipelinePhase::ExecutableTargets;
  }
}

static IREEVMPipelinePhase
getIREEVMPipelinePhase(IREE::HAL::PipelinePhase phase) {
  switch (phase) {
  default:
    return IREEVMPipelinePhase::HAL;
  case IREE::HAL::PipelinePhase::ExecutableSources:
    return IREEVMPipelinePhase::ExecutableSources;
  case IREE::HAL::PipelinePhase::ExecutableConfigurations:
    return IREEVMPipelinePhase::ExecutableConfigurations;
  case IREE::HAL::PipelinePhase::ExecutableTargets:
    return IREEVMPipelinePhase::ExecutableTargets;
  }
}

IREEVMPipelineHooks::operator IREE::HAL::PipelineHooks() const {
  IREE::HAL::PipelineHooks halHooks;

  auto beforePhase = this->beforePhase;
  halHooks.beforePhase = [beforePhase](IREE::HAL::PipelinePhase phase,
                                       OpPassManager &passManager) {
    if (beforePhase)
      beforePhase(getIREEVMPipelinePhase(phase), passManager);
  };

  auto afterPhase = this->afterPhase;
  halHooks.afterPhase = [afterPhase](IREE::HAL::PipelinePhase phase,
                                     OpPassManager &passManager) {
    if (afterPhase)
      afterPhase(getIREEVMPipelinePhase(phase), passManager);
  };

  return halHooks;
}

void buildIREEPrecompileTransformPassPipeline(
    const IREE::HAL::TargetRegistry &targetRegistry,
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    PreprocessingOptions preprocessingOptions,
    GlobalOptimizationOptions globalOptimizationOptions,
    SchedulingOptions schedulingOptions,
    IREE::HAL::TargetOptions halTargetOptions, IREEVMPipelineHooks &hooks,
    OpPassManager &passManager, IREEVMPipelinePhase compileFrom,
    IREEVMPipelinePhase compileTo) {
  // Input pipelines can result in changes to the exported functions and types
  // and must run before generating bindings.
  // After input processing, there should only be IREE legal types in
  // signatures.
  if (compileFrom < IREEVMPipelinePhase::Input) { // late-entry
    auto inputType = inputOptions.parseInputTypeMnemonic();
    IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "Input");
    if (hooks.beforePhase)
      hooks.beforePhase(IREEVMPipelinePhase::Input, passManager);
    if (hooks.pipelineExtensions) {
      hooks.pipelineExtensions->extendInputConversionPreprocessingPassPipeline(
          passManager, inputType);
    }

    switch (inputType) {
    case InputDialectOptions::Type::none:
      break;
    case InputDialectOptions::Type::auto_detect:
      // Run the auto pipeline that chooses from plugins using module contents.
      passManager.addPass(
          InputConversion::createAutoInputConversionPipelinePass(
              hooks.pipelineExtensions));
      break;
    case InputDialectOptions::Type::plugin: {
      // Explicitly use a single plugin.
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
    }

    InputConversion::TransformOptions inputTransformOptions;
    inputTransformOptions.options = inputOptions;

    InputConversion::buildCommonInputConversionPassPipeline(
        passManager, inputTransformOptions);
    if (hooks.afterPhase)
      hooks.afterPhase(IREEVMPipelinePhase::Input, passManager);
    IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Input");
  }
  if (compileTo == IREEVMPipelinePhase::Input)
    return; // early-exit

  // If the user specified a set of target devices we attach them to the module
  // IR so that they are available for all passes that may want to use this
  // information. If trying to compile in a generic mode the user should omit
  // specifying targets.
  IREE::HAL::AssignmentOptions halAssignmentOptions;
  halAssignmentOptions.legacyTargetBackends =
      halTargetOptions.legacyTargetBackends;
  halAssignmentOptions.targetDevices = halTargetOptions.targetDevices;
  halAssignmentOptions.defaultDevice = halTargetOptions.defaultDevice;
  IREE::HAL::buildHALDeviceAssignmentPassPipeline(passManager, targetRegistry,
                                                  halAssignmentOptions);

  // Now that inputs are legalized, generate wrapper for entry functions.
  if (compileFrom < IREEVMPipelinePhase::ABI) { // late-entry
    IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "ABI");
    if (hooks.beforePhase)
      hooks.beforePhase(IREEVMPipelinePhase::ABI, passManager);
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
    if (hooks.afterPhase)
      hooks.afterPhase(IREEVMPipelinePhase::ABI, passManager);
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
      IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "Preprocessing");
      if (hooks.beforePhase)
        hooks.beforePhase(IREEVMPipelinePhase::Preprocessing, passManager);
      Preprocessing::buildPreprocessingPassPipeline(
          passManager, preprocessingOptions, hooks.pipelineExtensions);
      if (hooks.afterPhase)
        hooks.afterPhase(IREEVMPipelinePhase::Preprocessing, passManager);
      IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Preprocessing");
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
      if (hooks.beforePhase)
        hooks.beforePhase(IREEVMPipelinePhase::GlobalOptimization, passManager);
      GlobalOptimization::buildGlobalOptimizationPassPipeline(
          passManager, globalTransformOptions);
      if (hooks.afterPhase)
        hooks.afterPhase(IREEVMPipelinePhase::GlobalOptimization, passManager);
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
    const IREE::HAL::TargetRegistry &targetRegistry,
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    PreprocessingOptions preprocessingOptions,
    GlobalOptimizationOptions globalOptimizationOptions,
    SchedulingOptions schedulingOptions,
    IREE::HAL::TargetOptions halTargetOptions,
    IREE::VM::TargetOptions vmTargetOptions, IREEVMPipelineHooks &hooks,
    OpPassManager &passManager, IREEVMPipelinePhase compileFrom,
    IREEVMPipelinePhase compileTo) {
  buildIREEPrecompileTransformPassPipeline(
      targetRegistry, bindingOptions, inputOptions, preprocessingOptions,
      globalOptimizationOptions, schedulingOptions, halTargetOptions, hooks,
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
      if (hooks.beforePhase)
        hooks.beforePhase(IREEVMPipelinePhase::Flow, passManager);
      IREE::Flow::buildFlowTransformPassPipeline(passManager, flowOptions);
      if (hooks.afterPhase)
        hooks.afterPhase(IREEVMPipelinePhase::Flow, passManager);
      IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Flow");
    }
    if (compileTo == IREEVMPipelinePhase::Flow)
      return; // early-exit

    if (compileFrom < IREEVMPipelinePhase::Stream) { // late-entry
      IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "Stream");
      if (hooks.beforePhase)
        hooks.beforePhase(IREEVMPipelinePhase::Stream, passManager);
      IREE::Stream::buildStreamTransformPassPipeline(passManager,
                                                     streamOptions);
      if (hooks.afterPhase)
        hooks.afterPhase(IREEVMPipelinePhase::Stream, passManager);
      IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Stream");
    }
    if (compileTo == IREEVMPipelinePhase::Stream)
      return; // early-exit
    break;
  }

  IREE::HAL::PipelinePhase halCompileFrom =
      getHALPipelinePhase(compileFrom, IREE::HAL::PipelinePhase::Start);
  IREE::HAL::PipelinePhase halCompileTo =
      getHALPipelinePhase(compileTo, IREE::HAL::PipelinePhase::End);

  if (compileFrom < IREEVMPipelinePhase::HAL) { // late-entry
    IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "HAL");
    if (hooks.beforePhase)
      hooks.beforePhase(IREEVMPipelinePhase::HAL, passManager);
    switch (schedulingOptions.executionModel) {
    case SchedulingOptions::ExecutionModel::HostOnly:
      // No HAL required.
      break;
    default:
    case SchedulingOptions::ExecutionModel::AsyncInternal:
    case SchedulingOptions::ExecutionModel::AsyncExternal:
      IREE::HAL::buildHALTransformPassPipeline(passManager, targetRegistry,
                                               halTargetOptions, hooks,
                                               halCompileFrom, halCompileTo);
      break;
    case SchedulingOptions::ExecutionModel::InlineStatic:
      IREE::HAL::Inline::buildHALInlineStaticTransformPassPipeline(
          passManager, targetRegistry, halTargetOptions);
      break;
    case SchedulingOptions::ExecutionModel::InlineDynamic:
      IREE::HAL::Loader::buildHALInlineDynamicTransformPassPipeline(
          passManager, targetRegistry, halTargetOptions);
      break;
    }
    if (hooks.afterPhase)
      hooks.afterPhase(IREEVMPipelinePhase::HAL, passManager);
    IREE_TRACE_ADD_END_FRAME_PASS(passManager, "HAL");
  }
  if (compileTo == IREEVMPipelinePhase::HAL ||
      halCompileTo != IREE::HAL::PipelinePhase::End) {
    return; // early-exit
  }

  if (compileFrom < IREEVMPipelinePhase::VM) { // late-entry
    IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "VM");
    if (hooks.beforePhase)
      hooks.beforePhase(IREEVMPipelinePhase::VM, passManager);
    IREE::VM::buildVMTransformPassPipeline(passManager, vmTargetOptions);
    passManager.addPass(IREE::Util::createDropCompilerHintsPass());
    if (hooks.afterPhase)
      hooks.afterPhase(IREEVMPipelinePhase::VM, passManager);
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
      IREE::HAL::TargetRegistry::getGlobal(), BindingOptions::FromFlags::get(),
      InputDialectOptions::FromFlags::get(),
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

} // namespace mlir::iree_compiler
