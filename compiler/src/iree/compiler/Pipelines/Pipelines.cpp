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
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/Modules/HAL/Inline/Transforms/Passes.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h"
#include "iree/compiler/Preprocessing/Passes.h"
#include "iree/compiler/Utils/TracingUtils.h"

#ifdef IREE_HAVE_MHLO_INPUT
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#endif  // IREE_HAVE_MHLO_INPUT
#ifdef IREE_HAVE_TORCH_INPUT
#include "iree/compiler/InputConversion/TMTensor/Passes.h"
#endif  // IREE_HAVE_TORCH_INPUT
#ifdef IREE_HAVE_TOSA_INPUT
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#endif  // IREE_HAVE_TOSA_INPUT

namespace mlir {
namespace iree_compiler {

void buildIREEVMTransformPassPipeline(
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    PreprocessingOptions preprocessingOptions,
    HighLevelOptimizationOptions highLevelOptimizationOptions,
    SchedulingOptions schedulingOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions, IREEVMPipelineHooks &hooks,
    OpPassManager &passManager, IREEVMPipelinePhase compileTo) {
  // Input pipelines can result in changes to the exported functions and types
  // and must run before generating bindings.
  // After input processing, there should only be IREE legal types in
  // signatures.
  IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "Input");
  switch (inputOptions.type) {
    case InputDialectOptions::Type::none:
      break;
#ifdef IREE_HAVE_MHLO_INPUT
    case InputDialectOptions::Type::mhlo:
      MHLO::buildMHLOInputConversionPassPipeline(passManager);
      break;
    case InputDialectOptions::Type::xla:
      MHLO::buildXLACleanupPassPipeline(passManager);
      MHLO::buildMHLOInputConversionPassPipeline(passManager);
      break;
#endif  // IREE_HAVE_MHLO_INPUT
#ifdef IREE_HAVE_TORCH_INPUT
    case InputDialectOptions::Type::tm_tensor:
      passManager.addNestedPass<func::FuncOp>(
          TMTensor::createConvertTMTensorToLinalgExtPass());
      break;
#endif  // IREE_HAVE_TORCH_INPUT
#ifdef IREE_HAVE_TOSA_INPUT
    case InputDialectOptions::Type::tosa:
      buildTOSAInputConversionPassPipeline(passManager);
      break;
#endif  // IREE_HAVE_TOSA_INPUT
  }
  buildCommonInputConversionPassPipeline(passManager);
  IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Input");
  if (compileTo == IREEVMPipelinePhase::Input) return;  // early-exit

  // Now that inputs are legalized, generate wrapper for entry functions.
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
  if (compileTo == IREEVMPipelinePhase::ABI) return;  // early-exit

  IREE::Flow::TransformOptions flowOptions;
  flowOptions.constExprHoisting =
      highLevelOptimizationOptions.constExprHoisting;
  flowOptions.numericPrecisionReduction =
      highLevelOptimizationOptions.numericPrecisionReduction;

  // Enable const-eval via hook. For debug builds, we assert if enabled without
  // a hook. For release, we just silently skip enabling const-eval.
  if (highLevelOptimizationOptions.constEval) {
    assert(hooks.buildConstEvalPassPipelineCallback &&
           "if const-eval is enabled the buildConstEvalPassPipelineCallback "
           "hook must be enabled");
  }
  if (highLevelOptimizationOptions.constEval &&
      hooks.buildConstEvalPassPipelineCallback) {
    flowOptions.buildConstEvalPassPipeline =
        hooks.buildConstEvalPassPipelineCallback;
  }

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

  switch (schedulingOptions.executionModel) {
    case SchedulingOptions::ExecutionModel::HostOnly:
      // No flow/stream processing (implies no tensors).
      break;
    default:
      IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "Preprocessing");
      IREE::buildPreprocessingPassPipeline(passManager, preprocessingOptions,
                                           hooks.pipelineExtensions);
      IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Preprocessing");
      if (compileTo == IREEVMPipelinePhase::Preprocessing)
        return;  // early-exit

      IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "Flow");
      IREE::Flow::buildFlowTransformPassPipeline(passManager, flowOptions);
      IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Flow");
      if (compileTo == IREEVMPipelinePhase::Flow) return;  // early-exit

      IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "Stream");
      IREE::Stream::buildStreamTransformPassPipeline(passManager,
                                                     streamOptions);
      IREE_TRACE_ADD_END_FRAME_PASS(passManager, "Stream");
      if (compileTo == IREEVMPipelinePhase::Stream) return;  // early-exit
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

  IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "HAL");
  switch (schedulingOptions.executionModel) {
    case SchedulingOptions::ExecutionModel::HostOnly:
      // No HAL required.
      break;
    default:
    case SchedulingOptions::ExecutionModel::AsyncInternal:
    case SchedulingOptions::ExecutionModel::AsyncExternal:
      IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions,
                                               halCompileTo);
      break;
    case SchedulingOptions::ExecutionModel::InlineStatic:
      IREE::HAL::Inline::buildHALInlineStaticTransformPassPipeline(
          passManager, executableOptions);
      break;
    case SchedulingOptions::ExecutionModel::InlineDynamic:
      IREE::HAL::Loader::buildHALInlineDynamicTransformPassPipeline(
          passManager, executableOptions);
      break;
  }
  IREE_TRACE_ADD_END_FRAME_PASS(passManager, "HAL");
  if (compileTo == IREEVMPipelinePhase::HAL ||
      halCompileTo != IREE::HAL::PipelinePhase::End) {
    return;  // early-exit
  }

  IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, "VM");
  IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
  passManager.addPass(IREE::Util::createDropCompilerHintsPass());
  IREE_TRACE_ADD_END_FRAME_PASS(passManager, "VM");
  if (compileTo == IREEVMPipelinePhase::VM) return;  // early-exit
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
  auto highLevelOptimizations = HighLevelOptimizationOptions::FromFlags::get();
  highLevelOptimizations.constEval = false;

  buildIREEVMTransformPassPipeline(
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

}  // namespace iree_compiler
}  // namespace mlir
