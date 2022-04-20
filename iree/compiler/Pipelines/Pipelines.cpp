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
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#ifdef IREE_HAVE_TORCH_MLIR_DIALECTS
#include "iree/compiler/InputConversion/TMTensor/Passes.h"
#endif
#include "iree/compiler/InputConversion/TOSA/Passes.h"

namespace mlir {
namespace iree_compiler {

void buildIREEVMTransformPassPipeline(
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    HighLevelOptimizationOptions highLevelOptimizationOptions,
    SchedulingOptions schedulingOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions, IREEVMPipelineHooks &hooks,
    OpPassManager &passManager) {
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
#ifdef IREE_HAVE_TORCH_MLIR_DIALECTS
    case InputDialectOptions::Type::tm_tensor:
      passManager.addNestedPass<func::FuncOp>(
          TMTensor::createConvertTMTensorToLinalgExtPass());
      break;
#endif
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

  IREE::Flow::buildFlowTransformPassPipeline(passManager, flowOptions);
  IREE::Stream::buildStreamTransformPassPipeline(passManager, streamOptions);
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
  passManager.addPass(IREE::Util::createDropCompilerHintsPass());
}

}  // namespace iree_compiler
}  // namespace mlir
