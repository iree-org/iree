// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PIPELINES_PIPELINES_H_
#define IREE_COMPILER_PIPELINES_PIPELINES_H_

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Pipelines/Options.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

class PipelineExtensions;

// Hooks for injecting behavior into the IREEVM pipeline. Since these are not
// derived from CLI options, we maintain them as a separate struct.
struct IREEVMPipelineHooks {
  // If the HighLevelOptimizationOptions::constEval option is true, then
  // this callback must be set to populate a pass manager to perform
  // constant eval. It typically just adds a ConstEval::createJitGlobalsPass()
  // pass. It must be injected like this to avoid circular dependencies from
  // the constant evaluator, which needs to recursively invoke these
  // pipelines.
  std::function<void(OpPassManager &)> buildConstEvalPassPipelineCallback;

  // Applies pipeline extensions to the built pipeline if not nullptr.
  PipelineExtensions *pipelineExtensions = nullptr;
};

enum class IREEVMPipelinePhase {
  Input,
  ABI,
  Preprocessing,
  Flow,
  Stream,
  ExecutableSources,
  ExecutableTargets,
  HAL,
  VM,
  End,
};

// Enumerates names and descriptions for pipeline phase values.
inline static void enumerateIREEVMPipelinePhases(
    std::function<void(IREEVMPipelinePhase, StringRef name, StringRef desc)>
        callback) {
  callback(IREEVMPipelinePhase::Input, "input",
           "Performs input processing and lowering into core IREE "
           "input dialects (linalg/etc).");
  callback(IREEVMPipelinePhase::ABI, "abi",
           "Adjusts program ABI for the specified execution environment.");
  callback(IREEVMPipelinePhase::Preprocessing, "preprocessing",
           "Compiles up to the `preprocessing` specified");
  callback(IREEVMPipelinePhase::Flow, "flow",
           "Compiles up to the `flow` dialect.");
  callback(IREEVMPipelinePhase::Stream, "stream",
           "Compiles up to the `stream` dialect.");
  callback(IREEVMPipelinePhase::ExecutableSources, "executable-sources",
           "Compiles up to just before `hal.executable`s are translated, "
           "excluding codegen.");
  callback(IREEVMPipelinePhase::ExecutableTargets, "executable-targets",
           "Compiles up to translated `hal.executable`s, including codegen.");
  callback(IREEVMPipelinePhase::HAL, "hal",
           "Compiles up to the `hal` dialect, including codegen.");
  callback(IREEVMPipelinePhase::VM, "vm", "Compiles up to the `vm` dialect.");
  callback(IREEVMPipelinePhase::End, "end",
           "Complete the full compilation pipeline.");
}

// Builds a pass pipeline to perform end-to-end compilation from a
// supported MLIR-based input to the IREE vm dialect.
//
// If a |runTo| phase is specified the pipeline will stop and output the full
// IR after the phase completes.
void buildIREEVMTransformPassPipeline(
    const IREE::HAL::TargetBackendRegistry &targetRegistry,
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    PreprocessingOptions preprocessingOptions,
    HighLevelOptimizationOptions highLevelOptimizationOptions,
    SchedulingOptions schedulingOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions, IREEVMPipelineHooks &hooks,
    OpPassManager &passManager,
    IREEVMPipelinePhase compileTo = IREEVMPipelinePhase::End);

// Builds the above with options initialized from flags.
void buildDefaultIREEVMTransformPassPipeline(OpPassManager &passManager);

// Registration hooks.
void registerIREEVMTransformPassPipeline();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_PIPELINES_PIPELINES_H_
