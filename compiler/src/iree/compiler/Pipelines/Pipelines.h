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
};

// Builds a pass pipeline to perform end-to-end compilation from a
// supported MLIR-based input to the IREE vm dialect.
void buildIREEVMTransformPassPipeline(
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    HighLevelOptimizationOptions highLevelOptimizationOptions,
    SchedulingOptions schedulingOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions, IREEVMPipelineHooks &hooks,
    OpPassManager &passManager);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_PIPELINES_PIPELINES_H_
