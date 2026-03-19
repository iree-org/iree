// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CPU_EXTERNALINTERFACES_CPUPIPELINEEXTERNALMODELS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CPU_EXTERNALINTERFACES_CPUPIPELINEEXTERNALMODELS_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {
struct CodegenPipelineOptions;
} // namespace mlir::iree_compiler

namespace mlir::iree_compiler::IREE::CPU {

/// Callback type for CPU pipeline builders. Returns success if the pipeline
/// was handled.
using CPUPipelineBuilder =
    LogicalResult (*)(Attribute pipelineAttr, OpPassManager &pm,
                      const CodegenPipelineOptions *options);

/// Registers a CPU pipeline builder callback. Called from the LLVMCPU backend
/// at pass registration time. The callback is invoked by the
/// PipelineAttrInterface external model on CPU::PipelineAttr.
void registerCPUPipelineBuilder(CPUPipelineBuilder builder);

/// Registers the external model attaching PipelineAttrInterface to
/// CPU::PipelineAttr.
void registerCPUPipelineExternalModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler::IREE::CPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_EXTERNALINTERFACES_CPUPIPELINEEXTERNALMODELS_H_
