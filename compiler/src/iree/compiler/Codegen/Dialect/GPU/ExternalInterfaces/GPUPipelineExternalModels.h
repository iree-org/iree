// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_GPUPIPELINEEXTERNALMODELS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_GPUPIPELINEEXTERNALMODELS_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {
struct CodegenPipelineOptions;
} // namespace mlir::iree_compiler

namespace mlir::iree_compiler::IREE::GPU {

/// Callback type for GPU pipeline builders. Returns success if the pipeline
/// was handled.
using GPUPipelineBuilder =
    LogicalResult (*)(Attribute pipelineAttr, OpPassManager &pm,
                      const CodegenPipelineOptions *options);

/// Registers a GPU pipeline builder callback. Called from the LLVMGPU backend
/// at pass registration time. The callback is invoked by the
/// PipelineAttrInterface external model on GPU::PipelineAttr.
void registerGPUPipelineBuilder(GPUPipelineBuilder builder);

/// Registers the external model attaching PipelineAttrInterface to
/// GPU::PipelineAttr.
void registerGPUPipelineExternalModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_GPUPIPELINEEXTERNALMODELS_H_
