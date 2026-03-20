// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_SPIRVPIPELINEEXTERNALMODELS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_SPIRVPIPELINEEXTERNALMODELS_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {
struct CodegenPipelineOptions;
} // namespace mlir::iree_compiler

namespace mlir::iree_compiler::IREE::GPU {

/// Callback type for SPIRV pipeline builders. Returns success if the pipeline
/// was handled.
using SPIRVPipelineBuilder =
    LogicalResult (*)(Attribute pipelineAttr, OpPassManager &pm,
                      const CodegenPipelineOptions *options);

/// Registers a SPIRV pipeline builder callback. Called from the SPIRV backend
/// at pass registration time. The callback is invoked by the
/// PipelineAttrInterface external model on GPU::SPIRVPipelineAttr.
void registerSPIRVPipelineBuilder(SPIRVPipelineBuilder builder);

/// Registers the external model attaching PipelineAttrInterface to
/// GPU::SPIRVPipelineAttr.
void registerSPIRVPipelineExternalModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_SPIRVPIPELINEEXTERNALMODELS_H_
