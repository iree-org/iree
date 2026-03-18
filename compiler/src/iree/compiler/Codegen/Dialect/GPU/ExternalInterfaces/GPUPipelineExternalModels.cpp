// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/ExternalInterfaces/GPUPipelineExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"

namespace mlir::iree_compiler::IREE::GPU {

/// Global builder callback registered by the LLVMGPU backend.
static GPUPipelineBuilder globalGPUPipelineBuilder = nullptr;

void registerGPUPipelineBuilder(GPUPipelineBuilder builder) {
  assert((!globalGPUPipelineBuilder || globalGPUPipelineBuilder == builder) &&
         "GPU pipeline builder already registered with a different callback");
  globalGPUPipelineBuilder = builder;
}

namespace {

struct GPUPipelineExternalModel final
    : Codegen::PipelineAttrInterface::ExternalModel<GPUPipelineExternalModel,
                                                    PipelineAttr> {
  LogicalResult buildPipeline(Attribute attr, OpPassManager &pm,
                              const CodegenPipelineOptions *options) const {
    assert(globalGPUPipelineBuilder &&
           "no GPU pipeline builder registered; ensure "
           "registerCodegenLLVMGPUPasses() was called");
    return globalGPUPipelineBuilder(attr, pm, options);
  }
};

} // namespace

void registerGPUPipelineExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, IREEGPUDialect *dialect) {
    PipelineAttr::attachInterface<GPUPipelineExternalModel>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::GPU
