// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/ExternalInterfaces/SPIRVPipelineExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"

namespace mlir::iree_compiler::IREE::GPU {

static SPIRVPipelineBuilder &getSPIRVPipelineBuilder() {
  static SPIRVPipelineBuilder builder = nullptr;
  return builder;
}

void registerSPIRVPipelineBuilder(SPIRVPipelineBuilder builder) {
  SPIRVPipelineBuilder &current = getSPIRVPipelineBuilder();
  assert((!current || current == builder) &&
         "SPIRV pipeline builder already registered with a different callback");
  current = builder;
}

namespace {

struct SPIRVPipelineExternalModel final
    : Codegen::PipelineAttrInterface::ExternalModel<SPIRVPipelineExternalModel,
                                                    SPIRVPipelineAttr> {
  LogicalResult buildPipeline(Attribute attr, OpPassManager &pm,
                              const CodegenPipelineOptions *options) const {
    SPIRVPipelineBuilder builder = getSPIRVPipelineBuilder();
    assert(builder && "no SPIRV pipeline builder registered; ensure "
                      "registerCodegenSPIRVPasses() was called");
    return builder(attr, pm, options);
  }
};

} // namespace

void registerSPIRVPipelineExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, IREEGPUDialect *dialect) {
    SPIRVPipelineAttr::attachInterface<SPIRVPipelineExternalModel>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::GPU
