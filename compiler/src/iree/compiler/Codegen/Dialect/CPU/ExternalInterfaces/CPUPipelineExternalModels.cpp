// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/ExternalInterfaces/CPUPipelineExternalModels.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"

namespace mlir::iree_compiler::IREE::CPU {

static CPUPipelineBuilder &getCPUPipelineBuilder() {
  static CPUPipelineBuilder builder = nullptr;
  return builder;
}

void registerCPUPipelineBuilder(CPUPipelineBuilder builder) {
  CPUPipelineBuilder &current = getCPUPipelineBuilder();
  assert((!current || current == builder) &&
         "CPU pipeline builder already registered with a different callback");
  current = builder;
}

namespace {

struct CPUPipelineExternalModel final
    : Codegen::PipelineAttrInterface::ExternalModel<CPUPipelineExternalModel,
                                                    PipelineAttr> {
  LogicalResult buildPipeline(Attribute attr, OpPassManager &pm,
                              const CodegenPipelineOptions *options) const {
    CPUPipelineBuilder builder = getCPUPipelineBuilder();
    assert(builder && "no CPU pipeline builder registered; ensure "
                      "registerCodegenLLVMCPUPasses() was called");
    return builder(attr, pm, options);
  }
};

} // namespace

void registerCPUPipelineExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, IREECPUDialect *dialect) {
    PipelineAttr::attachInterface<CPUPipelineExternalModel>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::CPU
