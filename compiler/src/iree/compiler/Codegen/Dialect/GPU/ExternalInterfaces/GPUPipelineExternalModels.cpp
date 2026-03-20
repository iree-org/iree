// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/ExternalInterfaces/GPUPipelineExternalModels.h"

#include <mutex>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"

namespace mlir::iree_compiler::IREE::GPU {

static GPUPipelineBuilder &getPipelineBuilder() {
  static GPUPipelineBuilder builder = nullptr;
  return builder;
}

static GPUConstraintEmitter &getConstraintEmitter() {
  static GPUConstraintEmitter emitter = nullptr;
  return emitter;
}

void registerGPUPipelineCallbacks(GPUPipelineBuilder builder,
                                  GPUConstraintEmitter constraintEmitter) {
  static std::once_flag onceFlag;
  std::call_once(onceFlag, [&] {
    if (builder) {
      getPipelineBuilder() = builder;
    }
    if (constraintEmitter) {
      getConstraintEmitter() = constraintEmitter;
    }
  });
}

namespace {

struct GPUPipelineExternalModel final
    : Codegen::PipelineAttrInterface::ExternalModel<GPUPipelineExternalModel,
                                                    PipelineAttr> {
  LogicalResult buildPipeline(Attribute attr, OpPassManager &pm,
                              const CodegenPipelineOptions *options) const {
    GPUPipelineBuilder builder = getPipelineBuilder();
    assert(builder && "no GPU pipeline builder registered; ensure "
                      "registerCodegenLLVMGPUPasses() was called");
    return builder(attr, pm, options);
  }

  LogicalResult emitConstraints(Attribute attr,
                                ArrayRef<Operation *> rootOps) const {
    GPUConstraintEmitter emitter = getConstraintEmitter();
    if (!emitter || rootOps.empty()) {
      return success();
    }
    return emitter(attr, rootOps);
  }
};

} // namespace

void registerGPUPipelineExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, IREEGPUDialect *dialect) {
    PipelineAttr::attachInterface<GPUPipelineExternalModel>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::GPU
