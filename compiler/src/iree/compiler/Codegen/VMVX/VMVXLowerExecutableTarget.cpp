// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "iree-vmvx-lower-executable-target"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VMVXLOWEREXECUTABLETARGETPASS
#include "iree/compiler/Codegen/VMVX/Passes.h.inc"

namespace {

/// Lowers an hal.executable.variant operation to scalar/native-vector code.
class VMVXLowerExecutableTargetPass final
    : public impl::VMVXLowerExecutableTargetPassBase<
          VMVXLowerExecutableTargetPass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<IREE::HAL::HALDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    bufferization::BufferizationDialect,
                    linalg::LinalgDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  void runOnOperation() override;
};
} // namespace

void VMVXLowerExecutableTargetPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();

  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo) {
    return;
  }

  std::optional<OpPassManager> maybePipeline =
      getFunctionOpInterfacePassManager(funcOp);
  if (!maybePipeline) {
    funcOp.emitOpError(
        "unhandled function-like container during executable lowering");
    return signalPassFailure();
  }
  OpPassManager &pipeline = maybePipeline.value();

  Attribute pipelineAttr = translationInfo.getPassPipeline();
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  VMVXCodegenPipelineOptions vmvxOptions(
      /*enableUKernels=*/target && hasUkernel(target.getConfiguration()));

  // No pipeline specified, nothing to do.
  if (isa<IREE::Codegen::NoPipelineAttr>(pipelineAttr)) {
    return;
  }

  // Check for a pipeline via PipelineAttrInterface.
  if (auto pipelineIface =
          dyn_cast<IREE::Codegen::PipelineAttrInterface>(pipelineAttr)) {
    if (failed(pipelineIface.buildPipeline(pipeline, &vmvxOptions))) {
      funcOp.emitOpError("failed to build pass pipeline");
      return signalPassFailure();
    }
  } else {
    // Not an interface implementor -- reject any remaining legacy pipeline.
    funcOp.emitOpError("Unsupported pipeline on VMVX target.");
    return signalPassFailure();
  }

  LDBG() << "Using pass pipeline: ";
  LLVM_DEBUG(pipeline.dump());
  if (failed(runPipeline(pipeline, funcOp))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler
