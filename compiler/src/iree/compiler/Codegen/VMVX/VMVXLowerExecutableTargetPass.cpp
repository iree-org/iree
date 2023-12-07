// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/VMVX/PassDetail.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler {

namespace {

/// Lowers an hal.executable.variant operation to scalar/native-vector code.
class VMVXLowerExecutableTargetPass
    : public VMVXLowerExecutableTargetBase<VMVXLowerExecutableTargetPass> {
public:
  VMVXLowerExecutableTargetPass() = default;
  VMVXLowerExecutableTargetPass(const VMVXLowerExecutableTargetPass &pass) {}
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
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();

  std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo =
      getIdenticalTranslationInfo(variantOp);
  if (!translationInfo) {
    variantOp.emitOpError(
        "unhandled compilation of entry point functions with different "
        "translation info");
    return signalPassFailure();
  }

  OpPassManager pipeline(IREE::HAL::ExecutableVariantOp::getOperationName());
  if (translationInfo.has_value()) {
    auto target = variantOp.getTarget();
    bool enableUKernels = hasUkernel(target);
    switch (translationInfo.value().getDispatchLoweringPassPipeline()) {
    // No pipleline specified, nothing to do.
    case IREE::Codegen::DispatchLoweringPassPipeline::None:
      return;
    case IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault:
      addVMVXDefaultPassPipeline(pipeline, enableUKernels);
      break;
    default:
      variantOp.emitOpError("Unsupported pipeline on VMVX target.");
      return signalPassFailure();
    }
  }

  if (failed(runPipeline(pipeline, variantOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createVMVXLowerExecutableTargetPass() {
  return std::make_unique<VMVXLowerExecutableTargetPass>();
}

} // namespace mlir::iree_compiler
