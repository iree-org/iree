// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/ROCDL/KernelConfig.h"
#include "iree/compiler/Codegen/ROCDL/PassDetail.h"
#include "iree/compiler/Codegen/ROCDL/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

namespace {
/// Selects a strategy for lowering an IREE hal.executable.variant to ROCDL.
class ROCDLSelectLoweringStrategyPass
    : public ROCDLSelectLoweringStrategyBase<ROCDLSelectLoweringStrategyPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp moduleOp = variantOp.getInnerModule();

    if (failed(initROCDLLaunchConfig(moduleOp))) {
      return signalPassFailure();
    }

    std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo =
        getIdenticalTranslationInfo(variantOp);
    if (!translationInfo) {
      moduleOp.emitError(
          "unsupported entry point functions with different translation info");
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createROCDLSelectLoweringStrategyPass() {
  return std::make_unique<ROCDLSelectLoweringStrategyPass>();
}

} // namespace mlir::iree_compiler
