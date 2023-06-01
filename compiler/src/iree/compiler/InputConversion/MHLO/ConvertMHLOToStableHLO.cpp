// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "mhlo/transforms/passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::MHLO {
namespace {
struct ConvertMHLOToStableHLOPass final
    : ConvertMHLOToStableHLOPassBase<ConvertMHLOToStableHLOPass> {
  void runOnOperation() override {
    OpPassManager pm(ModuleOp::getOperationName(),
                     OpPassManager::Nesting::Explicit);
    pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::stablehlo::StablehloDialect>();
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertMHLOToStableHLOPass() {
  return std::make_unique<ConvertMHLOToStableHLOPass>();
}
}  // namespace mlir::iree_compiler::MHLO
