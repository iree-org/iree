// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {
namespace {
struct AutoInputConversionPipelinePass final
    : AutoInputConversionPipelineBase<AutoInputConversionPipelinePass> {
  AutoInputConversionPipelinePass(PipelineExtensions *pipelineExtensions)
      : pipelineExtensions(pipelineExtensions) {}
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;

  PipelineExtensions *pipelineExtensions = nullptr;
};

void AutoInputConversionPipelinePass::runOnOperation() {
  if (!pipelineExtensions)
    return;

  ModuleOp module = getOperation();
  llvm::StringSet<> detectedTypeMnemonics;
  pipelineExtensions->populateDetectedCustomInputConversionTypes(
      module, detectedTypeMnemonics);
  if (detectedTypeMnemonics.empty())
    return;

  if (detectedTypeMnemonics.getNumItems() > 1) {
    // TODO(scotttodd): handle multiple typeMnemonics (use all?)
    auto diag = module.emitError(
        "mixture of input types not yet implemented, set "
        "'--iree-input-type=[type]' explicitly instead of using 'auto' or "
        "audit the input program to understand why dialects are mixed");
    diag << " (detected:";
    for (auto &s : detectedTypeMnemonics) {
      diag << " '" << s.first() << "'";
    }
    diag << ")";
    return signalPassFailure();
  }

  auto typeMnemonic = detectedTypeMnemonics.begin()->getKey();
  OpPassManager passManager(module.getOperationName());
  bool foundExtension =
      pipelineExtensions->extendCustomInputConversionPassPipeline(passManager,
                                                                  typeMnemonic);
  if (!foundExtension) {
    // We expect that callers properly validate supported extensions and
    // that if a plugin advertises support, it actually provides it.
    module.emitError() << "custom input conversion for extension '"
                       << typeMnemonic << "' not found";
    return signalPassFailure();
  }
  if (failed(runPipeline(passManager, module))) {
    return signalPassFailure();
  }
}

void AutoInputConversionPipelinePass::getDependentDialects(
    DialectRegistry &registry) const {
  if (pipelineExtensions) {
    pipelineExtensions->registerDialects(registry);
  }
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createAutoInputConversionPipelinePass() {
  return std::make_unique<AutoInputConversionPipelinePass>(nullptr);
}

std::unique_ptr<OperationPass<ModuleOp>>
createAutoInputConversionPipelinePass(PipelineExtensions *pipelineExtensions) {
  return std::make_unique<AutoInputConversionPipelinePass>(pipelineExtensions);
}

} // namespace mlir::iree_compiler
