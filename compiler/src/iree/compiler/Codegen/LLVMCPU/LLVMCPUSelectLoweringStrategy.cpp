// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir {
namespace iree_compiler {

namespace {
/// Selects the lowering strategy for a hal.executable.variant operation.
class LLVMCPUSelectLoweringStrategyPass
    : public LLVMCPUSelectLoweringStrategyBase<
          LLVMCPUSelectLoweringStrategyPass> {
public:
  LLVMCPUSelectLoweringStrategyPass() = default;
  LLVMCPUSelectLoweringStrategyPass(
      const LLVMCPUSelectLoweringStrategyPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override;
};
} // namespace

/// Verify that valid configuration is set for all ops within the module.
template <typename F>
static LogicalResult
verifyLoweringConfiguration(ModuleOp module,
                            IREE::Codegen::TranslationInfoAttr translationInfo,
                            F verificationFn) {
  auto walkResult = module.walk([&](Operation *op) -> WalkResult {
    IREE::Codegen::LoweringConfigAttr loweringConfig = getLoweringConfig(op);
    if (!loweringConfig)
      return WalkResult::advance();
    TilingConfig tilingConfig(loweringConfig);
    return verificationFn(op, tilingConfig, translationInfo,
                          ArrayRef<int64_t>{});
  });
  return failure(walkResult.wasInterrupted());
}

void LLVMCPUSelectLoweringStrategyPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();
  if (!moduleOp) {
    getOperation()->emitError(
        "Expected a variantOp root with an inner ModuleOp");
    return signalPassFailure();
  }

  // Set the strategy with default heuristics.
  if (failed(initCPULaunchConfig(moduleOp))) {
    return signalPassFailure();
  }

  // There might be multiple entry points in the module. Currently, all of
  // them need to have the same translation info.
  // TODO(ravishankarm): This is strange that this is not enforced
  // structurally, but something to address later on. The main issue is how
  // to invoke separate dynamic pass pipelines on  entry point functions, when
  // the passes might have module level changes. For now this restriction
  // is fine.
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo;
  for (auto &it : exportOps) {
    auto exportOp = it.second;
    if (IREE::Codegen::TranslationInfoAttr currTranslationInfo =
            getTranslationInfo(exportOp)) {
      if (translationInfo) {
        if (currTranslationInfo != translationInfo.value()) {
          moduleOp.emitOpError(
              "unhandled compilation of entry point functions with different "
              "translation info");
          return signalPassFailure();
        }
      } else {
        translationInfo = currTranslationInfo;
      }
    }
  }

  // Verify the configuration.
  if (translationInfo.has_value()) {
    LogicalResult verificationStatus = success();
    switch (translationInfo.value().getDispatchLoweringPassPipeline()) {
    case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert:
    case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingPadExpert:
      verificationStatus = verifyLoweringConfiguration(
          moduleOp, translationInfo.value(),
          verifyDoubleTilingExpertPassPipelineConfig);
      break;
    case IREE::Codegen::DispatchLoweringPassPipeline::
        CPUConvTileAndDecomposeExpert:
      verificationStatus =
          verifyLoweringConfiguration(moduleOp, translationInfo.value(),
                                      verifyConvTileAndDecomposeExpertConfig);
      break;
    default:
      break;
    }
    if (failed(verificationStatus)) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPUSelectLoweringStrategyPass() {
  return std::make_unique<LLVMCPUSelectLoweringStrategyPass>();
}

} // namespace iree_compiler
} // namespace mlir
