// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUSELECTLOWERINGSTRATEGYPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {
/// Selects the lowering strategy for a hal.executable.variant operation.
class LLVMCPUSelectLoweringStrategyPass
    : public impl::LLVMCPUSelectLoweringStrategyPassBase<
          LLVMCPUSelectLoweringStrategyPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::CPU::IREECPUDialect, IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override;
};
} // namespace

/// Verify that valid configuration is set for all ops within the funcOp.
template <typename F>
static LogicalResult
verifyLoweringConfiguration(FunctionOpInterface funcOp,
                            IREE::Codegen::TranslationInfoAttr translationInfo,
                            F verificationFn) {
  auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    if (isa<IREE::LinalgExt::CustomOp>(op)) {
      return WalkResult::advance();
    }
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
        getLoweringConfig(op);
    if (!loweringConfig)
      return WalkResult::advance();
    std::unique_ptr<TilingConfig> tilingConfig =
        TilingConfig::create(loweringConfig);
    if (!tilingConfig)
      return WalkResult::interrupt();
    return verificationFn(op, *tilingConfig, translationInfo,
                          ArrayRef<int64_t>{});
  });
  return failure(walkResult.wasInterrupted());
}

void LLVMCPUSelectLoweringStrategyPass::runOnOperation() {
  auto moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    // Set the strategy with default heuristics.
    if (failed(initCPULaunchConfig(funcOp))) {
      funcOp.emitOpError("failed to set lowering configuration");
      return signalPassFailure();
    }

    auto translationInfo = getTranslationInfo(funcOp);
    if (!translationInfo) {
      continue;
    }

    // Verify the configuration.
    LogicalResult verificationStatus = success();
    switch (translationInfo.getDispatchLoweringPassPipeline()) {
    case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert:
      verificationStatus = verifyLoweringConfiguration(
          funcOp, translationInfo, verifyDoubleTilingExpertPassPipelineConfig);
      break;
    case IREE::Codegen::DispatchLoweringPassPipeline::
        CPUConvTileAndDecomposeExpert:
      verificationStatus = verifyLoweringConfiguration(
          funcOp, translationInfo, verifyConvTileAndDecomposeExpertConfig);
      break;
    default:
      break;
    }
    if (failed(verificationStatus)) {
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler
