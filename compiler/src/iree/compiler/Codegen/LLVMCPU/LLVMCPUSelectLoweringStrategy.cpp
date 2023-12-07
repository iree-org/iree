// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler {

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
    // TODO(qedawkins): Once TransformStrategies is deprecated, drop the
    // unnecessary dialect registrations.
    // clang-format off
    registry.insert<IREE::Codegen::IREECodegenDialect,
                    IREE::HAL::HALDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    bufferization::BufferizationDialect,
                    linalg::LinalgDialect,
                    LLVM::LLVMDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    transform::TransformDialect,
                    vector::VectorDialect>();
    // clang-format on
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

  // Set the strategy with default heuristics.
  if (failed(initCPULaunchConfig(moduleOp))) {
    return signalPassFailure();
  }

  std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo =
      getIdenticalTranslationInfo(variantOp);
  if (!translationInfo) {
    moduleOp.emitOpError(
        "unhandled compilation of entry point functions with different "
        "translation info");
    return signalPassFailure();
  }

  // Verify the configuration.
  LogicalResult verificationStatus = success();
  switch (translationInfo.value().getDispatchLoweringPassPipeline()) {
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert:
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingPadExpert:
    verificationStatus =
        verifyLoweringConfiguration(moduleOp, translationInfo.value(),
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

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPUSelectLoweringStrategyPass() {
  return std::make_unique<LLVMCPUSelectLoweringStrategyPass>();
}

} // namespace mlir::iree_compiler
