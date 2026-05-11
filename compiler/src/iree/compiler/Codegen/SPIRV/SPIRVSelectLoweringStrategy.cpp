// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVSELECTLOWERINGSTRATEGYPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

using SPIRVPipeline = IREE::GPU::SPIRVLoweringPipeline;

namespace {
/// Lowers a hal.executable.variant inner module to SPIR-V scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code,
/// - then convert to SPIRV dialect.
class SPIRVSelectLoweringStrategyPass final
    : public impl::SPIRVSelectLoweringStrategyPassBase<
          SPIRVSelectLoweringStrategyPass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect,
                    IREE::GPU::IREEGPUDialect, spirv::SPIRVDialect>();
  }

  void runOnOperation() override;
};
} // namespace

/// Verify that valid configuration is set for all ops within the compiled
/// module.
template <typename F>
static LogicalResult
verifyLoweringConfiguration(FunctionOpInterface funcOp,
                            IREE::Codegen::TranslationInfoAttr translationInfo,
                            ArrayRef<int64_t> workgroupSize, F verificationFn) {
  auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    auto loweringConfig =
        getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(op);
    if (!loweringConfig) {
      return WalkResult::advance();
    }
    return verificationFn(op, loweringConfig, translationInfo, workgroupSize);
  });
  return failure(walkResult.wasInterrupted());
}

static LogicalResult
verifyTranslationInfo(FunctionOpInterface funcOp,
                      IREE::Codegen::TranslationInfoAttr translationInfo) {
  Attribute pipelineAttr = translationInfo.getPassPipeline();

  // Transform dialect encodes configuration into the schedule directly.
  if (isa<IREE::Codegen::TransformDialectCodegenPipelineAttr>(pipelineAttr)) {
    return success();
  }

  // Only SPIRV pipelines have additional verification. Other pipeline types
  // (e.g., custom pipelines via PipelineAttrInterface) skip verification.
  auto spirvPipeline = dyn_cast<IREE::GPU::SPIRVPipelineAttr>(pipelineAttr);
  if (!spirvPipeline) {
    return success();
  }

  SmallVector<int64_t> workgroupSizes =
      llvm::to_vector(translationInfo.getWorkgroupSize());
  switch (spirvPipeline.getValue()) {
  case SPIRVPipeline::BaseVectorize:
    return verifyLoweringConfiguration(funcOp, translationInfo, workgroupSizes,
                                       verifySPIRVBaseVectorizePassPipeline);
  case SPIRVPipeline::MatmulPromoteVectorize:
    return verifyLoweringConfiguration(
        funcOp, translationInfo, workgroupSizes,
        verifySPIRVMatmulPromoteVectorizePassPipeline);
  case SPIRVPipeline::CooperativeMatrixVectorize:
    return verifyLoweringConfiguration(
        funcOp, translationInfo, workgroupSizes,
        verifySPIRVCooperativeMatrixVectorizePassPipeline);
  default:
    break;
  }
  return success();
}

void SPIRVSelectLoweringStrategyPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    if (failed(initSPIRVLaunchConfig(funcOp))) {
      funcOp.emitOpError("failed to set lowering configuration");
      return signalPassFailure();
    }

    IREE::Codegen::TranslationInfoAttr translationInfo =
        getTranslationInfo(funcOp);
    if (!translationInfo) {
      continue;
    }

    // Verify the properties of each entry point based on the target pipeline.
    if (failed(verifyTranslationInfo(funcOp, translationInfo))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
