// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

namespace {
/// Lowers a hal.executable.variant inner module to SPIR-V scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code,
/// - then convert to SPIRV dialect.
class SPIRVSelectLoweringStrategyPass
    : public SPIRVSelectLoweringStrategyBase<SPIRVSelectLoweringStrategyPass> {
public:
  SPIRVSelectLoweringStrategyPass() = default;
  SPIRVSelectLoweringStrategyPass(const SPIRVSelectLoweringStrategyPass &pass) {
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO(qedawkins): Once TransformStrategies is deprecated, drop the
    // unnecessary dialect registrations.
    registry
        .insert<IREE::Codegen::IREECodegenDialect, affine::AffineDialect,
                gpu::GPUDialect, IREE::HAL::HALDialect, linalg::LinalgDialect,
                IREE::LinalgExt::IREELinalgExtDialect, memref::MemRefDialect,
                bufferization::BufferizationDialect, scf::SCFDialect,
                spirv::SPIRVDialect, transform::TransformDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

/// Verify that valid configuration is set for all ops within the compiled
/// module.
template <typename F>
static LogicalResult
verifyLoweringConfiguration(ModuleOp module,
                            IREE::Codegen::TranslationInfoAttr translationInfo,
                            ArrayRef<int64_t> workgroupSize, F verificationFn) {
  auto walkResult = module.walk([&](Operation *op) -> WalkResult {
    IREE::Codegen::LoweringConfigAttr loweringConfig = getLoweringConfig(op);
    if (!loweringConfig)
      return WalkResult::advance();
    return verificationFn(op, loweringConfig, translationInfo, workgroupSize);
  });
  return failure(walkResult.wasInterrupted());
}

static LogicalResult
verifyEntryPoint(ModuleOp moduleOp,
                 IREE::Codegen::TranslationInfoAttr translationInfo,
                 IREE::HAL::ExecutableExportOp exportOp) {
  if (translationInfo.getDispatchLoweringPassPipeline() ==
      CodeGenPipeline::TransformDialectCodegen) {
    // Transform dialect encodes configuration into the schedule directly.
    return success();
  }

  std::optional<mlir::ArrayAttr> workgroupSizeAttr =
      exportOp.getWorkgroupSize();
  if (!workgroupSizeAttr || workgroupSizeAttr->size() != 3) {
    return moduleOp.emitError(
        "expected workgroup size to have three dimensions for SPIR-V "
        "pipelines");
  }

  std::array<int64_t, 3> workgroupSizes;
  for (auto [index, attr] : llvm::enumerate(workgroupSizeAttr.value())) {
    workgroupSizes[index] = llvm::cast<IntegerAttr>(attr).getInt();
  }

  switch (translationInfo.getDispatchLoweringPassPipeline()) {
  case CodeGenPipeline::SPIRVBaseVectorize:
    return verifyLoweringConfiguration(moduleOp, translationInfo,
                                       workgroupSizes,
                                       verifySPIRVBaseVectorizePassPipeline);
  case CodeGenPipeline::SPIRVMatmulPromoteVectorize:
    return verifyLoweringConfiguration(
        moduleOp, translationInfo, workgroupSizes,
        verifySPIRVMatmulPromoteVectorizePassPipeline);
  case CodeGenPipeline::SPIRVCooperativeMatrixVectorize:
    return verifyLoweringConfiguration(
        moduleOp, translationInfo, workgroupSizes,
        verifySPIRVCooperativeMatrixVectorizePassPipeline);
  default:
    break;
  }
  return success();
}

void SPIRVSelectLoweringStrategyPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();

  if (failed(initSPIRVLaunchConfig(moduleOp))) {
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

  // Verify the properties of each entry point based on the target pipeline.
  for (auto exportOp : variantOp.getExportOps()) {
    if (failed(verifyEntryPoint(moduleOp, translationInfo.value(), exportOp))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVSelectLoweringStrategyPass() {
  return std::make_unique<SPIRVSelectLoweringStrategyPass>();
}

} // namespace mlir::iree_compiler
