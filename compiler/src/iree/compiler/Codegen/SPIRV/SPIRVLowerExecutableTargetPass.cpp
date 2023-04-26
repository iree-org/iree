// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-spirv-lower-executable-target-pass"

namespace mlir {
namespace iree_compiler {

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

namespace {
/// Lowers a hal.executable.variant inner module to SPIR-V scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code,
/// - then convert to SPIRV dialect.
class SPIRVLowerExecutableTargetPass
    : public SPIRVLowerExecutableTargetBase<SPIRVLowerExecutableTargetPass> {
 public:
  SPIRVLowerExecutableTargetPass() = default;
  SPIRVLowerExecutableTargetPass(const SPIRVLowerExecutableTargetPass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, affine::AffineDialect,
                gpu::GPUDialect, IREE::HAL::HALDialect, linalg::LinalgDialect,
                IREE::LinalgExt::IREELinalgExtDialect, memref::MemRefDialect,
                bufferization::BufferizationDialect, scf::SCFDialect,
                spirv::SPIRVDialect, transform::TransformDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override;

 private:
  Option<bool> testLoweringConfiguration{
      *this, "test-lowering-configuration",
      llvm::cl::desc("Flag used for lit-testing the configuration set for root "
                     "ops in hal.executable.variants. Defaults to false. Set "
                     "to true for lit tests; not for general usage"),
      llvm::cl::init(false)};
};
}  // namespace

/// Verify that valid configuration is set for all ops within the compiled
/// module.
template <typename F>
static LogicalResult verifyLoweringConfiguration(
    ModuleOp module, IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize, F verificationFn) {
  auto walkResult = module.walk([&](Operation *op) -> WalkResult {
    IREE::Codegen::LoweringConfigAttr loweringConfig = getLoweringConfig(op);
    if (!loweringConfig) return WalkResult::advance();
    return verificationFn(op, loweringConfig, translationInfo, workgroupSize);
  });
  return failure(walkResult.wasInterrupted());
}

static LogicalResult verifyEntryPoint(
    ModuleOp moduleOp, IREE::Codegen::TranslationInfoAttr translationInfo,
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
    workgroupSizes[index] = attr.cast<IntegerAttr>().getInt();
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

void SPIRVLowerExecutableTargetPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();

  OpPassManager pipeline(IREE::HAL::ExecutableVariantOp::getOperationName());

  if (failed(initSPIRVLaunchConfig(moduleOp))) {
    return signalPassFailure();
  }
  // There might be multiple entry points in the module. Currently, all of
  // them need to have the same pipeline.
  // TODO(ravishankarm): This is strange that this is not enforced
  // structurally, but something to address later on. For now this restriction
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
          moduleOp.emitError(
              "unhandled compilation of entry point function with different "
              "translation info within a module");
          return signalPassFailure();
        }
        continue;
      }

      // Verify the properties of each entry point based on the target
      // pipeline.
      if (failed(verifyEntryPoint(moduleOp, currTranslationInfo, exportOp))) {
        return signalPassFailure();
      }

      translationInfo = currTranslationInfo;
    }
  }

  if (!testLoweringConfiguration && translationInfo.has_value()) {
    switch (translationInfo.value().getDispatchLoweringPassPipeline()) {
      case CodeGenPipeline::SPIRVBaseDistribute:
        addSPIRVBaseDistributePassPipeline(pipeline);
        break;
      case CodeGenPipeline::SPIRVBaseVectorize:
        addSPIRVBaseVectorizePassPipeline(pipeline);
        break;
      case CodeGenPipeline::SPIRVCooperativeMatrixVectorize:
        addSPIRVCooperativeMatrixVectorizePassPipeline(
            pipeline, translationInfo.value().getSoftwarePipelineDepth(),
            translationInfo.value().getSoftwarePipelineStoreStage());
        break;
      case CodeGenPipeline::SPIRVMatmulPromoteVectorize:
        addSPIRVMatmulPromoteVectorizePassPipeline(
            pipeline, translationInfo.value().getSoftwarePipelineDepth(),
            translationInfo.value().getSoftwarePipelineStoreStage());
        break;
      case CodeGenPipeline::SPIRVSubgroupReduce:
        addSPIRVSubgroupReducePassPipeline(pipeline);
        break;
      case CodeGenPipeline::SPIRVWinogradVectorize:
        addSPIRVWinogradVectorizePassPipeline(pipeline);
        break;
      case CodeGenPipeline::TransformDialectCodegen:
        addSPIRVTransformDialectPassPipeline(pipeline);
        break;
      default:
        variantOp.emitOpError("Unsupported pipeline on GPU target.");
        return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Using SPIR-V lowering pass pipeline:\n";
    pipeline.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  if (failed(runPipeline(pipeline, variantOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVLowerExecutableTargetPass() {
  return std::make_unique<SPIRVLowerExecutableTargetPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
