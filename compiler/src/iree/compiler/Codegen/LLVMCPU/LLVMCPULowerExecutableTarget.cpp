// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {

namespace {
/// Lowers an hal.executable.variant operation to scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code
/// - then convert to LLVM dialect.
/// In due course this could be used to generate code for all backends.
class LLVMCPULowerExecutableTargetPass
    : public LLVMCPULowerExecutableTargetBase<
          LLVMCPULowerExecutableTargetPass> {
 public:
  LLVMCPULowerExecutableTargetPass() = default;
  LLVMCPULowerExecutableTargetPass(
      const LLVMCPULowerExecutableTargetPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<IREE::Codegen::IREECodegenDialect,
                    IREE::HAL::HALDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    bufferization::BufferizationDialect,
                    linalg::LinalgDialect,
                    linalg::transform::LinalgTransformDialect,
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

 private:
  Option<bool> testLoweringConfiguration{
      *this, "test-lowering-configuration",
      llvm::cl::desc(
          "Flag used for lit-testing the default configuration set for root "
          "ops in hal.executable.variants. Defaults to false and is set to "
          "true "
          "for lit tests. Not for general usage"),
      llvm::cl::init(false)};

  Option<std::string> useLoweringPipeline{
      *this, "use-lowering-pipeline",
      llvm::cl::desc(
          "List of passes to be applied for lowering the "
          "hal.executable.variant. Note that this is used for all "
          "hal.executable.variants, so might be useful when there is "
          "only one such operation. The specified pass pipeline is "
          "expected to work on the std.module op within the "
          "hal.executable.variant operation")};
};
}  // namespace

/// The pipeline parser doesnt like strings that have `'` or `"` in them. But it
/// is needed for demarcating the option value. So just drop them before sending
/// it one.
static StringRef sanitizePipelineString(StringRef input) {
  if (input.empty()) return input;
  // If first/last character is ' or ", drop them.
  if (input.front() == '\'' || input.front() == '"') {
    input = input.drop_front();
  }
  if (input.back() == '\'' || input.back() == '"') {
    input = input.drop_back();
  }
  return input;
}

/// Verify that valid configuration is set for all ops within the compiled
/// module.
template <typename F>
static LogicalResult verifyLoweringConfiguration(
    ModuleOp module, IREE::Codegen::TranslationInfoAttr translationInfo,
    F verificationFn) {
  auto walkResult = module.walk([&](Operation *op) -> WalkResult {
    IREE::Codegen::LoweringConfigAttr loweringConfig = getLoweringConfig(op);
    if (!loweringConfig) return WalkResult::advance();
    return verificationFn(op, loweringConfig, translationInfo,
                          ArrayRef<int64_t>{});
  });
  return failure(walkResult.wasInterrupted());
}

void LLVMCPULowerExecutableTargetPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();
  if (!variantOp || !moduleOp) {
    getOperation()->emitError(
        "Expected a variantOp root with an inner ModuleOp");
    return signalPassFailure();
  }

  OpPassManager executableLoweringPipeline(
      IREE::HAL::ExecutableVariantOp::getOperationName());

  if (!useLoweringPipeline.empty()) {
    OpPassManager &nestedModulePM = executableLoweringPipeline.nest<ModuleOp>();
    if (failed(parsePassPipeline(sanitizePipelineString(useLoweringPipeline),
                                 nestedModulePM))) {
      return signalPassFailure();
    }
  } else {
    // Use default heuristics.
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
        case IREE::Codegen::DispatchLoweringPassPipeline::
            CPUDoubleTilingPadExpert:
          verificationStatus = verifyLoweringConfiguration(
              moduleOp, translationInfo.value(),
              verifyDoubleTilingExpertPassPipelineConfig);
          break;
        case IREE::Codegen::DispatchLoweringPassPipeline::
            CPUConvTileAndDecomposeExpert:
          verificationStatus = verifyLoweringConfiguration(
              moduleOp, translationInfo.value(),
              verifyConvTileAndDecomposeExpertConfig);
          break;
        default:;
      }
      if (failed(verificationStatus)) {
        return signalPassFailure();
      }

      auto target = variantOp.getTarget();
      bool lowerToAVX2 = hasAVX2Feature(target);
      bool enableVectorMasking =
          isX86(target) || isRISCV(target) ||
          (isAArch64(target) && hasAnySVEFeature(target));
      bool enableMicrokernels = hasMicrokernels(target);
      if (!testLoweringConfiguration) {
        switch (translationInfo.value().getDispatchLoweringPassPipeline()) {
          case IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault:
          case IREE::Codegen::DispatchLoweringPassPipeline::None:
            addCPUDefaultPassPipeline(executableLoweringPipeline);
            break;
          case IREE::Codegen::DispatchLoweringPassPipeline::
              CPUBufferOpsTileAndVectorize:
            addCPUBufferOpsTileAndVectorizePipeline(executableLoweringPipeline,
                                                    enableVectorMasking);
            break;
          case IREE::Codegen::DispatchLoweringPassPipeline::
              CPUDoubleTilingExpert:
            addMultiTilingExpertPassPipeline(
                executableLoweringPipeline,
                static_cast<int>(StrategyTilingLevel::NumStrategyTileLevels),
                /*enablePeeling=*/false, enableVectorMasking, lowerToAVX2);
            break;
          case IREE::Codegen::DispatchLoweringPassPipeline::
              CPUDoubleTilingPadExpert:
            addDoubleTilingPadExpertPassPipeline(executableLoweringPipeline,
                                                 enableVectorMasking);
            break;
          case IREE::Codegen::DispatchLoweringPassPipeline::
              CPUDoubleTilingPeelingExpert:
            addMultiTilingExpertPassPipeline(
                executableLoweringPipeline,
                static_cast<int>(StrategyTilingLevel::NumStrategyTileLevels),
                /*enablePeeling=*/true, enableVectorMasking, lowerToAVX2);
            break;
          case IREE::Codegen::DispatchLoweringPassPipeline::
              CPUConvTileAndDecomposeExpert:
            addConvTileAndDecomposeExpertPassPipeline(
                executableLoweringPipeline, enableVectorMasking);
            break;
          case IREE::Codegen::DispatchLoweringPassPipeline::Mmt4dTilingExpert:
            addMmt4dTilingExpertPassPipeline(executableLoweringPipeline);
            break;
          case IREE::Codegen::DispatchLoweringPassPipeline::CPUDataTiling:
            addCPUDataTilingPipeline(executableLoweringPipeline);
            break;
          case IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault:
            addVMVXDefaultPassPipeline(executableLoweringPipeline,
                                       enableMicrokernels);
            break;
          // Transform-dialect pipelines.
          case IREE::Codegen::DispatchLoweringPassPipeline::
              TransformDialectCodegen:
            addTransformDialectPasses(executableLoweringPipeline);
            break;
          default:
            variantOp.emitOpError("Unsupported pipeline on CPU target.");
            return signalPassFailure();
        }
      }
    }
  }

  if (failed(runPipeline(executableLoweringPipeline, variantOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPULowerExecutableTargetPass() {
  return std::make_unique<LLVMCPULowerExecutableTargetPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
