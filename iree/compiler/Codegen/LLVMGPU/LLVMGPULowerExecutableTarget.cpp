// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
/// Lowers an hal.executable.variant operation to scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code
/// - then convert to NVVM/ROCDL dialect.
/// This should be merged with the equivalent pass in LinalgToLLVM. Fo
/// simplicity it is currently a separate pass.
class LLVMGPULowerExecutableTargetPass
    : public LLVMGPULowerExecutableTargetBase<
          LLVMGPULowerExecutableTargetPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect, linalg::LinalgDialect,
                    linalg_ext::LinalgExtDialect, vector::VectorDialect,
                    gpu::GPUDialect>();
  }

  LLVMGPULowerExecutableTargetPass() = default;
  LLVMGPULowerExecutableTargetPass(
      const LLVMGPULowerExecutableTargetPass &pass){};

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
};
}  // namespace

void LLVMGPULowerExecutableTargetPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();

  OpPassManager executableLoweringPipeline(
      IREE::HAL::ExecutableVariantOp::getOperationName());

  if (failed(initGPULaunchConfig(moduleOp))) {
    return signalPassFailure();
  }
  // There might be multiple entry points in the module. Currently, all of
  // them need to have the same pipeline.
  // TODO(ravishankarm): This is strange that this is not enforced
  // structurally, but something to address later on. For now this restriction
  // is fine.
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPoints =
      getAllEntryPoints(moduleOp);
  Optional<IREE::HAL::DispatchLoweringPassPipeline> passPipeline;
  for (auto &it : entryPoints) {
    auto entryPointOp = it.second;
    if (IREE::HAL::TranslationInfo translationInfo =
            getTranslationInfo(entryPointOp)) {
      IREE::HAL::DispatchLoweringPassPipeline currPipeline =
          translationInfo.passPipeline().getValue();
      if (passPipeline) {
        if (currPipeline != passPipeline.getValue()) {
          moduleOp.emitError(
              "unhandled compilation of entry point function with different "
              "pass pipelines within a module");
          return signalPassFailure();
        }
        continue;
      }
      passPipeline = currPipeline;
    }
  }

  executableLoweringPipeline.addPass(createSetNumWorkgroupsPass());
  executableLoweringPipeline.addPass(createCanonicalizerPass());
  if (!testLoweringConfiguration && passPipeline.hasValue()) {
    OpPassManager &nestedModulePM = executableLoweringPipeline.nest<ModuleOp>();
    switch (*passPipeline) {
      case IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUDistribute:
        addGPUSimpleDistributePassPipeline(nestedModulePM);
        break;
      case IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUVectorize:
        addGPUVectorizationPassPipeline(nestedModulePM);
        break;
      default:
        llvm_unreachable("Unsupported pipeline on GPU target.");
    }
  }

  if (failed(runPipeline(executableLoweringPipeline, variantOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMGPULowerExecutableTargetPass() {
  return std::make_unique<LLVMGPULowerExecutableTargetPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
