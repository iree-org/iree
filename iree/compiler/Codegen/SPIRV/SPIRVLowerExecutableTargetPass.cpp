// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-spirv-lower-executable-target-pass"

namespace mlir {
namespace iree_compiler {

namespace {
/// Lowers an hal.executable.variant operation to scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code
/// - then convert to SPIRV dialect.
class SPIRVLowerExecutableTargetPass
    : public SPIRVLowerExecutableTargetBase<SPIRVLowerExecutableTargetPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, IREE::HAL::HALDialect,
                    linalg::LinalgDialect, linalg_ext::LinalgExtDialect,
                    memref::MemRefDialect, scf::SCFDialect, ShapeDialect,
                    spirv::SPIRVDialect, vector::VectorDialect>();
  }

  SPIRVLowerExecutableTargetPass() = default;
  SPIRVLowerExecutableTargetPass(const SPIRVLowerExecutableTargetPass &pass){};

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

void SPIRVLowerExecutableTargetPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();

  OpPassManager executableLoweringPipeline(
      IREE::HAL::ExecutableVariantOp::getOperationName());

  if (failed(initSPIRVLaunchConfig(moduleOp))) {
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
      case IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute:
        addSPIRVDistributePassPipeline(nestedModulePM);
        break;
      case IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistributeToGlobalID:
        addSPIRVDistributeToGlobalIDPipeline(nestedModulePM);
        break;
      case IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize:
        addSPIRVVectorizationPassPipeline(nestedModulePM);
        break;
      default:
        llvm_unreachable("Unsupported pipeline on GPU target.");
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Using SPIRV Lowering Pass pipeline :\n";
    executableLoweringPipeline.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  if (failed(runPipeline(executableLoweringPipeline, variantOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVLowerExecutableTargetPass() {
  return std::make_unique<SPIRVLowerExecutableTargetPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
