// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect, linalg::LinalgDialect,
                    LLVM::LLVMDialect, vector::VectorDialect>();
  }

  LLVMCPULowerExecutableTargetPass(bool vectorize = true)
      : lowerToVectors(vectorize) {}
  LLVMCPULowerExecutableTargetPass(
      const LLVMCPULowerExecutableTargetPass &pass) {}

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

  ListOption<int> workloadPerWorkgroup{
      *this, "workload-per-workgroup", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc(
          "Specifies the workload per workgroup to use in x, y, z order. Is "
          "expected for use only with use-lowering-pipeline option")};

  /// TODO(ravishankarm): Option to not generate any `vector.` instructions. The
  /// VMVX backend uses the same lowering as the CPU pass but there is no
  /// lowering of these `vector.` operations to scalar code. So as a WAR do the
  /// same tiling scheme but avoid generating vector instructions. When VMVX can
  /// handle vector instructions, drop this options.
  bool lowerToVectors;
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

void LLVMCPULowerExecutableTargetPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();

  OpPassManager executableLoweringPipeline(
      IREE::HAL::ExecutableVariantOp::getOperationName());

  if (!useLoweringPipeline.empty()) {
    // Use the pass pipeline specified in the command line.
    SmallVector<int64_t, 4> workloadPerWorkgroupVec;
    workloadPerWorkgroupVec.assign(workloadPerWorkgroup.begin(),
                                   workloadPerWorkgroup.end());
    executableLoweringPipeline.addPass(
        createSetNumWorkgroupsPass(workloadPerWorkgroupVec));
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
    OpPassManager &nestedModulePM = executableLoweringPipeline.nest<ModuleOp>();
    if (!testLoweringConfiguration && passPipeline.hasValue()) {
      switch (passPipeline.getValue()) {
        case IREE::HAL::DispatchLoweringPassPipeline::CPUDefault:
          addCPUDefaultPassPipeline(nestedModulePM);
          break;
        case IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization:
          addCPUVectorizationPassPipeline(nestedModulePM, lowerToVectors);
          break;
        default:
          llvm_unreachable("Unsupported pipeline on CPU target.");
      }
    }
  }

  if (failed(runPipeline(executableLoweringPipeline, variantOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPULowerExecutableTargetPass(bool lowerToVectors) {
  return std::make_unique<LLVMCPULowerExecutableTargetPass>(lowerToVectors);
}

}  // namespace iree_compiler
}  // namespace mlir
