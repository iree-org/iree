// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {

namespace {
/// Lowers an hal.executable.target operation to scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code
/// - then convert to LLVM dialect.
/// In due course this could be used to generate code for all backends.
class LowerExecutableTargetPass
    : public PassWrapper<LowerExecutableTargetPass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect, linalg::LinalgDialect,
                    LLVM::LLVMDialect, vector::VectorDialect>();
  }

  LowerExecutableTargetPass(bool vectorize = true)
      : lowerToVectors(vectorize) {}
  LowerExecutableTargetPass(const LowerExecutableTargetPass &pass) {}

  void runOnOperation() override;

 private:
  Option<bool> testLoweringConfiguration{
      *this, "test-lowering-configuration",
      llvm::cl::desc(
          "Flag used for lit-testing the default configuration set for root "
          "ops in hal.executable.targets. Defaults to false and is set to true "
          "for lit tests. Not for general usage"),
      llvm::cl::init(false)};

  Option<std::string> useLoweringPipeline{
      *this, "use-lowering-pipeline",
      llvm::cl::desc("List of passes to be applied for lowering the "
                     "hal.executable.target. Note that this is used for all "
                     "hal.executable.targets, so might be useful when there is "
                     "only one such operation. The specified pass pipeline is "
                     "expected to work on the std.module op within the "
                     "hal.executable.target operation")};

  ListOption<int> workgroupSizes{
      *this, "workgroup-size", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc(
          "Specifies the workgroup size to use in x, y, z order. Is expected "
          "for use only with use-lowering-pipeline option")};

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

void LowerExecutableTargetPass::runOnOperation() {
  IREE::HAL::ExecutableTargetOp targetOp = getOperation();
  ModuleOp moduleOp = targetOp.getInnerModule();

  OpPassManager executableLoweringPipeline(
      IREE::HAL::ExecutableTargetOp::getOperationName());

  if (!useLoweringPipeline.empty()) {
    // Use the pass pipeline specified in the command line.
    SmallVector<int64_t, 4> dispatchWorkgroupSize;
    dispatchWorkgroupSize.assign(workgroupSizes.begin(), workgroupSizes.end());
    executableLoweringPipeline.addPass(
        createSetNumWorkgroupsPass(dispatchWorkgroupSize));
    OpPassManager &nestedModulePM = executableLoweringPipeline.nest<ModuleOp>();
    if (failed(parsePassPipeline(sanitizePipelineString(useLoweringPipeline),
                                 nestedModulePM))) {
      return signalPassFailure();
    }
  } else {
    // Use default heuristics.
    FailureOr<IREE::HAL::TranslateExecutableInfo> translationInfo =
        initCPULaunchConfig(moduleOp);
    if (failed(translationInfo)) {
      return signalPassFailure();
    }
    executableLoweringPipeline.addPass(
        createSetNumWorkgroupsPass(translationInfo->workgroupSize));

    OpPassManager &nestedModulePM = executableLoweringPipeline.nest<ModuleOp>();
    if (!testLoweringConfiguration) {
      switch (translationInfo->passPipeline) {
        case IREE::HAL::DispatchLoweringPassPipeline::CPUDefault:
          addCPUDefaultPassPipeline(nestedModulePM);
          break;
        case IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization:
          addCPUVectorizationPassPipeline(nestedModulePM, lowerToVectors);
          break;
      }
    }
  }

  if (failed(runPipeline(executableLoweringPipeline, targetOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLowerExecutableTargetPass(bool lowerToVectors) {
  return std::make_unique<LowerExecutableTargetPass>(lowerToVectors);
}

static PassRegistration<LowerExecutableTargetPass> pass(
    "iree-lower-executable-target-pass",
    "Perform lowering of executable target using one of the "
    "IREE::HAL::DispatchLoweringPassPipeline",
    [] { return std::make_unique<LowerExecutableTargetPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
