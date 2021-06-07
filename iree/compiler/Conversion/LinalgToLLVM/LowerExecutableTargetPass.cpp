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
                    LLVM::LLVMDialect>();
  }

  LowerExecutableTargetPass(bool vectorize = true)
      : lowerToVectors(vectorize) {}
  LowerExecutableTargetPass(const LowerExecutableTargetPass &pass) {
    invokeLoweringPipelines = pass.invokeLoweringPipelines;
    lowerToVectors = pass.lowerToVectors;
  }

  void runOnOperation() override;

 private:
  Option<bool> invokeLoweringPipelines{
      *this, "invoke-lowering-pipelines",
      llvm::cl::desc(
          "Invokes the pass pipeline to lower an hal.executable.target "
          "operation into scalar/native-vector code. Defaults to true, but "
          "can be set to false for testing purposes."),
      llvm::cl::init(true)};

  /// TODO(ravishankarm): Option to not generate any `vector.` instructions. The
  /// VMVX backend uses the same lowering as the CPU pass but there is no
  /// lowering of these `vector.` operations to scalar code. So as a WAR do the
  /// same tiling scheme but avoid generating vector instructions. When VMVX can
  /// handle vector instructions, drop this options.
  bool lowerToVectors;
};
}  // namespace

void LowerExecutableTargetPass::runOnOperation() {
  IREE::HAL::ExecutableTargetOp targetOp = getOperation();
  ModuleOp moduleOp = targetOp.getInnerModule();

  FailureOr<IREE::HAL::DispatchLoweringPassPipeline> setPipeline =
      initCPULaunchConfig(moduleOp);
  if (failed(setPipeline)) {
    return signalPassFailure();
  }

  OpPassManager executableLoweringPipeline(
      IREE::HAL::ExecutableTargetOp::getOperationName());
  executableLoweringPipeline.addPass(createSetNumWorkgroupsPass());
  OpPassManager &nestedModulePM = executableLoweringPipeline.nest<ModuleOp>();

  if (invokeLoweringPipelines) {
    IREE::HAL::DispatchLoweringPassPipeline passPipeline =
        setPipeline.getValue();
    switch (passPipeline) {
      case IREE::HAL::DispatchLoweringPassPipeline::CPUDefault:
        addCPUDefaultPassPipeline(nestedModulePM);
        break;
      case IREE::HAL::DispatchLoweringPassPipeline::CPUVectorization:
        addCPUVectorizationPassPipeline(nestedModulePM, lowerToVectors);
        break;
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
