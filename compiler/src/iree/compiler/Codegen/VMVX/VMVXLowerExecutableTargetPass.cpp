// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/VMVX/PassDetail.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "iree-vmvx-lower-executable-target"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler {

namespace {

/// Lowers an hal.executable.variant operation to scalar/native-vector code.
class VMVXLowerExecutableTargetPass
    : public VMVXLowerExecutableTargetBase<VMVXLowerExecutableTargetPass> {
public:
  VMVXLowerExecutableTargetPass() = default;
  VMVXLowerExecutableTargetPass(const VMVXLowerExecutableTargetPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<IREE::HAL::HALDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    bufferization::BufferizationDialect,
                    linalg::LinalgDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  void runOnOperation() override;
};
} // namespace

void VMVXLowerExecutableTargetPass::runOnOperation() {
  auto funcOp = getOperation();

  auto translationInfo = getTranslationInfo(funcOp);
  if (!translationInfo)
    return;

  std::optional<OpPassManager> maybePipeline =
      getFunctionOpInterfacePassManager(funcOp);
  if (!maybePipeline) {
    funcOp.emitOpError(
        "unhandled function-like container during executable lowering");
    return signalPassFailure();
  }
  OpPassManager &pipeline = maybePipeline.value();

  auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  bool enableUKernels = target && hasUkernel(target);
  switch (translationInfo.getDispatchLoweringPassPipeline()) {
  // No pipleline specified, nothing to do.
  case IREE::Codegen::DispatchLoweringPassPipeline::None:
    return;
  case IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault:
    addVMVXDefaultPassPipeline(pipeline, enableUKernels);
    break;
  default:
    funcOp.emitOpError("Unsupported pipeline on VMVX target.");
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Using Pass pipeline : ";
    pipeline.dump();
  });
  if (failed(runPipeline(pipeline, funcOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createVMVXLowerExecutableTargetPass() {
  return std::make_unique<VMVXLowerExecutableTargetPass>();
}

} // namespace mlir::iree_compiler
