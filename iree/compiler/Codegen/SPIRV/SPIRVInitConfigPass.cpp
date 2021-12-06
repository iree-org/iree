// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "iree-spirv-init-config-pass"

namespace mlir {
namespace iree_compiler {

namespace {

/// Initializes CodeGen configuration for a dispatch region.
class SPIRVInitConfigPass : public SPIRVInitConfigBase<SPIRVInitConfigPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp moduleOp = variantOp.getInnerModule();
    if (failed(initSPIRVLaunchConfig(moduleOp))) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVInitConfigPass() {
  return std::make_unique<SPIRVInitConfigPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
