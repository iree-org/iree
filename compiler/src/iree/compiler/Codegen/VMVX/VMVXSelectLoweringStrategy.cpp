// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/VMVX/KernelDispatch.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VMVXSELECTLOWERINGSTRATEGYPASS
#include "iree/compiler/Codegen/VMVX/Passes.h.inc"

namespace {
/// Selects the lowering strategy for a hal.executable.variant operation.
class VMVXSelectLoweringStrategyPass
    : public impl::VMVXSelectLoweringStrategyPassBase<
          VMVXSelectLoweringStrategyPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void VMVXSelectLoweringStrategyPass::runOnOperation() {
  auto moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    // Set the strategy with default heuristics.
    if (failed(initVMVXLaunchConfig(funcOp))) {
      funcOp.emitOpError("failed to set lowering configuration");
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler
