// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/Passes.h"

#include "compiler/plugins/target/LLVMCPU/LLVMTargetOptions.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/LLVMCodeGenUtils.h"
#include "llvm/Target/TargetMachine.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_TESTPROPAGATETARGETATTRSPASS
#include "compiler/plugins/target/LLVMCPU/Passes.h.inc"

namespace {
struct TestPropagateTargetAttrsPass final
    : impl::TestPropagateTargetAttrsPassBase<TestPropagateTargetAttrsPass> {
  void runOnOperation() override;
};
} // namespace

static std::optional<LLVMTarget>
getVariantTarget(IREE::HAL::ExecutableVariantOp variantOp) {
  DictionaryAttr configAttr = variantOp.getTarget().getConfiguration();
  LLVMTargetOptions emptyOptions = LLVMCPUTargetCLOptions().getTargetOptions();
  return LLVMTarget::loadFromConfigAttr(variantOp.getLoc(), configAttr,
                                        emptyOptions.target);
}

void TestPropagateTargetAttrsPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  auto maybeTarget = getVariantTarget(variantOp);
  if (!maybeTarget) {
    return signalPassFailure();
  }
  const LLVMTarget &target = *maybeTarget;

  std::unique_ptr<llvm::TargetMachine> targetMachine =
      createTargetMachine(target);
  if (!targetMachine) {
    mlir::emitError(variantOp.getLoc())
        << "failed to create target machine for target triple '"
        << target.getTriple() << "'";
    return signalPassFailure();
  }

  ModuleOp variantModOp = variantOp.getInnerModule();
  // Propagate target features and cpu to function ops.
  populateLLVMFuncTargetAttrs(variantModOp, *targetMachine);
}
} // namespace mlir::iree_compiler::IREE::HAL
