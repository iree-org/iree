// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct LLVMCPUCheckIRBeforeLLVMConversionPass
    : LLVMCPUCheckIRBeforeLLVMConversionBase<
          LLVMCPUCheckIRBeforeLLVMConversionPass> {
  void runOnOperation() override;
};
}  // namespace

void LLVMCPUCheckIRBeforeLLVMConversionPass::runOnOperation() {
  auto moduleOp = getOperation();
  // For now only check that there are no stack allocations.
  auto walkResult = moduleOp.walk([](memref::AllocaOp allocaOp) -> WalkResult {
    return allocaOp.emitOpError("expected no static allocations");
  });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUCheckIRBeforeLLVMConversionPass() {
  return std::make_unique<LLVMCPUCheckIRBeforeLLVMConversionPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
