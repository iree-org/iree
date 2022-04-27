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
  int64_t bits = 0;
  auto walkResult = moduleOp.walk([&](memref::AllocaOp allocaOp) -> WalkResult {
    auto type = allocaOp.getType().cast<ShapedType>();
    if (!type.hasStaticShape()) {
      return allocaOp.emitOpError(
          "expected no stack allocations with dynamic shapes");
    }
    bits += type.getSizeInBits();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }
  constexpr int k16KBInBits = 16 * 1024 * 8;
  if (bits >= k16KBInBits) {
    moduleOp.emitOpError(
        "expected total size of stack allocation is smaller than 16 KB");
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUCheckIRBeforeLLVMConversionPass() {
  return std::make_unique<LLVMCPUCheckIRBeforeLLVMConversionPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
