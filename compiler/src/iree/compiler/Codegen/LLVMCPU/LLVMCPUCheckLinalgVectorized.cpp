// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct LLVMCPUCheckLinalgVectorizedPass
    : LLVMCPUCheckLinalgVectorizedBase<LLVMCPUCheckLinalgVectorizedPass> {
  void runOnOperation() override;
};
}  // namespace

void LLVMCPUCheckLinalgVectorizedPass::runOnOperation() {
  auto funcOp = getOperation();
  auto walkResult = funcOp.walk([&](linalg::LinalgOp op) -> WalkResult {
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted()) {
    funcOp.emitWarning("one or more operations were found not vectorized");
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUCheckLinalgVectorizedPass() {
  return std::make_unique<LLVMCPUCheckLinalgVectorizedPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
