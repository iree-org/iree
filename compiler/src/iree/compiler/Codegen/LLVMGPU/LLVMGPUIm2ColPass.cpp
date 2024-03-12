// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir::iree_compiler {

namespace {

class LLVMGPUIm2ColPass : public LLVMGPUIm2ColBase<LLVMGPUIm2ColPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override;
};

} // namespace

void LLVMGPUIm2ColPass::runOnOperation() {
  auto operation = getOperation();
  SmallVector<linalg::LinalgOp> convOps;
  operation->walk([&](linalg::Conv2DNhwcHwcfOp convOp) { convOps.push_back(convOp);});

}

std::unique_ptr<OperationPass<>>
createLLVMGPUIm2ColPass() {
  return std::make_unique<LLVMGPUIm2ColPass>();
}

} // namespace mlir::iree_compiler