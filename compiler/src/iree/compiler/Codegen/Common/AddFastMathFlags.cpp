// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-add-fast-math-flags"

using namespace mlir;
using namespace mlir::iree_compiler;

/// Add `contract` FMF to operations that support it.
static void addContractFMF(Operation *op) {
  LLVM::FastmathFlags contract = LLVM::FastmathFlags::contract;
  TypeSwitch<Operation *>(op)
      .Case<LLVM::FMulOp, LLVM::FAddOp, LLVM::FSubOp, LLVM::FNegOp>(
          [&](auto llvmOp) { llvmOp.setFastmathFlags(contract); });
}

namespace {

/// Add the corresponding fast-math flags to operations given a floating-point
/// optimization mode.
// TODO: For now we only allow default flags, such as arithmetic reassociation.
struct AddFastMathFlagsPass
    : public AddFastMathFlagsBase<AddFastMathFlagsPass> {
public:
  using AddFastMathFlagsBase::AddFastMathFlagsBase;

  void runOnOperation() override {
    getOperation()->walk([](Operation *op) { addContractFMF(op); });
  }
};

} // namespace

std::unique_ptr<OperationPass<LLVM::LLVMFuncOp>>
mlir::iree_compiler::createAddFastMathFlagsPass() {
  return std::make_unique<AddFastMathFlagsPass>();
}
