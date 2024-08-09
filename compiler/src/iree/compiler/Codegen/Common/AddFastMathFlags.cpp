// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-add-fast-math-flags"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ADDFASTMATHFLAGSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

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
    : public impl::AddFastMathFlagsPassBase<AddFastMathFlagsPass> {
public:
  using impl::AddFastMathFlagsPassBase<
      AddFastMathFlagsPass>::AddFastMathFlagsPassBase;

  void runOnOperation() override {
    getOperation()->walk([](Operation *op) { addContractFMF(op); });
  }
};

} // namespace
} // namespace mlir::iree_compiler
