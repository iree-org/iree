// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct TestExecutablePreprocessingPass
    : public TestExecutablePreprocessingBase<TestExecutablePreprocessingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    // Replace i64 constants with whatever we source from the target
    // configuration. A real pipeline would use the target information to do
    // whatever it needed to the executable instead.
    getOperation()->walk([&](IREE::HAL::ExecutableVariantOp variantOp) {
      auto configAttr = variantOp.getTarget().getConfiguration();
      if (!configAttr) return;
      auto replacementAttr = configAttr.getAs<IntegerAttr>("replace_i64");
      if (!replacementAttr) {
        // Skip variants that don't request modification.
        return;
      }
      variantOp.walk([&](Operation *op) {
        if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
          if (constantOp.getType() == replacementAttr.getType()) {
            constantOp.setValueAttr(replacementAttr);
          }
        }
      });
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<void>> createTestExecutablePreprocessingPass() {
  return std::make_unique<TestExecutablePreprocessingPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
