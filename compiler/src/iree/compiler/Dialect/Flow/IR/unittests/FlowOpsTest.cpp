// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

namespace mlir::iree_compiler::IREE::Flow {
namespace {

class FlowOpsTest : public ::testing::Test {
protected:
  FlowOpsTest() {
    registry.insert<FlowDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.allowUnregisteredDialects();
    context.loadAllAvailableDialects();
  }

  MLIRContext *getContext() { return &context; }

private:
  MLIRContext context;
  DialectRegistry registry;
};

TEST_F(FlowOpsTest, MovementPolicyIsNotCached) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      R"mlir(
module {
  func.func @test(%arg0 : tensor<4xf32>) {
    flow.dispatch.region {
      flow.return
    }
    "test.use"(%arg0) : (tensor<4xf32>) -> ()
    func.return
  }
}
)mlir",
      getContext());
  ASSERT_TRUE(module);

  func::FuncOp function = module->lookupSymbol<func::FuncOp>("test");
  ASSERT_TRUE(function);
  auto dispatchOps = function.getOps<DispatchRegionOp>();
  ASSERT_FALSE(dispatchOps.empty());
  DispatchRegionOp dispatch = *dispatchOps.begin();
  Operation *use = nullptr;
  function.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "test.use") {
      use = op;
    }
  });
  ASSERT_NE(use, nullptr);

  DispatchRegionTiedUseAnalysis analysis(dispatch);
  BlockArgument argument = function.getArgument(0);
  EXPECT_FALSE(analysis.hasUseAfterDispatch(
      argument, nullptr, [&](Operation *op) { return op == use; }));
  EXPECT_TRUE(analysis.hasUseAfterDispatch(argument, nullptr,
                                           [](Operation *) { return false; }));
}

} // namespace
} // namespace mlir::iree_compiler::IREE::Flow
