// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/ScopedTransform.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestWrapScopePass : public PassWrapper<TestWrapScopePass, Pass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestWrapScopePass)

  TestWrapScopePass() = default;
  TestWrapScopePass(const TestWrapScopePass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-wrap-scope"; }
  StringRef getDescription() const final { return "Test wrap scope pass."; }
  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::transform::LinalgTransformDialect>();
  }

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() != opToWrap)
        return;
      linalg::transform::wrapInScope(op);
    });
  }

  Pass::Option<std::string> opToWrap{*this, "opname",
                                     llvm::cl::desc("Op to wrap")};
};

struct TestUnwrapScopePass : public PassWrapper<TestUnwrapScopePass, Pass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestUnwrapScopePass)
  StringRef getArgument() const final { return "test-unwrap-scope"; }
  StringRef getDescription() const final { return "Test unwrap scope pass."; }
  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void runOnOperation() override {
    getOperation()->walk(
        [](linalg::transform::ScopeOp scope) { (void)unwrapScope(scope); });
  }
};
} // namespace

namespace mlir {
namespace test_ext {
void registerTestLinalgTransformWrapScope() {
  PassRegistration<TestWrapScopePass>();
  PassRegistration<TestUnwrapScopePass>();
}
} // namespace test_ext
} // namespace mlir
