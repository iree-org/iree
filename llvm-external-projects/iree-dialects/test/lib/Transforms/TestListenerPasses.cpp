// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Transforms/Listener.h"
#include "iree-dialects/Transforms/ListenerCSE.h"
#include "iree-dialects/Transforms/ListenerGreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// The test listener prints stuff to `stdout` so that it can be checked by lit
/// tests.
struct TestListener : public RewriteListener {
  void notifyOperationReplaced(Operation *op, ValueRange newValues) override {
    llvm::outs() << "REPLACED " << op->getName() << "\n";
  }
  void notifyOperationRemoved(Operation *op) override {
    llvm::outs() << "REMOVED " << op->getName() << "\n";
  }
};

struct TestListenerCanonicalizePass
    : public PassWrapper<TestListenerCanonicalizePass, Pass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestListenerCanonicalizePass)

  TestListenerCanonicalizePass() = default;
  TestListenerCanonicalizePass(const TestListenerCanonicalizePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-listener-canonicalize"; }
  StringRef getDescription() const final { return "Test canonicalize pass."; }
  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void runOnOperation() override {
    TestListener listener;
    RewriteListener *listenerToUse = nullptr;
    if (withListener)
      listenerToUse = &listener;

    RewritePatternSet patterns(&getContext());
    for (Dialect *dialect : getContext().getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : getContext().getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, &getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            GreedyRewriteConfig(),
                                            listenerToUse)))
      signalPassFailure();
  }

  Pass::Option<bool> withListener{
      *this, "listener", llvm::cl::desc("Whether to run with a test listener"),
      llvm::cl::init(false)};
};

struct TestListenerCSEPass : public PassWrapper<TestListenerCSEPass, Pass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestListenerCSEPass)

  TestListenerCSEPass() = default;
  TestListenerCSEPass(const TestListenerCSEPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-listener-cse"; }
  StringRef getDescription() const final { return "Test CSE pass."; }
  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void runOnOperation() override {
    TestListener listener;
    RewriteListener *listenerToUse = nullptr;
    if (withListener)
      listenerToUse = &listener;

    if (failed(eliminateCommonSubexpressions(getOperation(),
                                             /*domInfo=*/nullptr,
                                             listenerToUse)))
      signalPassFailure();
  }

  Pass::Option<bool> withListener{
      *this, "listener", llvm::cl::desc("Whether to run with a test listener"),
      llvm::cl::init(false)};
};

} // namespace

namespace mlir {
namespace test_ext {
void registerTestListenerPasses() {
  PassRegistration<TestListenerCanonicalizePass>();
  PassRegistration<TestListenerCSEPass>();
}
} // namespace test_ext
} // namespace mlir
