// Copyright 2022right The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// #include "iree-dialects/Dialect/LinalgTransform/ScopedTransform.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestTopkSplitReductionPass
    : public PassWrapper<TestTopkSplitReductionPass, Pass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTopkSplitReductionPass)

  TestTopkSplitReductionPass() = default;
  TestTopkSplitReductionPass(const TestTopkSplitReductionPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-topk-split-reduction"; }
  StringRef getDescription() const final {
    return "Test topk split reduction pass.";
  }
  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    mlir::arith::ArithmeticDialect, math::MathDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::iree_compiler::IREE::LinalgExt::TopkSplitReductionControlFn
        splitReductionFn =
            [=](mlir::iree_compiler::IREE::LinalgExt::TopkOp topkOp) {
              return splitRatio.getValue();
            };
    patterns.add<mlir::iree_compiler::IREE::LinalgExt::TopkOpSplitReduction>(
        patterns.getContext(), splitReductionFn,
        LinalgTransformationFilter(
            ArrayRef<StringAttr>{},
            StringAttr::get(patterns.getContext(), "SPLIT_REDUCTION")));
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  Pass::Option<int64_t> splitRatio{*this, "split-ratio",
                                   llvm::cl::desc("Split reduction ratio")};
};

} // namespace

namespace mlir {
namespace test_ext {
void registerTestLinalgExtTransformSplitReduction() {
  PassRegistration<TestTopkSplitReductionPass>();
}
} // namespace test_ext
} // namespace mlir
