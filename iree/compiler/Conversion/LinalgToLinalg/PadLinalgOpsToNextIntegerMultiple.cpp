// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

namespace mlir {
namespace iree_compiler {

namespace {

struct PadLinalgOpsToNextIntegerMultiplePass
    : PassWrapper<PadLinalgOpsToNextIntegerMultiplePass, FunctionPass> {
  PadLinalgOpsToNextIntegerMultiplePass(int paddingMultiplier)
      : paddingMultiplier(paddingMultiplier) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<PadLinalgMatmulOp>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  int paddingMultiplier;
};

}  // namespace

std::unique_ptr<FunctionPass> createPadLinalgOpsToNextIntegerMultipleOf(
    int paddingMultiplie = 4) {
  return std::make_unique<PadLinalgOpsToNextIntegerMultiple>(paddingMultiplier);
}

static PassRegistration<PadLinalgOpsToNextIntegerMultiple> pass(
    "iree-flow-pad-tensor-to-subtensor-insert",
    "Convert linalg.pad_tensor into linalg.fill + subtensor_insert.");

}  // namespace iree_compiler

}  // namespace mlir
