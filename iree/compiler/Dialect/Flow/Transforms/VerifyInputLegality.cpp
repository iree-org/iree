// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
class VerifyInputLegalityPass
    : public VerifyInputLegalityBase<VerifyInputLegalityPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addLegalOp<tosa::ApplyScaleOp>();
    // We're already depending on the Tosa Dialect
    target.addIllegalDialect<tosa::TosaDialect>();
    // Avoid MHLO dependency
    target.addIllegalDialect("mhlo");
    target.addIllegalOp<UnrealizedConversionCastOp>();

    if (failed(iree_compiler::verifyAllOperationsAreLegal(getOperation(),
                                                          target))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createVerifyInputLegalityPass() {
  return std::make_unique<VerifyInputLegalityPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
