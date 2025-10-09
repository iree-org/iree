// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_VERIFYINPUTLEGALITYPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

struct VerifyInputLegalityPass
    : public IREE::Flow::impl::VerifyInputLegalityPassBase<
          VerifyInputLegalityPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addLegalOp<tosa::ApplyScaleOp>();
    // We're already depending on the Tosa Dialect
    target.addIllegalDialect<tosa::TosaDialect>();
    // Avoid StableHLO dependency
    target.addIllegalDialect("chlo");
    target.addIllegalDialect("stablehlo");
    target.addIllegalOp<UnrealizedConversionCastOp>();

    if (failed(iree_compiler::verifyAllOperationsAreLegal(getOperation(),
                                                          target))) {
      return signalPassFailure();
    }

    // Preserve all analyses since this is a read-only verification pass.
    markAllAnalysesPreserved();
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
