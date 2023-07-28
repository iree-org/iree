// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Pipelines/Options.h"
#include "iree/compiler/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static llvm::cl::opt<bool>
    clVerifyNoF64("iree-flow-verify-no-f64",
                  llvm::cl::desc("Verify that there are no ops that use f64."),
                  llvm::cl::init(true));

namespace {

class VerifyInputLegalityPass
    : public VerifyInputLegalityBase<VerifyInputLegalityPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([seenF64 =
                                              false](Operation *op) mutable {
      // Diagnose only the first encountered F64
      if (seenF64)
        return true;
      bool shouldVerifyNoF64 =
          clVerifyNoF64.getNumOccurrences()
              ? clVerifyNoF64
              : !GlobalOptimizationOptions::FromFlags::get().demoteF64ToF32;
      if (!shouldVerifyNoF64)
        return true;
      // TODO: Query the targets this will execute on to check if F64 is
      // supported.
      auto isF64 = [](Type type) {
        return llvm::isa<Float64Type>(getElementTypeOrSelf(type));
      };
      seenF64 |= llvm::any_of(op->getOperandTypes(), isF64);
      seenF64 |= llvm::any_of(op->getResultTypes(), isF64);
      if (seenF64)
        op->emitError() << "uses partially supported type f64; pass "
                           "--iree-flow-verify-no-f64=false to allow anyway";
      return !seenF64;
    });
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
  }
};
} // namespace

std::unique_ptr<Pass> createVerifyInputLegalityPass() {
  return std::make_unique<VerifyInputLegalityPass>();
}

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
