// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo-iree/Conversion/PassDetail.h"
#include "stablehlo-iree/Conversion/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_VERIFYCOMPILERSTABLEHLOINPUTLEGALITY
#include "stablehlo-iree/Conversion/Passes.h.inc"

namespace {

struct VerifyCompilerStableHloInputLegality final
    : impl::VerifyCompilerStableHloInputLegalityBase<
          VerifyCompilerStableHloInputLegality> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget conversionTarget(*context);
    RewritePatternSet conversionPatterns(context);

    // Note that we would prefer allow-lists of what we positively support.
    // However, it is so common to sneak input-level ops into the pipeline
    // that we explicitly deny the dialects we know about.
    conversionTarget.addIllegalDialect<mlir::stablehlo::StablehloDialect>();
    conversionTarget.addIllegalDialect<mlir::chlo::ChloDialect>();
    conversionTarget.addIllegalDialect<mlir::shape::ShapeDialect>();

    // NOTE: It is not fully illegal to tunnel input dialect ops through to
    // backends that expect them. When such situations arise, the container
    // op should be marked recursively legal here.
    SmallVector<Diagnostic> failuresDiagnostics;
    {
      ScopedDiagnosticHandler diag(
          context, [&](Diagnostic &d) -> LogicalResult {
            failuresDiagnostics.push_back(std::move(d));
            return success();
          });
      if (succeeded(applyPartialConversion(getOperation(), conversionTarget,
                                           std::move(conversionPatterns)))) {
        // Continue checking for other possible errors.
      }
    }

    // Also check for dialects that should have been converted to stablehlo
    // prior to being presented to IREE in the first place.
    // We don't want to use source dependencies for these dialects, so we just
    // use string matching.
    llvm::DenseSet<StringRef> invalidDialects{"tf", "mhlo"};
    llvm::DenseSet<Operation *> invalidOps;
    getOperation().walk([&](Operation *op) {
      if (invalidDialects.contains(op->getName().getDialectNamespace())) {
        invalidOps.insert(op);
      }
    });

    if (failuresDiagnostics.empty() && invalidOps.empty()) {
      return;
    }

    // Error fall-through. Attach all reported issues as notes.
    InFlightDiagnostic errorDiag =
        emitError(getOperation().getLoc())
        << "one or more illegal operations were found in the compiler input "
           "(are you missing an --iree-input-type= flag, or did you mean to "
           "pre-process through an IREE importer frontend?)";
    for (Diagnostic &failureDiag : failuresDiagnostics) {
      Diagnostic &note = errorDiag.attachNote(failureDiag.getLocation());
      for (auto &arg : failureDiag.getArguments()) {
        note.append(arg);
      }
    }
    for (auto &invalidOp : invalidOps) {
      Diagnostic &note = errorDiag.attachNote(invalidOp->getLoc());
      note.append(invalidOp->getName());
    }

    signalPassFailure();
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
