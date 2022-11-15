// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

struct VerifyCompilerMHLOInputLegalityPass
    : public VerifyCompilerMHLOInputLegalityBase<
          VerifyCompilerMHLOInputLegalityPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    ConversionTarget conversionTarget(*context);
    RewritePatternSet conversionPatterns(&getContext());

    // Note that we would prefer allow-lists of what we positively support.
    // However, it is so common to sneak input-level ops into the pipeline
    // that we explicitly deny the dialects we know about.
    conversionTarget.addIllegalDialect<mhlo::MhloDialect>();
    conversionTarget.addIllegalDialect<chlo::ChloDialect>();
    conversionTarget.addIllegalDialect<mlir::shape::ShapeDialect>();

    // NOTE: It is not fully illegal to tunnel input dialect ops through to
    // backends that expect them. When such situations arise, the container
    // op should be marked recursively legal here.
    SmallVector<Diagnostic> failures;
    {
      ScopedDiagnosticHandler diag(context,
                                   [&](Diagnostic &d) -> LogicalResult {
                                     failures.push_back(std::move(d));
                                     return success();
                                   });
      if (succeeded(applyPartialConversion(getOperation(), conversionTarget,
                                           std::move(conversionPatterns)))) {
        return;
      }
    }

    // Error fall-through. Attach all reported issues as notes.
    InFlightDiagnostic errorDiag =
        emitError(getOperation().getLoc())
        << "one or more illegal operations were found in the compiler input "
           "(are you missing an --iree-input-type= flag, or did you mean to "
           "pre-process through an IREE importer frontend?)";
    for (auto &failureDiag : failures) {
      Diagnostic &note = errorDiag.attachNote(failureDiag.getLocation());
      for (auto &arg : failureDiag.getArguments()) {
        note.append(arg);
      }
    }

    signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createVerifyCompilerMHLOInputLegality() {
  return std::make_unique<VerifyCompilerMHLOInputLegalityPass>();
}

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir
