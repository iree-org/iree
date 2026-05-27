// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/ErrorCheckingTrackingListener.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-codegen-error-checking-tracking-listener"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::iree_compiler {

void ErrorCheckingTrackingListener::notifyPayloadReplacementNotFound(
    Operation *op, ValueRange values, DiagnosedSilenceableFailure &&diag) {
  // Certain ops can dropped safely.
  if (isa<scf::ForOp>(op)) {
    LLVM_DEBUG(DBGS() << "Silently dropping scf.for op mapping\n");
    return;
  }

  SmallVector<Diagnostic> diags;
  diag.takeDiagnostics(diags);
  if (!status.succeeded()) {
    status.takeDiagnostics(diags);
  }
  status = DiagnosedSilenceableFailure::silenceableFailure(std::move(diags));

  status = emitSilenceableFailure(
      getTransformOp(), "!!! tracking listener failed to find replacement op");
  status.attachNote(op->getLoc()) << "replaced op";
  for (Value v : values) {
    status.attachNote(v.getLoc()) << "replacement value";
  }
}

} // namespace mlir::iree_compiler
