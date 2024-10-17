// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/Casting.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_VERIFYDISPATCHESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

struct VerifyDispatchesPass final
    : public impl::VerifyDispatchesPassBase<VerifyDispatchesPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void VerifyDispatchesPass::runOnOperation() {
  bool isError = false;
  auto funcOp = getOperation();
  // Keep track of if an op or any of its producers (that are inside a
  // dispatch) implement the `TilingInterface`
  llvm::DenseMap<Operation *, bool> isOpOrProducersTileable;
  funcOp.walk([&](Operation *op) {
    // Only check for ops directly nested in a dispatch
    IREE::Flow::DispatchRegionOp dispatchOp =
        op->getParentOfType<IREE::Flow::DispatchRegionOp>();
    if (!dispatchOp) {
      return;
    }

    bool isTileable = isa<TilingInterface>(op);
    bool hasTileableProducer = false;
    for (Value operand : op->getOperands()) {
      auto producer = operand.getDefiningOp();
      if (isOpOrProducersTileable[producer]) {
        hasTileableProducer = true;
        break;
      }
    }

    isOpOrProducersTileable[op] = isTileable || hasTileableProducer;
    // If this op is tileable, it won't cause any issues
    if (isTileable) {
      return;
    }

    bool hasTileableConsumer = false;
    for (Operation *user : op->getUsers()) {
      if (isa_and_nonnull<TilingInterface>(user)) {
        hasTileableConsumer = true;
      }
    }

    // If this op has a producer that is tileable and a consumer that is
    // tileable but isn't itsself tileable, then it will cause codegen issues
    if (hasTileableConsumer && hasTileableProducer) {
      auto error = dispatchOp->emitOpError(
          "contains a non-tileable op in a dispatch between tileable ops");
      error.attachNote(op->getLoc()) << " non-tileable op:\n";
      isError = true;
    }
  });

  if (isError) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
