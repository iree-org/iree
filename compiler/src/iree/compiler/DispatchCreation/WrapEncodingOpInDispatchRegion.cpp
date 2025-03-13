// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_WRAPENCODINGOPINDISPATCHREGIONPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

struct WrapEncodingOpInDispatchRegionPass
    : public impl::WrapEncodingOpInDispatchRegionPassBase<
          WrapEncodingOpInDispatchRegionPass> {

  void runOnOperation() override;
};

} // namespace

void WrapEncodingOpInDispatchRegionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  SmallVector<IREE::Encoding::SetEncodingOp> encodingOps;
  funcOp->walk([&](IREE::Encoding::SetEncodingOp encodingOp) {
    if (IREE::Flow::isNonNullAndOutsideDispatch(encodingOp)) {
      encodingOps.push_back(encodingOp);
    }
  });

  IRRewriter rewriter(context);
  for (auto encodingOp : encodingOps) {
    if (failed(IREE::Flow::wrapOpInDispatchRegion(rewriter, encodingOp))) {
      funcOp.emitOpError("failed to move encoding op into dispatch region");
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
