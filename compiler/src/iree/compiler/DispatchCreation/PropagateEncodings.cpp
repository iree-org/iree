// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-propagate-encodings"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_PROPAGATEENCODINGSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

// TODO(#20179): Support the propagation through interfaces. It is supposed to
// be done with data-flow analysis.
struct PropagateEncodingsPass
    : public DispatchCreation::impl::PropagateEncodingsPassBase<
          PropagateEncodingsPass> {
  void runOnOperation() override;
};

} // namespace

void PropagateEncodingsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *ctx = &getContext();
  RewritePatternSet propagationPatterns(ctx);
  // TODO(#21970): Add patterns to the pass. The pass is not used currently, but
  // it is a placeholder for future changes. So it is not deleted atm.
  GreedyRewriteConfig config;
  config.enableFolding(true).enableConstantCSE(false);
  if (failed(applyPatternsGreedily(funcOp, std::move(propagationPatterns),
                                   config))) {
    funcOp.emitOpError("failed to propagate encodings");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
