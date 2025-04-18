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

/// Pattern to swap `tensor.collapse_shape` -> `iree_encoding.set_encoding`
struct SwapEncodingOpWithTensorCollapseShapeOp
    : public OpRewritePattern<IREE::Encoding::SetEncodingOp> {
  using Base = OpRewritePattern<IREE::Encoding::SetEncodingOp>;
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override;
};

// TODO(#20179): Support the propagation through interfaces. It is supposed to
// be done with data-flow analysis.
struct PropagateEncodingsPass
    : public DispatchCreation::impl::PropagateEncodingsPassBase<
          PropagateEncodingsPass> {
  void runOnOperation() override;
};

} // namespace

LogicalResult SwapEncodingOpWithTensorCollapseShapeOp::matchAndRewrite(
    IREE::Encoding::SetEncodingOp encodingOp, PatternRewriter &rewriter) const {

  Value target = encodingOp.getSource();

  auto encoding = dyn_cast<IREE::Encoding::MatmulKAttr>(
      encodingOp.getResultType().getEncoding());

  auto attrPropInterface =
      llvm::dyn_cast_or_null<IREE::Encoding::EncodingPropagationAttrInterface>(
          encoding);

  if (!attrPropInterface || !attrPropInterface.isPropagable(target)) {
    return failure();
  }

  // Get the encoding attributes for the operands and results of the operation.
  FailureOr<IREE::Encoding::PropagationEncoding> propEncodings =
      attrPropInterface.generateEncodings(target);

  if (failed(propEncodings)) {
    return failure();
  }
  auto collapseOp =
      encodingOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();

  if (!collapseOp) {
    return rewriter.notifyMatchFailure(encodingOp,
                                       "expected a collapse_shape producer");
  }

  if (!IREE::Flow::isNonNullAndOutsideDispatch(encodingOp) ||
      !IREE::Flow::isNonNullAndOutsideDispatch(collapseOp)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "expected that both operations are outside dispatch");
  }

  auto propResult =
      llvm::dyn_cast_or_null<IREE::Encoding::EncodingPropagationOpInterface>(
          collapseOp.getOperation());

  if (!propResult) {
    return failure();
  }

  // Propagate the set encoding and generate the new encoding operations.
  FailureOr<IREE::Encoding::PropagationResult> maybeResult =
      propResult.propagateEncoding(rewriter, *propEncodings,
                                   cast<OpResult>(encodingOp.getSource()));
  if (failed(maybeResult)) {
    return failure();
  }

  rewriter.replaceOp(encodingOp, maybeResult->replacement);
  return success();
}

void PropagateEncodingsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *ctx = &getContext();
  RewritePatternSet propagationPatterns(ctx);
  propagationPatterns.insert<SwapEncodingOpWithTensorCollapseShapeOp>(ctx);
  GreedyRewriteConfig config;
  config.fold = true;
  config.cseConstants = false;
  if (failed(applyPatternsGreedily(funcOp, std::move(propagationPatterns),
                                   config))) {
    funcOp.emitOpError("failed to propagate encodings");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
