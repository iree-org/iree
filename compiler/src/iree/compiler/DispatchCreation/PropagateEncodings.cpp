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
  auto encoding = dyn_cast<IREE::Encoding::MatmulKAttr>(
      encodingOp.getResultType().getEncoding());
  if (!encoding) {
    return failure();
  }
  auto collapseOp =
      encodingOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();
  if (!collapseOp) {
    return failure();
  }
  if (!(IREE::Flow::isNonNullAndOutsideDispatch(encodingOp) &&
        IREE::Flow::isNonNullAndOutsideDispatch(collapseOp))) {
    return failure();
  }

  SmallVector<int32_t> kDims(encoding.getKDims().asArrayRef());
  llvm::SetVector<int32_t> kDimsSet(kDims.begin(), kDims.end());

  // Get a mapping from original iteration space to expanded iteration space.
  MLIRContext *ctx = rewriter.getContext();
  unsigned numDims = 0;
  SmallVector<int32_t> newKDims;
  for (auto [idx, reassociation] :
       llvm::enumerate(collapseOp.getReassociationIndices())) {
    if (kDimsSet.count(idx)) {
      newKDims.append(llvm::to_vector(
          llvm::seq<int32_t>(numDims, numDims + reassociation.size())));
    }
    numDims += reassociation.size();
  }

  auto newEncodingAttr = IREE::Encoding::MatmulKAttr::get(
      ctx, encoding.getOperandIndex(), encoding.getElementTypes(),
      rewriter.getDenseI32ArrayAttr(newKDims), encoding.getLayouts());

  // Create the new encoding op.
  RankedTensorType newEncodingType =
      collapseOp.getSrcType().cloneWithEncoding(newEncodingAttr);
  Value newEncodingOp = rewriter.create<IREE::Encoding::SetEncodingOp>(
      encodingOp.getLoc(), newEncodingType, collapseOp.getSrc());
  Value newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
      collapseOp.getLoc(), encodingOp.getResultType(), newEncodingOp,
      collapseOp.getReassociationIndices());
  rewriter.replaceOp(encodingOp, newCollapseOp);
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
