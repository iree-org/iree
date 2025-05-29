// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_IREEINJECTASSUMEALIGNMENTPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct InjectAssumeAlignmentForSubspanOp
    : public OpRewritePattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpRewritePattern<
      IREE::HAL::InterfaceBindingSubspanOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op,
                                PatternRewriter &rewriter) const override;
};

struct IREEInjectAssumeAlignmentPass final
    : public impl::IREEInjectAssumeAlignmentPassBase<
          IREEInjectAssumeAlignmentPass> {
  void runOnOperation() override;
  using Base::Base;
};
} // namespace

LogicalResult InjectAssumeAlignmentForSubspanOp::matchAndRewrite(
    IREE::HAL::InterfaceBindingSubspanOp op, PatternRewriter &rewriter) const {
  auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
  if (!resultType) {
    return rewriter.notifyMatchFailure(op, "result type is not a MemRefType");
  }
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  auto alignOp = rewriter.create<memref::AssumeAlignmentOp>(
      loc, op.getResult(), op.calculateAlignment().value());
  rewriter.replaceAllUsesExcept(op.getResult(), alignOp.getResult(), alignOp);
  return success();
}

void IREEInjectAssumeAlignmentPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<InjectAssumeAlignmentForSubspanOp>(&getContext());
  walkAndApplyPatterns(getOperation(), std::move(patterns));
}

} // namespace mlir::iree_compiler
