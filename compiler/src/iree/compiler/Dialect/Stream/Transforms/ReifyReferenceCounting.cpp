// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_REIFYREFERENCECOUNTINGPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-reify-reference-counting
//===----------------------------------------------------------------------===//

struct AsyncRetainOpPattern final
    : public OpRewritePattern<IREE::Stream::AsyncRetainOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::AsyncRetainOp retainOp,
                                PatternRewriter &rewriter) const override {
    rewriter.create<IREE::Stream::ResourceRetainOp>(
        retainOp.getLoc(), retainOp.getOperand(), retainOp.getOperandSize());
    rewriter.replaceOp(retainOp, {retainOp.getOperand()});
    return success();
  }
};

struct AsyncReleaseOpPattern final
    : public OpRewritePattern<IREE::Stream::AsyncReleaseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::AsyncReleaseOp releaseOp,
                                PatternRewriter &rewriter) const override {
    Value wasTerminal = rewriter.create<IREE::Stream::ResourceReleaseOp>(
        releaseOp.getLoc(), releaseOp.getOperand(), releaseOp.getOperandSize());

    auto ifOp = rewriter.create<scf::IfOp>(
        releaseOp.getLoc(), releaseOp.getResultTimepoint().getType(),
        wasTerminal,
        /*withElseRegion=*/true);

    {
      auto thenBuilder = ifOp.getThenBodyBuilder();
      auto deallocaOp = thenBuilder.create<IREE::Stream::ResourceDeallocaOp>(
          releaseOp.getLoc(), releaseOp.getOperand(),
          releaseOp.getOperandSize(), /*prefer_origin=*/true,
          releaseOp.getAwaitTimepoint(), releaseOp.getAffinityAttr());
      thenBuilder.create<scf::YieldOp>(releaseOp.getLoc(),
                                       deallocaOp.getResultTimepoint());
    }

    {
      auto elseBuilder = ifOp.getElseBodyBuilder();
      if (auto awaitTimepoint = releaseOp.getAwaitTimepoint()) {
        elseBuilder.create<scf::YieldOp>(releaseOp.getLoc(), awaitTimepoint);
      } else {
        Value immediateTimepoint =
            elseBuilder.create<IREE::Stream::TimepointImmediateOp>(
                releaseOp.getLoc());
        elseBuilder.create<scf::YieldOp>(releaseOp.getLoc(),
                                         immediateTimepoint);
      }
    }

    rewriter.replaceOp(releaseOp, {releaseOp.getOperand(), ifOp.getResult(0)});
    return success();
  }
};

struct ReifyReferenceCountingPass
    : public IREE::Stream::impl::ReifyReferenceCountingPassBase<
          ReifyReferenceCountingPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<AsyncRetainOpPattern>(context);
    patterns.add<AsyncReleaseOpPattern>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
