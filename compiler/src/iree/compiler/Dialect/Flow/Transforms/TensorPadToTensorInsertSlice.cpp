// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- TensorPadToInsertSlice.cpp ----- Pass to legalize tensor.pad -------===//
//
// Pass to convert tensor.pad to linalg.fill + tensor.insert_slice.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_TENSORPADTOTENSORINSERTSLICEPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {
/// Pattern to convert a tensor.tensor operation into a fill +
/// tensor.insert_slice. This is needed till tensor.pad op can be fused with its
/// consumers.
struct TensorPadOpConversion : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;
  TensorPadOpConversion(MLIRContext *context, bool skipSingleLinalgOpUses)
      : OpRewritePattern<tensor::PadOp>(context, skipSingleLinalgOpUses),
        skipSingleLinalgOpUses(skipSingleLinalgOpUses) {}

  LogicalResult matchAndRewrite(tensor::PadOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    // Check that the region is just a yield operation which is returning a
    // scalar that is not one of the arguments of the linalg operation.
    Region &region = padTensorOp.getRegion();
    Block &block = region.front();
    if (!llvm::hasSingleElement(block))
      return failure();
    auto yieldOp = cast<tensor::YieldOp>(block.getTerminator());
    Value yieldVal = yieldOp.getValue();
    if (llvm::any_of(block.getArguments(),
                     [&](Value v) { return v == yieldVal; })) {
      return failure();
    }

    if (padTensorOp->hasOneUse()) {
      Operation *use = padTensorOp->use_begin()->getOwner();
      if (skipSingleLinalgOpUses) {
        // TODO(#10312): Relax the condition to not check quantized ops. They
        // are going to be deprecated. We don't expect them being IREE's input.
        if (isa<linalg::LinalgOp>(use) &&
            !isa<linalg::Conv2DNhwcHwcfQOp, linalg::DepthwiseConv2DNhwcHwcQOp,
                 linalg::DepthwiseConv2DNhwcHwcmQOp>(use)) {
          return failure();
        }
      }
      // (pad + set_encoding) gets folded in to tensor.pack in the
      // MaterializeEncoding pass. Rewriting those pads into insert_slice would
      // defeat that.
      if (isa<IREE::Encoding::SetEncodingOp>(use)) {
        return failure();
      }
    }

    // Rewrite tensor.pad to tensor.empty + linalg.fill + tensor.insert_slice.
    return static_cast<LogicalResult>(
        linalg::rewriteInDestinationPassingStyle(rewriter, padTensorOp));
  }

private:
  // Option to skip the pattern when tensor.pad op has one use and is used by
  // a Linalg op.
  bool skipSingleLinalgOpUses = false;
};

struct TensorPadToTensorInsertSlicePass
    : public IREE::Flow::impl::TensorPadToTensorInsertSlicePassBase<
          TensorPadToTensorInsertSlicePass> {
  using IREE::Flow::impl::TensorPadToTensorInsertSlicePassBase<
      TensorPadToTensorInsertSlicePass>::TensorPadToTensorInsertSlicePassBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<TensorPadOpConversion>(context, skipSingleLinalgOpUses);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
