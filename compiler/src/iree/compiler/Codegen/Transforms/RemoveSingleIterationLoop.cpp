// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- RemoveSingleIterationLoop.cpp - Remove single iteration loops ------===//
//
// Removes loops that are known to be single-trip count even when the loop
// itself might be distributed.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-codegen-remove-single-iteration"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir::iree_compiler {

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, scf::ForOp op,
                                ValueRange blockArgs = {}) {
  Block *block = op.getBody();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Same as `replaceOpWithRegion` function but within an scf.if region.
static void replaceForWithIf(PatternRewriter &rewriter, scf::ForOp op,
                             ValueRange blockArgs = {}) {
  Block *block = op.getBody();
  ValueRange initArgs = op.getInitArgs();
  Value count =
      arith::CmpIOp::create(rewriter, op->getLoc(), arith::CmpIPredicate::sgt,
                            op.getUpperBound(), op.getLowerBound());
  auto ifOp =
      scf::IfOp::create(rewriter, op->getLoc(), op.getResultTypes(), count,
                        /*withElseRegion=*/initArgs.size() != 0);
  Operation *terminator = block->getTerminator();
  rewriter.inlineBlockBefore(block, &ifOp.getThenRegion().front(),
                             ifOp.getThenRegion().front().begin(), blockArgs);
  if (initArgs.size() == 0) {
    rewriter.eraseOp(terminator);
  } else {
    rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    scf::YieldOp::create(rewriter, ifOp.getLoc(), initArgs);
  }
  rewriter.replaceOp(op, ifOp);
}

namespace {
/// Rewriting pattern that replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<scf::ForOp> {

  SimplifyTrivialLoops(MLIRContext *context, ForControlFnRef controlFn)
      : OpRewritePattern(context), controlFn(controlFn) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (controlFn && !controlFn(op)) {
      return rewriter.notifyMatchFailure(
          op, "doesn't match according to the the control function");
    }
    if (!neverRunsSecondIteration(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "is not a single-iteration for loop");
    }
    // The second iteration is never run so the loop atmost can have 1
    // iteration. Inline its body and remove the loop.
    SmallVector<Value> blockArgs;
    blockArgs.reserve(op.getInitArgs().size() + 1);
    blockArgs.push_back(op.getLowerBound());
    llvm::append_range(blockArgs, op.getInitArgs());
    if (alwaysRunsFirstIteration(op)) {
      replaceOpWithRegion(rewriter, op, blockArgs);
    } else {
      replaceForWithIf(rewriter, op, blockArgs);
    }
    return success();
  }

private:
  ForControlFnRef controlFn;
};

} // namespace

void populateRemoveSingleIterationLoopPattern(RewritePatternSet &patterns,
                                              ForControlFnRef controlFn) {
  patterns.add<SimplifyTrivialLoops>(patterns.getContext(), controlFn);
}

} // namespace mlir::iree_compiler
