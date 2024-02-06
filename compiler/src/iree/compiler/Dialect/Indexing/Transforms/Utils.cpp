// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Indexing/Transforms/Utils.h"

#include "iree/compiler/Dialect/Indexing/IR/IndexingOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Indexing {

//===----------------------------------------------------------------------===//
// Common patterns
//===----------------------------------------------------------------------===//

namespace {

/// Concretizes tensor.pad op's result shape if its source op implements
/// OffsetSizeAndStrideOpInterface. For example, pad(extract_slice).
struct StripAssertAlignedRangeOpPattern final
    : public OpRewritePattern<AssertAlignedRangeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AssertAlignedRangeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getValue());
    return success();
  }
};
} // namespace

void populateStripIndexingAssertionPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns) {
  patterns.insert<StripAssertAlignedRangeOpPattern>(context);
}

} // namespace mlir::iree_compiler::IREE::Indexing
