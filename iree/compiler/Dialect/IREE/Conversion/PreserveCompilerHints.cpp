// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/IREE/Conversion/PreserveCompilerHints.h"

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {
class PreserveDoNotOptimize
    : public OpConversionPattern<IREE::DoNotOptimizeOp> {
 public:
  using OpConversionPattern<IREE::DoNotOptimizeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::DoNotOptimizeOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::DoNotOptimizeOp>(op, operands,
                                                       op->getAttrs());
    return success();
  }
};
}  // namespace

void setupCompilerHintsLegality(MLIRContext *context, ConversionTarget &target,
                                TypeConverter &typeConverter) {
  target.addDynamicallyLegalOp<IREE::DoNotOptimizeOp>(
      [&](IREE::DoNotOptimizeOp op) {
        return llvm::all_of(op.getResultTypes(), [&typeConverter](Type t) {
          return typeConverter.isLegal(t);
        });
      });
}

void populatePreserveCompilerHintsPatterns(MLIRContext *context,
                                           OwningRewritePatternList &patterns) {
  patterns.insert<PreserveDoNotOptimize>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
