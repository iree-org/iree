// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Conversion/PreserveCompilerHints.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

namespace {
class PreserveDoNotOptimize : public OpConversionPattern<DoNotOptimizeOp> {
 public:
  using OpConversionPattern<DoNotOptimizeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      DoNotOptimizeOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DoNotOptimizeOp>(op, operands, op->getAttrs());
    return success();
  }
};
}  // namespace

void setupCompilerHintsLegality(MLIRContext *context, ConversionTarget &target,
                                TypeConverter &typeConverter) {
  target.addDynamicallyLegalOp<DoNotOptimizeOp>([&](DoNotOptimizeOp op) {
    return llvm::all_of(op.getResultTypes(), [&typeConverter](Type t) {
      return typeConverter.isLegal(t);
    });
  });
}

void populatePreserveCompilerHintsPatterns(MLIRContext *context,
                                           OwningRewritePatternList &patterns) {
  patterns.insert<PreserveDoNotOptimize>(context);
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
