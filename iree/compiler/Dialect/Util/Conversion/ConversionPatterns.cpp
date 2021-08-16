// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// Hints
//===----------------------------------------------------------------------===//

class PreserveDoNotOptimize
    : public OpConversionPattern<IREE::Util::DoNotOptimizeOp> {
 public:
  using OpConversionPattern<IREE::Util::DoNotOptimizeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::DoNotOptimizeOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Util::DoNotOptimizeOp>(op, operands,
                                                             op->getAttrs());
    return success();
  }
};

}  // namespace

void populateUtilConversionPatterns(MLIRContext *context,
                                    TypeConverter &typeConverter,
                                    OwningRewritePatternList &patterns) {
  patterns.insert<PreserveDoNotOptimize>(context);
}

void populateUtilConversionPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    OwningRewritePatternList &patterns) {
  conversionTarget.addDynamicallyLegalOp<IREE::Util::DoNotOptimizeOp>(
      [&](IREE::Util::DoNotOptimizeOp op) {
        return llvm::all_of(op.getResultTypes(), [&typeConverter](Type t) {
          return typeConverter.isLegal(t);
        });
      });

  populateUtilConversionPatterns(context, typeConverter, patterns);
}

}  // namespace iree_compiler
}  // namespace mlir
