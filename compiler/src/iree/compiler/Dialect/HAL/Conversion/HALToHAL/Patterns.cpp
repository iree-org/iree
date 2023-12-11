// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/HALToHAL/Patterns.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct ConvertExecutableCalculateWorkgroupsOp
    : public OpConversionPattern<IREE::HAL::ExecutableCalculateWorkgroupsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::ExecutableCalculateWorkgroupsOp calculateOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto exportOp =
        SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
            calculateOp, calculateOp.getEntryPoint());
    if (!exportOp) {
      return rewriter.notifyMatchFailure(calculateOp,
                                         "target entry point not found");
    }
    auto workgroupCount = exportOp.calculateWorkgroupCount(
        calculateOp.getLoc(), adaptor.getDevice(), adaptor.getWorkload(),
        rewriter);
    rewriter.replaceOp(calculateOp, workgroupCount);
    return success();
  }
};

} // namespace

void populateHALToHALPatterns(MLIRContext *context,
                              ConversionTarget &conversionTarget,
                              TypeConverter &typeConverter,
                              RewritePatternSet &patterns) {
  conversionTarget.addIllegalOp<IREE::HAL::ExecutableCalculateWorkgroupsOp>();
  patterns.insert<ConvertExecutableCalculateWorkgroupsOp>(typeConverter,
                                                          context);
}

} // namespace mlir::iree_compiler
