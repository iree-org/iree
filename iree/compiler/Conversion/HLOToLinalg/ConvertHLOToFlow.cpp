// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/HLOToLinalg/ConvertHLOToFlow.h"

#include <iterator>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ConstOpLowering : public OpRewritePattern<mhlo::ConstOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConstOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.value());
    return success();
  }
};

}  // namespace

void setupDirectHLOToFlowLegality(MLIRContext *context,
                                  ConversionTarget &conversionTarget) {
  conversionTarget.addIllegalOp<mhlo::ConstOp>();
}

void populateHLOToFlowPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns) {
  patterns.insert<ConstOpLowering>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
