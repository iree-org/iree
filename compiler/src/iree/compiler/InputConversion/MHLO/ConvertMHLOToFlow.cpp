// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/MHLO/ConvertMHLOToFlow.h"

#include <iterator>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

namespace {

struct ConstOpLowering : public OpRewritePattern<mhlo::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

}  // namespace

void setupDirectMHLOToFlowLegality(MLIRContext *context,
                                   ConversionTarget &conversionTarget) {
  conversionTarget.addIllegalOp<mhlo::ConstantOp>();
}

void populateMHLOToFlowPatterns(MLIRContext *context,
                                RewritePatternSet &patterns) {
  patterns.insert<ConstOpLowering>(context);
}

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir
