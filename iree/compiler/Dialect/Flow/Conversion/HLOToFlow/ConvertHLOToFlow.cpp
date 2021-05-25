// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/HLOToFlow/ConvertHLOToFlow.h"

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

struct DynamicUpdateSliceOpLowering
    : public OpRewritePattern<mhlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DynamicUpdateSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto startIndices = llvm::to_vector<4>(
        llvm::map_range(op.start_indices(), [&](Value tensorValue) {
          return rewriter.createOrFold<IndexCastOp>(
              op.getLoc(),
              rewriter.createOrFold<tensor::ExtractOp>(op.getLoc(),
                                                       tensorValue),
              rewriter.getIndexType());
        }));
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorUpdateOp>(
        op, op.operand(), startIndices, op.update());
    return success();
  }
};

}  // namespace

void setupDirectHLOToFlowLegality(MLIRContext *context,
                                  ConversionTarget &conversionTarget) {
  conversionTarget.addIllegalOp<mhlo::ConstOp, mhlo::DynamicUpdateSliceOp>();
}

void populateHLOToFlowPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns) {
  patterns.insert<ConstOpLowering, DynamicUpdateSliceOpLowering>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
