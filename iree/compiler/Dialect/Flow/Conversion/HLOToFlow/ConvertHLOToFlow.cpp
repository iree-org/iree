// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/Flow/Conversion/HLOToFlow/ConvertHLOToFlow.h"

#include <iterator>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

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
              rewriter.createOrFold<ExtractElementOp>(op.getLoc(), tensorValue),
              rewriter.getIndexType());
        }));
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorUpdateOp>(
        op, op.getResult().getType(), op.update(), op.operand(), startIndices);
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
