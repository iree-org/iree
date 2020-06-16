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

#include "iree/compiler/Dialect/Flow/Conversion/StandardToFlow/ConvertStandardToFlow.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ExtractElementOpLowering : public OpRewritePattern<ExtractElementOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ExtractElementOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = op.getAggregate().getType().dyn_cast<TensorType>();
    if (!tensorType) {
      return rewriter.notifyMatchFailure(op, "expected tensor types");
    }
    // tensor<i1> is not valid to load, it needs to be converted to i8 or
    // something else instead.
    if (tensorType.getElementTypeBitWidth() == 1) {
      return rewriter.notifyMatchFailure(op, "expected non-i1 type");
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorLoadOp>(
        op, tensorType.getElementType(), op.aggregate(), op.indices());
    return success();
  }
};

}  // namespace

void setupDirectStandardToFlowLegality(MLIRContext *context,
                                       ConversionTarget &conversionTarget) {
  conversionTarget.addDynamicallyLegalOp<ExtractElementOp>(
      [](ExtractElementOp op) {
        return !op.getAggregate().getType().isa<TensorType>();
      });
}

void populateStandardToFlowPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns) {
  patterns.insert<ExtractElementOpLowering>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
