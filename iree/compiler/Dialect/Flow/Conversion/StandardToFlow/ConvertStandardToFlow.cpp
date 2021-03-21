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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// tensor::ExtractOp will be lowered to IREE::Flow::TensorLoadOp. If the type
/// is i1, it's not valid to load. In this case, we need to cast it to i8 before
/// the load, and truncate the value after the load.
struct ExtractElementOpLowering
    : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    // tensor<i1> is not valid to load, it needs to be converted to i8 or
    // something else instead.
    auto tensorType = op.tensor().getType().cast<TensorType>();
    if (tensorType.getElementType().isInteger(1)) {
      auto i1Type = rewriter.getI1Type();
      auto i8Type = rewriter.getIntegerType(8);
      auto convertedOperand = rewriter.createOrFold<ZeroExtendIOp>(
          op.getLoc(), args[0],
          RankedTensorType::get(tensorType.getShape(), i8Type));
      auto i8Value = rewriter.createOrFold<IREE::Flow::TensorLoadOp>(
          op.getLoc(), i8Type, convertedOperand, op.indices());
      rewriter.replaceOpWithNewOp<TruncateIOp>(op, i1Type, i8Value);
    } else {
      rewriter.replaceOpWithNewOp<IREE::Flow::TensorLoadOp>(
          op, tensorType.getElementType(), op.tensor(), op.indices());
    }
    return success();
  }
};

}  // namespace

void setupDirectStandardToFlowLegality(MLIRContext *context,
                                       ConversionTarget &conversionTarget) {
  conversionTarget.addIllegalOp<tensor::ExtractOp>();
}

void populateStandardToFlowPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns) {
  patterns.insert<ExtractElementOpLowering>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
