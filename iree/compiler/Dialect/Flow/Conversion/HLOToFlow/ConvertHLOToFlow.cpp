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
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ConstOpLowering : public OpRewritePattern<xla_hlo::ConstOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(xla_hlo::ConstOp op,
                                     PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.value());
    return matchSuccess();
  }
};

struct DynamicUpdateSliceOpLowering
    : public OpRewritePattern<xla_hlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(xla_hlo::DynamicUpdateSliceOp op,
                                     PatternRewriter &rewriter) const override {
    auto startIndices = llvm::to_vector<4>(
        llvm::map_range(op.start_indices(), [&](Value *tensorValue) {
          return rewriter.createOrFold<ExtractElementOp>(op.getLoc(),
                                                         tensorValue);
        }));
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorUpdateOp>(
        op, op.getResult()->getType(), op.update(), op.operand(), startIndices);
    return matchSuccess();
  }
};

}  // namespace

void setupDirectHLOToFlowLegality(MLIRContext *context,
                                  ConversionTarget &conversionTarget) {
  conversionTarget
      .addIllegalOp<xla_hlo::ConstOp, xla_hlo::DynamicUpdateSliceOp>();
}

void populateHLOToFlowPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns) {
  patterns.insert<ConstOpLowering, DynamicUpdateSliceOpLowering>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
