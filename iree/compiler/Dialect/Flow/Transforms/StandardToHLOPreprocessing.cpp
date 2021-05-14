// Copyright 2021 Google LLC
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

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

class SelectOpConverter : public OpRewritePattern<SelectOp> {
 public:
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto extractOp = op.condition().getDefiningOp<tensor::ExtractOp>();
    if (!extractOp) return failure();
    Value cond = extractOp.tensor();
    if (cond.getType().cast<RankedTensorType>().getRank() > 0) return failure();

    auto type = op.true_value().getType().cast<RankedTensorType>();
    if (!type.hasStaticShape()) return failure();

    if (type.getRank() > 0) {
      auto shape = type.getShape();
      auto condType = RankedTensorType::get(shape, rewriter.getI1Type());
      auto shapeAttr = DenseIntElementsAttr::get(
          RankedTensorType::get(shape.size(), rewriter.getI64Type()), shape);
      cond = rewriter.create<mhlo::BroadcastOp>(op.getLoc(), condType, cond,
                                                shapeAttr);
    }
    rewriter.replaceOpWithNewOp<mhlo::SelectOp>(op, cond, op.true_value(),
                                                op.false_value());
    return success();
  }
};

struct StandardToHLOPreprocessingPass
    : public StandardToHLOPreprocessingBase<StandardToHLOPreprocessingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<mhlo::MhloDialect, tensor::TensorDialect, StandardOpsDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<SelectOpConverter>(
        context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createStandardToHLOPreprocessingPass() {
  return std::make_unique<StandardToHLOPreprocessingPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
