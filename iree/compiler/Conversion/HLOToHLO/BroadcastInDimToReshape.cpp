// Copyright 2020 Google LLC
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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

namespace {
// Rewrites 1x1x1... brodcast_in_dim as reshape
class BroadcastInDimToReshape
    : public OpRewritePattern<mhlo::BroadcastInDimOp> {
 public:
  using OpRewritePattern<mhlo::BroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto resultShapeType =
        op.getResult().getType().dyn_cast_or_null<RankedTensorType>();
    auto inputShapeType =
        op.getOperand().getType().dyn_cast_or_null<RankedTensorType>();
    if (!resultShapeType || !inputShapeType) return failure();
    auto resultShape = resultShapeType.getShape();
    auto inputShape = inputShapeType.getShape();
    const int numDims = resultShape.size() - inputShape.size();
    for (int i = 0; i < numDims; ++i) {
      if (resultShape[i] != 1) return failure();
    }

    int64_t resultSize = 1, inputSize = 1;
    for (int i = 0; i < resultShape.size(); ++i) resultSize *= resultShape[i];
    for (int i = 0; i < inputShape.size(); ++i) inputSize *= inputShape[i];
    if (resultSize != inputSize) return failure();

    rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, op.getResult().getType(),
                                                 op.getOperand());
    return success();
  }
};
}  // namespace

// TODO(ataei): Consider upstream this pass as a canonicalization pass.
struct BroadcastInDimToReshapePass
    : public PassWrapper<BroadcastInDimToReshapePass, FunctionPass> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<BroadcastInDimToReshape>(context);
    applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};

std::unique_ptr<OperationPass<FuncOp>> createBroadcastInDimToReshapePass() {
  return std::make_unique<BroadcastInDimToReshapePass>();
}

static PassRegistration<BroadcastInDimToReshapePass> pass(
    "iree-codegen-broadcast-in-dim-to-reshape",
    "Rewrites mhlo.broadcast_in_dim as reshape when all added outer dims are "
    "1 and size doesn't change.");

}  // namespace iree_compiler
}  // namespace mlir
