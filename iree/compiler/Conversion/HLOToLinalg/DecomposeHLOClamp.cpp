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
/// A pass to decompose mhlo.clamp ops into mhlo.compare and
/// mhlo.select ops.
class DecomposeClampOp : public OpRewritePattern<mhlo::ClampOp> {
 public:
  using OpRewritePattern<mhlo::ClampOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ClampOp op,
                                PatternRewriter &rewriter) const override {
    auto minType = op.min().getType().dyn_cast<RankedTensorType>();
    auto operandType = op.operand().getType().dyn_cast<RankedTensorType>();
    auto maxType = op.max().getType().dyn_cast<RankedTensorType>();

    if (!operandType) return failure();

    // Reject implicitly broadcasted cases. They should be made explicit first.
    if (minType != operandType || maxType != operandType) return failure();

    // clamp(a, x, b) = min(max(a, x), b)
    Location loc = op.getLoc();
    Value cmpMin = rewriter.create<mhlo::CompareOp>(
        loc, op.min(), op.operand(), rewriter.getStringAttr("LT"));
    Value selectMin = rewriter.create<mhlo::SelectOp>(loc, operandType, cmpMin,
                                                      op.operand(), op.min());
    Value cmpMax = rewriter.create<mhlo::CompareOp>(
        loc, selectMin, op.max(), rewriter.getStringAttr("LT"));
    Value selectMax = rewriter.create<mhlo::SelectOp>(loc, operandType, cmpMax,
                                                      selectMin, op.max());
    rewriter.replaceOp(op, selectMax);
    return success();
  }
};

struct DecomposeHLOClampPass
    : public PassWrapper<DecomposeHLOClampPass, FunctionPass> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<DecomposeClampOp>(context);
    applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDecomposeHLOClampPass() {
  return std::make_unique<DecomposeHLOClampPass>();
}

static PassRegistration<DecomposeHLOClampPass> pass(
    "iree-codegen-decompose-hlo-clamp",
    "Decompose HLO clamp op into primitive ops");
}  // namespace iree_compiler
}  // namespace mlir
