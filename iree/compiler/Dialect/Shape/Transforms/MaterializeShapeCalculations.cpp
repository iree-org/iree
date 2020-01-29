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

#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

namespace {

// TODO(laurenzo): Replace this with a standard facility instead of a switch
// mess.
template <typename HloOp>
bool rewriteXlaBinaryElementwiseOpShape(Value &result,
                                        GetRankedShapeOp getShapeOp,
                                        Operation *inputOperation,
                                        OpBuilder &builder) {
  auto hlo_op = dyn_cast<HloOp>(inputOperation);
  if (!hlo_op) return false;

  if (hlo_op.broadcast_dimensions()) {
    // Has implicit broadcast - ignore for now.
    return false;
  }

  // No implicit broadcast. Tread as same element type.
  llvm::SmallVector<Value, 4> inputOperands(inputOperation->getOperands());
  result = buildCastInputsToResultShape(inputOperation->getLoc(),
                                        getShapeOp.getRankedShape(),
                                        inputOperands, builder);
  // Build may still have failed but match is successful.
  return true;
}

Value rewriteCustomOpShape(GetRankedShapeOp getShapeOp,
                           Operation *inputOperation, OpBuilder &builder) {
  // HLO binary elementwise ops.
  Value result;
  if (rewriteXlaBinaryElementwiseOpShape<xla_hlo::AddOp>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::Atan2Op>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::DivOp>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::MaxOp>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::MinOp>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::MulOp>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::PowOp>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::RemOp>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::ShiftLeftOp>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::ShiftRightArithmeticOp>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::ShiftRightLogicalOp>(
          result, getShapeOp, inputOperation, builder) ||
      rewriteXlaBinaryElementwiseOpShape<xla_hlo::SubOp>(
          result, getShapeOp, inputOperation, builder)) {
    return result;
  }

  return result;
}

class GetRankedShapePattern : public OpRewritePattern<Shape::GetRankedShapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(GetRankedShapeOp getShapeOp,
                                     PatternRewriter &rewriter) const override {
    // Check for static shape and elide.
    auto operandType =
        getShapeOp.operand().getType().dyn_cast<RankedTensorType>();
    auto shapeType = getShapeOp.shape().getType().dyn_cast<RankedShapeType>();
    if (operandType && shapeType && operandType.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<ConstRankedShapeOp>(getShapeOp, shapeType);
      return matchSuccess();
    }

    // Check for input operation.
    Operation *inputOperation = getShapeOp.operand().getDefiningOp();
    if (inputOperation) {
      return rewriteInputOp(getShapeOp, inputOperation, rewriter);
    }

    return matchFailure();
  }

 private:
  // Matches the case where the input to a GetRankedShapeOp is another
  // operation. This is the primary supported case as other rewrites should
  // have isolated function/block boundaries with TieShape ops.
  PatternMatchResult rewriteInputOp(GetRankedShapeOp getShapeOp,
                                    Operation *inputOperation,
                                    PatternRewriter &rewriter) const {
    // SameOperandsAndResultShape trait.
    if (inputOperation->hasTrait<OpTrait::SameOperandsAndResultShape>() ||
        inputOperation->hasTrait<OpTrait::SameOperandsAndResultType>()) {
      return rewriteSameOperandsAndResultShape(getShapeOp, inputOperation,
                                               rewriter);
    }

    auto customShape =
        rewriteCustomOpShape(getShapeOp, inputOperation, rewriter);
    if (customShape) {
      rewriter.replaceOp(getShapeOp, customShape);
      return matchSuccess();
    }

    return matchFailure();
  }

  PatternMatchResult rewriteSameOperandsAndResultShape(
      GetRankedShapeOp getShapeOp, Operation *inputOperation,
      PatternRewriter &rewriter) const {
    llvm::SmallVector<Value, 4> inputOperands(inputOperation->getOperands());
    auto combinedShapeOp = buildCastInputsToResultShape(
        inputOperation->getLoc(), getShapeOp.getRankedShape(), inputOperands,
        rewriter);
    if (!combinedShapeOp) return matchFailure();
    rewriter.replaceOp(getShapeOp, {combinedShapeOp});
    return matchSuccess();
  }
};

class MaterializeShapeCalculationsPass
    : public FunctionPass<MaterializeShapeCalculationsPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    // Always apply the canonicalizations for GetRankedShape.
    GetRankedShapeOp::getCanonicalizationPatterns(patterns, &getContext());
    CastCompatibleShapeOp::getCanonicalizationPatterns(patterns, &getContext());
    patterns.insert<GetRankedShapePattern>(&getContext());
    // TODO: apply repeatedly.
    applyPatternsGreedily(getFunction(), patterns);
  }
};

}  // namespace
}  // namespace Shape

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OpPassBase<FuncOp>> createMaterializeShapeCalculationsPass() {
  return std::make_unique<Shape::MaterializeShapeCalculationsPass>();
}

static PassRegistration<Shape::MaterializeShapeCalculationsPass> pass(
    "iree-shape-materialize-calculations", "Materializes shape calculations.");

}  // namespace iree_compiler
}  // namespace mlir
