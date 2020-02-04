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

#include <numeric>

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {
namespace {

// Returns a 1-d i64 elements attribute populated with numbers from start to
// end, excluding.
static DenseIntElementsAttr getI64ElementsAttrForSeq(int start, int end,
                                                     Builder &builder) {
  int size = end - start;

  SmallVector<int64_t, 4> vals;
  vals.resize(size);
  std::iota(vals.begin(), vals.end(), start);

  TensorType ty = RankedTensorType::get({size}, builder.getIntegerType(64));
  return DenseIntElementsAttr::get(ty, vals);
}

// Returns true if a given HLO elementwise op does not broadcast.
template <typename HloOpTy>
bool IsSameRankedTypeBinaryElementwiseOp(HloOpTy op) {
  if (op.broadcast_dimensions()) {
    // Has intra-operand broadcast.
    return false;
  }
  auto lhsType = op.lhs().getType().template dyn_cast<RankedTensorType>();
  auto rhsType = op.rhs().getType().template dyn_cast<RankedTensorType>();
  if (!lhsType || !rhsType) return false;

  return lhsType == rhsType;
}

// Converts a broadcasted binary elementwise HLO to have explicit broadcasting.
template <typename HloOpTy>
class BroadcastedRankedBinaryElementwiseConversion
    : public OpConversionPattern<HloOpTy> {
  using OpConversionPattern<HloOpTy>::OpConversionPattern;
  using ConversionPattern::matchFailure;
  using ConversionPattern::matchSuccess;

  PatternMatchResult matchAndRewrite(
      HloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto lhs = operands[0];
    auto rhs = operands[1];
    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getOperation()
                          ->getResultTypes()[0]
                          .template dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType || !resultType) {
      // This conversion only supports ranked.
      return matchFailure();
    }

    // Get the shapes of the operands. Note that we assume that a prior shape
    // inference pass has appropriately specialized the shapes and we use them
    // as-is versus recomputing the broadcast.
    auto lhsShape = rewriter.create<GetRankedShapeOp>(op.getLoc(), lhs);
    auto rhsShape = rewriter.create<GetRankedShapeOp>(op.getLoc(), rhs);
    auto resultShapeType =
        RankedShapeType::get(resultType.getShape(), rewriter.getIndexType());
    auto resultShapeDims = resultShapeType.getAllDims();

    // Rank broadcast as appropriate.
    Value broadcastedLhs = lhs;
    Value broadcastedRhs = rhs;
    DenseIntElementsAttr lhsBroadcastDims;
    DenseIntElementsAttr rhsBroadcastDims;
    if (op.broadcast_dimensions()) {
      auto lhsRank = lhsType.getRank();
      auto rhsRank = rhsType.getRank();
      auto higherRankBroadcastDims =
          getI64ElementsAttrForSeq(0, std::max(lhsRank, rhsRank), rewriter);
      if (lhsRank > rhsRank) {
        lhsBroadcastDims = higherRankBroadcastDims;
        rhsBroadcastDims = *op.broadcast_dimensions();
      } else if (rhsRank > lhsRank) {
        lhsBroadcastDims = *op.broadcast_dimensions();
        rhsBroadcastDims = higherRankBroadcastDims;
      } else {
        op.emitOpError() << "broadcast_dimensions implies rank broadcast "
                         << "but operands are of the same rank";
        return matchFailure();
      }
    } else if (lhsType != rhsType) {
      op.emitError() << "degenerate broadcast of same-rank operands "
                     << "not yet implemented";
      return matchFailure();
    }

    auto resultShape = rewriter.create<RankedBroadcastShapeOp>(
        op.getLoc(), resultShapeType, lhsShape, rhsShape, lhsBroadcastDims,
        rhsBroadcastDims);
    broadcastedLhs = rewriter.create<RankedBroadcastInDimOp>(
        op.getLoc(),
        RankedTensorType::get(resultShapeDims, lhsType.getElementType()),
        broadcastedLhs, resultShape, lhsBroadcastDims);
    broadcastedRhs = rewriter.create<RankedBroadcastInDimOp>(
        op.getLoc(),
        RankedTensorType::get(resultShapeDims, rhsType.getElementType()),
        broadcastedRhs, resultShape, rhsBroadcastDims);

    auto newOp = rewriter.create<HloOpTy>(
        op.getLoc(), resultType, broadcastedLhs, broadcastedRhs, nullptr);
    rewriter.replaceOp(op, {newOp});
    return matchSuccess();
  }
};

class ConvertHLOToShapePass : public FunctionPass<ConvertHLOToShapePass> {
  void runOnFunction() override {
    ConversionTarget conversionTarget(getContext());
    OwningRewritePatternList conversionPatterns;

    conversionTarget.addLegalDialect<ShapeDialect>();
    conversionTarget.addLegalDialect<StandardOpsDialect>();
    conversionTarget.addLegalDialect<xla_hlo::XlaHloDialect>();

#define CONVERT_BINARY_ELEMENTWISE_OP(HloOpTy)                             \
  conversionTarget.addDynamicallyLegalOp<HloOpTy>(                         \
      [](HloOpTy op) { return IsSameRankedTypeBinaryElementwiseOp(op); }); \
  conversionPatterns                                                       \
      .insert<BroadcastedRankedBinaryElementwiseConversion<HloOpTy>>(      \
          &getContext());

    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::AddOp);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::Atan2Op);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::DivOp);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::MaxOp);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::MinOp);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::MulOp);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::PowOp);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::RemOp);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::ShiftLeftOp);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::ShiftRightArithmeticOp);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::ShiftRightLogicalOp);
    CONVERT_BINARY_ELEMENTWISE_OP(xla_hlo::SubOp);

    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace Shape

// Converts shape-sensitive HLOs to be based on facilities in the shape
// dialect.
std::unique_ptr<OpPassBase<FuncOp>> createConvertHLOToShapePass() {
  return std::make_unique<Shape::ConvertHLOToShapePass>();
}

static PassRegistration<Shape::ConvertHLOToShapePass> pass(
    "iree-shape-convert-hlo",
    "Converts dynamic shape dependent HLO ops to shaped variants.");

}  // namespace iree_compiler
}  // namespace mlir
