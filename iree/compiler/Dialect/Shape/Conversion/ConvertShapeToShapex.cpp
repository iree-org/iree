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

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {
namespace {

// This conversion is currently quite limited, such as not handling multiple
// basic blocks in general, due to doing a type conversion that the MLIR core
// conversion infra doesn't handle well.
//
// In particular, we convert `!shape.shape` to `!shapex.ranked_shape<...>`, but
// the contents of the `...` are context-dependent. Thus, one could say that
// this pass does a context-dependent type conversion.
//
// The current MLIR conversion infra doesn't handle context-dependent type
// conversions.
//
// I can see two solutions:
//
// 1. Extend the MLIR conversion infra to better support context-dependent type
// conversions. One way to do this would be for the conversion infra to convert
// blocks in RPO and use the type of the converted successor operand in a
// dominating predecessor as the type for the block argument when converting a
// block. A similar thing could be done with an RPO traversal of the callgraph.
// This algorithm wouldn't work in the presence of recursively dead cycles. And
// of course linkage boundaries cannot have a context-dependent type conversion
// (by definition).
//
// 2. Avoid needing to convert to !shapex.ranked_shape in the first place. This
// could be accomplished by generalizing !shape.shape to be able to support the
// use case of !shapex.ranked_shape. One important requirement here is that
// !shapex.ranked_shape models a partially-specified shape (hardcoded for the
// ranked case). !shape.shape could be extended to capture partially-specified
// shapes in the type, such as allowing `!shape.shape<*>` to model an unranked
// shape (which is the default; no information), `!shape.shape<?x?x5x?>` to
// model a rank-4 shape with dimension 2 being of extent 5, etc.
//
// Once we have this, we could do this lowering from generic !shape.shape to
// statically-known ranked shapes more progressively and treat it more like a
// type refinement algorithm.
//
// The main risk is that we are trying to shove too much stuff into the
// !shape.shape type. There's a risk that "progressive lowering" becomes "no
// clear boundaries" and we end up with code deep into the compiler continuously
// needing to doublecheck that the !shape.shape's at this point are in fact
// statically known to be ranked, or silently making that assumption and
// triggering assertions on verifier-valid IR. Pipelines and legalization
// targets could make these assertions not fire in practice, but it would
// be a maintenance burden.

class ConvertConstShapeOp : public OpConversionPattern<shape::ConstShapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      shape::ConstShapeOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t, 4> extents;
    for (APInt extent : op.shape()) {
      extents.push_back(extent.getZExtValue());
    }
    auto rsType = RankedShapeType::get(extents, rewriter.getContext());
    rewriter.replaceOpWithNewOp<ConstRankedShapeOp>(op, rsType);
    return success();
  }
};

class ConvertShapeOfOp : public OpConversionPattern<shape::ShapeOfOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      shape::ShapeOfOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto tensorType = operands[0].getType().dyn_cast<RankedTensorType>();
    if (!tensorType) {
      return failure();
    }
    auto resultType =
        RankedShapeType::get(tensorType.getShape(), rewriter.getContext());
    // TODO(jpienaar): The following needs to be re-evaluated once the patch
    // train from 2020/07/23 integrates properly. This is required to make
    // it forward and backwards compatible. Also, tests need to be added once
    // upstream integrates (and this can be tested).
    // rewriter.replaceOpWithNewOp<Shape::GetRankedShapeOp>(op, resultType,
    //                                                      operands[0]);
    auto getRanked = rewriter.create<Shape::GetRankedShapeOp>(
        op.getLoc(), resultType, operands[0]);

    // For FromExtentTensorOp users, just forward the result from GetRanked.
    SmallPtrSet<Operation *, 2> toDelete;
    for (auto use : op.getOperation()->getUsers()) {
      if (isa<FromExtentTensorOp>(use)) {
        use->replaceAllUsesWith(getRanked);
        toDelete.insert(use);
      }
    }
    for (Operation *use : toDelete) {
      rewriter.eraseOp(use);
    }

    rewriter.replaceOp(op.getOperation(), getRanked.getResult());
    return success();
  }
};

class ConvertSplitAtOp : public OpConversionPattern<shape::SplitAtOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      shape::SplitAtOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IntegerAttr indexAttr;
    if (!matchPattern(op.index(), m_Constant(&indexAttr))) {
      return rewriter.notifyMatchFailure(op, "requires constant `index`");
    }
    auto rank = operands[0].getType().cast<RankedShapeType>().getRank();
    int64_t index = indexAttr.getInt();
    if (index < 0) {
      index += rank;
    }
    auto head_indices = llvm::to_vector<4>(llvm::seq<int64_t>(0, index));
    auto tail_indices = llvm::to_vector<4>(llvm::seq<int64_t>(index, rank));
    Value head = rewriter.create<GatherExtentsOp>(
        op.getLoc(), operands[0], rewriter.getI64TensorAttr(head_indices));
    Value tail = rewriter.create<GatherExtentsOp>(
        op.getLoc(), operands[0], rewriter.getI64TensorAttr(tail_indices));
    rewriter.replaceOp(op, {head, tail});
    return success();
  }
};

class ConvertBroadcastOp : public OpConversionPattern<shape::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      shape::BroadcastOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Value lhs = operands[0];
    Value rhs = operands[1];
    auto lhsType = lhs.getType().cast<RankedShapeType>();
    auto rhsType = rhs.getType().cast<RankedShapeType>();
    // Establish invariant that rank(lhs) <= rank(rhs)
    if (lhsType.getRank() > rhsType.getRank()) {
      std::swap(lhsType, rhsType);
      std::swap(lhs, rhs);
    }
    SmallVector<int64_t, 6> resultShape;
    OpTrait::util::getBroadcastedShape(lhsType.getAllDims(),
                                       rhsType.getAllDims(), resultShape);
    auto resultType = RankedShapeType::get(resultShape, rewriter.getContext());
    auto iota = llvm::to_vector<4>(llvm::seq<int64_t>(0, rhsType.getRank()));
    rewriter.replaceOpWithNewOp<RankedBroadcastShapeOp>(
        op, resultType, lhs, rhs,
        /*lhs_broadcast_dimensions=*/
        rewriter.getI64TensorAttr(makeArrayRef(iota).drop_front(
            rhsType.getRank() - lhsType.getRank())),
        /*rhs_broadcast_dimensions=*/
        rewriter.getI64TensorAttr(iota));
    return success();
  }
};

class ConvertConcatOp : public OpConversionPattern<shape::ConcatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      shape::ConcatOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto resultRank = operands[0].getType().cast<RankedShapeType>().getRank() +
                      operands[1].getType().cast<RankedShapeType>().getRank();
    auto indices = llvm::to_vector<4>(llvm::seq<int64_t>(0, resultRank));
    rewriter.replaceOpWithNewOp<Shape::GatherExtentsOp>(
        op, ValueRange({operands[0], operands[1]}),
        rewriter.getI64TensorAttr(indices));
    return success();
  }
};

class ConvertToExtentTensorOp
    : public OpConversionPattern<shape::ToExtentTensorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      shape::ToExtentTensorOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Shape::ToExtentTensorOp>(op, op.getType(),
                                                         operands[0]);
    return success();
  }
};

// Currently, upstream shape lowering can use tensor<?xindex> to represent a
// shape, and will insert tensor_cast ops to convert to specific extent tensor
// types. However, not all tensor_cast ops are shape-related.
class ConvertTensorCastOp : public OpConversionPattern<TensorCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      TensorCastOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!operands[0].getType().isa<RankedShapeType>())
      return rewriter.notifyMatchFailure(op, "not a shape-related tensor_cast");
    rewriter.replaceOpWithNewOp<Shape::ToExtentTensorOp>(op, op.getType(),
                                                         operands[0]);
    return success();
  }
};

class ConvertShapeToShapex
    : public PassWrapper<ConvertShapeToShapex, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Conversion target definition.
    ConversionTarget conversionTarget(*context);
    conversionTarget.addIllegalDialect<shape::ShapeDialect>();
    conversionTarget.addLegalDialect<iree_compiler::ShapeDialect>();

    // Patterns.
    OwningRewritePatternList patterns;
    patterns.insert<ConvertConstShapeOp>(context);
    patterns.insert<ConvertShapeOfOp>(context);
    patterns.insert<ConvertSplitAtOp>(context);
    patterns.insert<ConvertBroadcastOp>(context);
    patterns.insert<ConvertConcatOp>(context);
    patterns.insert<ConvertToExtentTensorOp>(context);
    patterns.insert<ConvertTensorCastOp>(context);

    if (failed(applyPartialConversion(module, conversionTarget, patterns))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertShapeToShapexPass() {
  return std::make_unique<ConvertShapeToShapex>();
}

static PassRegistration<ConvertShapeToShapex> registration(
    "convert-shape-to-shapex", "Convert `shape` dialect to `shapex` dialect");

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
