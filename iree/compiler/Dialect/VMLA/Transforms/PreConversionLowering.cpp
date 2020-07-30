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
#include "iree/compiler/Dialect/VMLA/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLADialect.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {

namespace {

// Convert instances of `mhlo.dot` to `mhlo.dot_general`.
//
// TODO(silvasean): This logically is part of a future HLO client -> HLO server
// type of pass in the mhlo dialect proper.
struct LowerDotOp : public OpRewritePattern<mhlo::DotOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DotOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType) {
      return failure();
    }
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
      return failure();
    }
    // TODO(silvasean): Move this helper to MLIR core.
    auto make1DElementsAttr = [&rewriter](ArrayRef<int64_t> integers) {
      auto type = RankedTensorType::get({static_cast<int64_t>(integers.size())},
                                        rewriter.getIntegerType(64));
      return DenseIntElementsAttr::get(type, integers);
    };
    auto dimensionNumbers = mhlo::DotDimensionNumbers::get(
        /*lhs_batching_dimensions=*/make1DElementsAttr({}),
        /*rhs_batching_dimensions=*/make1DElementsAttr({}),
        /*lhs_contracting_dimensions=*/make1DElementsAttr({1}),
        /*rhs_contracting_dimensions=*/make1DElementsAttr({0}),
        rewriter.getContext());
    rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
        op, op.getType(), lhs, rhs, dimensionNumbers,
        op.precision_config().hasValue() ? op.precision_config().getValue()
                                         : nullptr);
    return success();
  }
};

// Inserts transposes on the operands of DotGeneralOp's such that the resulting
// batch dimensions are all the leading dimensions and all the contracting
// dimensions are all the trailing dimensions.
//
// Furthermore, all batch, contracting, and free dimensions are flattened into
// single dimensions, with an appropriate reshape back to the original
// dimensions.
//
// This results in a very simple corresponding VMLA op in the runtime.
// [1 batch dimension, 1 free dimension, 1 contracting dimension].
//
// The result doesn't have a DotGeneralOp, but rather a
// VMLA::BatchMatMulPseudoOp which represents this transformation.
//
// TODO(silvasean): Move this to a "prepare" pass and test separately.
struct LowerDotGeneralOp : public OpRewritePattern<mhlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    Type elementType = lhsType.getElementType();
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(op, "requires ranked types");
    }
    mhlo::DotDimensionNumbers dimNumbers = op.dot_dimension_numbers();
    auto extract1DVector = [](DenseIntElementsAttr elements) {
      SmallVector<int64_t, 6> ret;
      for (const APInt &element : elements) {
        ret.push_back(element.getLimitedValue());
      }
      return ret;
    };
    auto lhsBatchingDims =
        extract1DVector(dimNumbers.lhs_batching_dimensions());
    auto rhsBatchingDims =
        extract1DVector(dimNumbers.rhs_batching_dimensions());
    auto lhsContractingDims =
        extract1DVector(dimNumbers.lhs_contracting_dimensions());
    auto rhsContractingDims =
        extract1DVector(dimNumbers.rhs_contracting_dimensions());
    // TODO(silvasean): Move this helper to MLIR core.
    auto make1DElementsAttr = [&rewriter](ArrayRef<int64_t> integers) {
      auto type = RankedTensorType::get({static_cast<int64_t>(integers.size())},
                                        rewriter.getIntegerType(64));
      return DenseIntElementsAttr::get(type, integers);
    };
    auto totalElements = [&](ArrayRef<Value> extents) {
      Value numElements = rewriter.create<mlir::ConstantOp>(
          op.getLoc(), IntegerAttr::get(rewriter.getIndexType(), 1));
      for (Value extent : extents) {
        numElements =
            rewriter.create<mlir::MulIOp>(op.getLoc(), numElements, extent);
      }
      return numElements;
    };
    auto handleOneSide = [&](ArrayRef<int64_t> batchingDims,
                             ArrayRef<int64_t> contractingDims, Value &value,
                             RankedTensorType &type,
                             SmallVectorImpl<int64_t> &outFreeDims,
                             SmallVectorImpl<Value> &outFreeDimExtents,
                             SmallVectorImpl<Value> &outBatchingDimExtents) {
      outBatchingDimExtents.clear();
      RankedTensorType untransposedType = type;
      SmallVector<int64_t, 6> permutation;
      llvm::BitVector freeDims(untransposedType.getRank(), true);
      SmallVector<Value, 6> contractingDimExtents;
      Value valueShape =
          rewriter.create<Shape::GetRankedShapeOp>(op.getLoc(), value);
      auto getExtentValue = [&](int64_t dim) {
        return rewriter.create<Shape::RankedDimOp>(op.getLoc(), valueShape,
                                                   dim);
      };
      for (auto dims : {batchingDims, contractingDims}) {
        for (int64_t dim : dims) {
          freeDims.reset(dim);
        }
      }
      for (int64_t dim : batchingDims) {
        permutation.push_back(dim);
        outBatchingDimExtents.push_back(getExtentValue(dim));
      }
      for (int64_t dim : freeDims.set_bits()) {
        permutation.push_back(dim);
        outFreeDims.push_back(dim);
        outFreeDimExtents.push_back(getExtentValue(dim));
      }
      for (int64_t dim : contractingDims) {
        permutation.push_back(dim);
        contractingDimExtents.push_back(getExtentValue(dim));
      }
      // Construct the type that the transpose will result in.
      SmallVector<int64_t, 6> transposeStaticShape;
      for (int64_t index : permutation) {
        (void)index;
        transposeStaticShape.push_back(-1);
      }
      auto transposeType =
          RankedTensorType::get(transposeStaticShape, elementType);
      auto transpose = rewriter.create<mhlo::TransposeOp>(
          op.getLoc(), transposeType, value, make1DElementsAttr(permutation));

      SmallVector<Value, 6> reshapeShape;
      reshapeShape.push_back(totalElements(outBatchingDimExtents));
      reshapeShape.push_back(totalElements(outFreeDimExtents));
      reshapeShape.push_back(totalElements(contractingDimExtents));
      auto reshapeType = RankedTensorType::get(
          {static_cast<int64_t>(-1), static_cast<int64_t>(-1),
           static_cast<int64_t>(-1)},
          elementType);
      auto reshapeRankedShape = rewriter.create<Shape::MakeRankedShapeOp>(
          op.getLoc(),
          Shape::RankedShapeType::get(reshapeType.getShape(),
                                      rewriter.getContext()),
          reshapeShape);
      auto reshapeShapeExtentTensor = rewriter.create<Shape::ToExtentTensorOp>(
          op.getLoc(), reshapeRankedShape);
      value = rewriter.create<mhlo::DynamicReshapeOp>(
          op.getLoc(), reshapeType, transpose, reshapeShapeExtentTensor);
    };
    SmallVector<Value, 6> batchingDimExtents;
    SmallVector<int64_t, 6> lhsFreeDims;
    SmallVector<Value, 6> lhsFreeDimExtents;
    handleOneSide(lhsBatchingDims, lhsContractingDims, lhs, lhsType,
                  lhsFreeDims, lhsFreeDimExtents, batchingDimExtents);
    SmallVector<int64_t, 6> rhsFreeDims;
    SmallVector<Value, 6> rhsFreeDimExtents;
    handleOneSide(rhsBatchingDims, rhsContractingDims, rhs, rhsType,
                  rhsFreeDims, rhsFreeDimExtents, batchingDimExtents);

    auto dstStaticShape = llvm::to_vector<6>(
        llvm::makeArrayRef({static_cast<int64_t>(-1), static_cast<int64_t>(-1),
                            static_cast<int64_t>(-1)}));
    auto dstType = RankedTensorType::get(dstStaticShape, elementType);
    Value dst = rewriter.create<IREE::VMLA::BatchMatMulPseudoOp>(
        op.getLoc(), dstType, lhs, rhs);
    RankedTensorType transposeType = RankedTensorType::get(
        {dstStaticShape[0], dstStaticShape[2], dstStaticShape[1]}, elementType);
    auto transpose = rewriter.create<mhlo::TransposeOp>(
        op.getLoc(), transposeType, dst, make1DElementsAttr({0, 2, 1}));
    auto reshapeShape = batchingDimExtents;
    reshapeShape.append(lhsFreeDimExtents.begin(), lhsFreeDimExtents.end());
    reshapeShape.append(rhsFreeDimExtents.begin(), rhsFreeDimExtents.end());
    SmallVector<int64_t, 6> reshapeStaticShape;
    for (int i = 0, e = batchingDimExtents.size() + lhsFreeDimExtents.size() +
                        rhsFreeDimExtents.size();
         i < e; i++) {
      reshapeStaticShape.push_back(-1);
    }
    auto reshapeRankedShape = rewriter.create<Shape::MakeRankedShapeOp>(
        op.getLoc(),
        Shape::RankedShapeType::get(reshapeStaticShape, rewriter.getContext()),
        reshapeShape);
    auto reshapeShapeExtentTensor = rewriter.create<Shape::ToExtentTensorOp>(
        op.getLoc(), reshapeRankedShape);
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
        op, op.getType(), transpose, reshapeShapeExtentTensor);
    return success();
  }
};

class LowerBroadcastInDimOp : public OpRewritePattern<mhlo::BroadcastInDimOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getType().cast<RankedTensorType>();
    auto shapeType =
        Shape::RankedShapeType::get(type.getShape(), rewriter.getContext());
    auto shape =
        rewriter.create<Shape::ConstRankedShapeOp>(op.getLoc(), shapeType);
    rewriter.replaceOpWithNewOp<Shape::RankedBroadcastInDimOp>(
        op, op.getType(), op.operand(), shape, op.broadcast_dimensions());
    return success();
  }
};

// Lower mhlo::BroadcastOp via mhlo::BroadcastInDimOp.
class LowerBroadcastOp : public OpRewritePattern<mhlo::BroadcastOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getOperand().getType().cast<RankedTensorType>();
    auto resultType = op.getType().cast<RankedTensorType>();
    auto broadcastDimensions = llvm::to_vector<6>(llvm::seq<int64_t>(
        resultType.getRank() - type.getRank(), resultType.getRank()));
    rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
        op, op.getType(), op.getOperand(),
        rewriter.getI64TensorAttr(broadcastDimensions));
    return success();
  }
};

class PreConversionLoweringPass
    : public PassWrapper<PreConversionLoweringPass, OperationPass<FuncOp>> {
 public:
  void runOnOperation() {
    MLIRContext *context = &getContext();

    // These patterns should be run greedily as they are not dialect
    // conversions.
    OwningRewritePatternList greedyPatterns;
    mhlo::PopulateComplexLoweringPatterns(context, &greedyPatterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), greedyPatterns))) {
      return signalPassFailure();
    }

    OwningRewritePatternList patterns;
    ConversionTarget target(*context);
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<IREE::VMLA::VMLADialect>();
    target.addLegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<ShapeDialect>();

    target.addIllegalOp<mhlo::DotGeneralOp>();
    patterns.insert<LowerDotGeneralOp>(context);
    target.addIllegalOp<mhlo::DotOp>();
    patterns.insert<LowerDotOp>(context);
    target.addIllegalOp<mhlo::BroadcastInDimOp>();
    patterns.insert<LowerBroadcastInDimOp>(context);
    target.addIllegalOp<mhlo::BroadcastOp>();
    patterns.insert<LowerBroadcastOp>(context);

    if (failed(applyPartialConversion(getOperation(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<PreConversionLoweringPass> pass(
    "iree-vmla-pre-conversion-lowering",
    "Tensor-level pattern-based lowerings.");

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createPreConversionLoweringPass() {
  return std::make_unique<PreConversionLoweringPass>();
}

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
