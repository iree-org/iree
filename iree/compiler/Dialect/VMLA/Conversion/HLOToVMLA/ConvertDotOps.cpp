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

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/VMLA/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VMLA/Conversion/HLOToVMLA/ConvertHLOToVMLA.h"
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
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Convert instances of `xla_hlo.dot` to `xla_hlo.dot_general`.
//
// TODO(silvasean): This logically is part of a future HLO client -> HLO server
// type of pass in the xla_hlo dialect proper.
struct CanonicalizeDotOp : public OpRewritePattern<xla_hlo::DotOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(xla_hlo::DotOp op,
                                     PatternRewriter &rewriter) const override {
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType) {
      return matchFailure();
    }
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
      return matchFailure();
    }
    // TODO(silvasean): Move this helper to MLIR core.
    auto make1DElementsAttr = [&rewriter](ArrayRef<int64_t> integers) {
      auto type = RankedTensorType::get({static_cast<int64_t>(integers.size())},
                                        rewriter.getIntegerType(64));
      return DenseIntElementsAttr::get(type, integers);
    };
    auto dimensionNumbers = xla_hlo::DotDimensionNumbers::get(
        /*lhs_batching_dimensions=*/make1DElementsAttr({}),
        /*rhs_batching_dimensions=*/make1DElementsAttr({}),
        /*lhs_contracting_dimensions=*/make1DElementsAttr({1}),
        /*rhs_contracting_dimensions=*/make1DElementsAttr({0}),
        rewriter.getContext());
    rewriter.replaceOpWithNewOp<xla_hlo::DotGeneralOp>(
        op, op.getType(), lhs, rhs, dimensionNumbers,
        op.precision_config().hasValue() ? op.precision_config().getValue()
                                         : nullptr);
    return matchSuccess();
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
struct CanonicalizeDotGeneralOp
    : public OpRewritePattern<xla_hlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(xla_hlo::DotGeneralOp op,
                                     PatternRewriter &rewriter) const override {
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    Type elementType = lhsType.getElementType();
    if (!lhsType || !rhsType) {
      return matchFailure();
    }
    // TODO(silvasean): Extend to support dynamic shapes.
    // This op is a really good case for testing our e2e dynamic shape support.
    // There's interesting questions at the TF2XLA level too.
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
      return matchFailure();
    }
    xla_hlo::DotDimensionNumbers dimNumbers = op.dot_dimension_numbers();
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
    auto totalElements = [](ArrayRef<int64_t> extents) {
      int64_t numElements = 1;
      for (int64_t extent : extents) {
        numElements *= extent;
      }
      return numElements;
    };
    auto handleOneSide = [&](ArrayRef<int64_t> batchingDims,
                             ArrayRef<int64_t> contractingDims, Value &value,
                             RankedTensorType &type,
                             SmallVectorImpl<int64_t> &outFreeDims,
                             SmallVectorImpl<int64_t> &outFreeDimExtents,
                             SmallVectorImpl<int64_t> &outBatchingDimExtents) {
      outBatchingDimExtents.clear();
      RankedTensorType untransposedType = type;
      SmallVector<int64_t, 6> permutation;
      llvm::BitVector freeDims(untransposedType.getRank(), true);
      SmallVector<int64_t, 6> contractingDimExtents;
      for (auto dims : {batchingDims, contractingDims}) {
        for (int64_t dim : dims) {
          freeDims.reset(dim);
        }
      }
      for (int64_t dim : batchingDims) {
        permutation.push_back(dim);
        outBatchingDimExtents.push_back(untransposedType.getDimSize(dim));
      }
      for (int64_t dim : freeDims.set_bits()) {
        permutation.push_back(dim);
        outFreeDims.push_back(dim);
        outFreeDimExtents.push_back(untransposedType.getDimSize(dim));
      }
      for (int64_t dim : contractingDims) {
        permutation.push_back(dim);
        contractingDimExtents.push_back(untransposedType.getDimSize(dim));
      }
      // Construct the type that the transpose will result in.
      SmallVector<int64_t, 6> transposeShape;
      for (int64_t index : permutation) {
        transposeShape.push_back(type.getDimSize(index));
      }
      auto transposeType = RankedTensorType::get(transposeShape, elementType);
      auto transpose = rewriter.create<xla_hlo::TransposeOp>(
          op.getLoc(), transposeType, value, make1DElementsAttr(permutation));

      auto reshapeType =
          RankedTensorType::get({totalElements(outBatchingDimExtents),
                                 totalElements(outFreeDimExtents),
                                 totalElements(contractingDimExtents)},
                                elementType);
      value = rewriter.create<xla_hlo::ReshapeOp>(op.getLoc(), reshapeType,
                                                  transpose);
    };
    SmallVector<int64_t, 6> batchingDimExtents;
    SmallVector<int64_t, 6> lhsFreeDims;
    SmallVector<int64_t, 6> lhsFreeDimExtents;
    handleOneSide(lhsBatchingDims, lhsContractingDims, lhs, lhsType,
                  lhsFreeDims, lhsFreeDimExtents, batchingDimExtents);
    SmallVector<int64_t, 6> rhsFreeDims;
    SmallVector<int64_t, 6> rhsFreeDimExtents;
    handleOneSide(rhsBatchingDims, rhsContractingDims, rhs, rhsType,
                  rhsFreeDims, rhsFreeDimExtents, batchingDimExtents);

    auto dstShape = llvm::to_vector<6>(llvm::makeArrayRef(
        {totalElements(batchingDimExtents), totalElements(rhsFreeDimExtents),
         totalElements(lhsFreeDimExtents)}));
    auto dstType = RankedTensorType::get(dstShape, elementType);
    Value dst = rewriter.create<IREE::VMLA::BatchMatMulPseudoOp>(
        op.getLoc(), dstType, lhs, rhs);
    RankedTensorType transposeType = RankedTensorType::get(
        {dstShape[0], dstShape[2], dstShape[1]}, elementType);
    auto transpose = rewriter.create<xla_hlo::TransposeOp>(
        op.getLoc(), transposeType, dst, make1DElementsAttr({0, 2, 1}));
    auto reshapeShape = batchingDimExtents;
    reshapeShape.append(lhsFreeDimExtents.begin(), lhsFreeDimExtents.end());
    reshapeShape.append(rhsFreeDimExtents.begin(), rhsFreeDimExtents.end());
    auto reshapeType = RankedTensorType::get(reshapeShape, elementType);
    rewriter.replaceOpWithNewOp<xla_hlo::ReshapeOp>(op, reshapeType, transpose);
    return matchSuccess();
  }
};

}  // namespace

void populateHLODotToVMLAPatterns(MLIRContext *context,
                                  OwningRewritePatternList &patterns,
                                  TypeConverter &typeConverter) {
  // Tensor-level preparation for lowering to the runtime BatchMatMul op.
  patterns.insert<CanonicalizeDotGeneralOp>(context);
  patterns.insert<CanonicalizeDotOp>(context);

  // Lowering from tensor ops to VMLA runtime ops.
  patterns.insert<VMLAOpConversion<IREE::VMLA::BatchMatMulPseudoOp,
                                   IREE::VMLA::BatchMatMulOp>>(context,
                                                               typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
