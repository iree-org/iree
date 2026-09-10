// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowerings for the StableHLO ops whose shapes are known only at runtime.

#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/LegalizeToLinalgUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

// Copied verbatim from StablehloLegalizeToLinalg.cpp, where it is file-local.
Value extractIndexFromTensor(OpBuilder &builder, Location loc, Value tensor,
                             ShapedType originalType,
                             ArrayRef<Value> tensorIndex = {}) {
  Value extracted =
      tensor::ExtractOp::create(builder, loc, tensor, tensorIndex);
  if (extracted.getType().isIndex()) {
    return extracted;
  }
  return originalType.getElementType().isUnsignedInteger()
             ? builder.createOrFold<arith::IndexCastUIOp>(
                   loc, builder.getIndexType(), extracted)
             : builder.createOrFold<arith::IndexCastOp>(
                   loc, builder.getIndexType(), extracted);
}

// Reads element `i` of a 1-D shape operand as an index.
Value extractIndex(OpBuilder &b, Location loc, Value shapeTensor, int64_t i) {
  Value index = arith::ConstantIndexOp::create(b, loc, i);
  Value element = tensor::ExtractOp::create(b, loc, shapeTensor, index);
  if (element.getType().isIndex()) {
    return element;
  }
  return arith::IndexCastOp::create(b, loc, b.getIndexType(), element);
}

// tensor.reshape takes the shape tensor as-is, and Flow lowers it.
struct DynamicReshapeOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicReshapeOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
        op, resultType, adaptor.getOperand(), adaptor.getOutputShape());
    return success();
  }
};

// tensor.pad has no interior padding, and whether the interior amounts are
// zero is unknown here, so the strided insert covers every case.
struct DynamicPadOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicPadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicPadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    auto operandType =
        dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
    if (!resultType || !operandType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }
    int64_t rank = operandType.getRank();

    Value paddingValue = rewriter.createOrFold<tensor::ExtractOp>(
        loc, adaptor.getPaddingValue());

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> dynamicDims;
    SmallVector<OpFoldResult> insertOffsets, insertSizes, insertStrides;
    SmallVector<OpFoldResult> extractOffsets, extractSizes;
    for (int64_t i = 0; i < rank; ++i) {
      Value dim = tensor::DimOp::create(rewriter, loc, adaptor.getOperand(), i);
      Value low = extractIndex(rewriter, loc, adaptor.getEdgePaddingLow(), i);
      Value high = extractIndex(rewriter, loc, adaptor.getEdgePaddingHigh(), i);
      Value interior =
          extractIndex(rewriter, loc, adaptor.getInteriorPadding(), i);

      Value lowPos = arith::MaxSIOp::create(rewriter, loc, low, zero);
      Value highPos = arith::MaxSIOp::create(rewriter, loc, high, zero);
      Value lowNeg = arith::MaxSIOp::create(
          rewriter, loc, arith::SubIOp::create(rewriter, loc, zero, low), zero);

      Value dimMinusOne = arith::SubIOp::create(rewriter, loc, dim, one);
      Value clampedDimMinusOne =
          arith::MaxSIOp::create(rewriter, loc, dimMinusOne, zero);
      Value interiorTotal =
          arith::MulIOp::create(rewriter, loc, clampedDimMinusOne, interior);
      Value dimAndInterior =
          arith::AddIOp::create(rewriter, loc, dim, interiorTotal);

      // The scratch tensor's size depends on lowPos/highPos, which are
      // runtime values even where the result type's dim is static, so every
      // dim of the scratch tensor is dynamic.
      Value dimAndLowPos =
          arith::AddIOp::create(rewriter, loc, dimAndInterior, lowPos);
      Value filledDim =
          arith::AddIOp::create(rewriter, loc, dimAndLowPos, highPos);
      dynamicDims.push_back(filledDim);

      if (resultType.isDynamicDim(i)) {
        Value dimAndLow =
            arith::AddIOp::create(rewriter, loc, dimAndInterior, low);
        Value resultDim = arith::AddIOp::create(rewriter, loc, dimAndLow, high);
        extractSizes.push_back(resultDim);
      } else {
        extractSizes.push_back(rewriter.getIndexAttr(resultType.getDimSize(i)));
      }

      insertOffsets.push_back(lowPos);
      insertSizes.push_back(
          tensor::getMixedSize(rewriter, loc, adaptor.getOperand(), i));
      insertStrides.push_back(
          arith::AddIOp::create(rewriter, loc, interior, one).getResult());
      extractOffsets.push_back(lowNeg);
    }

    SmallVector<int64_t> scratchShape(rank, ShapedType::kDynamic);
    Value empty = tensor::EmptyOp::create(
        rewriter, loc, scratchShape, resultType.getElementType(), dynamicDims);
    Value filled =
        linalg::FillOp::create(rewriter, loc, paddingValue, empty).result();
    Value inserted = tensor::InsertSliceOp::create(
                         rewriter, loc, adaptor.getOperand(), filled,
                         insertOffsets, insertSizes, insertStrides)
                         .getResult();

    // Negative edge padding crops, which the insert cannot express.
    SmallVector<OpFoldResult> extractStrides(rank, rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultType, inserted, extractOffsets, extractSizes, extractStrides);
    return success();
  }
};

// Copied from StablehloLegalizeToLinalg.cpp's GatherConversion. The operand
// rank comes from the operand type and the unranked branch is gone.
//
// Upstream's GatherConversion needs slice_sizes only for the operand rank;
// the result shape carries the rest.
struct DynamicGatherOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicGatherOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicGatherOp gatherOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = gatherOp.getLoc();
    Value startIndices = adaptor.getStartIndices();
    Value operand = adaptor.getOperand();
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(gatherOp.getType());
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto startIndicesType = dyn_cast<RankedTensorType>(startIndices.getType());
    if (!resultType || !operandType || !startIndicesType) {
      return rewriter.notifyMatchFailure(gatherOp, "unranked operands");
    }
    int64_t resultRank = resultType.getRank();
    int64_t operandRank = operandType.getRank();

    auto dims = gatherOp.getDimensionNumbers();
    int64_t indexVectorDim = dims.getIndexVectorDim();
    ArrayRef<int64_t> offsetDims = dims.getOffsetDims();
    ArrayRef<int64_t> collapsedSliceDims = dims.getCollapsedSliceDims();
    ArrayRef<int64_t> operandBatchingDims = dims.getOperandBatchingDims();
    ArrayRef<int64_t> startIndicesBatchingDims =
        dims.getStartIndicesBatchingDims();
    ArrayRef<int64_t> startIndexMap = dims.getStartIndexMap();

    SmallVector<Value> constants;
    for (int64_t i = 0, e = std::max({resultRank, operandRank, int64_t{2}});
         i < e; ++i) {
      constants.push_back(arith::ConstantIndexOp::create(rewriter, loc, i));
    }

    Value emptyOp = mlir::stablehlo::getEmptyTensorFor(
        rewriter, loc, resultType, gatherOp, adaptor.getOperands());

    SmallVector<AffineMap, 1> indexingMaps(
        {rewriter.getMultiDimIdentityMap(resultRank)});
    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{}, /*outputs=*/emptyOp, indexingMaps,
        mlir::stablehlo::getNParallelLoopsAttrs(resultRank),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(gatherOp));

    Region &region = linalgOp.getRegion();
    Block *block = rewriter.createBlock(&region, region.end());
    block->addArguments(resultType.getElementType(), loc);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(block);

    SmallVector<int64_t> batchDims;
    for (int64_t dim = 0; dim < resultRank; ++dim) {
      if (!llvm::is_contained(offsetDims, dim)) {
        batchDims.push_back(dim);
      }
    }
    SmallVector<Value> linalgIndices;
    for (int64_t i = 0; i < resultRank; ++i) {
      linalgIndices.push_back(linalg::IndexOp::create(rewriter, loc, i));
    }
    SmallVector<Value> gatherIndex;
    for (int64_t dim : batchDims) {
      gatherIndex.push_back(linalgIndices[dim]);
    }

    SmallVector<Value> indexFromStartIndices;
    for (size_t i = 0, e = startIndexMap.size(); i != e; ++i) {
      SmallVector<Value> gCombine(gatherIndex);
      if (indexVectorDim != startIndicesType.getRank()) {
        gCombine.insert(gCombine.begin() + indexVectorDim, constants[i]);
      }
      indexFromStartIndices.push_back(extractIndexFromTensor(
          rewriter, loc, startIndices, gatherOp.getStartIndices().getType(),
          gCombine));
    }

    SmallVector<Value> remappedIndexFromIndices(operandRank, constants[0]);
    for (auto [idx, value] : llvm::enumerate(startIndexMap)) {
      remappedIndexFromIndices[value] = indexFromStartIndices[idx];
    }

    SmallVector<Value> indexFromBatching(operandRank, constants[0]);
    for (auto [operandDim, indicesDim] :
         llvm::zip_equal(operandBatchingDims, startIndicesBatchingDims)) {
      indexFromBatching[operandDim] =
          gatherIndex[indicesDim + (indicesDim < indexVectorDim ? 0 : 1)];
    }

    auto isCollapsedOrBatching = [&](int64_t dim) {
      return llvm::is_contained(collapsedSliceDims, dim) ||
             llvm::is_contained(operandBatchingDims, dim);
    };
    SmallVector<unsigned> remappedOffsetDims;
    for (int64_t i = 0; i < operandRank; ++i) {
      if (!isCollapsedOrBatching(i)) {
        remappedOffsetDims.push_back(static_cast<unsigned>(i));
      }
    }

    // Clamp start indices to [0, operand_dim - slice_size]; the slice size is
    // the matching result dim, or 1 for a collapsed dim.
    for (int i = 0, operandIndexDim = 0; i < operandRank; ++i) {
      Value outputDimSize = constants[1];
      if (!isCollapsedOrBatching(i)) {
        outputDimSize = rewriter.createOrFold<tensor::DimOp>(
            loc, emptyOp, offsetDims[operandIndexDim++]);
      }
      if (remappedIndexFromIndices[i] == constants[0]) {
        continue;
      }
      Value operandDimSize =
          rewriter.createOrFold<tensor::DimOp>(loc, operand, i);
      Value largestValidIndex = rewriter.createOrFold<arith::SubIOp>(
          loc, operandDimSize, outputDimSize);
      remappedIndexFromIndices[i] = arith::MinSIOp::create(
          rewriter, loc,
          arith::MaxSIOp::create(rewriter, loc, constants[0],
                                 remappedIndexFromIndices[i]),
          largestValidIndex);
    }

    SmallVector<Value> indexFromOffset(operandRank, constants[0]);
    for (auto [remappedOffsetDim, offsetDim] :
         llvm::zip_equal(remappedOffsetDims, offsetDims)) {
      indexFromOffset[remappedOffsetDim] = linalgIndices[offsetDim];
    }

    SmallVector<Value> combinedIndex;
    for (int64_t i = 0; i < operandRank; ++i) {
      combinedIndex.push_back(rewriter.createOrFold<arith::AddIOp>(
          loc, rewriter.getIndexType(),
          rewriter.createOrFold<arith::AddIOp>(loc, rewriter.getIndexType(),
                                               remappedIndexFromIndices[i],
                                               indexFromBatching[i]),
          indexFromOffset[i]));
    }
    Value element =
        tensor::ExtractOp::create(rewriter, loc, operand, combinedIndex);
    linalg::YieldOp::create(rewriter, loc, element);

    rewriter.replaceOp(gatherOp, linalgOp.getResults());
    return success();
  }
};

// An affine indexing map cannot branch on a runtime size, so when nothing
// says whether an operand dim expands, gather each element.
struct DynamicBroadcastInDimGatherConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = adaptor.getOperand();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!operandType || !resultType) {
      return rewriter.notifyMatchFailure(op, "unranked");
    }

    // Step in only when some operand dim is dynamic and unannotated; upstream
    // handles the rest.
    ArrayRef<int64_t> bcastDims = op.getBroadcastDimensions();
    SmallVector<std::optional<bool>> expanding(operandType.getRank());
    for (auto [idx, dim] : llvm::enumerate(operandType.getShape())) {
      if (!ShapedType::isDynamic(dim)) {
        expanding[idx] = (dim == 1);
      }
    }
    if (auto known = op.getKnownExpandingDimensions()) {
      for (int64_t i : *known) {
        expanding[i] = true;
      }
    }
    if (auto known = op.getKnownNonexpandingDimensions()) {
      for (int64_t i : *known) {
        expanding[i] = false;
      }
    }
    if (llvm::all_of(expanding, [](auto v) { return v.has_value(); })) {
      return rewriter.notifyMatchFailure(op, "expansion is decidable");
    }

    Value empty = mlir::stablehlo::getEmptyTensorFor(rewriter, loc, resultType,
                                                     op, adaptor.getOperands());
    int64_t resultRank = resultType.getRank();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Operand dim sizes, read once outside the loop nest.
    SmallVector<Value> operandDims;
    for (int64_t i = 0, e = operandType.getRank(); i < e; ++i) {
      operandDims.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, operand, i));
    }

    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{}, /*outputs=*/empty,
        SmallVector<AffineMap>{rewriter.getMultiDimIdentityMap(resultRank)},
        mlir::stablehlo::getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &b, Location nestedLoc, ValueRange) {
          SmallVector<Value> index;
          for (auto [operandDim, resultDim] : llvm::enumerate(bcastDims)) {
            Value resultIndex =
                linalg::IndexOp::create(b, nestedLoc, resultDim);
            if (expanding[operandDim].has_value()) {
              index.push_back(*expanding[operandDim] ? zero : resultIndex);
              continue;
            }
            Value isOne =
                arith::CmpIOp::create(b, nestedLoc, arith::CmpIPredicate::eq,
                                      operandDims[operandDim], one);
            index.push_back(arith::SelectOp::create(b, nestedLoc, isOne, zero,
                                                    resultIndex));
          }
          Value element =
              tensor::ExtractOp::create(b, nestedLoc, operand, index);
          linalg::YieldOp::create(b, nestedLoc, element);
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

} // namespace

void populateDynamicShapeConversionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet *patterns) {
  // Higher benefit than the upstream patterns, which decline these forms.
  patterns
      ->add<DynamicReshapeOpConversion, DynamicPadOpConversion,
            DynamicGatherOpConversion, DynamicBroadcastInDimGatherConversion>(
          typeConverter, context, PatternBenefit{1000});
}

} // namespace mlir::iree_compiler::stablehlo
