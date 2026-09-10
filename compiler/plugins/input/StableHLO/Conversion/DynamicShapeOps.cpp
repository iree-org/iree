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

// Returns the reduction op combining the resultIndex-th and the
// (resultIndex + numInputs)-th block arguments, or null if the body does not
// have that shape. Mirrors upstream's ReduceWindowOpConversion::getReductionOp.
Operation *getReduceWindowReductionOp(mlir::stablehlo::ReduceWindowOp op,
                                      int64_t resultIndex) {
  auto returnOp =
      cast<mlir::stablehlo::ReturnOp>(op.getBody().front().getTerminator());
  Operation *computeOp = returnOp.getResults()[resultIndex].getDefiningOp();
  if (!computeOp || computeOp->getNumOperands() != 2) {
    return nullptr;
  }
  auto arg0 = dyn_cast<BlockArgument>(computeOp->getOperand(0));
  auto arg1 = dyn_cast<BlockArgument>(computeOp->getOperand(1));
  if (!arg0 || !arg1) {
    return nullptr;
  }
  int64_t otherArgIndex = resultIndex + op.getInputs().size();
  if (arg0.getArgNumber() == resultIndex &&
      arg1.getArgNumber() == otherArgIndex) {
    return computeOp;
  }
  if (arg0.getArgNumber() == otherArgIndex &&
      arg1.getArgNumber() == resultIndex &&
      computeOp->hasTrait<OpTrait::IsCommutative>()) {
    return computeOp;
  }
  return nullptr;
}

// Upstream's ReduceWindowOpConversion (StablehloToLinalgReduce.cpp) lowers
// these to a named pooling op, dynamic result dims included, so this pattern
// stands aside.
bool matchesUpstreamPooling(mlir::stablehlo::ReduceWindowOp op) {
  int64_t rank = cast<ShapedType>(op.getResultTypes()[0]).getRank();
  if (rank != 4 && rank != 5) {
    return false;
  }
  if (op.getPadding() && !mlir::stablehlo::isSplatValue(*op.getPadding(), 0)) {
    return false;
  }
  if (auto bd = op.getBaseDilations();
      bd && !llvm::all_of(*bd, [](int64_t v) { return v == 1; })) {
    return false;
  }
  int64_t lastDim = rank - 1;
  if (op.getWindowDimensions()[0] != 1 ||
      op.getWindowDimensions()[lastDim] != 1) {
    return false;
  }
  if (auto ws = op.getWindowStrides();
      ws && (ws.value()[0] != 1 || ws.value()[lastDim] != 1)) {
    return false;
  }
  for (auto [index, input] : llvm::enumerate(op.getInputs())) {
    if (!cast<ShapedType>(input.getType()).getElementType().isF32()) {
      return false;
    }
    Operation *reduceOp = getReduceWindowReductionOp(op, index);
    if (!reduceOp || !isa<mlir::stablehlo::MinOp, mlir::stablehlo::MaxOp,
                          mlir::stablehlo::AddOp>(*reduceOp)) {
      return false;
    }
  }
  return true;
}

// Copied from StablehloToLinalgReduce.cpp's
// ReduceWindowOpOnTensorsGenericConversion; the seed and the result dims
// differ.
//
// Upstream's reduce_window converter needs a static result only to seed the
// output.
struct DynamicReduceWindowOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReduceWindowOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceWindowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (matchesUpstreamPooling(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "upstream pooling lowering applies");
    }
    MLIRContext *ctx = op->getContext();
    Location loc = op.getLoc();
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return failure();
    }
    // Static results take the upstream path.
    if (llvm::all_of(resultTypes, [](Type t) {
          return cast<ShapedType>(t).hasStaticShape();
        })) {
      return rewriter.notifyMatchFailure(op, "static result");
    }
    SmallVector<Value> initValues = adaptor.getInitValues();
    size_t numOperands = initValues.size();

    SmallVector<int64_t> windowDimensions(op.getWindowDimensions());
    int64_t rank = windowDimensions.size();
    SmallVector<int64_t> padding(2 * rank, 0);
    if (op.getPadding()) {
      padding = llvm::to_vector(op.getPadding()->getValues<int64_t>());
    }
    SmallVector<int64_t> baseDilations(rank, 1);
    if (op.getBaseDilations()) {
      baseDilations = llvm::to_vector(*op.getBaseDilations());
    }
    SmallVector<int64_t> windowStrides(rank, 1);
    if (op.getWindowStrides()) {
      windowStrides = llvm::to_vector(*op.getWindowStrides());
    }
    SmallVector<int64_t> windowDilations(rank, 1);
    if (op.getWindowDilations()) {
      windowDilations = llvm::to_vector(*op.getWindowDilations());
    }

    SmallVector<AffineExpr> srcExprs, windowExprs, dstExprs;
    SmallVector<int64_t> filteredWindowDims;
    int windowDim = 0;
    for (int64_t i = 0; i < rank; ++i) {
      AffineExpr srcExpr = getAffineDimExpr(i, ctx);
      if (windowStrides[i] != 1) {
        srcExpr = srcExpr * windowStrides[i];
      }
      if (windowDimensions[i] != 1) {
        filteredWindowDims.push_back(windowDimensions[i]);
        AffineExpr windowExpr = getAffineDimExpr(rank + windowDim, ctx);
        windowExprs.push_back(windowExpr);
        if (windowDilations[i] != 1) {
          windowExpr = windowExpr * windowDilations[i];
        }
        srcExpr = srcExpr + windowExpr;
        ++windowDim;
      }
      srcExprs.push_back(srcExpr);
      dstExprs.push_back(getAffineDimExpr(i, ctx));
    }
    SmallVector<AffineMap> inferredMaps(3, AffineMap::get(ctx));
    if (rank > 0) {
      inferredMaps =
          AffineMap::inferFromExprList({srcExprs, windowExprs, dstExprs}, ctx);
    }
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.append(numOperands, inferredMaps[0]);
    indexingMaps.push_back(inferredMaps[1]);
    indexingMaps.append(numOperands, inferredMaps[2]);

    // Pad and dilate through stablehlo.pad, as upstream does; it accepts
    // dynamic operands.
    SmallVector<Value> inputs = llvm::to_vector(adaptor.getInputs());
    bool needsPad =
        llvm::any_of(padding, [](int64_t v) { return v != 0; }) ||
        llvm::any_of(baseDilations, [](int64_t v) { return v != 1; });
    if (needsPad) {
      SmallVector<int64_t> lows(rank), highs(rank), interiors(rank);
      for (int64_t i = 0; i < rank; ++i) {
        lows[i] = padding[2 * i];
        highs[i] = padding[2 * i + 1];
        interiors[i] = baseDilations[i] - 1;
      }
      for (auto [input, initValue] : llvm::zip(inputs, initValues)) {
        input = mlir::stablehlo::PadOp::create(rewriter, loc, input, initValue,
                                               lows, highs, interiors);
      }
    }

    // Seed each output: result dim i is (padded_i - dilated_window_i) /
    // stride_i + 1 where the type leaves it dynamic.
    SmallVector<Value> seeds;
    for (auto [initValue, resultTypeIt, input] :
         llvm::zip(initValues, resultTypes, inputs)) {
      auto resultTy = cast<RankedTensorType>(resultTypeIt);
      SmallVector<Value> dynamicDims;
      for (int64_t i = 0; i < rank; ++i) {
        if (!resultTy.isDynamicDim(i)) {
          continue;
        }
        Value padded = rewriter.createOrFold<tensor::DimOp>(loc, input, i);
        int64_t dilatedWindow =
            (windowDimensions[i] - 1) * windowDilations[i] + 1;
        Value span = arith::SubIOp::create(
            rewriter, loc, padded,
            arith::ConstantIndexOp::create(rewriter, loc, dilatedWindow));
        Value steps = arith::DivSIOp::create(
            rewriter, loc, span,
            arith::ConstantIndexOp::create(rewriter, loc, windowStrides[i]));
        dynamicDims.push_back(arith::AddIOp::create(
            rewriter, loc, steps,
            arith::ConstantIndexOp::create(rewriter, loc, 1)));
      }
      Value empty =
          tensor::EmptyOp::create(rewriter, loc, resultTy, dynamicDims);
      Value scalarInit =
          rewriter.createOrFold<tensor::ExtractOp>(loc, initValue);
      seeds.push_back(
          linalg::FillOp::create(rewriter, loc, scalarInit, empty).result());
    }

    inputs.push_back(tensor::EmptyOp::create(rewriter, loc, filteredWindowDims,
                                             rewriter.getF32Type()));
    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, resultTypes, inputs, seeds, indexingMaps,
        mlir::stablehlo::getParallelAndReductionIterators(
            rank + filteredWindowDims.size(), filteredWindowDims.size()),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    Region &region = linalgOp.getRegion();
    rewriter.cloneRegionBefore(op.getBody(), region, region.end());
    TypeConverter::SignatureConversion signatureConverter(
        inputs.size() + op->getNumResults() - 1);
    for (auto [i, type] : llvm::enumerate(resultTypes)) {
      signatureConverter.addInputs(inputs.size() + i - 1,
                                   cast<ShapedType>(type).getElementType());
    }
    signatureConverter.addInputs(
        cast<ShapedType>(inputs.back().getType()).getElementType());
    for (auto [i, input] :
         llvm::enumerate(ArrayRef<Value>(inputs).drop_back())) {
      signatureConverter.addInputs(
          i, cast<ShapedType>(input.getType()).getElementType());
    }
    rewriter.applySignatureConversion(&region.front(), signatureConverter,
                                      getTypeConverter());
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

} // namespace

void populateDynamicShapeConversionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet *patterns) {
  // These patterns get first refusal. Where an upstream lowering applies,
  // the pattern stands aside.
  patterns
      ->add<DynamicReshapeOpConversion, DynamicPadOpConversion,
            DynamicGatherOpConversion, DynamicBroadcastInDimGatherConversion,
            DynamicReduceWindowOpConversion>(typeConverter, context,
                                             PatternBenefit{1000});
}

} // namespace mlir::iree_compiler::stablehlo
