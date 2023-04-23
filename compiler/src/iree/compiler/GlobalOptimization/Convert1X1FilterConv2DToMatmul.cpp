// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-convert-1x1-filter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir::linalg::detail;

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

static SmallVector<int64_t> getDimOrder(AffineMap map) {
  SmallVector<int64_t> dimOrder;
  for (AffineExpr expr : map.getResults())
    dimOrder.push_back(cast<AffineDimExpr>(expr).getPosition());
  return dimOrder;
}

static SmallVector<StringRef>
getShortDimTypeNames(DenseMap<unsigned, ConvolutionDimType> convDimMap,
                     SmallVector<int64_t> dimOrder) {
  SmallVector<StringRef> typeNames;
  for (auto dim : dimOrder)
    typeNames.push_back(getShortDimTypeName(convDimMap[dim]));
  return typeNames;
}

static SmallVector<int64_t> getIm2ColDimOrder(
    AffineMap inputMap, DenseMap<unsigned, ConvolutionDimType> convDimMap,
    SmallVector<int64_t> filterDimOrder, SmallVector<int64_t> outputDimOrder) {
  SmallVector<int64_t> inputDimOrder;
  int outputIndex = 0;
  int filterIndex = 0;
  bool foundInputChannel = false;
  for (int filterEnd = filterDimOrder.size(), outputEnd = outputDimOrder.size();
       filterIndex < filterEnd || outputIndex < outputEnd;) {
    if (outputIndex >= outputEnd) {
      for (; filterIndex < filterEnd; filterIndex++)
        inputDimOrder.push_back(filterDimOrder[filterIndex]);
      break;
    }
    if (filterIndex >= filterEnd) {
      for (; outputIndex < outputEnd; outputIndex++)
        inputDimOrder.push_back(outputDimOrder[outputIndex]);
      break;
    }
    if (isa<ConvolutionDimType::Batch>(
            convDimMap[outputDimOrder[outputIndex]])) {
      inputDimOrder.push_back(outputDimOrder[outputIndex++]);
      continue;
    }
    if (foundInputChannel) {
      if (isa<ConvolutionDimType::FilterLoop>(
              convDimMap[filterDimOrder[filterIndex]])) {
        inputDimOrder.push_back(filterDimOrder[filterIndex++]);
        continue;
      }
      if (isa<ConvolutionDimType::OutputImage>(
              convDimMap[outputDimOrder[outputIndex]])) {
        inputDimOrder.push_back(outputDimOrder[outputIndex++]);
        continue;
      }
      if (isa<ConvolutionDimType::InputChannel>(
              convDimMap[filterDimOrder[filterIndex]])) {
        inputDimOrder.push_back(filterDimOrder[filterIndex++]);
        continue;
      }
    } else {
      if (isa<ConvolutionDimType::InputChannel>(
              convDimMap[filterDimOrder[filterIndex]])) {
        inputDimOrder.push_back(filterDimOrder[filterIndex++]);
        foundInputChannel = true;
        continue;
      }
      if (isa<ConvolutionDimType::OutputImage>(
              convDimMap[outputDimOrder[outputIndex]])) {
        inputDimOrder.push_back(outputDimOrder[outputIndex++]);
        continue;
      }
      if (isa<ConvolutionDimType::FilterLoop>(
              convDimMap[filterDimOrder[filterIndex]])) {
        inputDimOrder.push_back(filterDimOrder[filterIndex++]);
        continue;
      }
    }
    // Else it is necessarily an OutputChannel for both so increment both
    // indices.
    filterIndex++;
    outputIndex++;
  }
  return inputDimOrder;
}

class Convert1x1FilterConvGenericToMatmul
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    mlir::linalg::ConvolutionDimensions dimensions;
    if (!linalg::detail::getMatchConvolutionMessage(
             linalg::detail::isConvolutionInterfaceImpl(genericOp, &dimensions))
             .empty())
      return failure();

    if (dimensions.outputImage.size() < 2 || dimensions.filterLoop.size() < 2)
      return failure();

    assert(dimensions.outputImage.size() == dimensions.filterLoop.size());

    auto inputType = genericOp.getInputs()[0].getType().cast<ShapedType>();
    auto filterType = genericOp.getInputs()[1].getType().cast<ShapedType>();
    auto outputType = genericOp.getOutputs()[0].getType().cast<ShapedType>();

    if (!filterType.hasStaticShape())
      return failure();

    if (!inputType.hasStaticShape())
      return failure();

    if (!llvm::all_of(dimensions.dilations,
                      [](unsigned element) { return element == 1; }))
      return failure();

    if (!llvm::all_of(dimensions.strides,
                      [](unsigned element) { return element == 1; }))
      return failure();

    auto loc = genericOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    Value input = genericOp.getInputs()[0];
    Value filter = genericOp.getInputs()[1];
    Value output = genericOp.getOutputs()[0];

    auto dimSizes = genericOp.getStaticLoopRanges();

    if (!llvm::all_of(dimensions.filterLoop,
                      [&](int64_t index) { return dimSizes[index] == 1; }))
      return failure();

    auto convDimMap = getConvolutionDimTypeMap(dimensions);

    auto indexingMaps = genericOp.getIndexingMapsArray();
    AffineMap inputMap = indexingMaps[0];
    AffineMap filterMap = indexingMaps[1];
    AffineMap outputMap = indexingMaps[2];

    SmallVector<int64_t> filterDimOrder = getDimOrder(filterMap);
    SmallVector<int64_t> outputDimOrder = getDimOrder(outputMap);
    SmallVector<int64_t> im2ColDimOrder =
        getIm2ColDimOrder(inputMap, convDimMap, filterDimOrder, outputDimOrder);

    LLVM_DEBUG({
      DBGS() << "im2col: Selected dimension orders.\n";
      llvm::interleaveComma(im2ColDimOrder, DBGS() << "im2ColDimOrder: ");
      llvm::dbgs() << "\n";
      llvm::interleaveComma(getShortDimTypeNames(convDimMap, im2ColDimOrder),
                            DBGS() << "im2ColDimTypes: ");
      llvm::dbgs() << "\n";
      llvm::interleaveComma(filterDimOrder, DBGS() << "filterDimOrder: ");
      llvm::dbgs() << "\n";
      llvm::interleaveComma(getShortDimTypeNames(convDimMap, filterDimOrder),
                            DBGS() << "filterDimTypes: ");
      llvm::dbgs() << "\n";
      llvm::interleaveComma(outputDimOrder, DBGS() << "outputDimOrder: ");
      llvm::dbgs() << "\n";
      llvm::interleaveComma(getShortDimTypeNames(convDimMap, outputDimOrder),
                            DBGS() << "outputDimTypes: ");
      llvm::dbgs() << "\n";
    });

    SmallVector<ReassociationIndices> im2ColReassocIndices;
    SmallVector<int64_t> im2ColShape;
    SmallVector<ReassociationIndices> filterReassocIndices;
    SmallVector<int64_t> filterShape;
    SmallVector<ReassociationIndices> outputReassocIndices;
    SmallVector<int64_t> outputShape;

    auto findDimPosFromIndex = [](int64_t dimPos, int64_t index,
                                  SmallVector<int64_t> dimOrder) {
      for (int i = index, e = dimOrder.size(); i < e; i++) {
        if (dimOrder[i] == dimPos)
          return i;
      }
      return static_cast<int>(dimOrder.size());
    };

    auto updateReassociationAndShapeVecs =
        [&](int64_t dim, int64_t im2ColIndex, int64_t prevIndex,
            int64_t nextIndex, bool prevCollapsible,
            SmallVector<int64_t> dimOrder, SmallVector<int64_t> &shapeVec,
            SmallVector<ReassociationIndices> &reassociationMap) {
          auto dimSize = dimSizes[dim];
          if (prevIndex == nextIndex && prevCollapsible) {
            im2ColReassocIndices.back().push_back(im2ColIndex);
            im2ColShape[im2ColShape.size() - 1] *= dimSize;
            reassociationMap.back().push_back(prevIndex);
            shapeVec[shapeVec.size() - 1] *= dimSize;
          } else {
            im2ColReassocIndices.push_back({im2ColIndex});
            im2ColShape.push_back(dimSize);
            for (auto i = prevIndex; i < nextIndex + 1; i++) {
              reassociationMap.push_back({i});
              shapeVec.push_back(dimSizes[dimOrder[i]]);
            }
          }
        };

    int64_t filterDimPos = 0;
    int64_t outputDimPos = 0;
    bool prevWasFilterCollapsible = false;
    bool prevWasOutputCollapsible = false;
    for (auto [im2ColIndex, dim] : llvm::enumerate(im2ColDimOrder)) {
      if (isa<ConvolutionDimType::InputChannel, ConvolutionDimType::FilterLoop>(
              convDimMap[dim])) {
        prevWasOutputCollapsible = false;
        auto nextFilter =
            findDimPosFromIndex(dim, filterDimPos, filterDimOrder);
        if (nextFilter >= filterDimOrder.size())
          return rewriter.notifyMatchFailure(
              genericOp, "filter layout does not match im2col layout");
        updateReassociationAndShapeVecs(dim, im2ColIndex, filterDimPos,
                                        nextFilter, prevWasFilterCollapsible,
                                        filterDimOrder, filterShape,
                                        filterReassocIndices);

        filterDimPos = nextFilter + 1;
        prevWasFilterCollapsible = true;
        continue;
      } else if (isa<ConvolutionDimType::OutputImage>(convDimMap[dim])) {
        prevWasFilterCollapsible = false;
        auto nextOutput =
            findDimPosFromIndex(dim, outputDimPos, outputDimOrder);
        if (nextOutput >= outputDimOrder.size())
          return rewriter.notifyMatchFailure(
              genericOp, "output layout does not match im2col layout");
        updateReassociationAndShapeVecs(dim, im2ColIndex, outputDimPos,
                                        nextOutput, prevWasOutputCollapsible,
                                        outputDimOrder, outputShape,
                                        outputReassocIndices);

        outputDimPos = nextOutput + 1;
        prevWasOutputCollapsible = true;
        continue;
      } else if (isa<ConvolutionDimType::Batch>(convDimMap[dim])) {
        auto nextOutput =
            findDimPosFromIndex(dim, outputDimPos, outputDimOrder);
        if (nextOutput >= outputDimOrder.size())
          return rewriter.notifyMatchFailure(
              genericOp, "output layout does not match im2col layout");
        updateReassociationAndShapeVecs(dim, im2ColIndex, outputDimPos,
                                        nextOutput, prevWasOutputCollapsible,
                                        outputDimOrder, outputShape,
                                        outputReassocIndices);

        outputDimPos = nextOutput + 1;
      }
      prevWasFilterCollapsible = false;
      prevWasOutputCollapsible = false;
    }
    for (int i = filterDimPos, e = filterDimOrder.size(); i < e; i++) {
      filterReassocIndices.push_back({i});
      filterShape.push_back(dimSizes[filterDimOrder[i]]);
    }
    for (int i = outputDimPos, e = outputDimOrder.size(); i < e; i++) {
      outputReassocIndices.push_back({i});
      outputShape.push_back(dimSizes[outputDimOrder[i]]);
    }

    auto reshapedFilterType =
        RankedTensorType::get(filterShape, filterType.getElementType());
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, filterReassocIndices);

    auto reshapedOutputType =
        RankedTensorType::get(outputShape, outputType.getElementType());
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, outputReassocIndices);

    auto getGroupedDimOrder =
        [](SmallVector<int64_t> dimOrder,
           SmallVector<ReassociationIndices> reassociationMap) {
          SmallVector<ReassociationIndices> groupedDimOrder;
          for (auto group : reassociationMap) {
            ReassociationIndices dims;
            for (auto i : group)
              dims.push_back(dimOrder[i]);
            groupedDimOrder.push_back(dims);
          }
          return groupedDimOrder;
        };
    SmallVector<ReassociationIndices> groupedIm2ColDimOrder =
        getGroupedDimOrder(im2ColDimOrder, im2ColReassocIndices);

    SmallVector<ReassociationIndices> im2ColCollapseIndices;
    int collapseIndex = 0;
    for (auto group : groupedIm2ColDimOrder) {
      im2ColCollapseIndices.emplace_back();
      for (auto i : group)
        if (!isa<ConvolutionDimType::FilterLoop>(convDimMap[i]))
          im2ColCollapseIndices.back().push_back(collapseIndex++);
    }

    auto reshapedInputType =
        RankedTensorType::get(im2ColShape, inputType.getElementType());
    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, im2ColCollapseIndices);

    SmallVector<ReassociationIndices> groupedFilterDimOrder =
        getGroupedDimOrder(filterDimOrder, filterReassocIndices);
    SmallVector<ReassociationIndices> groupedOutputDimOrder =
        getGroupedDimOrder(outputDimOrder, outputReassocIndices);

    LLVM_DEBUG({
      DBGS() << "im2col: Selected dimension groupings.\n";
      DBGS() << "im2ColDimMap: ";
      for (auto group : groupedIm2ColDimOrder) {
        llvm::dbgs() << "{";
        llvm::interleaveComma(group, llvm::dbgs());
        llvm::dbgs() << "} ";
      }
      llvm::dbgs() << "\n";
      DBGS() << "filterDimMap: ";
      for (auto group : groupedFilterDimOrder) {
        llvm::dbgs() << "{";
        llvm::interleaveComma(group, llvm::dbgs());
        llvm::dbgs() << "} ";
      }
      llvm::dbgs() << "\n";
      DBGS() << "outputDimMap: ";
      for (auto group : groupedOutputDimOrder) {
        llvm::dbgs() << "{";
        llvm::interleaveComma(group, llvm::dbgs());
        llvm::dbgs() << "} ";
      }
      llvm::dbgs() << "\n";
    });

    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    SmallVector<utils::IteratorType> iteratorTypes;
    DenseMap<int64_t, AffineExpr> leadingGroupDimToIterationDim;
    int iterationDim = 0;

    auto advanceOperandIndex =
        [&](int &l, int r, SmallVector<ReassociationIndices> groupedDimOrder) {
          while (l < groupedDimOrder.size() &&
                 leadingGroupDimToIterationDim.count(groupedDimOrder[l][0]))
            l++;
          for (; l < r; l++) {
            auto groupDim = groupedDimOrder[l][0];
            if (leadingGroupDimToIterationDim.count(groupDim))
              continue;
            leadingGroupDimToIterationDim[groupDim] =
                rewriter.getAffineDimExpr(iterationDim++);
            if (isa<ConvolutionDimType::FilterLoop,
                    ConvolutionDimType::InputChannel>(convDimMap[groupDim]))
              iteratorTypes.push_back(reduction);
            else
              iteratorTypes.push_back(parallel);
          }
        };

    // If the inner most dim of the input is a reduced dimension, assume we
    // should keep the filter on the right hand side of the matrix
    // multiplication.
    bool filterRHS =
        isa<ConvolutionDimType::FilterLoop, ConvolutionDimType::InputChannel>(
            convDimMap[im2ColDimOrder.back()]);

    int inputIndex = 0;
    int filterIndex = 0;
    int outputIndex = 0;
    int inputEnd = im2ColShape.size();
    int filterEnd = filterShape.size();
    int outputEnd = outputShape.size();
    while (inputIndex < inputEnd && filterIndex < filterEnd &&
           outputIndex < outputEnd) {
      ConvolutionDimType lhsDimType =
          filterRHS ? convDimMap[groupedIm2ColDimOrder[inputIndex][0]]
                    : convDimMap[groupedFilterDimOrder[filterIndex][0]];
      bool isLastLhsDim = filterRHS
                              ? groupedIm2ColDimOrder.size() - 1 == inputIndex
                              : filterReassocIndices.size() - 1 == filterIndex;
      LLVM_DEBUG({
        DBGS() << "Current dim to iteration dim map:\n";
        for (auto [key, value] : leadingGroupDimToIterationDim) {
          DBGS() << key << " -> " << value << "\n";
        }
        DBGS() << "Input Index: " << inputIndex << "\n";
        DBGS() << "Filter Index: " << filterIndex << "\n";
        DBGS() << "Output Index: " << outputIndex << "\n";
        DBGS() << "Is last LHS Dim: " << (isLastLhsDim ? "true" : "false")
               << "\n";
        DBGS() << "LHS Dim Type: " << getShortDimTypeName(lhsDimType) << "\n";
      });
      if (isLastLhsDim &&
          isa<ConvolutionDimType::FilterLoop, ConvolutionDimType::InputChannel>(
              lhsDimType))
        break;

      if (groupedFilterDimOrder[filterIndex][0] ==
          groupedIm2ColDimOrder[inputIndex][0]) {
        advanceOperandIndex(filterIndex, filterIndex + 1,
                            groupedFilterDimOrder);
      } else if (groupedOutputDimOrder[outputIndex][0] ==
                 groupedFilterDimOrder[filterIndex][0]) {
        advanceOperandIndex(outputIndex, outputIndex + 1,
                            groupedOutputDimOrder);
      } else if (groupedOutputDimOrder[outputIndex][0] ==
                 groupedIm2ColDimOrder[inputIndex][0]) {
        advanceOperandIndex(inputIndex, inputIndex + 1, groupedIm2ColDimOrder);
      } else {
        // Greedily prefer iterating over dims of the output.
        advanceOperandIndex(outputIndex, outputIndex + 1,
                            groupedOutputDimOrder);
      }

      // Catch up all indices based on the iterator dim map.
      advanceOperandIndex(outputIndex, outputIndex, groupedOutputDimOrder);
      advanceOperandIndex(filterIndex, filterIndex, groupedFilterDimOrder);
      advanceOperandIndex(inputIndex, inputIndex, groupedIm2ColDimOrder);
    }

    // Fill in any missing iteration dimensions.
    advanceOperandIndex(outputIndex, outputEnd, groupedOutputDimOrder);
    advanceOperandIndex(filterIndex, filterEnd, groupedFilterDimOrder);
    advanceOperandIndex(inputIndex, inputEnd, groupedIm2ColDimOrder);

    SmallVector<AffineExpr> inputExprs;
    SmallVector<AffineExpr> filterExprs;
    SmallVector<AffineExpr> outputExprs;

    for (auto group : groupedIm2ColDimOrder)
      inputExprs.push_back(leadingGroupDimToIterationDim[group[0]]);
    for (auto group : groupedFilterDimOrder)
      filterExprs.push_back(leadingGroupDimToIterationDim[group[0]]);
    for (auto group : groupedOutputDimOrder)
      outputExprs.push_back(leadingGroupDimToIterationDim[group[0]]);

    auto newInputMap = AffineMap::get(iterationDim, 0, inputExprs, context);
    auto newFilterMap = AffineMap::get(iterationDim, 0, filterExprs, context);
    auto newOutputMap = AffineMap::get(iterationDim, 0, outputExprs, context);

    LLVM_DEBUG({
      DBGS() << "im2col: Indexing maps.\n";
      if (filterRHS) {
        DBGS() << "im2ColMap: " << newInputMap << "\n";
        DBGS() << "filterMap: " << newFilterMap << "\n";
      } else {
        DBGS() << "filterMap: " << newFilterMap << "\n";
        DBGS() << "im2ColMap: " << newInputMap << "\n";
      }
      DBGS() << "outputMap: " << newOutputMap << "\n";
    });

    auto newGenericOp = rewriter.create<linalg::GenericOp>(
        loc, reshapedOutputType,
        /*inputs=*/
        filterRHS ? ValueRange{reshapedInput, reshapedFilter}
                  : ValueRange{reshapedFilter, reshapedInput},
        /*outputs=*/ValueRange{reshapedOutput},
        filterRHS
            ? ArrayRef<AffineMap>{newInputMap, newFilterMap, newOutputMap}
            : ArrayRef<AffineMap>{newFilterMap, newInputMap, newOutputMap},
        iteratorTypes);
    IRMapping mapper;
    genericOp.getRegion().cloneInto(&newGenericOp.getRegion(), mapper);

    if (!filterRHS)
      swapGenericBinaryInputArgs(newGenericOp);

    Value result = newGenericOp.getResults().front();

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputType, result, outputReassocIndices);

    rewriter.replaceOp(genericOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
template <typename Conv2DOpType>
class Convert1x1FilterConvToMatmul : public OpRewritePattern<Conv2DOpType> {
public:
  using OpRewritePattern<Conv2DOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOpType convOp,
                                PatternRewriter &rewriter) const override {
    auto inputShapeType = llvm::dyn_cast<RankedTensorType>(
        convOp.getDpsInputOperand(0)->get().getType());
    auto filterShapeType = llvm::dyn_cast<RankedTensorType>(
        convOp.getDpsInputOperand(1)->get().getType());
    auto outputShapeType = llvm::dyn_cast<RankedTensorType>(
        convOp.getDpsInitOperand(0)->get().getType());

    const bool isNCHW = isa<linalg::Conv2DNchwFchwOp>(convOp);
    const bool isNHWC = isa<linalg::Conv2DNhwcHwcfOp>(convOp);
    if (!isNCHW & !isNHWC)
      return failure();

    if (!inputShapeType || !filterShapeType || !outputShapeType)
      return failure();

    auto inputShape = inputShapeType.getShape();
    auto filterShape = filterShapeType.getShape();
    auto outputShape = outputShapeType.getShape();

    // Adjusting dimension indices based on Conv2DOpType.
    const int nIndex = 0;
    const int kcIndex = isNHWC ? 2 : 1;
    const int kfIndex = isNHWC ? 3 : 0;
    const int khIndex = isNHWC ? 0 : 2;
    const int kwIndex = isNHWC ? 1 : 3;
    const int ohIndex = isNHWC ? 1 : 2;
    const int owIndex = isNHWC ? 2 : 3;
    const int ocIndex = isNHWC ? 3 : 1;

    bool isInputHWDynamic = ShapedType::isDynamic(inputShape[ohIndex]) &&
                            ShapedType::isDynamic(inputShape[owIndex]);

    // We cannot merge the width and height if they are both dynamic as we
    // cannot expand them back to their dynamic values.
    if (isInputHWDynamic)
      return failure();

    if (filterShape[khIndex] != 1 || filterShape[kwIndex] != 1)
      return failure();

    // TODO(ataei): Support conversion to linalg.batch_matmul.
    if (inputShape[0] != 1) {
      auto newGenericOp = linalg::generalizeNamedOp(rewriter, convOp);
      if (failed(newGenericOp)) {
        return failure();
      }
      return success();
    }

    if (!llvm::all_of(convOp.getStrides(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();
    if (!llvm::all_of(convOp.getDilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    auto combineDims = [](int64_t a, int64_t b) {
      if (ShapedType::isDynamic(a) || ShapedType::isDynamic(b))
        return ShapedType::kDynamic;
      return a * b;
    };

    SmallVector<ReassociationIndices> reassociationInputOutputIndices;
    SmallVector<ReassociationIndices> reassociationFilterIndices;
    SmallVector<int64_t> reshapedInputShape(2, 0);
    SmallVector<int64_t> reshapedFilterShape(2, 0);
    SmallVector<int64_t> reshapedOutputShape(2, 0);
    if (isNHWC) {
      // Generate reassociation indices.
      reassociationInputOutputIndices = {{nIndex, ohIndex, owIndex}, {ocIndex}};
      reassociationFilterIndices = {{khIndex, kwIndex, kcIndex}, {kfIndex}};

      // Generate matmul shapes from 1x1 conv.
      reshapedInputShape = {
          combineDims(inputShape[ohIndex], inputShape[owIndex]),
          inputShape[ocIndex]};
      reshapedFilterShape = {filterShape[kcIndex], filterShape[kfIndex]};
      reshapedOutputShape = {
          combineDims(outputShape[ohIndex], outputShape[owIndex]),
          outputShape[ocIndex]};
    } else if (isNCHW) {
      // Generate reassociation indices.
      reassociationInputOutputIndices = {{nIndex, ocIndex}, {ohIndex, owIndex}};
      reassociationFilterIndices = {{kfIndex}, {kcIndex, khIndex, kwIndex}};

      // Generate matmul shapes from 1x1 conv.
      reshapedInputShape = {
          inputShape[ocIndex],
          combineDims(inputShape[ohIndex], inputShape[owIndex])};
      reshapedFilterShape = {filterShape[kfIndex], filterShape[kcIndex]};
      reshapedOutputShape = {
          outputShape[ocIndex],
          combineDims(outputShape[ohIndex], outputShape[owIndex])};
    }

    auto reshapedInputType = RankedTensorType::get(
        reshapedInputShape, inputShapeType.getElementType());

    auto reshapedFilterType = RankedTensorType::get(
        reshapedFilterShape, filterShapeType.getElementType());

    auto reshapedOutputType = RankedTensorType::get(
        reshapedOutputShape, outputShapeType.getElementType());

    Value input = convOp.getDpsInputOperand(0)->get();
    Value filter = convOp.getDpsInputOperand(1)->get();
    Value output = convOp.getDpsInitOperand(0)->get();
    auto loc = convOp.getLoc();

    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, reassociationInputOutputIndices);
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, reassociationFilterIndices);
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, reassociationInputOutputIndices);

    SmallVector<Value, 2> matmulInput;
    if (isNHWC) {
      matmulInput = {reshapedInput, reshapedFilter};
    } else if (isNCHW) {
      matmulInput = {reshapedFilter, reshapedInput};
    }
    auto matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType, matmulInput, ArrayRef<Value>{reshapedOutput});

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputShapeType, matmulResult.getResults()[0],
        reassociationInputOutputIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

struct Convert1X1FilterConv2DToMatmulPass
    : public Convert1X1FilterConv2DToMatmulBase<
          Convert1X1FilterConv2DToMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<Convert1x1FilterConvToMatmul<linalg::Conv2DNhwcHwcfOp>,
                    Convert1x1FilterConvToMatmul<linalg::Conv2DNchwFchwOp>>(
        context);
    patterns.insert<Convert1x1FilterConvGenericToMatmul>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> createConvert1X1FilterConv2DToMatmulPass() {
  return std::make_unique<Convert1X1FilterConv2DToMatmulPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
