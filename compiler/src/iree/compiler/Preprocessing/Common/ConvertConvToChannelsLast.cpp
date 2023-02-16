// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-preprocessing-convert-conv-to-channels-last"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::Preprocessing {

using TransposeIndices = SmallVector<int64_t, 4>;
using ConvBuilderFn = std::function<Value(
    OpBuilder &b, Location loc, linalg::LinalgOp srcConv, Value input,
    Value filter, Value output, AffineMap inputMap, AffineMap filterMap,
    AffineMap outputMap, SmallVector<unsigned> newDimOrder,
    SmallVector<utils::IteratorType> newIteratorTypes)>;
using linalg::detail::MatchConvolutionResult;

// Helper function to build a convolution-like generic op with the given
// indexing maps and packed inputs. |newDimOrder| specifies a permutation
// on the map iterators for the new linalg op.
static Value
defaultConvBuilderFn(OpBuilder &b, Location loc, linalg::LinalgOp srcConv,
                     Value input, Value filter, Value output,
                     AffineMap inputMap, AffineMap filterMap,
                     AffineMap outputMap, SmallVector<unsigned> newDimOrder,
                     SmallVector<utils::IteratorType> newIteratorTypes) {
  AffineMap newInputMap = inputMap;
  AffineMap newFilterMap = filterMap;
  AffineMap newOutputMap = outputMap;
  if (!newDimOrder.empty()) {
    DenseMap<AffineExpr, AffineExpr> dimMap;
    for (auto [newDim, oldDim] : llvm::enumerate(newDimOrder)) {
      dimMap[b.getAffineDimExpr(oldDim)] = b.getAffineDimExpr(newDim);
    }
    newInputMap = inputMap.replace(dimMap,
                                   /*numResultDims=*/newDimOrder.size(),
                                   /*numResultSymbols=*/0);
    newFilterMap = filterMap.replace(dimMap,
                                     /*numResultDims=*/newDimOrder.size(),
                                     /*numResultSymbols=*/0);
    newOutputMap = outputMap.replace(dimMap,
                                     /*numResultDims=*/newDimOrder.size(),
                                     /*numResultSymbols=*/0);
  }
  SmallVector<utils::IteratorType> iterators = srcConv.getIteratorTypesArray();
  iterators.append(newIteratorTypes);
  auto genericConv = b.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{input, filter}, output,
      ArrayRef<AffineMap>{newInputMap, newFilterMap, newOutputMap}, iterators);
  IRMapping mapper;
  srcConv->getRegion(0).cloneInto(&genericConv.getRegion(), mapper);
  return genericConv.getResult(0);
}

template <typename sourceNamedConvTy, typename targetNamedConvTy>
static Value
namedConvBuilderFn(OpBuilder &b, Location loc, linalg::LinalgOp srcConv,
                   Value input, Value filter, Value output, AffineMap inputMap,
                   AffineMap filterMap, AffineMap outputMap,
                   SmallVector<unsigned> newDimOrder,
                   SmallVector<utils::IteratorType> newIteratorTypes) {
  sourceNamedConvTy namedConv = cast<sourceNamedConvTy>(srcConv);
  return b
      .create<targetNamedConvTy>(
          loc, output.getType(), ValueRange{input, filter}, output,
          namedConv.getStrides(), namedConv.getDilations())
      .getResult(0);
}

// Normalizes the given permutation vector. Expects the input permutation is
// shifted by a constant, for example, [4, 6, 5] -> [0, 2, 1].
static TransposeIndices getNormalizedIndices(TransposeIndices targetIndices) {
  int startDim = *std::min_element(targetIndices.begin(), targetIndices.end());
  TransposeIndices normalized(targetIndices.size());
  for (auto i : llvm::enumerate(targetIndices)) {
    normalized[i.index()] = i.value() - startDim;
  }
  return normalized;
}

// Inverts the given shifted permutation vector. For example,
// [2, 0, 1] + 4 -> [1, 2, 0] + 4
// [6, 4, 5] -> [5, 6, 4]
static TransposeIndices invertIndices(TransposeIndices targetIndices) {
  int startDim = *std::min_element(targetIndices.begin(), targetIndices.end());
  TransposeIndices inverted(targetIndices.size());
  for (auto i : llvm::enumerate(targetIndices)) {
    inverted[i.value() - startDim] = i.index() + startDim;
  }
  return inverted;
}

// Indicates whether the given permutation vector is a minor identity
// for a permutation of the given |rank|.
static bool isInnerIdentityIndices(TransposeIndices indices, int64_t rank) {
  if (indices.empty()) {
    return true;
  }
  int64_t base = indices[0];
  return llvm::all_of(
             llvm::enumerate(indices),
             [base](auto e) { return e.value() - base == e.index(); }) &&
         indices.back() == rank - 1;
}

// Helper to shuffle vectors according to the transpose indices.
template <typename T>
static SmallVector<T> shuffleFromIndices(SmallVector<T> unshuffled,
                                         TransposeIndices targetIndices) {
  int startDim = *std::min_element(targetIndices.begin(), targetIndices.end());
  SmallVector<T> shuffled(unshuffled);
  for (auto i : llvm::enumerate(targetIndices)) {
    shuffled[i.index() + startDim] = unshuffled[i.value()];
  }
  return shuffled;
}

// Helper to separate the elements in |vec| affected by the |targetIndices|.
// For example,
// vec = [d0, d1, d2]
// targetIndices = [2, 0]
// ret = [d1 | d2, d0]
template <typename T>
static SmallVector<T> getPackedVector(SmallVector<T> vec,
                                      TransposeIndices targetIndices) {
  SmallVector<T> packedShape;
  for (auto [i, val] : llvm::enumerate(vec)) {
    if (!llvm::is_contained(targetIndices, i)) {
      packedShape.push_back(val);
    }
  }
  for (auto i : targetIndices) {
    packedShape.push_back(vec[i]);
  }
  return packedShape;
}

// Helper to construct a reassociation map for a collapse shape assuming all
// outer dimensions are tiled to `1`.
static SmallVector<ReassociationIndices, 4>
getUnitOuterDimPackReassociationMap(TransposeIndices targetIndices,
                                    int64_t rank) {
  int startDim = *std::min_element(targetIndices.begin(), targetIndices.end());
  int dimCount = targetIndices.size();
  SmallVector<ReassociationIndices, 4> reassociationMap;
  for (int i = 0; i <= startDim; i++) {
    reassociationMap.push_back({i});
  }
  for (int i = startDim + 1; i < dimCount + startDim + 1; i++) {
    reassociationMap[startDim].push_back(i);
  }
  for (int i = dimCount + startDim + 1; i < dimCount + rank; i++) {
    reassociationMap.push_back({i});
  }
  return reassociationMap;
}

// Transpose the given tensor based on the given transpose indices using a
// tensor.pack. Additionally returns a new AffineMap for the packed value
// assuming otherwise the same iteration space.
static std::tuple<Value, std::optional<tensor::PackOp>, AffineMap>
createTransposeAsTensorPack(
    PatternRewriter &rewriter, Location loc, Value input, AffineMap inputMap,
    TransposeIndices targetIndices, int tilingFactor,
    llvm::DenseMap<int64_t, int64_t> innerDimToDomainDim) {
  if (isInnerIdentityIndices(targetIndices, inputMap.getNumResults())) {
    return std::make_tuple(input, std::nullopt, inputMap);
  }

  RankedTensorType inType = input.getType().cast<RankedTensorType>();
  auto elementType = inType.getElementType();
  auto inputShape(inType.getShape());

  SmallVector<OpFoldResult> transposedTileSizes(
      targetIndices.size(), rewriter.getIndexAttr(tilingFactor));
  if (tilingFactor <= 0) {
    for (auto [index, i] : llvm::enumerate(targetIndices)) {
      if (ShapedType::isDynamic(inputShape[i])) {
        transposedTileSizes[index] =
            rewriter.create<tensor::DimOp>(loc, input, i).getResult();
      } else {
        transposedTileSizes[index] = rewriter.getIndexAttr(inputShape[i]);
      }
    }
  }

  // Pack the input tensor.
  auto empty = tensor::PackOp::createDestinationTensor(
      rewriter, loc, input, transposedTileSizes, targetIndices,
      SmallVector<int64_t>{});
  auto packedInput = rewriter.create<tensor::PackOp>(
      loc, input, empty, targetIndices, transposedTileSizes,
      /*padding=*/std::nullopt, SmallVector<int64_t>{});

  SmallVector<AffineExpr> mapResults(inputMap.getResults());
  AffineMap transposedMap;

  Value packedOperand = packedInput;
  // Collapse the unit dims created by tensor.pack if the pack is just a
  // transpose.
  if (tilingFactor <= 0) {
    auto reassociationMap =
        getUnitOuterDimPackReassociationMap(targetIndices, inType.getRank());
    auto transposedInputShape =
        getPackedVector<int64_t>(llvm::to_vector(inputShape), targetIndices);
    packedOperand =
        rewriter
            .create<tensor::CollapseShapeOp>(
                loc, RankedTensorType::get(transposedInputShape, elementType),
                packedOperand, reassociationMap)
            .getResult();
    transposedMap =
        AffineMap::get(inputMap.getNumDims(), inputMap.getNumSymbols(),
                       getPackedVector<AffineExpr>(mapResults, targetIndices),
                       input.getContext());
  } else {
    for (auto innerDim : targetIndices) {
      mapResults.push_back(rewriter.getAffineDimExpr(
          innerDimToDomainDim[inputMap.getDimPosition(innerDim)]));
    }
    transposedMap = AffineMap::get(
        inputMap.getNumDims() + innerDimToDomainDim.size(),
        inputMap.getNumSymbols(), mapResults, input.getContext());
  }

  return std::make_tuple(packedOperand, packedInput, transposedMap);
}

// Unpacks the given |output| value based on the |packOp| on the corresponding
// destination-passing-style init. If the pack is just being used as a
// transpose, this will first expand the shape of the output to include the
// unit dimensions necessary for the unpack.
static Value createTransposeAsTensorUnPack(PatternRewriter &rewriter,
                                           Location loc, Value output,
                                           tensor::PackOp packOp,
                                           int tilingFactor) {
  Value packedOutput = output;
  if (tilingFactor <= 0) {
    RankedTensorType outType = output.getType().cast<RankedTensorType>();
    auto elementType = outType.getElementType();
    auto outputShape(outType.getShape());
    int64_t rank = outType.getRank();
    TransposeIndices targetIndices(packOp.getInnerDimsPos());

    int startDim =
        *std::min_element(targetIndices.begin(), targetIndices.end());
    SmallVector<int64_t> expandedOutputShape;
    for (int i = 0, e = startDim; i < e; i++) {
      expandedOutputShape.push_back(outputShape[i]);
    }
    for (int i = 0, e = targetIndices.size(); i < e; i++) {
      expandedOutputShape.push_back(1);
    }
    for (int i = startDim, e = rank; i < e; i++) {
      expandedOutputShape.push_back(outputShape[i]);
    }

    auto reassociationMap =
        getUnitOuterDimPackReassociationMap(targetIndices, rank);
    packedOutput =
        rewriter
            .create<tensor::ExpandShapeOp>(
                loc, RankedTensorType::get(expandedOutputShape, elementType),
                output, reassociationMap)
            .getResult();
  }

  Value empty = tensor::UnPackOp::createDestinationTensor(
      rewriter, loc, packedOutput, packOp.getMixedTiles(),
      packOp.getInnerDimsPos(), packOp.getOuterDimsPerm());

  auto unpackedOutput = rewriter.create<tensor::UnPackOp>(
      loc, packedOutput, empty, packOp.getInnerDimsPos(),
      packOp.getMixedTiles(), packOp.getOuterDimsPerm());
  return unpackedOutput.getResult();
}

// Given a list of categorized dimensions and an affine map, returns a
// permutation on the map results such that all dimensions in one of the
// categories is accounted for and in the order given by the categories.
// For example
//
// map = ... (d0, d1, d2, d3)
// transposeDimTargets = [[d2], [d1]]
// ret = [2, 1]
static TransposeIndices collectChannelTransposeIndices(
    AffineMap map, SmallVector<SmallVector<unsigned, 2>> transposeDimTargets) {
  SmallVector<TransposeIndices> channelIndices(transposeDimTargets.size());
  for (auto [index, result] : llvm::enumerate(map.getResults())) {
    if (isa<AffineDimExpr>(result)) {
      for (auto [channelVec, dimCategory] :
           llvm::zip_equal(channelIndices, transposeDimTargets)) {
        if (llvm::is_contained(dimCategory,
                               cast<AffineDimExpr>(result).getPosition())) {
          channelVec.push_back(index);
          break;
        }
      }
    }
  }

  TransposeIndices indices;
  for (auto channelVec : channelIndices) {
    indices.append(channelVec);
  }
  return indices;
}

// Helper to transpose the input and output channel dimensions to be inner
// most on the input and output channels. Takes a tiling factor to tile the
// channel dimensions by with a pack. if |tilingFactor| is <= 0, then it
// fully transposes the channel dimensions. Takes a builder function to allow
// switching between named op conversions and conversions for generics while
// using the same transposing/packing logic.
static LogicalResult
transposeConvLikeLinalgOp(PatternRewriter &rewriter, linalg::LinalgOp convOp,
                          int tilingFactor,
                          ConvBuilderFn convBuilder = defaultConvBuilderFn) {
  Location loc = convOp.getLoc();

  linalg::ConvolutionDimensions convDims;
  auto errString = getMatchConvolutionMessage(
      linalg::detail::isConvolutionInterfaceImpl(convOp, &convDims));
  if (!errString.empty()) {
    return failure();
  }

  if (convDims.inputChannel.size() > 1) {
    return failure();
  }

  if (convDims.outputChannel.size() > 1) {
    return failure();
  }

  // TODO: Support depthwise convolutions
  if (!convDims.depth.empty()) {
    return failure();
  }

  Value input = convOp->getOperand(0);
  Value filter = convOp->getOperand(1);
  Value output = convOp->getOperand(2);

  auto inputMap = convOp.getIndexingMapsArray()[0];
  auto filterMap = convOp.getIndexingMapsArray()[1];
  auto outputMap = convOp.getIndexingMapsArray()[2];

  auto inputIndices =
      collectChannelTransposeIndices(inputMap, {convDims.inputChannel});
  auto filterIndices = collectChannelTransposeIndices(
      filterMap, {convDims.inputChannel, convDims.outputChannel});
  auto outputIndices =
      collectChannelTransposeIndices(outputMap, {convDims.outputChannel});

  llvm::errs() << "Checking identity\n";
  // If the dimensions to transpose/pack are already in the correct order and
  // inner most, nothing to do.
  if (isInnerIdentityIndices(inputIndices, inputMap.getNumResults()) &&
      isInnerIdentityIndices(filterIndices, filterMap.getNumResults()) &&
      isInnerIdentityIndices(outputIndices, outputMap.getNumResults())) {
    return failure();
  }

  int nDims = outputMap.getNumDims();
  llvm::DenseMap<int64_t, int64_t> innerDimsToDomainDims;
  for (auto [index, dim] : llvm::enumerate(convDims.inputChannel)) {
    innerDimsToDomainDims[dim] = nDims + index;
  }
  for (auto [index, dim] : llvm::enumerate(convDims.outputChannel)) {
    innerDimsToDomainDims[dim] = nDims + index + convDims.inputChannel.size();
  }

  auto [transposedInput, inputPack, transposedInputMap] =
      createTransposeAsTensorPack(rewriter, loc, input, inputMap, inputIndices,
                                  tilingFactor, innerDimsToDomainDims);
  auto [transposedFilter, filterPack, transposedFilterMap] =
      createTransposeAsTensorPack(rewriter, loc, filter, filterMap,
                                  filterIndices, tilingFactor,
                                  innerDimsToDomainDims);
  auto [transposedOutput, outputPack, transposedOutputMap] =
      createTransposeAsTensorPack(rewriter, loc, output, outputMap,
                                  outputIndices, tilingFactor,
                                  innerDimsToDomainDims);

  llvm::errs() << "Checking do something\n";
  // Don't transpose if there's no change to the op.
  if (transposedInputMap == inputMap && transposedFilterMap == filterMap &&
      transposedOutputMap == outputMap) {
    return failure();
  }

  Value convDest = transposedOutput;
  // If the output is a fill op, pack the fill directly.
  if (auto fillOp = output.getDefiningOp<linalg::FillOp>()) {
    if (outputPack) {
      auto outputDest = outputPack->getDest().getDefiningOp<tensor::EmptyOp>();
      auto elementType = outputDest.getType().getElementType();

      auto dimToTileMapping = outputPack->getDimAndTileMapping();
      SmallVector<OpFoldResult> mixedSizes = outputDest.getMixedSizes();
      SmallVector<OpFoldResult> packedSizes;
      for (auto [index, size] : llvm::enumerate(mixedSizes)) {
        if (!dimToTileMapping.count(index) || tilingFactor > 0) {
          packedSizes.push_back(size);
        }
      }

      auto emptyOp =
          rewriter.create<tensor::EmptyOp>(loc, packedSizes, elementType);

      convDest = rewriter
                     .create<linalg::FillOp>(loc, fillOp.getInputs(),
                                             emptyOp.getResult())
                     .result();
    }
  }

  SmallVector<unsigned> newDimOrder;
  SmallVector<utils::IteratorType> newIteratorTypes;
  if (tilingFactor <= 0) {
    newDimOrder.append(convDims.batch);
    newDimOrder.append(convDims.outputImage);
    newDimOrder.append(convDims.outputChannel);
    newDimOrder.append(convDims.filterLoop);
    newDimOrder.append(convDims.inputChannel);
  } else {
    newIteratorTypes.append(convDims.inputChannel.size(),
                            utils::IteratorType::reduction);
    newIteratorTypes.append(convDims.outputChannel.size(),
                            utils::IteratorType::parallel);
  }

  // Invoke the builder function. For named op -> named op conversions, this
  // will construct the target named op, else it constructs a convolution like
  // generic.
  Value transposedConvResult =
      convBuilder(rewriter, loc, convOp, transposedInput, transposedFilter,
                  convDest, transposedInputMap, transposedFilterMap,
                  transposedOutputMap, newDimOrder, newIteratorTypes);

  Value returnToNCHW = transposedConvResult;
  if (outputPack) {
    returnToNCHW = createTransposeAsTensorUnPack(
        rewriter, loc, transposedConvResult, *outputPack, tilingFactor);
  }

  llvm::errs() << "done\n";
  rewriter.replaceOp(convOp, returnToNCHW);
  return success();
}

namespace {

//=====================================================================
// Convolution packing patterns
//=====================================================================

// Named op -> named op conversions if a default inner tile size is specified.

struct ConvertLinalgConvNchwFchw : OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  using OpRewritePattern::OpRewritePattern;
  ConvertLinalgConvNchwFchw(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    return transposeConvLikeLinalgOp(
        rewriter, convOp, /*tilingFactor=*/-1,
        namedConvBuilderFn<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>);
  }
};

// Default convolution-like transposing pattern for any linalg op to a generic.
// Has a lower benefit than the named op conversions.

struct ConvertLinalgConvOp : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  ConvertLinalgConvOp(MLIRContext *context, int tile,
                      PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
        tilingFactor(tile) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return transposeConvLikeLinalgOp(rewriter, op, tilingFactor);
  }

private:
  int tilingFactor;
};

//=====================================================================
// Generalization patterns
//=====================================================================

// Returns a reassocation map of the given |rank| such that all dimensions
// in the set |innerDims| map to two dimensions. For example,
//
// rank = 4, innerDims = {1, 3}
// map = [[0], [1, 2], [3], [4, 5]]
template <typename SetTy>
static SmallVector<ReassociationIndices>
getTilingReassociationMap(int64_t rank, SetTy innerDims) {
  SmallVector<ReassociationIndices> map;
  int64_t nTiled = 0;
  for (int64_t i = 0, e = rank; i < e; i++) {
    if (innerDims.contains(i)) {
      map.push_back({i + nTiled++, i + nTiled});
      continue;
    }
    map.push_back({i + nTiled});
  }
  return map;
}

// Generalizes pack operations such that all tiled dimensions have unit outer
// dims. Produces a transpose on the tiled dimensions followed by an
// expand_shape to introduce the outer unit dims. For example,
//
// tensor.pack inner_dims_pos = [1] inner_tiles = [64]
//   : tensor<32x64x16xf32> to tensor<32x1x16x64xf32>
//
// Generalizes to:
//
// linalg.transpose ... tensor<32x64x16xf32> to tensor<32x16x64xf32>
// tensor.expand_shape ... tensor<32x16x64xf32> to tensor<32x1x16x64xf32>
class GeneralizeOuterUnitDimsPackOp final
    : public OpRewritePattern<tensor::PackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  GeneralizeOuterUnitDimsPackOp(MLIRContext *context,
                                PatternBenefit benefit = 2)
      : OpRewritePattern<tensor::PackOp>(context, benefit) {}

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (!packOp.getOuterDimsPerm().empty())
      return failure();
    if (packOp.getPaddingValue())
      return failure();

    RankedTensorType destType =
        packOp.getDest().getType().cast<RankedTensorType>();
    ArrayRef<int64_t> destShape = destType.getShape();
    ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();

    if (llvm::any_of(innerDimsPos, [destShape](int64_t index) {
          return destShape[index] != 1;
        })) {
      return rewriter.notifyMatchFailure(packOp,
                                         "require the tiled outer dimensions "
                                         "of the packed tensor are all unit");
    }

    // Collect the set of transposed dimensions.
    llvm::DenseSet<int64_t> innerDims;
    for (auto innerDim : innerDimsPos) {
      innerDims.insert(innerDim);
    }

    // Construct the permutation for the transpose. It is constructed as
    // [untiled_outer_dims, inner_dims_pos].
    int64_t srcRank = packOp.getSourceRank();
    TransposeIndices perm;
    for (int i = 0, e = srcRank; i < e; i++) {
      if (!innerDims.count(i)) {
        perm.push_back(i);
      }
    }
    perm.append(innerDimsPos.begin(), innerDimsPos.end());
    Location loc = packOp.getLoc();

    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, loc, packOp.getSource());
    applyPermutationToVector(mixedSizes, perm);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, mixedSizes,
                                                   destType.getElementType());
    Value transposed =
        rewriter
            .create<linalg::TransposeOp>(loc, packOp.getSource(), empty, perm)
            .getResult()[0];
    // Expand the unit dimensions for the result of the pack.
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        packOp, destType, transposed,
        getTilingReassociationMap(srcRank, innerDims));
    return success();
  }
};

// Generalizes pack operations without padding and outer dimension permutations
// For example:
//
// tensor.pack inner_dims_pos = [1] inner_tiles = [32]
//   : tensor<32x64x16xf32> to tensor<32x2x16x32xf32>
//
// Generalizes to:
//
// tensor.expand_shape ... tensor<32x64x16xf32> to tensor<32x2x32x16xf32>
// linalg.transpose ... tensor<32x2x32x16xf32> to tensor<32x2x16x32xf32>
class GeneralizeUnpermutedAndUnpaddedPackOp final
    : public OpRewritePattern<tensor::PackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  GeneralizeUnpermutedAndUnpaddedPackOp(MLIRContext *context,
                                        PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::PackOp>(context, benefit) {}

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (!packOp.getOuterDimsPerm().empty())
      return failure();
    if (packOp.getPaddingValue())
      return failure();

    RankedTensorType srcType =
        packOp.getSource().getType().cast<RankedTensorType>();
    int64_t rank = srcType.getRank();
    auto innerDimsPos = packOp.getInnerDimsPos();
    llvm::DenseMap<int64_t, int64_t> innerDims;
    for (auto [index, innerDim] : llvm::enumerate(innerDimsPos))
      innerDims[innerDim] = index;

    llvm::DenseMap<int64_t, int64_t> innerDimsToExpandedDims;
    TransposeIndices perm;
    int64_t nTiled = 0;
    for (int i = 0, e = rank; i < e; i++) {
      perm.push_back(i + nTiled);
      if (innerDims.count(i))
        innerDimsToExpandedDims[i] = i + ++nTiled;
    }
    for (auto i : innerDimsPos)
      perm.push_back(innerDimsToExpandedDims[i]);

    RankedTensorType destType =
        packOp.getDest().getType().cast<RankedTensorType>();
    SmallVector<int64_t> destShape(destType.getShape());
    applyPermutationToVector<int64_t>(destShape, invertPermutationVector(perm));

    auto expand = rewriter.create<tensor::ExpandShapeOp>(
        packOp.getLoc(),
        RankedTensorType::get(destShape, destType.getElementType()),
        packOp.getSource(), getTilingReassociationMap(rank, innerDims));

    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(packOp, expand,
                                                     packOp.getDest(), perm);
    return success();
  }
};

// Generalizes unpack operations if all tiled dimensions have unit outer dims,
// and thus no padding. Produces a collapse_shape to remove the unit dimensions
// followed by a transpose. For example:
//
// tensor.unpack inner_dims_pos = [1] inner_tiles = [64]
//   : tensor<32x1x16x64xf32> to tensor<32x64x16xf32>
//
// Generalizes to:
//
// tensor.collapse_shape ... tensor<32x1x16x64xf32> to tensor<32x16x64xf32>
// linalg.transpose ... tensor<32x16x64xf32> to tensor<32x64x16xf32>
class GeneralizeOuterUnitDimsUnPackOp final
    : public OpRewritePattern<tensor::UnPackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  GeneralizeOuterUnitDimsUnPackOp(MLIRContext *context,
                                  PatternBenefit benefit = 2)
      : OpRewritePattern<tensor::UnPackOp>(context, benefit) {}

  LogicalResult matchAndRewrite(tensor::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    if (!unpackOp.getOuterDimsPerm().empty())
      return failure();

    RankedTensorType srcType =
        unpackOp.getSource().getType().cast<RankedTensorType>();
    ArrayRef<int64_t> srcShape = srcType.getShape();

    ArrayRef<int64_t> innerDimsPos = unpackOp.getInnerDimsPos();
    if (llvm::any_of(innerDimsPos, [srcShape](int64_t index) {
          return srcShape[index] != 1;
        })) {
      return rewriter.notifyMatchFailure(unpackOp,
                                         "require the tiled outer dimensions "
                                         "of the packed tensor are all unit");
    }

    llvm::DenseSet<int64_t> innerDims;
    for (auto innerDim : innerDimsPos) {
      innerDims.insert(innerDim);
    }

    RankedTensorType destType =
        unpackOp.getDest().getType().cast<RankedTensorType>();
    TransposeIndices perm;
    for (int i = 0, e = destType.getRank(); i < e; i++) {
      if (!innerDims.count(i)) {
        perm.push_back(i);
      }
    }
    perm.append(innerDimsPos.begin(), innerDimsPos.end());
    Location loc = unpackOp.getLoc();

    ArrayRef<int64_t> destShape = destType.getShape();
    auto collapsedType = RankedTensorType::get(
        applyPermutation(destShape, perm), destType.getElementType());

    auto collapse = rewriter.create<tensor::CollapseShapeOp>(
        loc, collapsedType, unpackOp.getSource(),
        getTilingReassociationMap(destType.getRank(), innerDims));
    rewriter
        .replaceOpWithNewOp<linalg::TransposeOp>(unpackOp, collapse,
                                                 unpackOp.getDest(),
                                                 invertPermutationVector(perm))
        .getResult()[0];
    return success();
  }
};

// Generalizes unpack operations without padding and outer dimension
// permutations. For example:
//
// tensor.unpack inner_dims_pos = [1] inner_tiles = [32]
//   : tensor<32x2x16x32xf32> to tensor<32x64x16xf32>
//
// Generalizes to:
//
// linalg.transpose ... tensor<32x2x16x32xf32> to tensor<32x2x32x16xf32>
// tensor.collapse_shape ... tensor<32x2x32x16xf32> to tensor<32x64x16xf32>
class GeneralizeUnpermutedAndUnpaddedUnPackOp final
    : public OpRewritePattern<tensor::UnPackOp> {
public:
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    if (!unpackOp.getOuterDimsPerm().empty())
      return failure();

    if (!unpackOp.getDest().getDefiningOp<tensor::EmptyOp>())
      return failure();

    RankedTensorType destType =
        unpackOp.getDest().getType().cast<RankedTensorType>();
    int64_t rank = destType.getRank();
    auto innerDimsPos = unpackOp.getInnerDimsPos();
    llvm::DenseMap<int64_t, int64_t> innerDims;
    for (auto [index, innerDim] : llvm::enumerate(innerDimsPos))
      innerDims[innerDim] = index;

    TransposeIndices perm;
    for (int i = 0, e = rank; i < e; i++) {
      perm.push_back(i);
      if (innerDims.count(i)) {
        perm.push_back(rank + innerDims[i]);
      }
    }

    Location loc = unpackOp.getLoc();
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, loc, unpackOp.getSource());
    applyPermutationToVector<OpFoldResult>(mixedSizes, perm);
    auto elType = getElementTypeOrSelf(unpackOp.getDest());

    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, mixedSizes, elType);

    Value transpose = rewriter
                          .create<linalg::TransposeOp>(
                              loc, unpackOp.getSource(), emptyOp, perm)
                          ->getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        unpackOp, destType, transpose,
        getTilingReassociationMap(rank, innerDims));
    return success();
  }
};

struct ConvertConvToChannelsLastPass
    : public ConvertConvToChannelsLastBase<ConvertConvToChannelsLastPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
  }
  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    tilingFactor = tileSize;
    return success();
  }

  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = &getContext();

    // First pack/transpose all convolution like ops, trying to do a named op
    // to named op conversion if possible.
    {
      RewritePatternSet patterns(context);
      if (tilingFactor <= 0) {
        patterns.insert<ConvertLinalgConvNchwFchw>(context);
      }
      patterns.insert<ConvertLinalgConvOp>(context, tilingFactor);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LDBG("after converting convolutions to channels last\n" << op);

    // Propagate packs introduced by the conversion patterns through adjacent
    // pads. Note that packs introduced by the above patterns will never include
    // padding.
    {
      RewritePatternSet patterns(context);
      linalg::populateDataLayoutPropagationPatterns(
          patterns, [](Operation *op) { return true; });
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LDBG("after propagating packs/unpacks\n" << op);

    // Run pack/unpack canonicalization to try to cancel any packs.
    {
      RewritePatternSet patterns(context);
      tensor::PackOp::getCanonicalizationPatterns(patterns, context);
      tensor::UnPackOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LDBG("after canonicalizing packs/unpacks\n" << op);

    // Generalize leftover packs and unpacks to allow for transpose propagation
    // and unit dim folding to handle them more effectively.
    {
      RewritePatternSet patterns(context);
      patterns.insert<GeneralizeUnpermutedAndUnpaddedPackOp>(context);
      patterns.insert<GeneralizeUnpermutedAndUnpaddedUnPackOp>(context);
      patterns.insert<GeneralizeOuterUnitDimsPackOp>(context);
      patterns.insert<GeneralizeOuterUnitDimsUnPackOp>(context);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LDBG("after generalizing all remaining packs/unpacks\n" << op);
  }

private:
  int64_t tilingFactor;
};

} // namespace

std::unique_ptr<Pass> createConvertConvToChannelsLastPass() {
  return std::make_unique<ConvertConvToChannelsLastPass>();
}

} // namespace mlir::iree_compiler::Preprocessing
