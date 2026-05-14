// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-ext-convert-conv-to-im2col-op"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_CONVERTCONVTOIM2COLOPPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

static bool hasAllOneValues(ArrayRef<int64_t> attr) {
  return llvm::all_of(attr, [](int64_t element) { return element == 1; });
}

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = isa<IntegerType>(x.getType());
  if (isInt) {
    return arith::AddIOp::create(builder, loc, x, y);
  }
  return arith::AddFOp::create(builder, loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = isa<IntegerType>(x.getType());
  if (isInt) {
    return arith::MulIOp::create(builder, loc, x, y);
  }
  return arith::MulFOp::create(builder, loc, x, y);
}

// TODO : Upstream utility that does this pruning is broken for LinalgOp. Drop
// this if that gets fixed.
static SmallVector<NamedAttribute> getPrunedAttributeList(linalg::LinalgOp op) {
  const StringLiteral memoAttr =
      linalg::LinalgDialect::kMemoizedIndexingMapsAttrName;
  SmallVector<NamedAttribute> prunedAttributeList;
  for (auto attr : op->getDiscardableAttrs()) {
    if (attr.getName() != memoAttr) {
      prunedAttributeList.push_back(attr);
    }
  }
  return prunedAttributeList;
}

// Computes `inputKPerm` for the real shared K dims before synthetic conv-batch
// M dims are inserted.
static SmallVector<int64_t>
computeInputKPerm(AffineMap inputMap, AffineMap filterMap,
                  const mlir::linalg::ConvolutionDimensions &convDims) {
  // Get reduction dims from input and filter in order of appearance.
  auto reductionDims =
      llvm::concat<const unsigned>(convDims.inputChannel, convDims.filterLoop);
  SmallVector<int64_t> inputReductionDims;
  for (AffineExpr dimExpr : inputMap.getResults()) {
    for (unsigned reductionDim : reductionDims) {
      if (dimExpr.isFunctionOfDim(reductionDim)) {
        inputReductionDims.push_back(reductionDim);
      }
    }
  }
  SmallVector<int64_t> filterReductionDims;
  for (AffineExpr dimExpr : filterMap.getResults()) {
    for (unsigned reductionDim : reductionDims) {
      if (dimExpr.isFunctionOfDim(reductionDim)) {
        filterReductionDims.push_back(reductionDim);
      }
    }
  }

  // Compute the permutation that maps inputSharedDims to filterSharedDims.
  SmallVector<int64_t> inputKPerm;
  for (int64_t dim : filterReductionDims) {
    auto it = llvm::find(inputReductionDims, dim);
    assert(it != inputReductionDims.end() &&
           "Filter dimension not found in input shared dimensions");
    inputKPerm.push_back(std::distance(inputReductionDims.begin(), it));
  }
  return inputKPerm;
}

/// Returns the first dim in `dims` that `expr` depends on, if any.
static std::optional<unsigned> findFunctionOfDim(AffineExpr expr,
                                                 ArrayRef<unsigned> dims) {
  for (unsigned dim : dims) {
    if (expr.isFunctionOfDim(dim)) {
      return dim;
    }
  }
  return std::nullopt;
}

/// Returns the first result position in `map` that depends on `dim`, if any.
static std::optional<int64_t> getResultPositionForDim(AffineMap map,
                                                      unsigned dim) {
  for (auto [idx, expr] : llvm::enumerate(map.getResults())) {
    if (expr.isFunctionOfDim(dim)) {
      return idx;
    }
  }
  return std::nullopt;
}

/// Expands `input_k_perm` after conv batch dims are reclassified as M dims.
///
/// Conv batch dims have synthetic unit kernel-window coordinates prepended to
/// the K output order. `input_k_perm` maps K output order to input shared-dim
/// order, so this shifts existing output-order indices by the number of
/// synthetic dims and inserts each synthetic dim at its input-map position.
static FailureOr<SmallVector<int64_t>> expandInputKPermForBatchMDims(
    AffineMap inputMap, ArrayRef<int64_t> inputKPerm,
    const mlir::linalg::ConvolutionDimensions &convDims) {
  if (convDims.batch.empty()) {
    return SmallVector<int64_t>(inputKPerm);
  }

  SmallVector<int64_t> inverseInputKPerm = invertPermutationVector(inputKPerm);
  int64_t numSyntheticBatchDims = convDims.batch.size();
  for (int64_t &outputOrderIndex : inverseInputKPerm) {
    outputOrderIndex += numSyntheticBatchDims;
  }

  SmallVector<unsigned> inputKPermDims;
  llvm::append_range(inputKPermDims, convDims.inputChannel);
  llvm::append_range(inputKPermDims, convDims.filterLoop);
  int64_t numSharedDimsSeen = 0;
  int64_t syntheticOutputIndex = numSyntheticBatchDims;

  for (AffineExpr inputExpr : llvm::reverse(inputMap.getResults())) {
    if (findFunctionOfDim(inputExpr, convDims.batch)) {
      inverseInputKPerm.insert(inverseInputKPerm.end() - numSharedDimsSeen,
                               --syntheticOutputIndex);
      ++numSharedDimsSeen;
      continue;
    }
    if (findFunctionOfDim(inputExpr, inputKPermDims)) {
      ++numSharedDimsSeen;
    }
  }

  if (syntheticOutputIndex != 0 ||
      inverseInputKPerm.size() != inputKPerm.size() + numSyntheticBatchDims) {
    return failure();
  }
  return invertPermutationVector(inverseInputKPerm);
}

namespace {

using ControlFnTy = std::function<bool(Operation *)>;
// Converts non-depthwise convs into into linalg.generic (for img2col packing)
// and linalg.matmul.
// The following explains this for a linalg.conv_2d_nhwc_hwcf op.
//
// A convolution operation can be written as a matrix-matrix multiplication by
// unfolding the cross correlation between input and filter and explicitly copy
// overlapped sliding window inputs.
//
// Consider 2D input X with single channel input and output and 2x2 filter W:
// [x(0, 0)  , x(0, 1)  , ...,   x(0, n)  ]
// [x(1, 0)  , x(1, 1)  , ...,   x(1, n)  ]
// [.        ,  .       ,.   ,      .     ]            [w(0, 0), w(0, 1)]
// [.        ,  .       , .  ,      .     ]    (conv)  [w(1, 0), w(1, 1)]
// [.        ,  .       ,   .,      .     ]
// [x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]
//
// The packed input data (img2col) is a matrix with |rows| = output spatial
// size, |columns| = filter spatial size. To compute the output Y(i, j) we need
// to calculate the dot product between filter window at input X(x, y)) and the
// filter which will look like the following where r.h.s is the img2col matrix
// and l.h.s is the flattened filter:
//
// clang-format off
// [x(0, 0), x(0, 1), x(1, 0), x(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)] (matmul) [w(0, 0), w(0, 1), w(1, 0), w(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)]
// [   .   ,    .   ,    .   ,    .   ]
// clang-format on
//
// In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
// and output (N, Ho, Wo, D) the convolution is the following matrix-matrix
// multiplication ((N x Ho x Wo), Kh x Kw x C) * (Kh x Kw x C, D). The
// convolution batch dimension only indexes the image/output operands, so it is
// represented as an M dimension in the im2col/GEMM metadata rather than as a
// GEMM batch dimension.

class ConvertConvGeneric final
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  ConvertConvGeneric(MLIRContext *context, std::optional<ControlFnTy> controlFn)
      : OpInterfaceRewritePattern(context), controlFn(controlFn) {}
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (controlFn.has_value() && !controlFn.value()(linalgOp)) {
      return rewriter.notifyMatchFailure(linalgOp, "controlFn failed.");
    }

    auto igemmConvDetailsOrFailure =
        LinalgExt::getIGEMMGenericConvDetails(linalgOp);
    if (failed(igemmConvDetailsOrFailure)) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "Failed to extract IGEMM details");
    }

    LinalgExt::IGEMMGenericConvDetails igemmConvDetails =
        *igemmConvDetailsOrFailure;

    SmallVector<AffineMap> igemmContractionMaps =
        igemmConvDetails.igemmContractionMaps;
    mlir::linalg::ConvolutionDimensions convDims = igemmConvDetails.convDims;
    SmallVector<ReassociationIndices> filterReassocIndices =
        igemmConvDetails.filterReassocIndices;
    bool isOutputChannelFirst = igemmConvDetails.isOutputChannelFirst;
    SmallVector<int64_t> igemmLoopBounds = igemmConvDetails.igemmLoopBounds;
    SmallVector<utils::IteratorType> igemmLoopIterators =
        igemmConvDetails.igemmLoopIterators;

    Value input = linalgOp.getDpsInputs()[0];
    Value filter = linalgOp.getDpsInputs()[1];
    Value output = linalgOp.getDpsInits()[0];
    auto inputType = cast<ShapedType>(input.getType());
    auto filterType = cast<ShapedType>(filter.getType());
    auto outputType = cast<ShapedType>(output.getType());

    ArrayRef<int64_t> filterShape = filterType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    AffineMap inputMap = indexingMaps[0];
    AffineMap filterMap = indexingMaps[1];
    AffineMap outputMap = indexingMaps[2];

    SmallVector<int64_t> outputPerm = igemmConvDetails.im2colOutputPerm;
    SmallVector<int64_t> batchPos;
    SmallVector<int64_t> mPos;
    SmallVector<int64_t> mShape;
    SmallVector<int64_t> im2colStrides;
    SmallVector<int64_t> im2colDilations;
    SmallVector<OpFoldResult> kernelSizes;
    int64_t numSyntheticBatchMDims = 0;

    DenseMap<unsigned, int64_t> outputImageDimToIndex;
    for (auto [idx, dim] : llvm::enumerate(convDims.outputImage)) {
      outputImageDimToIndex[dim] = idx;
    }

    for (auto [inputDim, inputExpr] : llvm::enumerate(inputMap.getResults())) {
      if (findFunctionOfDim(inputExpr, convDims.depth)) {
        batchPos.push_back(inputDim);
        continue;
      }
      if (findFunctionOfDim(inputExpr, convDims.batch)) {
        mPos.push_back(inputDim);
        mShape.push_back(inputShape[inputDim]);
        im2colStrides.push_back(1);
        im2colDilations.push_back(1);
        kernelSizes.push_back(rewriter.getIndexAttr(1));
        ++numSyntheticBatchMDims;
        continue;
      }
      if (std::optional<unsigned> outputImage =
              findFunctionOfDim(inputExpr, convDims.outputImage)) {
        std::optional<int64_t> outputDim =
            getResultPositionForDim(outputMap, *outputImage);
        if (!outputDim) {
          return rewriter.notifyMatchFailure(
              linalgOp, "Failed to infer output image shape.");
        }
        auto outputImageIt = outputImageDimToIndex.find(*outputImage);
        int64_t outputImageIndex = outputImageIt->second;
        unsigned filterLoop = convDims.filterLoop[outputImageIndex];
        std::optional<int64_t> filterDim = filterMap.getResultPosition(
            getAffineDimExpr(filterLoop, filterMap.getContext()));
        if (!filterDim) {
          return rewriter.notifyMatchFailure(linalgOp,
                                             "Failed to infer filter shape.");
        }
        mPos.push_back(inputDim);
        mShape.push_back(outputShape[*outputDim]);
        im2colStrides.push_back(convDims.strides[outputImageIndex]);
        im2colDilations.push_back(convDims.dilations[outputImageIndex]);
        kernelSizes.push_back(rewriter.getIndexAttr(filterShape[*filterDim]));
      }
    }

    SmallVector<int64_t> kPos;
    for (auto reductionDim : convDims.inputChannel) {
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(reductionDim)) {
          kPos.push_back(idx);
        }
      }
    }
    SmallVector<int64_t> inputKPerm =
        computeInputKPerm(inputMap, filterMap, convDims);
    FailureOr<SmallVector<int64_t>> expandedInputKPerm =
        expandInputKPermForBatchMDims(inputMap, inputKPerm, convDims);
    if (failed(expandedInputKPerm)) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "Failed to expand input K perm.");
    }
    inputKPerm = *expandedInputKPerm;

    // Build unified offsets and output_sizes for the im2col op.
    // Canonical output dim order: [batch_pos dims, M, inputChannel K,
    // filterLoop K]. For convolutions, batch_pos is used for depth-like dims;
    // convolution batch dims are M dims with synthetic unit window metadata.
    // At conv-to-im2col time all offsets are zero.

    // Classify each original filter dim as parallel, inputChannel, or
    // filterLoop. The canonical im2col output order places inputChannel K
    // dims before filterLoop K dims.
    llvm::SmallDenseSet<int64_t, 4> parallelFilterDims;
    for (auto iterDim :
         llvm::concat<const unsigned>(convDims.depth, convDims.outputChannel)) {
      std::optional<int64_t> maybeDim = filterMap.getResultPosition(
          getAffineDimExpr(iterDim, filterMap.getContext()));
      if (maybeDim) {
        parallelFilterDims.insert(maybeDim.value());
      }
    }
    llvm::SmallDenseSet<int64_t, 4> inputChannelFilterDims;
    for (unsigned iterDim : convDims.inputChannel) {
      std::optional<int64_t> maybeDim = filterMap.getResultPosition(
          getAffineDimExpr(iterDim, filterMap.getContext()));
      if (maybeDim) {
        inputChannelFilterDims.insert(maybeDim.value());
      }
    }

    // Collect K output dim inner sizes from filter reassociation indices,
    // separated into inputChannel and filterLoop groups. The canonical
    // im2col output order is: [batch_pos dims, M, inputChannel K,
    // filterLoop K].
    SmallVector<SmallVector<int64_t>> inputChannelInnerSizes;
    SmallVector<SmallVector<int64_t>> filterLoopInnerSizes;
    for (const auto &indices : filterReassocIndices) {
      bool isParallel =
          indices.size() == 1 && parallelFilterDims.contains(indices[0]);
      if (isParallel) {
        continue;
      }
      SmallVector<int64_t> innerSizes;
      for (int64_t idx : indices) {
        innerSizes.push_back(filterShape[idx]);
      }
      // Classify as inputChannel if all filter dims in the group are
      // inputChannel dims; otherwise classify as filterLoop.
      bool isInputChannel = llvm::all_of(indices, [&](int64_t idx) {
        return inputChannelFilterDims.contains(idx);
      });
      if (isInputChannel) {
        inputChannelInnerSizes.push_back(innerSizes);
      } else {
        filterLoopInnerSizes.push_back(innerSizes);
      }
    }

    SmallVector<SmallVector<OpFoldResult>> kOutputSizes;
    for (const auto &innerSizes : inputChannelInnerSizes) {
      kOutputSizes.push_back(getAsIndexOpFoldResult(getContext(), innerSizes));
    }
    for (const auto &innerSizes : filterLoopInnerSizes) {
      kOutputSizes.push_back(getAsIndexOpFoldResult(getContext(), innerSizes));
    }
    assert(!kOutputSizes.empty() &&
           "expected at least one K output dim for convolution");
    if (numSyntheticBatchMDims > 0) {
      kOutputSizes.front().insert(kOutputSizes.front().begin(),
                                  numSyntheticBatchMDims,
                                  rewriter.getIndexAttr(1));
    }

    int64_t numOutputDims =
        batchPos.size() + mShape.size() + kOutputSizes.size();
    SmallVector<OpFoldResult> offsets(numOutputDims, rewriter.getIndexAttr(0));
    SmallVector<SmallVector<OpFoldResult>> outputSizes;
    // Batch dims: each has a single inner size.
    for (int64_t dim : batchPos) {
      outputSizes.push_back({rewriter.getIndexAttr(inputShape[dim])});
    }
    // M dims: each convolution batch or spatial output dim is a separate output
    // dimension.
    for (int64_t m : mShape) {
      outputSizes.push_back({rewriter.getIndexAttr(m)});
    }
    // Synthetic conv-batch unit K coords first, then inputChannel K dims, then
    // filterLoop K dims.
    for (const auto &innerSizes : kOutputSizes) {
      outputSizes.push_back(innerSizes);
    }

    auto loc = linalgOp.getLoc();
    // Shape of the resulting tensor from im2col. Each output dim is the
    // product of its inner sizes.
    SmallVector<int64_t> colTensorShape;
    for (const auto &innerSizes : outputSizes) {
      int64_t dimSize = 1;
      for (OpFoldResult s : innerSizes) {
        std::optional<int64_t> constVal = getConstantIntValue(s);
        if (!constVal) {
          return rewriter.notifyMatchFailure(
              linalgOp, "dynamic inner sizes not supported");
        }
        dimSize *= *constVal;
      }
      colTensorShape.push_back(dimSize);
    }

    applyPermutationToVector(colTensorShape, outputPerm);
    Value colTensor = tensor::EmptyOp::create(rewriter, loc, colTensorShape,
                                              inputType.getElementType());
    Value img2ColTensor =
        IREE::LinalgExt::Im2colOp::create(
            rewriter, loc, input, /*output=*/colTensor, im2colStrides,
            im2colDilations, kernelSizes, offsets, outputSizes, batchPos, mPos,
            kPos, inputKPerm, outputPerm)
            .getResult(0);

    Value reshapedFilter = tensor::CollapseShapeOp::create(
        rewriter, loc, filter, filterReassocIndices);

    auto genericGEMMOp = linalg::GenericOp::create(
        rewriter, loc, outputType,
        /*inputs=*/
        isOutputChannelFirst ? ValueRange{reshapedFilter, img2ColTensor}
                             : ValueRange{img2ColTensor, reshapedFilter},
        /*outputs=*/ValueRange{output}, igemmContractionMaps,
        igemmLoopIterators,
        [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value lhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[0],
                                           args[2].getType(),
                                           /*isUnsignedCast=*/false);
          Value rhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[1],
                                           args[2].getType(),
                                           /*isUnsignedCast=*/false);
          Value mul = createMul(nestedLoc, lhs, rhs, nestedBuilder);
          Value add = createAdd(nestedLoc, mul, args[2], nestedBuilder);
          linalg::YieldOp::create(nestedBuilder, nestedLoc, add);
        });
    genericGEMMOp->setDiscardableAttrs(getPrunedAttributeList(linalgOp));

    rewriter.replaceOp(linalgOp, genericGEMMOp.getResults().front());
    return success();
  }

private:
  std::optional<ControlFnTy> controlFn;
};

struct ConvertConvToIm2ColOpPass final
    : impl::ConvertConvToIm2ColOpPassBase<ConvertConvToIm2ColOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateConvToIm2colOpPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateConvToIm2colOpPatterns(RewritePatternSet &patterns,
                                    std::optional<ControlFnTy> controlFn) {
  patterns.insert<ConvertConvGeneric>(patterns.getContext(),
                                      std::move(controlFn));
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
