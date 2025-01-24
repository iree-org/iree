// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_CONVERTCONV2DTOIM2COLOPPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

static bool hasAllOneValues(ArrayRef<int64_t> attr) {
  return llvm::all_of(attr, [](int64_t element) { return element == 1; });
}

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = llvm::isa<IntegerType>(x.getType());
  if (isInt)
    return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = llvm::isa<IntegerType>(x.getType());
  if (isInt)
    return builder.create<arith::MulIOp>(loc, x, y);
  return builder.create<arith::MulFOp>(loc, x, y);
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

static SmallVector<int64_t> getBasisFromShape(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> basis(shape.size());
  int64_t cummulativeProduct = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    cummulativeProduct *= shape[i];
    basis[i] = cummulativeProduct;
  }
  // Shift left with innermost basis one.
  basis.push_back(1);
  // Drop the outermost basis.
  return llvm::to_vector(llvm::ArrayRef(basis).drop_front());
}

namespace {

using ControlFnTy = std::function<bool(Operation *)>;
// Converts non-depthwise convs into into linalg.generic (for img2col packing)
// and linalg.matmul.
// The following explains this for a linalg.conv_2d_nhwc_hwcf op.
//
// A convolution operaton can be written as a matrix-matrix multiplication by
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
// and l.h.s is the flattned filter:
//
// clang-format off
// [x(0, 0), x(0, 1), x(1, 0), x(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)] (matmul) [w(0, 0), w(0, 1), w(1, 0), w(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)]
// [   .   ,    .   ,    .   ,    .   ]
// clang-format on
//
// In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
// and output (N, Ho, Wo, D) the convolutin is the following matrix-matrix
// multiplication (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in
// the N input. For the case where N > 1 its a batched matrxi-matrix
// multplication.

class ConvertConvGeneric final
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  ConvertConvGeneric(MLIRContext *context, std::optional<ControlFnTy> controlFn)
      : OpInterfaceRewritePattern(context), controlFn(controlFn) {}
  LogicalResult matchAndRewrite(linalg::LinalgOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (controlFn.has_value() && !controlFn.value()(genericOp)) {
      return rewriter.notifyMatchFailure(genericOp, "controlFn failed.");
    }

    auto convDimsOrFailure = linalg::inferConvolutionDims(genericOp);
    if (failed(convDimsOrFailure))
      return failure();
    Value input = genericOp.getDpsInputs()[0];
    Value filter = genericOp.getDpsInputs()[1];
    Value output = genericOp.getDpsInits()[0];
    auto inputType = llvm::cast<ShapedType>(input.getType());
    auto filterType = llvm::cast<ShapedType>(filter.getType());
    auto outputType = llvm::cast<ShapedType>(output.getType());

    if (!filterType.hasStaticShape() || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(genericOp, [](Diagnostic &diag) {
        diag << "[unimplemented] "
             << "expected 'filterType' and 'inputType' to have static shapes.";
      });
    }

    // TODO: Support dilation.
    if (!hasAllOneValues(convDimsOrFailure->dilations))
      return rewriter.notifyMatchFailure(genericOp, [](Diagnostic &diag) {
        diag << "[unimplemented] "
             << "expected no dilations (expected dilations to all be one).";
      });
    // TODO: Support depthwise.
    if (convDimsOrFailure->depth.size() != 0)
      return rewriter.notifyMatchFailure(genericOp, [](Diagnostic &diag) {
        diag << "[unimplemented] expected no depth";
      });
    auto filterShape = filterType.getShape();
    auto outputShape = outputType.getShape();
    auto indexingMaps = genericOp.getIndexingMapsArray();
    auto inputMap = indexingMaps[0];
    auto filterMap = indexingMaps[1];
    auto outputMap = indexingMaps[2];

    SmallVector<int64_t> colTensorShape;
    SmallVector<OpFoldResult> kernelSizes;
    for (auto filterLoop : convDimsOrFailure->filterLoop) {
      std::optional<int64_t> maybeDim = filterMap.getResultPosition(
          getAffineDimExpr(filterLoop, filterMap.getContext()));
      kernelSizes.push_back(
          rewriter.getIndexAttr(filterShape[maybeDim.value()]));
    }
    SmallVector<int64_t> batchPos;
    for (auto batch : convDimsOrFailure->batch) {
      std::optional<int64_t> maybeBatch = inputMap.getResultPosition(
          getAffineDimExpr(batch, inputMap.getContext()));
      if (maybeBatch) {
        batchPos.push_back(maybeBatch.value());
      }
    }
    for (auto batch : convDimsOrFailure->batch) {
      std::optional<int64_t> maybeBatch = outputMap.getResultPosition(
          getAffineDimExpr(batch, outputMap.getContext()));
      if (maybeBatch) {
        colTensorShape.push_back(outputShape[maybeBatch.value()]);
      }
    }

    SmallVector<int64_t> mPos;
    for (auto outputImage : convDimsOrFailure->outputImage) {
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(outputImage)) {
          mPos.push_back(idx);
        }
      }
    }

    SmallVector<int64_t> inputkPos;
    for (auto reductionDim : convDimsOrFailure->inputChannel) {
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(reductionDim)) {
          inputkPos.push_back(idx);
        }
      }
    }
    // begin utility
    SmallVector<int64_t> reductionDims;
    for (auto iter : llvm::enumerate(genericOp.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(iter.value())) {
        reductionDims.push_back(iter.index());
      }
    }
    SmallVector<int64_t> filterkPos;
    for (auto reductionDim : reductionDims) {
      std::optional<int64_t> maybeDim = filterMap.getResultPosition(
          getAffineDimExpr(reductionDim, filterMap.getContext()));
      filterkPos.push_back(maybeDim.value());
    }
    // group together adjacent reduction dimensions in the filter
    SmallVector<ReassociationIndices> collapsedFilterReductionDim;
    int64_t prevFilterIndex = filterkPos[0];
    int64_t currCollapsedIndex = 0;
    collapsedFilterReductionDim.push_back({filterkPos[0]});
    SmallVector<int64_t> kShape = {filterShape[filterkPos[0]]};
    for (auto currPos : llvm::ArrayRef(filterkPos).drop_front()) {
      if (prevFilterIndex == currPos - 1) {
        collapsedFilterReductionDim[currCollapsedIndex].push_back(currPos);
        kShape[currCollapsedIndex] *= filterShape[currPos];
      } else {
        collapsedFilterReductionDim.push_back({currPos});
        kShape.push_back(filterShape[currPos]);
        ++currCollapsedIndex;
      }
      prevFilterIndex = currPos;
    }
    // end utility
    SmallVector<int64_t> mShape;
    for (auto outputImage : convDimsOrFailure->outputImage) {
      for (auto [idx, e] : llvm::enumerate(outputMap.getResults())) {
        if (e.isFunctionOfDim(outputImage)) {
          mShape.push_back(outputShape[idx]);
          colTensorShape.push_back(outputShape[idx]);
        }
      }
    }
    SmallVector<OpFoldResult> mBasis =
        getAsIndexOpFoldResult(getContext(), getBasisFromShape(mShape));
    for (auto kb : kShape)
      colTensorShape.push_back(kb);

    SmallVector<OpFoldResult> kBasis =
        getAsIndexOpFoldResult(getContext(), getBasisFromShape(kShape));

    SmallVector<OpFoldResult> kOffset(kBasis.size(), rewriter.getIndexAttr(0));

    SmallVector<OpFoldResult> mOffset(mBasis.size(), rewriter.getIndexAttr(0));
    auto loc = genericOp.getLoc();
    Value colTensor = rewriter.create<tensor::EmptyOp>(
        loc, colTensorShape, inputType.getElementType());
    Value img2ColTensor =
        rewriter
            .create<IREE::LinalgExt::Im2colOp>(
                loc, input, /*output=*/colTensor, convDimsOrFailure->strides,
                convDimsOrFailure->dilations, kernelSizes, mOffset, mBasis,
                kOffset, kBasis, batchPos, mPos, inputkPos)
            .getResult(0);

    int64_t numBDims = (convDimsOrFailure->batch).size();
    int64_t numMDims = (convDimsOrFailure->outputImage).size();
    int64_t numNDims = (convDimsOrFailure->outputChannel).size();
    int64_t numParallelDim = numBDims + numMDims + numNDims;
    int64_t numKDims = collapsedFilterReductionDim.size();
    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    SmallVector<utils::IteratorType> filterIterators;
    SmallVector<int64_t> filterNdims;
    for (auto outputChannel : convDimsOrFailure->outputChannel) {
      std::optional<int64_t> maybeDim = filterMap.getResultPosition(
          getAffineDimExpr(outputChannel, filterMap.getContext()));
      filterNdims.push_back(maybeDim.value());
    }
    SmallVector<ReassociationIndices> filterReassocIndices;
    // Insert the parallel dims towards the end
    int64_t filterNdimPos = 0;
    for (auto collapsedDim : collapsedFilterReductionDim) {
      for (int i = filterNdimPos; i < filterNdims.size(); i++) {
        if (filterNdims[i] < collapsedDim[0]) {
          filterReassocIndices.push_back({filterNdims[i]});
          filterIterators.push_back(parallel);
          filterNdimPos = i + 1;
        } else {
          break;
        }
      }
      filterIterators.push_back(reduction);
      filterReassocIndices.push_back(collapsedDim);
    }
    // insert any leftover parallel Dims in the end.
    for (int i = filterNdimPos; i < filterNdims.size(); i++) {
      filterReassocIndices.push_back({filterNdims[i]});
      filterIterators.push_back(parallel);
    }
    SmallVector<int64_t> reshapedFilterShape(filterReassocIndices.size(), 1);
    for (auto [idx, indices] : llvm::enumerate(filterReassocIndices)) {
      for (auto index : indices) {
        reshapedFilterShape[idx] *= filterShape[index];
      }
    }

    auto reshapedFilterType =
        RankedTensorType::get(reshapedFilterShape, inputType.getElementType());

    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, filterReassocIndices);

    SmallVector<utils::IteratorType> genericIterators(numParallelDim, parallel);
    genericIterators.insert(genericIterators.end(), numKDims, reduction);

    SmallVector<AffineExpr> dims(numParallelDim + numKDims);
    bindDimsList<AffineExpr>(getContext(), dims);
    auto resultMap = AffineMap::get(
        numParallelDim + numKDims, 0,
        SmallVector<AffineExpr>(dims.begin(), dims.begin() + numParallelDim),
        getContext());

    bool isOutputChannelFirst = false;
    auto outputChannelPos = convDimsOrFailure->outputChannel;
    auto outputImagePos = convDimsOrFailure->outputImage;
    if (outputChannelPos.back() < outputImagePos[0])
      isOutputChannelFirst = true;

    // prepare the input map.
    SmallVector<AffineExpr> inputDims;
    // Add the batch dimensions.
    inputDims.insert(inputDims.end(), dims.begin(), dims.begin() + numBDims);
    int64_t starting_m_pos =
        isOutputChannelFirst ? numBDims + numNDims : numBDims;
    // Add the M dims.
    inputDims.insert(inputDims.end(), dims.begin() + starting_m_pos,
                     dims.begin() + starting_m_pos + numMDims);
    // Add the reduction dims.
    inputDims.insert(inputDims.end(), dims.begin() + numParallelDim,
                     dims.end());
    auto inputMapGEMM =
        AffineMap::get(numParallelDim + numKDims, 0, inputDims, getContext());

    // prepare filter map.
    SmallVector<AffineExpr> filterDims;
    int64_t curr_n_pos = isOutputChannelFirst ? numBDims : numBDims + numMDims;
    int64_t curr_k_pos = numBDims + numMDims + numNDims;

    for (auto iter : filterIterators) {
      if (iter == parallel) {
        filterDims.push_back(dims[curr_n_pos++]);
      } else if (iter == reduction) {
        filterDims.push_back(dims[curr_k_pos++]);
      }
    }
    auto filterMapGEMM =
        AffineMap::get(numParallelDim + numKDims, 0, filterDims, getContext());

    SmallVector<AffineMap> indexingGEMMMaps;
    if (isOutputChannelFirst) {
      indexingGEMMMaps.push_back(filterMapGEMM);
      indexingGEMMMaps.push_back(inputMapGEMM);
    } else {
      indexingGEMMMaps.push_back(inputMapGEMM);
      indexingGEMMMaps.push_back(filterMapGEMM);
    }
    indexingGEMMMaps.push_back(resultMap);
    auto genericGEMMOp = rewriter.create<linalg::GenericOp>(
        loc, outputType,
        /*inputs=*/
        isOutputChannelFirst ? ValueRange{reshapedFilter, img2ColTensor}
                             : ValueRange{img2ColTensor, reshapedFilter},
        /*outputs=*/ValueRange{output}, indexingGEMMMaps, genericIterators,
        [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value lhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[0],
                                           args[2].getType(),
                                           /*isUnsignedCast=*/false);
          Value rhs = convertScalarToDtype(nestedBuilder, nestedLoc, args[1],
                                           args[2].getType(),
                                           /*isUnsignedCast=*/false);
          Value mul = createMul(nestedLoc, lhs, rhs, nestedBuilder);
          Value add = createAdd(nestedLoc, mul, args[2], nestedBuilder);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
        });
    genericGEMMOp->setDiscardableAttrs(getPrunedAttributeList(genericOp));
    Value result = genericGEMMOp.getResults().front();

    rewriter.replaceOp(genericOp, result);
    return success();
  }

private:
  std::optional<ControlFnTy> controlFn;
};

struct ConvertConv2DToIm2ColOpPass final
    : impl::ConvertConv2DToIm2ColOpPassBase<ConvertConv2DToIm2ColOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateConv2DToIm2colOpPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateConv2DToIm2colOpPatterns(RewritePatternSet &patterns,
                                      std::optional<ControlFnTy> controlFn) {
  patterns.insert<ConvertConvGeneric>(patterns.getContext(),
                                      std::move(controlFn));
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
