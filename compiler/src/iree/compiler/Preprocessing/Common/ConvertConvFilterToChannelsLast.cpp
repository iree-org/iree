// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-preprocessing-convert-conv-filter-to-channels-last"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_CONVERTCONVFILTERTOCHANNELSLASTPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc"

static AffineMap applyPermutationToResults(AffineMap map,
                                           ArrayRef<int64_t> perm) {
  unsigned numDims = map.getNumDims();
  ArrayRef<AffineExpr> mapResults = map.getResults();
  SmallVector<AffineExpr> exprs;
  for (int i = 0, e = perm.size(); i < e; ++i) {
    exprs.push_back(mapResults[perm[i]]);
  }
  return AffineMap::get(numDims, map.getNumSymbols(), exprs, map.getContext());
}

static Value createTransposeOp(RewriterBase &rewriter, Location loc,
                               Value tensor, ArrayRef<int64_t> perm) {
  SmallVector<OpFoldResult> dimSizes =
      tensor::getMixedSizes(rewriter, loc, tensor);
  applyPermutationToVector(dimSizes, perm);

  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto emptyTensor = tensor::EmptyOp::create(rewriter, loc, dimSizes,
                                             tensorType.getElementType());
  return linalg::TransposeOp::create(rewriter, loc, tensor, emptyTensor, perm)
      .getResult()[0];
}

static LogicalResult
convertConvFilterToTargetLayout(linalg::Conv2DNhwcHwcfOp convOp,
                                RewriterBase &rewriter,
                                SmallVector<int64_t> &perm) {
  Location loc = convOp.getLoc();

  Value input = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value output = convOp.getOutputs()[0];

  AffineMap inputMap = convOp.getIndexingMapsArray()[0];
  AffineMap filterMap = convOp.getIndexingMapsArray()[1];
  AffineMap outputMap = convOp.getIndexingMapsArray()[2];

  AffineMap transposedFilterMap = applyPermutationToResults(filterMap, perm);
  Value transposedFilter = createTransposeOp(rewriter, loc, filter, perm);

  SmallVector<utils::IteratorType> iterators = convOp.getIteratorTypesArray();

  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, output.getType(), ValueRange{input, transposedFilter},
      output, ArrayRef<AffineMap>{inputMap, transposedFilterMap, outputMap},
      iterators);

  // Reuse the same payload as the original convolution op.
  rewriter.inlineRegionBefore(convOp->getRegion(0), genericOp.getRegion(),
                              genericOp.getRegion().begin());

  rewriter.replaceOp(convOp, genericOp->getResults());
  return success();
}

namespace {
struct ConvertHwcfToHwfc : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> perm = {0, 1, 3, 2};
    return convertConvFilterToTargetLayout(convOp, rewriter, perm);
  }
};

struct ConvertHwcfToFhwc : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> perm = {3, 0, 1, 2};
    return convertConvFilterToTargetLayout(convOp, rewriter, perm);
  }
};

struct ConvertGenericChwfToFhwc : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    if (!linalgOp || !linalg::isaConvolutionOpInterface(linalgOp)) {
      return failure();
    }

    FailureOr<mlir::linalg::ConvolutionDimensions> convolutionDims =
        mlir::linalg::inferConvolutionDims(linalgOp);
    if (failed(convolutionDims)) {
      return failure();
    }

    OpOperand *input = linalgOp.getDpsInputOperand(0);
    OpOperand *filter = linalgOp.getDpsInputOperand(1);
    OpOperand *output = linalgOp.getDpsInitOperand(0);

    AffineMap inputMap = linalgOp.getMatchingIndexingMap(input);
    AffineMap filterMap = linalgOp.getMatchingIndexingMap(filter);
    AffineMap outputMap = linalgOp.getMatchingIndexingMap(output);

    Value inputVal = input->get();
    Value filterVal = filter->get();
    Value outputVal = output->get();

    ArrayRef<int64_t> inputShape =
        cast<ShapedType>(inputVal.getType()).getShape();
    ArrayRef<int64_t> filterShape =
        cast<ShapedType>(filterVal.getType()).getShape();
    ArrayRef<int64_t> outputShape =
        cast<ShapedType>(outputVal.getType()).getShape();

    // TODO(vivian): Once the matmul shape check below is dropped, the
    // dynamic-shape check can also be removed.
    if (ShapedType::isDynamicShape(inputShape) ||
        ShapedType::isDynamicShape(filterShape) ||
        ShapedType::isDynamicShape(outputShape)) {
      return failure();
    }

    auto getDimPositions = [&](ArrayRef<unsigned> dims, const AffineMap &map) {
      SmallVector<int64_t> positions;
      for (auto dim : dims) {
        for (auto [idx, e] : llvm::enumerate(map.getResults())) {
          if (e.isFunctionOfDim(dim)) {
            positions.push_back(idx);
          }
        }
      }
      return positions;
    };

    // Only transpose when the input channel is the last dimension of conv
    // input.
    SmallVector<int64_t> cInputPos =
        getDimPositions(convolutionDims->inputChannel, inputMap);
    if (cInputPos.back() != inputShape.size() - 1) {
      return failure();
    }

    // Only transpose when the filter is `CHWF` layout.
    SmallVector<int64_t> fFilterPos =
        getDimPositions(convolutionDims->outputChannel, filterMap);
    SmallVector<int64_t> cFilterPos =
        getDimPositions(convolutionDims->inputChannel, filterMap);
    SmallVector<int64_t> kFilterPos =
        getDimPositions(convolutionDims->filterLoop, filterMap);
    int64_t fPos = fFilterPos.back();
    int64_t cPos = cFilterPos.back();
    int64_t kPos = kFilterPos.back();
    if (cPos > kPos || fPos != filterShape.size() - 1) {
      return failure();
    }

    // Don't transpose if it is a matmul and the input shape is small.
    // TODO(vivian): Solve the fusion of transpose op and remove this check.
    SmallVector<int64_t> imagePos =
        getDimPositions(convolutionDims->outputImage, outputMap);
    SmallVector<int64_t> batchPos =
        getDimPositions(convolutionDims->batch, outputMap);
    SmallVector<int64_t> mPos = imagePos;
    mPos.append(batchPos.begin(), batchPos.end());

    auto getProduct = [](ArrayRef<int64_t> shape, ArrayRef<int64_t> pos) {
      return llvm::accumulate(pos, int64_t{1}, [&](int64_t a, int64_t idx) {
        return a * shape[idx];
      });
    };

    int64_t mSize = getProduct(outputShape, mPos);
    int64_t nSize = getProduct(filterShape, fFilterPos);
    int64_t kSize = getProduct(filterShape, cFilterPos);
    int64_t filterProd = getProduct(filterShape, kFilterPos);
    bool smallShape = mSize < 384 || nSize < 384 || kSize < 384;
    if (filterProd == 1 && smallShape) {
      return failure();
    }

    // Swap the input and output channel dimension.
    SmallVector<int64_t> perm =
        llvm::to_vector(llvm::seq<int64_t>(0, filterShape.size()));
    std::swap(perm[cPos], perm[fPos]);

    Location loc = linalgOp.getLoc();

    AffineMap transposedFilterMap = applyPermutationToResults(filterMap, perm);
    Value transposedFilter = createTransposeOp(rewriter, loc, filterVal, perm);

    // Insert compute_barrier.start to avoid propagation of reshape ops and
    // undesirable fusion.
    auto barrierStartOp = IREE::TensorExt::ComputeBarrierStartOp::create(
        rewriter, loc, transposedFilter);

    SmallVector<utils::IteratorType> iterators =
        linalgOp.getIteratorTypesArray();

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, outputVal.getType(),
        ValueRange{inputVal, barrierStartOp.getResult()}, outputVal,
        ArrayRef<AffineMap>{inputMap, transposedFilterMap, outputMap},
        iterators);

    // Reuse the same payload as the original convolution op.
    rewriter.inlineRegionBefore(linalgOp->getRegion(0), genericOp.getRegion(),
                                genericOp.getRegion().begin());

    // Reorder the indexing dimensions so that the input channel loops appears
    // after the filter loops.
    unsigned numParallelLoop = genericOp.getNumParallelLoops();
    SmallVector<unsigned> interchange =
        llvm::to_vector(llvm::seq<unsigned>(0, numParallelLoop));
    interchange.append(convolutionDims->filterLoop.begin(),
                       convolutionDims->filterLoop.end());
    interchange.append(convolutionDims->inputChannel.begin(),
                       convolutionDims->inputChannel.end());

    FailureOr<linalg::GenericOp> reorderOp =
        linalg::interchangeGenericOp(rewriter, genericOp, interchange);
    if (failed(reorderOp))
      return failure();

    rewriter.replaceOp(linalgOp, reorderOp->getResults());
    return success();
  }
};

class ConvertConvFilterToChannelsLastPass
    : public iree_compiler::Preprocessing::impl::
          ConvertConvFilterToChannelsLastPassBase<
              ConvertConvFilterToChannelsLastPass> {
public:
  using iree_compiler::Preprocessing::impl::
      ConvertConvFilterToChannelsLastPassBase<
          ConvertConvFilterToChannelsLastPass>::
          ConvertConvFilterToChannelsLastPassBase;

  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    if (filterLayout == "hwfc") {
      LDBG() << "Converting filter layout to hwfc.";
      patterns.add<ConvertHwcfToHwfc>(context);
    } else if (filterLayout == "fhwc") {
      LDBG() << "Converting filter layout to fhwc.";
      patterns.add<ConvertHwcfToFhwc, ConvertGenericChwfToFhwc>(context);
    } else {
      LDBG() << "convert-filter-to-channels-last pass didn't apply since an "
                "unsupported layout is given. Please use hwfc or fhwc as pass "
                "filter-layout option.";
      return signalPassFailure();
    }

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::Preprocessing
