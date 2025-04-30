// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-preprocessing-convert-conv-filter-to-channels-last"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

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
  auto emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, dimSizes, tensorType.getElementType());
  return rewriter.create<linalg::TransposeOp>(loc, tensor, emptyTensor, perm)
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

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{input, transposedFilter}, output,
      ArrayRef<AffineMap>{inputMap, transposedFilterMap, outputMap}, iterators);

  // Reuse the same payload as the original convolution op.
  rewriter.inlineRegionBefore(convOp->getRegion(0), genericOp.getRegion(),
                              genericOp.getRegion().begin());

  rewriter.replaceOp(convOp, genericOp->getResults());
  return success();
}

namespace {
struct ConvertHwcfToHwfc : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> perm = {0, 1, 3, 2};
    return convertConvFilterToTargetLayout(convOp, rewriter, perm);
  }
};

struct ConvertHwcfToFhwc : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> perm = {3, 0, 1, 2};
    return convertConvFilterToTargetLayout(convOp, rewriter, perm);
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
      LDBG("Converting filter layout to hwfc.");
      patterns.add<ConvertHwcfToHwfc>(context);
    } else if (filterLayout == "fhwc") {
      LDBG("Converting filter layout to fhwc.");
      patterns.add<ConvertHwcfToFhwc>(context);
    } else {
      LDBG("convert-filter-to-channels-last pass didn't apply since an "
           "unsupported layout is given. Please use hwfc or fhwc as pass "
           "filter-layout option.");
      return signalPassFailure();
    }

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::Preprocessing
