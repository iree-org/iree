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

// Computes `inputKPerm` that maps the input spatial and channel dimension order
// to filter's.
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
// multiplication (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in
// the N batches. For the case where N > 1 its a batched matrxi-matrix
// multiplication.

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

    SmallVector<OpFoldResult> kernelSizes;
    for (auto filterLoop : convDims.filterLoop) {
      std::optional<int64_t> maybeDim = filterMap.getResultPosition(
          getAffineDimExpr(filterLoop, filterMap.getContext()));
      if (!maybeDim) {
        return rewriter.notifyMatchFailure(linalgOp,
                                           "Failed to infer filter shape.");
      }
      kernelSizes.push_back(
          rewriter.getIndexAttr(filterShape[maybeDim.value()]));
    }

    // Batch dims for the im2col also include the depth/group dimensions of the
    // conv.
    SmallVector<int64_t> outputPerm = igemmConvDetails.im2colOutputPerm;
    auto im2colBatchIterDims =
        llvm::to_vector(llvm::concat<unsigned>(convDims.depth, convDims.batch));
    SmallVector<int64_t> batchPos(im2colBatchIterDims.size());
    for (int64_t convDim : im2colBatchIterDims) {
      AffineExpr convDimExpr = getAffineDimExpr(convDim, getContext());
      int64_t im2colInputDim = inputMap.getResultPosition(convDimExpr).value();

      AffineExpr igemmDimExpr = igemmConvDetails.convToIgemmDimMap.at(convDim);
      int64_t igemmInputDim = igemmConvDetails.getIgemmInputImageMap()
                                  .getResultPosition(igemmDimExpr)
                                  .value();
      batchPos[outputPerm[igemmInputDim]] = im2colInputDim;
    }

    SmallVector<int64_t> mPos;
    SmallVector<int64_t> mShape;
    for (auto outputImage : convDims.outputImage) {
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(outputImage)) {
          mPos.push_back(idx);
        }
      }
      for (auto [idx, e] : llvm::enumerate(outputMap.getResults())) {
        if (e.isFunctionOfDim(outputImage)) {
          mShape.push_back(outputShape[idx]);
        }
      }
    }

    // Detect if M dims were collapsed by checking if multiple outputImage
    // conv dims map to the same IGEMM dim.
    llvm::SmallDenseSet<unsigned, 4> seenIgemmDims;
    bool mCollapsed = false;
    for (unsigned d : convDims.outputImage) {
      auto igemmExpr =
          cast<AffineDimExpr>(igemmConvDetails.convToIgemmDimMap.at(d));
      if (!seenIgemmDims.insert(igemmExpr.getPosition()).second) {
        mCollapsed = true;
        break;
      }
    }

    // Save original spatial sizes before flattening (needed for output_sizes).
    SmallVector<int64_t> originalMShape(mShape);

    // Flatten mShape when M dims are collapsed.
    if (mCollapsed) {
      int64_t flatM = 1;
      for (int64_t s : mShape) {
        if (ShapedType::isDynamic(s))
          return rewriter.notifyMatchFailure(
              linalgOp, "dynamic M dims cannot be flattened");
        flatM *= s;
      }
      mShape = {flatM};
    }

    SmallVector<int64_t> kPos;
    for (auto reductionDim : convDims.inputChannel) {
      for (auto [idx, e] : llvm::enumerate(inputMap.getResults())) {
        if (e.isFunctionOfDim(reductionDim)) {
          kPos.push_back(idx);
        }
      }
    }
    // Get collapsed K shape from igemmLoopBounds (for the output tensor shape).
    int64_t reductionBoundIndex =
        llvm::count(igemmLoopIterators, utils::IteratorType::parallel);
    SmallVector<int64_t> kShape(igemmLoopBounds.begin() + reductionBoundIndex,
                                igemmLoopBounds.end());

    // Build per-K-output-dim decomposed sizes from original filter shape.
    // Each IGEMM reduction group becomes one K output dim, and the output_sizes
    // inner list for that dim contains the original (pre-collapse) filter sizes.
    // This is needed for correct delinearization in decomposition/vectorization.
    DenseSet<unsigned> convReductionDims;
    for (unsigned d : convDims.filterLoop) {
      convReductionDims.insert(d);
    }
    for (unsigned d : convDims.inputChannel) {
      convReductionDims.insert(d);
    }

    SmallVector<SmallVector<int64_t>> kPerDimDecomposedSizes;
    for (const auto &indices : filterReassocIndices) {
      // Check if this filter reassociation group is a reduction group.
      bool isReduction = false;
      for (int64_t idx : indices) {
        // Map filter dim position to conv dim via filter map.
        AffineExpr expr = filterMap.getResult(idx);
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          if (convReductionDims.contains(dimExpr.getPosition())) {
            isReduction = true;
            break;
          }
        }
      }
      // Assert that all dims in this group are either entirely reduction or
      // entirely parallel. Mixed groups would produce incorrect im2col layout.
      assert(llvm::all_of(indices,
                          [&](int64_t idx) {
                            AffineExpr expr = filterMap.getResult(idx);
                            auto dimExpr = dyn_cast<AffineDimExpr>(expr);
                            if (!dimExpr) {
                              return true;
                            }
                            return convReductionDims.contains(
                                       dimExpr.getPosition()) == isReduction;
                          }) &&
             "filter reassociation group mixes reduction and parallel dims");
      if (isReduction) {
        SmallVector<int64_t> groupSizes;
        for (int64_t idx : indices) {
          groupSizes.push_back(filterShape[idx]);
        }
        kPerDimDecomposedSizes.push_back(std::move(groupSizes));
      }
    }

    SmallVector<int64_t> inputKPerm =
        computeInputKPerm(inputMap, filterMap, convDims);

    auto loc = linalgOp.getLoc();
    // Shape of the resulting tensor from im2col.
    SmallVector<int64_t> colTensorShape;
    for (int64_t dim : batchPos) {
      colTensorShape.push_back(inputShape[dim]);
    }
    colTensorShape.append(mShape);
    colTensorShape.append(kShape);

    // Build offsets (all zeros) and output_sizes (nested sizes per output dim).
    int64_t outputRank =
        static_cast<int64_t>(batchPos.size() + mShape.size() + kShape.size());
    SmallVector<OpFoldResult> offsets(outputRank, rewriter.getIndexAttr(0));
    SmallVector<SmallVector<OpFoldResult>> outputSizes;
    // Batch dims: each has a single size equal to the input batch dim size.
    for (int64_t dim : batchPos) {
      outputSizes.push_back({rewriter.getIndexAttr(inputShape[dim])});
    }
    // M dims: if collapsed, one output dim with original spatial sizes
    // (pre-flattening); if expanded, one output dim per spatial dim.
    if (mCollapsed) {
      SmallVector<OpFoldResult> mSizes;
      for (int64_t s : originalMShape) {
        mSizes.push_back(rewriter.getIndexAttr(s));
      }
      outputSizes.push_back(std::move(mSizes));
    } else {
      for (int64_t s : originalMShape) {
        outputSizes.push_back({rewriter.getIndexAttr(s)});
      }
    }
    // K dims: one output_sizes entry per K output dim (IGEMM reduction group).
    // Each entry contains the decomposed filter dim sizes for that group.
    for (const auto &groupSizes : kPerDimDecomposedSizes) {
      SmallVector<OpFoldResult> kSizes;
      for (int64_t s : groupSizes) {
        kSizes.push_back(rewriter.getIndexAttr(s));
      }
      outputSizes.push_back(std::move(kSizes));
    }

    applyPermutationToVector(colTensorShape, outputPerm);
    Value colTensor = tensor::EmptyOp::create(rewriter, loc, colTensorShape,
                                              inputType.getElementType());
    Value img2ColTensor =
        IREE::LinalgExt::Im2colOp::create(
            rewriter, loc, input, /*output=*/colTensor, convDims.strides,
            convDims.dilations, kernelSizes, offsets, outputSizes,
            batchPos, mPos, kPos, inputKPerm, outputPerm)
            .getResult(0);

    Value reshapedFilter = tensor::CollapseShapeOp::create(
        rewriter, loc, filter, filterReassocIndices);

    // When M dims are collapsed, we need to collapse the conv output tensor
    // for the GEMM and expand the GEMM result back to the original shape.
    Value gemmOutput = output;
    ShapedType gemmOutputType = outputType;
    SmallVector<ReassociationIndices> outputReassoc;
    if (mCollapsed) {
      // Find outputImage positions in the output tensor.
      DenseSet<unsigned> oiDimSet(convDims.outputImage.begin(),
                                  convDims.outputImage.end());
      SmallVector<int64_t> oiOutputPositions;
      for (auto [idx, e] : llvm::enumerate(outputMap.getResults())) {
        if (oiDimSet.contains(cast<AffineDimExpr>(e).getPosition())) {
          oiOutputPositions.push_back(idx);
        }
      }
      int64_t oiStart = oiOutputPositions.front();
      int64_t oiEnd = oiOutputPositions.back();
      // Non-contiguous output image positions are not supported. This is
      // unreachable for standard convolutions but guards against exotic layouts.
      if (oiEnd - oiStart + 1 !=
          static_cast<int64_t>(oiOutputPositions.size())) {
        return rewriter.notifyMatchFailure(
            linalgOp, "non-contiguous outputImage positions");
      }

      // Build reassociation indices: group outputImage positions together.
      for (int64_t i = 0; i < outputType.getRank(); ++i) {
        if (i == oiStart) {
          ReassociationIndices group;
          for (int64_t j = oiStart; j <= oiEnd; ++j) {
            group.push_back(j);
          }
          outputReassoc.push_back(group);
          i = oiEnd;
        } else {
          outputReassoc.push_back({i});
        }
      }
      gemmOutput = tensor::CollapseShapeOp::create(rewriter, loc, output,
                                                    outputReassoc);
      gemmOutputType = cast<ShapedType>(gemmOutput.getType());
    }

    auto genericGEMMOp = linalg::GenericOp::create(
        rewriter, loc, gemmOutputType,
        /*inputs=*/
        isOutputChannelFirst ? ValueRange{reshapedFilter, img2ColTensor}
                             : ValueRange{img2ColTensor, reshapedFilter},
        /*outputs=*/ValueRange{gemmOutput}, igemmContractionMaps,
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
    Value result = genericGEMMOp.getResults().front();

    // Expand GEMM result back to original conv output shape.
    if (mCollapsed) {
      result = tensor::ExpandShapeOp::create(rewriter, loc, outputType, result,
                                             outputReassoc);
    }

    rewriter.replaceOp(linalgOp, result);
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
