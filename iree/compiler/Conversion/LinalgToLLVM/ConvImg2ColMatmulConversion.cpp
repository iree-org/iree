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

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

// clang-format off
//
// Convert linalg.conv_2d_input_nhwc_filter_hwcf op into img2col packing
// operation (linalg.generic) + linalg.matmul. See details below:
// A convolution operaton can be written as a matrix-matrix multiplication by
// unfolding the cross corrolation between input and filter and explcitiy copy
// overlaped sliding window inputs.
// Consider 2D input X with single channel input and output and 2x2 filter W:
// [x(0, 0)  , x(0, 1)  , ...,   x(0, n)  ]
// [x(1, 0)  , x(1, 1)  , ...,   x(1, n)  ]
// [.        ,  .       ,.   ,      .     ]            [w(0, 0), w(0, 1)]
// [.        ,  .       , .  ,      .     ]    (conv)  [w(1, 0), w(1, 1)]
// [.        ,  .       ,   .,      .     ]
// [x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]
// The packed input data (img2col) is a matrix with |rows| = output spatial
// size, |columns| = filter spatial size. To compute the output Y(i, j) we need
// to calculate the dot product between filter window at input X(x, y)) and the
// filter which will look like the following where r.h.s is the img2col matrix and
// l.h.s is the flattned filter:
// [x(0, 0), x(0, 1), x(1, 0), x(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)] (matmul) [w(0, 0), w(0, 1), w(1, 0), w(1, 1)]
// [x(0, 1), x(1, 1), x(0, 2), x(1, 2)]
// [   .   ,    .   ,    .   ,    .   ]
// In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
// and output (N, Ho, Wo, D) the convolutin is the following matrix-matrix multiplication
// (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in the N input.
// For the case where N > 1 its a batched matrxi-matrix multplication.
//
// clang-format on
class ConvImg2ColMatmulConversion
    : public OpRewritePattern<linalg::ConvInputNHWCFilterHWCFOp> {
 public:
  using OpRewritePattern<linalg::ConvInputNHWCFilterHWCFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ConvInputNHWCFilterHWCFOp op,
                                PatternRewriter &rewriter) const override {
    ShapedType filterShapeType = op.getInputShapedType(1);
    ShapedType inputShapeType = op.getInputShapedType(0);
    ShapedType outputShapeType = op.getOutputShapedType(0);
    if (!filterShapeType || !inputShapeType) return failure();
    if (!filterShapeType.hasStaticShape() || !inputShapeType.hasStaticShape())
      return failure();

    // TODO(ataei): Support for batched version.
    if (inputShapeType.getShape()[0] > 1) return failure();

    auto loc = op.getLoc();

    int numBatchDims = 1;
    int numSpatialDims = 2;
    int numInputFeatureDims = 1;
    int numOutputFeatureDims = 1;
    auto inputFeatures =
        inputShapeType.getShape()[numBatchDims + numSpatialDims];
    auto outputFeatures = filterShapeType.getShape().back();
    Value input = op.getInput(0);
    Value filter = op.getInput(1);
    Value output = op.getOutput(0);

    // Col buffer shape (n, d1, d1, d2, ...dn, k1, k2, k3, ...kn, ci)
    SmallVector<int64_t, 4> colBufferShape;
    int64_t spatialSize = 1, filterSpatialSize = 1;
    colBufferShape.push_back(outputShapeType.getShape()[0]);
    for (int i = 0; i < numSpatialDims; ++i) {
      auto dimSize = outputShapeType.getShape()[i + numBatchDims];
      colBufferShape.push_back(dimSize);
      spatialSize *= dimSize;
    }
    for (int i = 0; i < numSpatialDims; ++i) {
      auto dimSize = filterShapeType.getShape()[i];
      colBufferShape.push_back(dimSize);
      filterSpatialSize *= dimSize;
    }
    colBufferShape.push_back(filterShapeType.getShape()[numSpatialDims]);

    auto ColBufferMemrefType =
        MemRefType::get(colBufferShape, filterShapeType.getElementType());

    Value result = rewriter.create<AllocaOp>(loc, ColBufferMemrefType);

    // (n, d1, d2, d3, ..., dn, k1, k2, k3, ...kn, ci) ->
    // (n, d_1 * stride_1 + k_1, d_2 * stride_2 + k_2, ...d_n * stride_n + k_n,
    // ci)
    SmallVector<AffineExpr, 4> inputExprs;
    inputExprs.push_back(rewriter.getAffineDimExpr(0));
    int spatialDimsOffset = 1;
    auto kernelDimsOffset = spatialDimsOffset + numSpatialDims;
    for (unsigned i = 0; i < numSpatialDims; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i + spatialDimsOffset) *
                               op.strides().getValue<int64_t>({i}) +
                           rewriter.getAffineDimExpr(i + kernelDimsOffset));
    }
    inputExprs.push_back(
        rewriter.getAffineDimExpr(kernelDimsOffset + numSpatialDims));

    auto nloops = colBufferShape.size();

    SmallVector<StringRef, 3> loopAttributeTypes(nloops, "parallel");

    SmallVector<AffineMap, 4> indexingMaps;
    indexingMaps.emplace_back(
        AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()));
    indexingMaps.emplace_back(AffineMap::getMultiDimIdentityMap(
        ColBufferMemrefType.getRank(), rewriter.getContext()));

    rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/ArrayRef<Type>{},
        /*inputs=*/input, /*outputs=*/result, indexingMaps, loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });

    auto getIndicesVector = [](int start, int end) {
      return llvm::to_vector<2>(llvm::seq<int64_t>(start, end));
    };

    SmallVector<linalg::ReassociationIndices, 4> lhsCollapsedDimsList = {
        getIndicesVector(0, numBatchDims + numSpatialDims),
        getIndicesVector(
            numBatchDims + numSpatialDims,
            numBatchDims + numSpatialDims * 2 + numInputFeatureDims)};
    SmallVector<linalg::ReassociationIndices, 4> rhsCollapsedDimsList = {
        getIndicesVector(0, numSpatialDims + numInputFeatureDims),
        getIndicesVector(
            numSpatialDims + numInputFeatureDims,
            numSpatialDims + numInputFeatureDims + numOutputFeatureDims)};

    SmallVector<linalg::ReassociationIndices, 4> resultCollapsedDimsList = {
        getIndicesVector(0, numBatchDims + numSpatialDims),
        getIndicesVector(numBatchDims + numSpatialDims,
                         numBatchDims + numSpatialDims + numOutputFeatureDims)};

    auto reshapedColBufferType =
        MemRefType::get({spatialSize, filterSpatialSize * inputFeatures},
                        filterShapeType.getElementType());

    auto reshapedfilterType =
        MemRefType::get({filterSpatialSize * inputFeatures, outputFeatures},
                        filterShapeType.getElementType());

    auto reshapedOutputType = MemRefType::get({spatialSize, outputFeatures},
                                              filterShapeType.getElementType());

    Value reshapedLhs = rewriter.create<linalg::ReshapeOp>(
        loc, reshapedColBufferType, result, lhsCollapsedDimsList);

    Value reshapedRhs = rewriter.create<linalg::ReshapeOp>(
        loc, reshapedfilterType, filter, rhsCollapsedDimsList);

    Value reshapedResult = rewriter.create<linalg::ReshapeOp>(
        loc, reshapedOutputType, output, resultCollapsedDimsList);

    rewriter.create<linalg::MatmulOp>(
        loc, ArrayRef<Value>{reshapedLhs, reshapedRhs}, reshapedResult);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvImg2ColMatmulConversionPass
    : PassWrapper<ConvImg2ColMatmulConversionPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnFunction() override;
};
}  // namespace

void populateConvImg2ColMatmulConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<ConvImg2ColMatmulConversion>(context);
}

void ConvImg2ColMatmulConversionPass::runOnFunction() {
  auto funcOp = getOperation();
  auto context = funcOp.getContext();
  OwningRewritePatternList patterns;
  populateConvImg2ColMatmulConversionPatterns(context, patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<FunctionPass> createConvImg2ColMatmulConversionPass() {
  return std::make_unique<ConvImg2ColMatmulConversionPass>();
}

static PassRegistration<ConvImg2ColMatmulConversionPass> pass(
    "iree-codegen-linalg-to-llvm-conv-img2col-conversion-pass",
    "Convert linalg.conv_2d_input_nhwc_filter_hwcf on to img2col followd by "
    "linalg.matmul.",
    [] { return std::make_unique<ConvImg2ColMatmulConversionPass>(); });
}  // namespace iree_compiler
}  // namespace mlir
