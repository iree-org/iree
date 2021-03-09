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

//===- XLAToLinalgOnTensors.cpp - Pass to convert XLA to Linalg on tensors-===//
//
// Pass to convert from XLA to linalg on tensers. Uses the patterns from
// tensorflow/compiler/mlir/xla/transforms/legalize_to_linalg.cc along with
// some IREE specific patterns.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree/compiler/Conversion/HLOToLinalg/HLOToLinalgOnTensorPasses.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// mhlo.torch_index_select conversion patterns.
//===----------------------------------------------------------------------===//

static Value getOutputTensor(OpBuilder &builder, Location loc, Value opResult) {
  ShapedType outputType = opResult.getType().cast<ShapedType>();
  if (outputType.hasStaticShape()) {
    return builder.create<linalg::InitTensorOp>(loc, outputType.getShape(),
                                                outputType.getElementType());
  }
  // Check for tie-shape operations for the result to get the shape of the
  // output.
  SmallVector<Value, 4> dynamicSizes;
  for (Operation *user : opResult.getUsers()) {
    auto tieShapeOp = dyn_cast<Shape::TieShapeOp>(user);
    if (!tieShapeOp) continue;
    auto makeShapeOp =
        tieShapeOp.shape().getDefiningOp<Shape::MakeRankedShapeOp>();
    if (!makeShapeOp) continue;
    dynamicSizes = llvm::to_vector<4>(makeShapeOp.dynamic_dimensions());
    break;
  }
  if (outputType.getNumDynamicDims() != dynamicSizes.size()) return nullptr;
  return builder.create<linalg::InitTensorOp>(
      loc, dynamicSizes, outputType.getShape(), outputType.getElementType());
}

namespace {

/// Converts xla-hlo.torch_index_select op to a linalg.indexed_generic op.
struct TorchIndexSelectOpConversion
    : public OpConversionPattern<mhlo::TorchIndexSelectOp> {
  using OpConversionPattern<mhlo::TorchIndexSelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::TorchIndexSelectOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    mhlo::TorchIndexSelectOp::Adaptor adaptor(args);
    int axis = op.dim();
    int batch = op.batch_dims();
    auto indexShapeType = adaptor.index().getType().dyn_cast<ShapedType>();
    int nIndices = indexShapeType.getRank();
    auto inputShapeType = adaptor.input().getType().dyn_cast<ShapedType>();
    if (axis < 0) axis += inputShapeType.getRank();
    if (batch < 0) batch += nIndices;

    Location loc = op.getLoc();
    Value output = op.getResult();
    int rank = output.getType().cast<ShapedType>().getRank();
    SmallVector<AffineMap, 2> indexingMaps;
    SmallVector<AffineExpr, 4> exprs;
    for (int i = 0; i < batch; ++i)
      exprs.push_back(rewriter.getAffineDimExpr(i));
    for (int i = 0, e = nIndices - batch; i < e; ++i)
      exprs.push_back(rewriter.getAffineDimExpr(axis + i));
    indexingMaps.emplace_back(
        AffineMap::get(rank, /*symbolCount=*/0, exprs, rewriter.getContext()));
    indexingMaps.emplace_back(rewriter.getMultiDimIdentityMap(rank));
    SmallVector<StringRef, 3> loopTypes(rank, getParallelIteratorTypeName());
    ShapedType outputType = op.getResult().getType().cast<ShapedType>();
    Value initOp = getOutputTensor(rewriter, loc, op.getResult());
    if (!initOp) return failure();
    auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
        loc, /*resultTensors=*/ArrayRef<Type>{op.getResult().getType()},
        /*inputs=*/adaptor.index(),
        /*outputBuffers=*/initOp, indexingMaps, loopTypes);

    SmallVector<Type, 4> bodyArgTypes, opResultTypes;
    SmallVector<Value, 2> linalgOpArgs = {adaptor.index()};
    // Add a block to the region.
    auto *region = &linalgOp.region();
    auto *block = rewriter.createBlock(region, region->end());
    bodyArgTypes.append(rank, rewriter.getIndexType());
    for (auto blockArgs : linalgOpArgs) {
      bodyArgTypes.push_back(
          blockArgs.getType().cast<ShapedType>().getElementType());
    }
    block->addArguments(bodyArgTypes);
    block->addArguments(outputType.getElementType());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(block);

    SmallVector<Value, 4> indices;
    Value castedValue = rewriter.create<IndexCastOp>(
        loc, block->getArgument(rank), rewriter.getIndexType());
    for (int i = 0; i < axis; ++i) {
      indices.push_back(block->getArgument(i));
    }
    indices.push_back(castedValue);
    for (int i = axis + nIndices - batch; i < rank; ++i) {
      indices.push_back(block->getArgument(i));
    }

    Value res =
        rewriter.create<tensor::ExtractOp>(loc, adaptor.input(), indices);
    rewriter.create<linalg::YieldOp>(loc, res);

    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.conv conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Returns true if the given `dimensionNumbers` from a mhlo.convolution op
/// follows a canonical form:
///
/// * Input dimensions have order: (batch_count, spatial_dims,
///   input_channel_count).
/// * Filter dimensions have order: (spatial_dims, input_channel_count,
///   output_channel_count).
/// * Output dimensions have order: (batch_count, spatial_dims,
///   output_channel_count).
static bool hasCanonicalDimensionNumbers(
    const mhlo::ConvDimensionNumbers &dimensionNumbers) {
  const int inputSpatialRank =
      llvm::size(dimensionNumbers.input_spatial_dimensions());
  // The dimensions for input should follow the order of
  // batch_count, spatial_dims..., input_feature_count.
  if (dimensionNumbers.input_batch_dimension().getInt() != 0 ||
      dimensionNumbers.input_feature_dimension().getInt() !=
          (inputSpatialRank + 1)) {
    return false;
  }

  const int kernelSpatialRank =
      llvm::size(dimensionNumbers.kernel_spatial_dimensions());
  // The dimensions for filter should follow the order of
  // spatial_dims..., input_feature_count, num_output_feature_count.
  if (dimensionNumbers.kernel_input_feature_dimension().getInt() !=
          kernelSpatialRank ||
      dimensionNumbers.kernel_output_feature_dimension().getInt() !=
          (kernelSpatialRank + 1)) {
    return false;
  }

  const int outputSpatialRank =
      llvm::size(dimensionNumbers.output_spatial_dimensions());
  // The dimensions for output should follow the order of
  // batch_count, spatial_dims.., output_feature_count.
  if (dimensionNumbers.output_batch_dimension().getInt() != 0 ||
      dimensionNumbers.output_feature_dimension().getInt() !=
          (outputSpatialRank + 1)) {
    return false;
  }

  if (inputSpatialRank != outputSpatialRank ||
      inputSpatialRank != kernelSpatialRank) {
    return false;
  }

  auto inputSpatialDim = dimensionNumbers.input_spatial_dimensions().begin();
  auto kernelSpatialDim = dimensionNumbers.kernel_spatial_dimensions().begin();
  auto outputSpatialDim = dimensionNumbers.output_spatial_dimensions().begin();
  // Check spatial dims are ordred correctly.
  for (int i = 0; i < inputSpatialRank; ++i) {
    const int dim = i + 1;
    if ((*inputSpatialDim++).getZExtValue() != dim ||
        (*outputSpatialDim++).getZExtValue() != dim ||
        (*kernelSpatialDim++).getZExtValue() != i) {
      return false;
    }
  }

  return true;
}

/// Returns true if the given `attr` is a splat of the given `value`.
static bool isSplatValue(DenseIntElementsAttr attr, uint64_t value) {
  return attr.isSplat() && attr.getSplatValue<uint64_t>() == value;
}

/// Converts mhlo.convolution operation to linalg.depthwise_conv_nhwc op.
struct DepthwiseConvOpConversion : public OpConversionPattern<mhlo::ConvOp> {
  using OpConversionPattern<mhlo::ConvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override;
};

LogicalResult DepthwiseConvOpConversion::matchAndRewrite(
    mhlo::ConvOp op, ArrayRef<Value> args,
    ConversionPatternRewriter &rewriter) const {
  if (op.batch_group_count() != 1) return failure();

  if (op.padding() && !isSplatValue(*op.padding(), 0)) {
    return rewriter.notifyMatchFailure(op, "non-zero padding unsupported yet");
  }

  if ((op.lhs_dilation() && !isSplatValue(*op.lhs_dilation(), 1)) ||
      (op.rhs_dilation() && !isSplatValue(*op.rhs_dilation(), 1))) {
    return rewriter.notifyMatchFailure(op, "non-one dialation unsupported yet");
  }

  if (const mhlo::ConvDimensionNumbers &dimension_numbers =
          op.dimension_numbers()) {
    // Make sure that this is 2-D convolution.
    const int spatialRank =
        llvm::size(dimension_numbers.input_spatial_dimensions());
    if (spatialRank != 2) {
      return rewriter.notifyMatchFailure(op, "only support 2-D cases for now");
    }

    // Make sure that this is depthwise convolution.
    int64_t inputFeatureDim =
        dimension_numbers.input_feature_dimension().getInt();
    int64_t inputFeatureCount =
        op.lhs().getType().cast<ShapedType>().getDimSize(inputFeatureDim);
    if (op.feature_group_count() != inputFeatureCount) {
      return rewriter.notifyMatchFailure(op, "not depth-wise convolution");
    }

    // Make sure that this convolution has a canonical form.
    if (!hasCanonicalDimensionNumbers(dimension_numbers)) {
      return rewriter.notifyMatchFailure(op, "does not have canonical form");
    }
  }

  DenseIntElementsAttr windowStrides;
  if (op.window_strides()) {
    windowStrides = op.window_strides().getValue();
  } else {
    windowStrides = rewriter.getI64VectorAttr({1, 1});
  }

  mhlo::ConvOp::Adaptor adaptor(args);
  Location loc = op.getLoc();
  Value input = adaptor.lhs();
  Value filter = adaptor.rhs();
  auto resultType = op.getResult().getType().cast<RankedTensorType>();
  if (!resultType.hasStaticShape()) {
    return rewriter.notifyMatchFailure(op, "expected output has static shapes");
  }

  auto filterDims =
      llvm::to_vector<4>(op.rhs().getType().cast<ShapedType>().getShape());

  auto getIndicesVector = [](int start, int end) {
    return llvm::to_vector<2>(llvm::seq<int64_t>(start, end));
  };

  if (filterDims[2] * filterDims[3] != op.feature_group_count()) {
    // For cases where channel multiplier != 1
    auto outputDims = resultType.getShape();
    auto channelMultiplier = filterDims[3];
    SmallVector<int64_t> reshapedOutputDims;
    reshapedOutputDims.assign(outputDims.begin(), outputDims.end());
    reshapedOutputDims.push_back(channelMultiplier);
    reshapedOutputDims[3] /= channelMultiplier;

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, reshapedOutputDims, resultType.getElementType());
    auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
    Value zero = rewriter.create<ConstantOp>(loc, zeroAttr);
    Value zeroTensor =
        rewriter.create<linalg::FillOp>(loc, initTensor, zero).getResult(0);

    auto reshapedOutputType =
        RankedTensorType::get(reshapedOutputDims, resultType.getElementType());
    auto conv = rewriter.create<linalg::DepthwiseConvInputNHWCFilterHWCFOp>(
        op.getLoc(), reshapedOutputType, ValueRange{input, filter},
        ValueRange{zeroTensor}, windowStrides);

    // Create a Linalg reshape op that converts the output from 5 dimensions
    // into 4 dimensions (by collapsing the last two dimensions). This is needed
    // because linalg.depthwise_conv_2d_input_nhwc_filter_hwcf returns 5
    // dimensions for the output.
    SmallVector<linalg::ReassociationIndices, 4> collapsedDimList = {
        getIndicesVector(0, 1), getIndicesVector(1, 2), getIndicesVector(2, 3),
        getIndicesVector(3, 5)};
    rewriter.replaceOpWithNewOp<linalg::TensorReshapeOp>(
        op, resultType, conv.getResult(0), collapsedDimList);
  } else {
    // For cases where channel multiplier == 1
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
    Value zero = rewriter.create<ConstantOp>(loc, zeroAttr);
    Value zeroTensor =
        rewriter.create<linalg::FillOp>(loc, initTensor, zero).getResult(0);

    // Create a Linalg reshape op that converts the filter from 4 dimensions
    // into 3 dimensions (by droping the unit dimension). This is needed because
    // linalg.depthwise_conv_2d_input_nhwc_filter_hwc expects 3 dimensions for
    // the filter.

    filterDims[2] = op.feature_group_count();
    filterDims.pop_back();

    RankedTensorType filterShape =
        RankedTensorType::get(filterDims, op.getType().getElementType());

    SmallVector<linalg::ReassociationIndices, 4> collapsedDimList = {
        getIndicesVector(0, 1), getIndicesVector(1, 2), getIndicesVector(2, 4)};

    Value reshapedFilter = rewriter.create<linalg::TensorReshapeOp>(
        loc, filterShape, filter, collapsedDimList);

    rewriter.replaceOpWithNewOp<linalg::DepthwiseConvInputNHWCFilterHWCOp>(
        op, resultType, ValueRange{input, reshapedFilter},
        ValueRange{zeroTensor}, windowStrides);
  }

  return success();
}

/// Pattern to convert a linalg.pad_tensor operation into a fill + subtensor
/// insert. This is needed till pad_tensor op can be fused with its consumers.
struct PadTensorOpConversion : public OpConversionPattern<linalg::PadTensorOp> {
  using OpConversionPattern<linalg::PadTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::PadTensorOp padTensorOp, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    linalg::PadTensorOpAdaptor padOpAdaptor(args,
                                            padTensorOp->getAttrDictionary());
    // Check that the region is just a yield operation which is returning a
    // scalar that is not one of the arguments of the linalg operation.
    Region &region = padTensorOp.region();
    Block &block = region.front();
    if (!llvm::hasSingleElement(block)) return failure();
    auto yieldOp = cast<linalg::YieldOp>(block.getTerminator());
    if (!llvm::hasSingleElement(yieldOp.values())) return failure();
    Value yieldVal = yieldOp.values().front();
    if (llvm::any_of(block.getArguments(),
                     [&](Value v) { return v == yieldVal; })) {
      return failure();
    }

    OpBuilder::InsertionGuard g(rewriter);
    Location loc = padTensorOp.getLoc();
    auto lowPad = padTensorOp.getMixedLowPad();
    auto highPad = padTensorOp.getMixedHighPad();
    Value source = padOpAdaptor.source();
    RankedTensorType sourceType = padTensorOp.getSourceType();
    int64_t rank = sourceType.getRank();

    // TODO(ravishankarm): Use shape inference interface to get this.
    SmallVector<OpFoldResult> sourceShape;
    SmallVector<Value> outputShape;
    for (int64_t dim : llvm::seq<int64_t>(0, rank)) {
      SmallVector<Value> mapValues;
      Value sourceDim = rewriter.createOrFold<DimOp>(loc, source, dim);
      mapValues.push_back(sourceDim);
      sourceShape.push_back(sourceDim);
      AffineExpr expr = rewriter.getAffineDimExpr(0);
      unsigned numSymbols = 0;
      auto addValueOrAttr = [&](AffineExpr e, OpFoldResult valueOrAttr) {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          e = e + attr.cast<IntegerAttr>().getInt();
          return e;
        }
        e = e + rewriter.getAffineSymbolExpr(numSymbols++);
        mapValues.push_back(valueOrAttr.get<Value>());
        return e;
      };
      expr = addValueOrAttr(expr, lowPad[dim]);
      expr = addValueOrAttr(expr, highPad[dim]);
      outputShape.push_back(linalg::applyMapToValues(
          rewriter, loc, AffineMap::get(1, numSymbols, expr), mapValues)[0]);
    }
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, outputShape, sourceType.getElementType());
    Value fill =
        rewriter.create<linalg::FillOp>(loc, initTensor, yieldVal).getResult(0);
    SmallVector<OpFoldResult> strides(rank, rewriter.getI64IntegerAttr(1));
    Value replacement = rewriter.create<SubTensorInsertOp>(
        loc, source, fill, lowPad, sourceShape, strides);
    if (padTensorOp.getResultType() != replacement.getType()) {
      replacement = rewriter.create<tensor::CastOp>(
          loc, padTensorOp.getResultType(), replacement);
    }
    rewriter.replaceOp(padTensorOp, replacement);
    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.concatenate conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Converts mhlo.concatenate operation to subtensor ops + subtensor_insert ops.
struct ConcatenateOpConversion
    : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConcatenateOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected static shape for output");
    }

    Location loc = op.getLoc();
    int dim = op.dimension();
    int rank = resultType.getRank();
    SmallVector<Value, 3> offsets, sizes, strides;
    for (int i = 0; i < rank; ++i) {
      offsets.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
      sizes.push_back(rewriter.create<DimOp>(loc, args[0], i));
      strides.push_back(rewriter.create<ConstantIndexOp>(loc, 1));
    }
    Value resultDimSize = rewriter.create<ConstantIndexOp>(loc, 0);
    for (auto arg : args) {
      auto size = rewriter.create<DimOp>(loc, arg, dim);
      resultDimSize = rewriter.create<AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
    Value zero = rewriter.create<ConstantOp>(loc, zeroAttr);
    Value result =
        rewriter.create<linalg::FillOp>(loc, initTensor, zero).getResult(0);

    Value accBound = rewriter.create<ConstantIndexOp>(loc, 0);
    for (auto arg : args) {
      offsets[dim] = accBound;
      sizes[dim] = rewriter.create<DimOp>(loc, arg, dim);
      result = rewriter.create<SubTensorInsertOp>(loc, arg, result, offsets,
                                                  sizes, strides);
      accBound = rewriter.create<AddIOp>(loc, accBound, sizes[dim]);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};
}  // namespace

struct ConvertHLOToLinalgOnTensorsPass
    : public PassWrapper<ConvertHLOToLinalgOnTensorsPass, FunctionPass> {
  ConvertHLOToLinalgOnTensorsPass(bool useLinalgOnTensorsPath = false)
      : useLinalgOnTensorsPath(useLinalgOnTensorsPath){};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, linalg::LinalgDialect,
                    mhlo::MhloDialect, ShapeDialect, math::MathDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    MLIRContext *context = &getContext();
    populateHLOToLinalgOnTensorsConversionPatterns(context, patterns);
    if (useLinalgOnTensorsPath) {
      patterns.insert<PadTensorOpConversion>(context);
    }

    ConversionTarget target(getContext());
    // Don't convert the body of reduction ops.
    target.addDynamicallyLegalDialect<mhlo::MhloDialect>(
        Optional<ConversionTarget::DynamicLegalityCallbackFn>(
            [](Operation *op) {
              auto parentOp = op->getParentRegion()->getParentOp();
              return isa<mhlo::ReduceWindowOp>(parentOp);
            }));
    // Let the rest fall through.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (useLinalgOnTensorsPath) {
      // Set linalg.pad_tensor illegal for now.
      target.addIllegalOp<linalg::PadTensorOp>();
    }

    if (failed(applyPartialConversion(getFunction(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
  bool useLinalgOnTensorsPath;
};

/// This pass is just added for lit-testing when using the linalg on tensors
/// path. Remove when the linalg on tensors path becomes default.
struct ConvertHLOToLinalgOnTensorsPassExperimental
    : public ConvertHLOToLinalgOnTensorsPass {
  ConvertHLOToLinalgOnTensorsPassExperimental()
      : ConvertHLOToLinalgOnTensorsPass(true){};
};

/// Convert mhlo.constant op into std.const.
struct ConstOpConversion : public OpRewritePattern<mhlo::ConstOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConstOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.value());
    return success();
  }
};

}  // namespace

void populateHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  mhlo::populateHLOToLinalgConversionPattern(context, &patterns);
  patterns.insert<TorchIndexSelectOpConversion, ConstOpConversion,
                  ConcatenateOpConversion, DepthwiseConvOpConversion>(context);
}

static llvm::cl::opt<bool> clUseLinalgOnTensorsPath(
    "iree-linalg-on-tensors-path",
    llvm::cl::desc("Convert from MHLO to Linalg on tensors for linalg on "
                   "tensor codegen path"),
    llvm::cl::init(false));

std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass(
    bool useLinalgOnTensorsPath) {
  return std::make_unique<ConvertHLOToLinalgOnTensorsPass>(
      useLinalgOnTensorsPath);
}

static PassRegistration<ConvertHLOToLinalgOnTensorsPass> legalize_pass(
    "iree-codegen-hlo-to-linalg-on-tensors",
    "Convert from XLA-HLO ops to Linalg ops on tensors", []() {
      return std::make_unique<ConvertHLOToLinalgOnTensorsPass>(
          clUseLinalgOnTensorsPath);
    });

}  // namespace iree_compiler
}  // namespace mlir
