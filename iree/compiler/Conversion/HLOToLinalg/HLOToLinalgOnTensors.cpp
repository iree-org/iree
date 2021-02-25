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
/// Note: this only supports channel multiplier == 1.
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
  int rank = resultType.getRank();

  SmallVector<Value, 8> dynSizes;
  for (int i = 0, e = rank; i < e; ++i) {
    if (!resultType.isDynamicDim(i)) continue;
    if (i != 0 && i != e - 1) {
      return rewriter.notifyMatchFailure(
          op, "expected output spatial dims to be static shapes");
    }
    dynSizes.push_back(rewriter.create<DimOp>(loc, input, i));
  }
  Value initTensor = rewriter.create<linalg::InitTensorOp>(
      loc, dynSizes, resultType.getShape(), resultType.getElementType());
  auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
  Value zero = rewriter.create<ConstantOp>(loc, zeroAttr);
  Value zeroTensor =
      rewriter.create<linalg::FillOp>(loc, initTensor, zero).getResult(0);

  // Create a Linalg reshape op that converts the filter from 4 dimensions
  // into 3 dimensions (by droping the unit dimension). This is needed because
  // linalg.depthwise_conv_2d_nhwc expects 3 dimensions for the filter.

  auto filterDims =
      llvm::to_vector<4>(op.rhs().getType().cast<ShapedType>().getShape());
  if (filterDims[2] * filterDims[3] != op.feature_group_count()) {
    return rewriter.notifyMatchFailure(
        op, "non-one channel multiplier unsupported yet");
  }
  filterDims[2] = op.feature_group_count();
  filterDims.pop_back();

  RankedTensorType filterShape =
      RankedTensorType::get(filterDims, op.getType().getElementType());

  auto getIndicesVector = [](int start, int end) {
    return llvm::to_vector<2>(llvm::seq<int64_t>(start, end));
  };

  SmallVector<linalg::ReassociationIndices, 4> collapsedDimList = {
      getIndicesVector(0, 1), getIndicesVector(1, 2), getIndicesVector(2, 4)};

  Value reshapedFilter = rewriter.create<linalg::TensorReshapeOp>(
      loc, filterShape, filter, collapsedDimList);

  rewriter.replaceOpWithNewOp<linalg::DepthwiseConvInputNHWCFilterHWCOp>(
      op, resultType, ValueRange{input, reshapedFilter}, ValueRange{zeroTensor},
      windowStrides);

  return success();
}
}  // namespace

struct ConvertHLOToLinalgOnTensorsPass
    : public PassWrapper<ConvertHLOToLinalgOnTensorsPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, mhlo::MhloDialect, ShapeDialect,
                    math::MathDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateHLOToLinalgOnTensorsConversionPatterns(&getContext(), patterns);

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

    if (failed(applyPartialConversion(getFunction(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
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
                  DepthwiseConvOpConversion>(context);
}

std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass() {
  return std::make_unique<ConvertHLOToLinalgOnTensorsPass>();
}

static PassRegistration<ConvertHLOToLinalgOnTensorsPass> legalize_pass(
    "iree-codegen-hlo-to-linalg-on-tensors",
    "Convert from XLA-HLO ops to Linalg ops on tensors");

}  // namespace iree_compiler
}  // namespace mlir
