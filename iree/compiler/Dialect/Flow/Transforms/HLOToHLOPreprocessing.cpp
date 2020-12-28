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

#include <numeric>

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

static llvm::cl::opt<bool> extractPadFromConv(
    "iree-flow-extract-pad-from-conv",
    llvm::cl::desc("Extract padding attributes from conv op"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> orderConvFeatures(
    "iree-flow-order-conv-features",
    llvm::cl::desc("Guarantees input/output features ordered for conv kernel"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> conv1x1toDot(
    "iree-flow-1x1-conv-to-dot",
    llvm::cl::desc("Rewrites mhlo.conv with 1x1 filter into mhlo.dot"),
    llvm::cl::init(true));

static bool isAllZero(DenseIntElementsAttr attr) {
  if (!attr.isSplat()) return false;
  return attr.getSplatValue<IntegerAttr>().getInt() == 0;
}

static bool isIota(ArrayRef<int64_t> array) {
  for (auto it : llvm::enumerate(array)) {
    if (it.index() != it.value()) {
      return false;
    }
  }
  return true;
}

/// Returns true if the conv op has padding attribute, and that it has
/// non-zero entries.
static bool hasPadding(mhlo::ConvOp op) {
  Optional<DenseIntElementsAttr> padding = op.padding();
  if (!padding) return false;
  return llvm::any_of(padding.getValue(),
                      [](APInt v) -> bool { return !v.isNullValue(); });
}

static DenseIntElementsAttr make1DElementsAttr(PatternRewriter &rewriter,
                                               ArrayRef<int64_t> integers) {
  auto type = RankedTensorType::get({static_cast<int64_t>(integers.size())},
                                    rewriter.getIntegerType(64));
  return DenseIntElementsAttr::get(type, integers);
}

class DecomposeLog1PPattern : public OpRewritePattern<mhlo::Log1pOp> {
 public:
  using OpRewritePattern<mhlo::Log1pOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::Log1pOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = op.operand().getType().cast<TensorType>();
    DenseElementsAttr attr =
        DenseElementsAttr::get(type, rewriter.getF32FloatAttr(1.0));
    auto one = rewriter.create<ConstantOp>(loc, attr);
    auto x = rewriter.create<mhlo::AddOp>(loc, op.operand(), one);
    rewriter.replaceOpWithNewOp<mhlo::LogOp>(op, x);
    return success();
  }
};

class DecomposeExpM1Pattern : public OpRewritePattern<mhlo::Expm1Op> {
 public:
  using OpRewritePattern<mhlo::Expm1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::Expm1Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = op.operand().getType().cast<TensorType>();
    DenseElementsAttr attr =
        DenseElementsAttr::get(type, rewriter.getF32FloatAttr(1.0));
    auto one = rewriter.create<ConstantOp>(loc, attr);
    auto x = rewriter.create<mhlo::ExpOp>(loc, op.operand());
    rewriter.replaceOpWithNewOp<mhlo::SubOp>(op, x, one);
    return success();
  }
};

class ExtractConvOpPaddingAttributes : public OpRewritePattern<mhlo::ConvOp> {
 public:
  using OpRewritePattern<mhlo::ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasPadding(op)) return failure();
    auto inputType = op.lhs().getType().cast<ShapedType>();
    int rank = inputType.getRank();

    // TODO(suderman): Add proper support for padding + dilation for codegen.
    // We can't extract padding if the left hand side has dilation.
    if (op.lhs_dilation().hasValue()) {
      for (auto val : op.lhs_dilation().getValue().getIntValues()) {
        if (val != 1) {
          return failure();
        }
      }
    }

    SmallVector<int64_t, 4> paddingLow, paddingHigh, interiorPadding, shape;
    paddingLow.append(rank, 0);
    paddingHigh.append(rank, 0);
    interiorPadding.append(rank, 0);
    for (auto iter :
         llvm::enumerate(op.dimension_numbers().input_spatial_dimensions())) {
      unsigned idx = iter.index();
      unsigned dim = iter.value().getZExtValue();
      paddingLow[dim] = op.paddingAttr().getValue<int64_t>({idx, 0});
      paddingHigh[dim] = op.paddingAttr().getValue<int64_t>({idx, 1});
    }
    for (unsigned i = 0; i < rank; ++i) {
      // mhlo.pad doesn't support dynamic shape.
      if (inputType.isDynamicDim(i)) return failure();
      int size = inputType.getShape()[i];
      shape.push_back(size + paddingLow[i] + paddingHigh[i]);
    }

    auto toDenseAttr = [&rewriter](ArrayRef<int64_t> elements) {
      return DenseIntElementsAttr::get(
          RankedTensorType::get(elements.size(), rewriter.getIntegerType(64)),
          elements);
    };

    auto loc = op.getLoc();
    auto padResultType =
        RankedTensorType::get(shape, inputType.getElementType());
    Attribute zeroAttr = rewriter.getZeroAttr(
        RankedTensorType::get({}, inputType.getElementType()));
    auto zero = rewriter.create<ConstantOp>(loc, zeroAttr);
    auto padOp = rewriter.create<mhlo::PadOp>(
        loc, padResultType, op.lhs(), zero, toDenseAttr(paddingLow),
        toDenseAttr(paddingHigh), toDenseAttr(interiorPadding));
    auto resultType = op.getResult().getType();
    auto newOp = rewriter.create<mhlo::ConvOp>(
        op.getLoc(), resultType, padOp.getResult(), op.rhs(),
        op.window_stridesAttr(), /*padding=*/nullptr, op.lhs_dilationAttr(),
        op.rhs_dilationAttr(), /*window_reversal=*/nullptr,
        op.dimension_numbersAttr(), op.feature_group_countAttr(),
        op.batch_group_countAttr(), op.precision_configAttr());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Guarantee that the input dimensions are ordered batch, spatial_dims, feature
// dim.
class ReorderConvOpInputDimensions : public OpRewritePattern<mhlo::ConvOp> {
 public:
  using OpRewritePattern<mhlo::ConvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = op.lhs().getType().cast<ShapedType>();
    auto lhsShape = lhsType.getShape();
    if (!lhsType.hasRank()) {
      return failure();
    }

    auto dimensionNumbers = op.dimension_numbers();
    auto inputSpatialDimensions = dimensionNumbers.input_spatial_dimensions();
    llvm::SmallVector<int64_t, 4> spatialDims;
    for (auto dim : inputSpatialDimensions) {
      spatialDims.push_back(dim.getSExtValue());
    }

    // Compute the permutation required to create a standard order.
    llvm::SmallVector<int64_t, 4> permutations;
    permutations.push_back(
        dimensionNumbers.input_batch_dimension().getValue().getSExtValue());
    permutations.append(spatialDims.begin(), spatialDims.end());
    permutations.push_back(
        dimensionNumbers.input_feature_dimension().getValue().getSExtValue());

    // If the permutation is iota then no reordering is required.
    if (isIota(permutations)) {
      return failure();
    }

    llvm::SmallVector<int64_t, 4> transposeShape;
    for (auto p : permutations) {
      transposeShape.push_back(lhsShape[p]);
    }

    auto transposed = rewriter.create<mhlo::TransposeOp>(
        op.getLoc(),
        RankedTensorType::get(transposeShape, lhsType.getElementType()),
        op.lhs(), rewriter.getI64TensorAttr(permutations));

    llvm::SmallVector<int64_t, 4> newSpatialDimensions(spatialDims.size());
    std::iota(newSpatialDimensions.begin(), newSpatialDimensions.end(), 1);

    auto newDimensionNumbers = mhlo::ConvDimensionNumbers::get(
        /*input_batch_dimension=*/rewriter.getI64IntegerAttr(0),
        /*input_feature_dimension=*/
        rewriter.getI64IntegerAttr(newSpatialDimensions.size() + 1),
        /*input_spatial_dimensions=*/
        rewriter.getI64TensorAttr(newSpatialDimensions),
        dimensionNumbers.kernel_input_feature_dimension(),
        dimensionNumbers.kernel_output_feature_dimension(),
        dimensionNumbers.kernel_spatial_dimensions(),
        dimensionNumbers.output_batch_dimension(),
        dimensionNumbers.output_feature_dimension(),
        dimensionNumbers.output_spatial_dimensions(), op.getContext());

    SmallVector<Value, 2> operands = {transposed, op.rhs()};
    auto newConv = rewriter.create<mhlo::ConvOp>(op.getLoc(), op.getType(),
                                                 operands, op.getAttrs());
    newConv.dimension_numbersAttr(newDimensionNumbers);
    rewriter.replaceOp(op, newConv.getResult());

    return success();
  }
};

struct ReorderConvOpKernelDimensions : public OpRewritePattern<mhlo::ConvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvOp op,
                                PatternRewriter &rewriter) const override {
    auto kernel = op.rhs();
    auto kernelType = kernel.getType().cast<ShapedType>();
    if (!kernelType.hasRank()) return failure();
    auto kernelShape = kernelType.getShape();

    auto dimensionNumbers = op.dimension_numbers();

    auto inputSpatialDimensions = dimensionNumbers.kernel_spatial_dimensions();
    llvm::SmallVector<int64_t, 4> spatialDims;
    for (auto dim : inputSpatialDimensions) {
      spatialDims.push_back(dim.getSExtValue());
    }

    auto inputFeatureDimension =
        dimensionNumbers.kernel_input_feature_dimension().getInt();
    auto outputFeatureDimension =
        dimensionNumbers.kernel_output_feature_dimension().getInt();

    // Compute the permutation for the transpose.
    llvm::SmallVector<int64_t, 4> permutation(spatialDims);
    permutation.push_back(inputFeatureDimension);
    permutation.push_back(outputFeatureDimension);

    // If the permutation is iota, then no transpose is required.
    if (isIota(permutation)) return failure();

    llvm::SmallVector<int64_t, 4> transposeShape;
    for (auto perm : permutation) {
      transposeShape.push_back(kernelShape[perm]);
    }

    llvm::SmallVector<int64_t, 4> newSpatialDimensions(spatialDims.size());
    std::iota(newSpatialDimensions.begin(), newSpatialDimensions.end(), 0);

    auto transposeKernel = rewriter.create<mhlo::TransposeOp>(
        op.getLoc(),
        RankedTensorType::get(transposeShape, kernelType.getElementType()),
        kernel, rewriter.getI64TensorAttr(permutation));

    auto newDimensionNumbers = mhlo::ConvDimensionNumbers::get(
        dimensionNumbers.input_batch_dimension(),
        dimensionNumbers.input_feature_dimension(),
        dimensionNumbers.input_spatial_dimensions(),
        /*kernel_input_feature_dimension=*/
        rewriter.getI64IntegerAttr(newSpatialDimensions.size()),
        /*kernel_output_feature_dimension=*/
        rewriter.getI64IntegerAttr(newSpatialDimensions.size() + 1),
        rewriter.getI64TensorAttr(newSpatialDimensions),
        dimensionNumbers.output_batch_dimension(),
        dimensionNumbers.output_feature_dimension(),
        dimensionNumbers.output_spatial_dimensions(), op.getContext());

    SmallVector<Value, 2> operands = {op.lhs(), transposeKernel};
    mhlo::ConvOp newConv = rewriter.create<mhlo::ConvOp>(
        op.getLoc(), op.getType(), operands, op.getAttrs());
    newConv.dimension_numbersAttr(newDimensionNumbers);

    rewriter.replaceOp(op, {newConv.getResult()});
    return success();
  }
};

// Guarantee that the output dimensions are ordered batch, spatial_dims, feature
// dim.
class ReorderConvOpOutputDimensions : public OpRewritePattern<mhlo::ConvOp> {
 public:
  using OpRewritePattern<mhlo::ConvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType().cast<ShapedType>();
    auto resultShape = resultType.getShape();
    if (!resultType.hasRank()) {
      return failure();
    }

    auto dimensionNumbers = op.dimension_numbers();
    auto outputSpatialDimensions = dimensionNumbers.output_spatial_dimensions();
    llvm::SmallVector<int64_t, 4> spatialDims;
    for (auto dim : outputSpatialDimensions) {
      spatialDims.push_back(dim.getSExtValue());
    }

    // Compute the permutation to transpose to an ordered output.
    llvm::SmallVector<int64_t, 4> permutation;
    permutation.push_back(
        dimensionNumbers.output_batch_dimension().getValue().getSExtValue());
    permutation.append(spatialDims.begin(), spatialDims.end());
    permutation.push_back(
        dimensionNumbers.output_feature_dimension().getValue().getSExtValue());

    // If the permutation is iota then no reordering is required.
    if (isIota(permutation)) {
      return failure();
    }

    // Compute what the new conv shape should be.
    llvm::SmallVector<int64_t, 4> convShape;
    for (auto p : permutation) {
      convShape.push_back(resultShape[p]);
    }

    // Compute the inverse transpose to unordered and ordered output.
    llvm::SmallVector<int64_t, 4> invertPermutation(permutation.size());
    for (auto it : llvm::enumerate(permutation)) {
      invertPermutation[it.value()] = it.index();
    }

    llvm::SmallVector<int64_t, 4> newSpatialDimensions(spatialDims.size());
    std::iota(newSpatialDimensions.begin(), newSpatialDimensions.end(), 1);

    auto newDimensionNumbers = mhlo::ConvDimensionNumbers::get(
        dimensionNumbers.input_batch_dimension(),
        dimensionNumbers.input_feature_dimension(),
        dimensionNumbers.input_spatial_dimensions(),
        dimensionNumbers.kernel_input_feature_dimension(),
        dimensionNumbers.kernel_output_feature_dimension(),
        dimensionNumbers.kernel_spatial_dimensions(),
        /*output_batch_dimension=*/rewriter.getI64IntegerAttr(0),
        /*output_feature_dimension=*/
        rewriter.getI64IntegerAttr(newSpatialDimensions.size() + 1),
        /*output_spatial_dimensions=*/
        rewriter.getI64TensorAttr(newSpatialDimensions), op.getContext());

    SmallVector<Value, 2> operands = {op.lhs(), op.rhs()};
    auto newConv = rewriter.create<mhlo::ConvOp>(
        op.getLoc(),
        RankedTensorType::get(convShape, resultType.getElementType()), operands,
        op.getAttrs());
    newConv.dimension_numbersAttr(newDimensionNumbers);

    auto transposed = rewriter.create<mhlo::TransposeOp>(
        op.getLoc(), resultType, newConv,
        rewriter.getI64TensorAttr(invertPermutation));

    rewriter.replaceOp(op, transposed.getResult());
    return success();
  }
};

class ExtractReduceWindowOpPaddingAttributes
    : public OpRewritePattern<mhlo::ReduceWindowOp> {
 public:
  using OpRewritePattern<mhlo::ReduceWindowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ReduceWindowOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.padding()) return failure();

    if (op.base_dilations() || op.window_dilations()) return failure();
    if (isAllZero(op.paddingAttr())) return failure();

    auto inputType = op.operand().getType().cast<ShapedType>();
    int rank = inputType.getRank();
    SmallVector<int64_t, 4> paddingLow, paddingHigh, interiorPadding, shape;
    for (unsigned i = 0; i < rank; ++i) {
      // mhlo.pad doesn't support dynamic shape.
      if (inputType.isDynamicDim(i)) return failure();
      interiorPadding.push_back(0);
      paddingLow.push_back(op.paddingAttr().getValue<int64_t>({i, 0}));
      paddingHigh.push_back(op.paddingAttr().getValue<int64_t>({i, 1}));
      int size = inputType.getShape()[i];
      shape.push_back(size + paddingLow.back() + paddingHigh.back());
    }

    auto toDenseAttr = [&rewriter](ArrayRef<int64_t> elements) {
      return DenseIntElementsAttr::get(
          RankedTensorType::get(elements.size(), rewriter.getIntegerType(64)),
          elements);
    };

    auto loc = op.getLoc();
    auto padResultType =
        RankedTensorType::get(shape, inputType.getElementType());
    auto padOp = rewriter.create<mhlo::PadOp>(
        loc, padResultType, op.operand(), op.init_value(),
        toDenseAttr(paddingLow), toDenseAttr(paddingHigh),
        toDenseAttr(interiorPadding));
    auto newOp = rewriter.create<mhlo::ReduceWindowOp>(
        loc, op.getResult().getType(), padOp, op.init_value(),
        op.window_dimensions(), op.window_stridesAttr(),
        op.base_dilationsAttr(), op.window_dilationsAttr(),
        /*padding=*/nullptr);
    rewriter.inlineRegionBefore(op.body(), newOp.body(), newOp.body().begin());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Rewrites an n-d (n, d1, d2, d3, ..., ci) * (1, 1, 1, ..., ci, co)
// as (n * d1 * d2 * d3, ..., ci) . (ci, co)
class Lower1x1ConvolutionToDotOp : public OpRewritePattern<mhlo::ConvOp> {
 public:
  using OpRewritePattern<mhlo::ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvOp op,
                                PatternRewriter &rewriter) const override {
    // Only 1x1 convolution no groups will match.
    if (op.feature_group_count() != 1) return failure();

    Value input = op.lhs();
    Value filter = op.rhs();
    Value output = op.getResult();
    auto inputShapeType = input.getType().dyn_cast_or_null<RankedTensorType>();
    auto filterShapeType =
        filter.getType().dyn_cast_or_null<RankedTensorType>();
    auto outputShapeType =
        output.getType().dyn_cast_or_null<RankedTensorType>();

    if (!inputShapeType || !filterShapeType || !outputShapeType) {
      return failure();
    }

    auto inputShape = inputShapeType.getShape();
    auto filterShape = filterShapeType.getShape();

    auto inputBatchDim =
        op.dimension_numbers().input_batch_dimension().getInt();
    auto inputFeatureDim =
        op.dimension_numbers().input_feature_dimension().getInt();
    auto kernelInputFeatureDim =
        op.dimension_numbers().kernel_input_feature_dimension().getInt();
    auto kernelOutputFeatureDim =
        op.dimension_numbers().kernel_output_feature_dimension().getInt();

    // Match input (n, d1, d2, ..., ci) format
    if (inputFeatureDim != (inputShape.size() - 1) || inputBatchDim != 0) {
      return failure();
    }

    // Match filter (k1, k2, ..., ci, co) format
    if (kernelInputFeatureDim != (filterShape.size() - 2) ||
        kernelOutputFeatureDim != (filterShape.size() - 1)) {
      return failure();
    }

    // Check 1x1x... kernel spatial size.
    for (auto dim : op.dimension_numbers().kernel_spatial_dimensions()) {
      if (filterShape[dim.getZExtValue()] != 1) return failure();
    }

    // Check dilation & strides are ones.
    if (op.window_strides()) {
      for (auto stride : op.window_strides()->getValues<int64_t>()) {
        if (stride != 1) return failure();
      }
    }
    if (op.rhs_dilation()) {
      for (auto dilation : op.rhs_dilation()->getValues<int64_t>()) {
        if (dilation != 1) return failure();
      }
    }

    int64_t spatialSize = inputShape[0];
    for (auto dim : op.dimension_numbers().input_spatial_dimensions()) {
      spatialSize *= inputShape[dim.getZExtValue()];
    }

    Type reshapedInputType =
        RankedTensorType::get({spatialSize, inputShape[inputFeatureDim]},
                              inputShapeType.getElementType());
    Type reshapedFilterTYpe =
        RankedTensorType::get({filterShape[kernelInputFeatureDim],
                               filterShape[kernelOutputFeatureDim]},
                              filterShapeType.getElementType());
    Type dotResultType = RankedTensorType::get(
        {spatialSize, filterShape[kernelOutputFeatureDim]},
        outputShapeType.getElementType());

    Value reshapedInput =
        rewriter.create<mhlo::ReshapeOp>(op.getLoc(), reshapedInputType, input);
    Value reshapedFilter = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), reshapedFilterTYpe, filter);

    Value dotResult = rewriter.create<mhlo::DotOp>(
        op.getLoc(), dotResultType, reshapedInput, reshapedFilter,
        rewriter.getStrArrayAttr({"HIGHEST", "HIGHEST"}));

    Value reshapedResult = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), outputShapeType, dotResult);

    rewriter.replaceOp(op, reshapedResult);

    return success();
  }
};

// Adjust the shape of depthwise_conv filter where is applied by mhlo.
class AdjustDepthwiseFilterShape : public OpRewritePattern<mhlo::ConvOp> {
 public:
  using OpRewritePattern<mhlo::ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvOp op,
                                PatternRewriter &rewriter) const override {
    const auto featureInDim =
        op.dimension_numbers().kernel_input_feature_dimension().getInt();
    const auto featureOutDim =
        op.dimension_numbers().kernel_output_feature_dimension().getInt();
    const auto &kernelShape = op.rhs().getType().cast<ShapedType>().getShape();
    if (kernelShape[featureInDim] != 1) return failure();

    const auto groupCount = op.feature_group_count();
    if (groupCount == 1) return failure();
    if (kernelShape[featureOutDim] % groupCount != 0) return failure();

    SmallVector<int64_t, 4> newShape(kernelShape.begin(), kernelShape.end());
    newShape[featureInDim] = groupCount;
    newShape[featureOutDim] /= groupCount;
    auto loc = op.getLoc();
    auto elemType = op.rhs().getType().cast<ShapedType>().getElementType();
    auto reshapeOp = rewriter.create<mhlo::ReshapeOp>(
        loc, RankedTensorType::get(newShape, elemType), op.rhs());
    auto resultType = op.getResult().getType();
    SmallVector<Value, 2> operands = {op.lhs(), reshapeOp.getResult()};
    auto newOp = rewriter.create<mhlo::ConvOp>(op.getLoc(), resultType,
                                               operands, op.getAttrs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Rewrites rank-3 mhlo.dot_general so lhs contraction dimension is
// inner most (2) and rhs contraction dimension is dim right after batch
// dimension. The pattern inserts transposes so the dot_general always has the
// form: {batch_dim, parallel, contraction}.{batch_dim, contraction, parallel}
class TransposeRank3GenericDotGeneral
    : public OpRewritePattern<mhlo::DotGeneralOp> {
 public:
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsShapeType = op.lhs().getType().dyn_cast<RankedTensorType>();
    auto rhsShapeType = op.rhs().getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();

    if (!lhsShapeType || !rhsShapeType || !resultType) return failure();
    if (resultType.getRank() != 3) return failure();

    if (op.dot_dimension_numbers().lhs_contracting_dimensions().size() != 1 ||
        op.dot_dimension_numbers().rhs_contracting_dimensions().size() != 1)
      return failure();

    int64_t lhsBatchDim = (*op.dot_dimension_numbers()
                                .lhs_batching_dimensions()
                                .int_value_begin())
                              .getSExtValue();
    int64_t rhsBatchDim = (*op.dot_dimension_numbers()
                                .rhs_batching_dimensions()
                                .int_value_begin())
                              .getSExtValue();
    int64_t lhsContractionDim = (*op.dot_dimension_numbers()
                                      .lhs_contracting_dimensions()
                                      .int_value_begin())
                                    .getSExtValue();
    int64_t rhsContractionDim = (*op.dot_dimension_numbers()
                                      .rhs_contracting_dimensions()
                                      .int_value_begin())
                                    .getSExtValue();
    // Only accept rank-3 tensors with dim order when dims are :
    // lhs : {batch_dim, contraction, parallel}
    // rhs : {batch_dim, parallel, contraction}
    if (lhsBatchDim != 0 || rhsBatchDim != 0) return failure();
    // No transposes are needed.
    if (lhsContractionDim == 2 && rhsContractionDim == 1) return failure();

    Value lhs = op.lhs(), rhs = op.rhs();

    // transpose {batch_dim, contraction, parallel} case.
    if (lhsContractionDim == 1) {
      Type transposedType = RankedTensorType::get(
          {lhsShapeType.getDimSize(0), lhsShapeType.getDimSize(2),
           lhsShapeType.getDimSize(1)},
          resultType.getElementType());
      lhs = rewriter.create<mhlo::TransposeOp>(
          op.getLoc(), transposedType, lhs,
          make1DElementsAttr(rewriter, {0, 2, 1}));
    }

    // transpose {batch_dim, contraction, parallel} case.
    if (rhsContractionDim == 2) {
      Type transposedType = RankedTensorType::get(
          {rhsShapeType.getDimSize(0), rhsShapeType.getDimSize(2),
           rhsShapeType.getDimSize(1)},
          resultType.getElementType());
      rhs = rewriter.create<mhlo::TransposeOp>(
          op.getLoc(), transposedType, rhs,
          make1DElementsAttr(rewriter, {0, 2, 1}));
    }

    auto dimensionNumbers = mhlo::DotDimensionNumbers::get(
        /*lhs_batching_dimensions=*/make1DElementsAttr(rewriter, {0}),
        /*rhs_batching_dimensions=*/make1DElementsAttr(rewriter, {0}),
        /*lhs_contracting_dimensions=*/make1DElementsAttr(rewriter, {2}),
        /*rhs_contracting_dimensions=*/
        make1DElementsAttr(rewriter, {1}), rewriter.getContext());

    Value result = rewriter.create<mhlo::DotGeneralOp>(
        op.getLoc(), op.getType(), lhs, rhs, dimensionNumbers,
        op.precision_configAttr());
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Rewrite mhlo.dot_general to operate on rank-3 tensors when reduction dims are
// in consecutive order and not spliting the domain. This pattern inserts
// reshapes to collapse consecutive reduction and parallel dims to always
// generate a rank-3 dot_general op.
class RankReducedDotGeneral : public OpRewritePattern<mhlo::DotGeneralOp> {
 public:
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsShapeType = op.lhs().getType().dyn_cast<ShapedType>();
    auto rhsShapeType = op.rhs().getType().dyn_cast<ShapedType>();
    auto resultType = op.getResult().getType().dyn_cast<ShapedType>();

    if (!lhsShapeType || !rhsShapeType || !resultType) return failure();
    if (!lhsShapeType.hasStaticShape() || !rhsShapeType.hasStaticShape())
      return failure();
    if (resultType.getRank() <= 3) return failure();

    mhlo::DotDimensionNumbers dimNumbers = op.dot_dimension_numbers();
    auto lhsBatchingDims = llvm::to_vector<4>(
        llvm::map_range(dimNumbers.lhs_batching_dimensions(),
                        [](APInt v) { return v.getSExtValue(); }));
    auto rhsBatchingDims = llvm::to_vector<4>(
        llvm::map_range(dimNumbers.rhs_batching_dimensions(),
                        [](APInt v) { return v.getSExtValue(); }));
    auto lhsContractingDims = llvm::to_vector<4>(
        llvm::map_range(dimNumbers.lhs_contracting_dimensions(),
                        [](APInt v) { return v.getSExtValue(); }));
    auto rhsContractingDims = llvm::to_vector<4>(
        llvm::map_range(dimNumbers.rhs_contracting_dimensions(),
                        [](APInt v) { return v.getSExtValue(); }));

    if (lhsBatchingDims.empty() || rhsBatchingDims.empty()) return failure();

    llvm::sort(lhsBatchingDims);
    llvm::sort(lhsContractingDims);
    llvm::sort(rhsBatchingDims);
    llvm::sort(rhsContractingDims);

    auto isConsecutive = [](ArrayRef<int64_t> array) {
      for (int i = 1; i < array.size(); ++i) {
        if (array[i] - array[i - 1] != 1) return false;
      }
      return true;
    };

    auto isDomainSplit = [](ArrayRef<int64_t> shape,
                            ArrayRef<int64_t> batchingDims,
                            ArrayRef<int64_t> contractingDims) {
      // Batching and contracting are contiguous.
      if ((contractingDims.front() - batchingDims.back()) == 1) return false;
      // Contracting dims are inner most.
      if (contractingDims.back() == (shape.size() - 1)) return false;
      return true;
    };

    if (!isConsecutive(lhsBatchingDims) || !isConsecutive(lhsContractingDims) ||
        !isConsecutive(rhsBatchingDims) || !isConsecutive(rhsContractingDims))
      return failure();

    if (isDomainSplit(lhsShapeType.getShape(), lhsBatchingDims,
                      lhsContractingDims) ||
        isDomainSplit(rhsShapeType.getShape(), rhsBatchingDims,
                      rhsContractingDims))
      return failure();

    // Collapsing shape into a rank-3 tensor, returns newCollabsedShape
    // contraction and parallel dim indices.
    auto computeCollapsedShape = [](ArrayRef<int64_t> shape,
                                    ArrayRef<int64_t> batchingDims,
                                    ArrayRef<int64_t> contractingDims) {
      auto newRank =
          shape.size() - batchingDims.size() - contractingDims.size() + 2;
      auto batchingSize = std::accumulate(
          batchingDims.begin(), batchingDims.end(), 1,
          [shape](const int64_t accum, const int64_t index) -> int64_t {
            return accum * shape[index];
          });
      auto contractingSize = std::accumulate(
          contractingDims.begin(), contractingDims.end(), 1,
          [shape](const int64_t accum, const int64_t index) -> int64_t {
            return accum * shape[index];
          });

      int parallelDimIndex, contractingDimIndex, parallelDimSize = 1;
      if (contractingDims.front() - batchingDims.back() > 1) {
        parallelDimIndex = 1;
        contractingDimIndex = 2;
        for (int i = batchingDims.back() + 1; i < contractingDims.front();
             ++i) {
          parallelDimSize *= shape[i];
        }
      } else {
        contractingDimIndex = 1;
        parallelDimIndex = 2;
        for (int i = contractingDims.back() + 1; i < shape.size(); ++i) {
          parallelDimSize *= shape[i];
        }
      }
      llvm::SmallVector<int64_t, 4> newShape(newRank);
      newShape[0] = batchingSize;
      newShape[contractingDimIndex] = contractingSize;
      newShape[parallelDimIndex] = parallelDimSize;
      return std::make_tuple(newShape, contractingDimIndex, parallelDimIndex);
    };

    int lhsContractingDimIndex, rhsContractingDimIndex, lhsParallelDimIndex,
        rhsParallelDimIndex;
    SmallVector<int64_t, 4> lhsNewShape, rhsNewShape;
    std::tie(lhsNewShape, lhsContractingDimIndex, lhsParallelDimIndex) =
        computeCollapsedShape(lhsShapeType.getShape(), lhsBatchingDims,
                              lhsContractingDims);

    std::tie(rhsNewShape, rhsContractingDimIndex, rhsParallelDimIndex) =
        computeCollapsedShape(rhsShapeType.getShape(), rhsBatchingDims,
                              rhsContractingDims);
    SmallVector<int64_t, 4> resultNewShape = {lhsNewShape[0],
                                              lhsNewShape[lhsParallelDimIndex],
                                              rhsNewShape[rhsParallelDimIndex]};
    Type dotGeneralResultType =
        RankedTensorType::get(resultNewShape, resultType.getElementType());

    auto loc = op.getLoc();
    Value reshapedLhs = rewriter.create<mhlo::ReshapeOp>(
        loc, RankedTensorType::get(lhsNewShape, lhsShapeType.getElementType()),
        op.lhs());
    Value reshapedRhs = rewriter.create<mhlo::ReshapeOp>(
        loc, RankedTensorType::get(rhsNewShape, rhsShapeType.getElementType()),
        op.rhs());
    auto dimensionNumbers = mhlo::DotDimensionNumbers::get(
        /*lhs_batching_dimensions=*/make1DElementsAttr(rewriter, {0}),
        /*rhs_batching_dimensions=*/make1DElementsAttr(rewriter, {0}),
        /*lhs_contracting_dimensions=*/
        make1DElementsAttr(rewriter, {lhsContractingDimIndex}),
        /*rhs_contracting_dimensions=*/
        make1DElementsAttr(rewriter, {rhsContractingDimIndex}),
        rewriter.getContext());
    Value dotGeneralResult = rewriter.create<mhlo::DotGeneralOp>(
        loc, dotGeneralResultType, reshapedLhs, reshapedRhs, dimensionNumbers,
        op.precision_configAttr());

    Value result =
        rewriter.create<mhlo::ReshapeOp>(loc, resultType, dotGeneralResult);
    rewriter.replaceOp(op, result);

    return success();
  }
};  // namespace

// clang-format off
//
// Reorder BroadcastInDimOp and N-ary elementwise op.
//
// Rewrites the following pattern (take binary elementwise op as example)
//
// %bcastx = "mhlo.broadcast_in_dim"(%x) {broadcast_dimensions = %[[BCAST_DIMS]]} : (%[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
// %bcasty = "mhlo.broadcast_in_dim"(%y) {broadcast_dimensions = %[[BCAST_DIMS]]} : (%[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
// %result = "BinaryElementwiseOpT"(%bcastx, %bcasty) : (%[[SHAPE_AFTER_BCAST]], %[[SHAPE_AFTER_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
//
// into
//
// %z = "BinaryElementwiseOpT"(%x, %y) : (%[[SHAPE_BEFORE_BCAST]], %[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_BEFORE_BCAST]]
// %result = "mhlo.broadcast_in_dim"(%z) {broadcast_dimensions = %[[BCAST_DIMS]]} : (%[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
//
// clang-format on
template <typename ElementwiseOpT>
class ReorderBroadcastInDimOpAndElementwiseOp
    : public OpRewritePattern<ElementwiseOpT> {
 public:
  using OpRewritePattern<ElementwiseOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseOpT op,
                                PatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    assert(operation->getNumOperands() >= 1 && operation->getNumResults() == 1);

    // Verify if all operands are from BroadcastInDimOp and its
    // broadcast_dimensions is the same.
    llvm::SmallVector<mhlo::BroadcastInDimOp, 2> bcastOps;
    for (auto operand : operation->getOperands()) {
      if (auto bcastOp = operand.getDefiningOp<mhlo::BroadcastInDimOp>()) {
        bcastOps.push_back(bcastOp);
      } else {
        return failure();
      }
    }

    if (llvm::any_of(bcastOps, [&bcastOps](mhlo::BroadcastInDimOp bcastOp) {
          return bcastOp.broadcast_dimensions() !=
                 bcastOps[0].broadcast_dimensions();
        })) {
      return failure();
    }

    // Verify if all operands of BroadcastInDimOp are of same type and have
    // static shape.
    auto bcastOperandType =
        bcastOps[0].operand().getType().template dyn_cast<ShapedType>();
    llvm::SmallVector<Value, 2> bcastOperands;
    for (auto bcastOp : bcastOps) {
      auto bcastOperand = bcastOp.operand();
      auto type = bcastOperand.getType().template dyn_cast<ShapedType>();
      if (!type || !type.hasStaticShape() || type != bcastOperandType) {
        return failure();
      }
      bcastOperands.push_back(bcastOperand);
    }

    // Some elementwise ops, mhlo::RealOp for example, do not have
    // SameOperandsAndResultType trait, so resultType might be different
    // from bcastOperandType.
    auto elementType = getElementTypeOrSelf(op.getResult());
    auto resultShape = bcastOperandType.getShape();
    auto resultType = RankedTensorType::get(resultShape, elementType);

    Value result =
        rewriter.create<ElementwiseOpT>(op.getLoc(), resultType, bcastOperands);
    rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
        op, op.getType(), result, bcastOps[0].broadcast_dimensions());

    for (auto bcastOp : bcastOps) {
      if (bcastOp.getOperation()->use_empty()) {
        rewriter.eraseOp(bcastOp);
      }
    }

    return success();
  }
};

struct HLOToHLOPreprocessing
    : public PassWrapper<HLOToHLOPreprocessing, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, mhlo::MhloDialect,
                    tensor::TensorDialect>();
  }

  void runOnFunction() override {
    MLIRContext *context = &getContext();
    ConversionTarget conversionTarget(*context);
    OwningRewritePatternList conversionPatterns;
    // Note that various input modalities may do their own legalization of
    // CHLO. Converting here allows IREE to accept CHLO dialect regardless of
    // whether it was legalized away at a higher level.
    chlo::PopulateLegalizeChloToHloPatterns(context, &conversionPatterns);
    conversionTarget.addLegalDialect<shape::ShapeDialect, mhlo::MhloDialect,
                                     mlir::StandardOpsDialect,
                                     mlir::tensor::TensorDialect>();
    conversionTarget.addIllegalDialect<chlo::HloClientDialect>();
    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }

    OwningRewritePatternList patterns;
    mhlo::PopulateUnfuseBatchNormPatterns(context, &patterns);
    mhlo::PopulateComplexLoweringPatterns(context, &patterns);
    mhlo::PopulateGatherToTorchIndexSelectPatterns(context, &patterns);
    patterns.insert<ExtractReduceWindowOpPaddingAttributes,
                    AdjustDepthwiseFilterShape, DecomposeLog1PPattern,
                    DecomposeExpM1Pattern>(context);

    // dot_general canoncalization patterns.
    mhlo::PopulateGeneralDotOpLoweringPatterns(&patterns, context);
    patterns.insert<RankReducedDotGeneral, TransposeRank3GenericDotGeneral>(
        context);

    // Unary elementwise op.
    patterns.insert<
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::AbsOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::CeilOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ConvertOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ClzOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::CosOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ExpOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::Expm1Op>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::FloorOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ImagOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::IsFiniteOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::LogOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::Log1pOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::LogisticOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::NotOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::NegOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::PopulationCountOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::RealOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::RoundOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::RsqrtOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::SignOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::SinOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::SqrtOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::TanhOp>>(context);
    // Binary elementwise op.
    patterns.insert<
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::AddOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::Atan2Op>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ComplexOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::DivOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::MaxOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::MinOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::MulOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::PowOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::RemOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ShiftLeftOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ShiftRightArithmeticOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ShiftRightLogicalOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::SubOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::AndOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::OrOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::XorOp>>(context);
    if (extractPadFromConv) {
      patterns.insert<ExtractConvOpPaddingAttributes>(context);
    }
    if (orderConvFeatures) {
      patterns.insert<ReorderConvOpInputDimensions>(context);
      patterns.insert<ReorderConvOpKernelDimensions>(context);
      patterns.insert<ReorderConvOpOutputDimensions>(context);
    }
    if (conv1x1toDot) {
      patterns.insert<Lower1x1ConvolutionToDotOp>(context);
    }
    applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createHLOPreprocessingPass() {
  return std::make_unique<HLOToHLOPreprocessing>();
}

static PassRegistration<HLOToHLOPreprocessing> legalize_pass(
    "iree-flow-hlo-to-hlo-preprocessing",
    "Apply hlo to hlo transformations for some hlo ops");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
