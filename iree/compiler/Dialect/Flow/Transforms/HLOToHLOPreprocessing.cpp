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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

static llvm::cl::opt<bool> extractPadFromConv(
    "iree-flow-extract-pad-from-conv",
    llvm::cl::desc("Extract padding attributes from conv op"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> conv1x1toDot(
    "iree-flow-1x1-conv-to-dot",
    llvm::cl::desc("Rewrites mhlo.conv with 1x1 filter into mhlo.dot"),
    llvm::cl::init(false));

static bool isAllZero(DenseIntElementsAttr attr) {
  if (!attr.isSplat()) return false;
  return attr.getSplatValue<IntegerAttr>().getInt() == 0;
}

/// Returns true if the linalg op has padding attribute, and that it has
/// non-zero entries.
template <typename OpTy>
static bool hasPadding(OpTy op) {
  Optional<DenseIntElementsAttr> padding = op.padding();
  if (!padding) return false;
  return llvm::any_of(padding.getValue(),
                      [](APInt v) -> bool { return !v.isNullValue(); });
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

class ExtractConvOpPaddingAttributes : public OpRewritePattern<mhlo::ConvOp> {
 public:
  using OpRewritePattern<mhlo::ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasPadding(op)) return failure();
    auto inputType = op.lhs().getType().cast<ShapedType>();
    int rank = inputType.getRank();
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
        op.rhs_dilationAttr(), op.dimension_numbersAttr(),
        op.feature_group_countAttr(), op.batch_group_countAttr(),
        op.precision_configAttr());
    rewriter.replaceOp(op, newOp.getResult());
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

    const auto groupCount = op.feature_group_count().getZExtValue();
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

struct HLOToHLOPreprocessing
    : public PassWrapper<HLOToHLOPreprocessing, FunctionPass> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns;
    mhlo::PopulateUnfuseBatchNormPatterns(context, &patterns);
    mhlo::PopulateComplexLoweringPatterns(context, &patterns);
    mhlo::PopulateGatherToTorchIndexSelectPatterns(context, &patterns);
    // Note that various input modalities may do their own legalization of
    // CHLO. Converting here allows IREE to accept CHLO dialect regardless of
    // whether it was legalized away at a higher level.
    chlo::PopulateLegalizeChloToHloPatterns(context, &patterns);
    patterns.insert<ExtractReduceWindowOpPaddingAttributes,
                    AdjustDepthwiseFilterShape, DecomposeLog1PPattern>(context);
    if (extractPadFromConv) {
      patterns.insert<ExtractConvOpPaddingAttributes>(context);
    }
    if (conv1x1toDot) {
      patterns.insert<Lower1x1ConvolutionToDotOp>(context);
    }
    applyPatternsAndFoldGreedily(getOperation(), patterns);
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
