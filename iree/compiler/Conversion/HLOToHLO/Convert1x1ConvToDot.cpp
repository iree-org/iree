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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Rewrites an n-d (n, d1, d2, d3, ..., ci) * (1, 1, 1, ..., ci, co)
// as (n * d1 * d2 * d3, ..., ci) . (ci, co)
// TODO(#4876): this pattern should be replaced by a pattern that converts
// linalg.conv to linalg.matmul.
class Convert1x1ConvolutionToDotOp : public OpRewritePattern<mhlo::ConvOp> {
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

struct Convert1x1ConvToDotPass
    : public PassWrapper<Convert1x1ConvToDotPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect>();
  }

  void runOnFunction() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<Convert1x1ConvolutionToDotOp>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createConvert1x1ConvToDotPass() {
  return std::make_unique<Convert1x1ConvToDotPass>();
}

static PassRegistration<Convert1x1ConvToDotPass> pass(
    "iree-codegen-convert-1x1-conv-to-dot",
    "Convert mhlo.convolution ops with 1x1 kernels into mhlo.dot ops");

}  // namespace iree_compiler
}  // namespace mlir
