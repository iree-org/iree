// Copyright 2021 Google LLC
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

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
class Convert1x1ConvolutionMatmulOp
    : public OpRewritePattern<linalg::ConvInputNHWCFilterHWCFOp> {
 public:
  using OpRewritePattern<linalg::ConvInputNHWCFilterHWCFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ConvInputNHWCFilterHWCFOp convOp,
                                PatternRewriter &rewriter) const override {
    ShapedType inputShapeType = convOp.getInputShapedType(0);
    ShapedType filterShapeType = convOp.getInputShapedType(1);
    ShapedType outputShapeType = convOp.getOutputShapedType(0);

    auto inputShape = inputShapeType.getShape();
    auto filterShape = filterShapeType.getShape();
    auto outputShape = outputShapeType.getShape();

    if (filterShape[0] != 1 || filterShape[1] != 1) return failure();

    // TODO(ataei): Support conversion to linalg.batch_matmul.
    if (inputShape[0] != 1) return failure();

    if (!llvm::all_of(convOp.strides(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();
    if (!llvm::all_of(convOp.dilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    SmallVector<linalg::ReassociationIndices, 4> reassociationIndices = {
        {0, 1, 2}, {3}};

    auto reshapedInputType =
        RankedTensorType::get({inputShape[1] * inputShape[2], inputShape[3]},
                              inputShapeType.getElementType());

    auto reshapedFilterType = RankedTensorType::get(
        {filterShape[2], filterShape[3]}, filterShapeType.getElementType());

    auto reshapedOutputType =
        RankedTensorType::get({outputShape[1] * outputShape[2], outputShape[3]},
                              outputShapeType.getElementType());

    Value input = convOp.getInput(0);
    Value filter = convOp.getInput(1);
    Value output = convOp.getOutput(0);
    auto loc = convOp.getLoc();

    Value reshapedInput = rewriter.create<linalg::TensorReshapeOp>(
        loc, reshapedInputType, input, reassociationIndices);
    Value reshapedFilter = rewriter.create<linalg::TensorReshapeOp>(
        loc, reshapedFilterType, filter, reassociationIndices);
    Value reshapedOutput = rewriter.create<linalg::TensorReshapeOp>(
        loc, reshapedOutputType, output, reassociationIndices);

    auto matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType, ArrayRef<Value>{reshapedInput, reshapedFilter},
        ArrayRef<Value>{reshapedOutput});

    auto reshapedResult = rewriter.create<linalg::TensorReshapeOp>(
        loc, outputShapeType, matmulResult.getResults()[0],
        reassociationIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

struct Convert1x1ConvToMatmulPass
    : public PassWrapper<Convert1x1ConvToMatmulPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnFunction() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<Convert1x1ConvolutionMatmulOp>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createConvert1x1ConvToMatmulPass() {
  return std::make_unique<Convert1x1ConvToMatmulPass>();
}

static PassRegistration<Convert1x1ConvToMatmulPass> pass(
    "iree-codegen-convert-1x1-conv-to-matmul",
    "Convert linalg convolution ops with 1x1 kernels into linalg matrix "
    "multiplication ops.");

}  // namespace iree_compiler
}  // namespace mlir
