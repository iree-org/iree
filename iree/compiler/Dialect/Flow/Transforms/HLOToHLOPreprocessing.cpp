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
    "iree-extract-pad-from-conv",
    llvm::cl::desc("Extract padding attributes from conv op"),
    llvm::cl::init(true));

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
    // Note that various input modalities may do their own legalization of
    // CHLO. Converting here allows IREE to accept CHLO dialect regardless of
    // whether it was legalized away at a higher level.
    chlo::PopulateLegalizeChloToHloPatterns(context, &patterns);
    patterns.insert<ExtractReduceWindowOpPaddingAttributes,
                    AdjustDepthwiseFilterShape>(context);
    if (extractPadFromConv) {
      patterns.insert<ExtractConvOpPaddingAttributes>(context);
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
