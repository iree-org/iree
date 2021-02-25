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
// mhlo.pad conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Returns the constant value associated with the init value if the defining
/// operation is a constant.
static Attribute getInitValueAsConst(Value init) {
  Attribute attr;
  if (!matchPattern(init, m_Constant(&attr))) return {};
  if (attr.getType().isa<IntegerType, FloatType>()) return attr;

  auto splatAttr = attr.dyn_cast<SplatElementsAttr>();
  if (!splatAttr) return {};
  auto type = splatAttr.getType().dyn_cast<ShapedType>();
  if (!type) return {};
  if (auto intType = type.getElementType().dyn_cast<IntegerType>()) {
    return IntegerAttr::get(intType, splatAttr.getSplatValue<APInt>());
  } else if (auto floatType = type.getElementType().dyn_cast<FloatType>()) {
    return FloatAttr::get(floatType, splatAttr.getSplatValue<APFloat>());
  }
  return {};
}

/// Converts mhlo.pad operation to linalg.pad_tensor op.
struct PadOpConversion : public OpConversionPattern<mhlo::PadOp> {
  using OpConversionPattern<mhlo::PadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::PadOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    mhlo::PadOp::Adaptor adaptor(args);
    if (llvm::any_of(op.interior_padding().getValues<APInt>(),
                     [](APInt intVal) { return intVal.getZExtValue() != 0; })) {
      return rewriter.notifyMatchFailure(op, "expected no interior padding");
    }
    auto loc = op.getLoc();

    Attribute paddingConstVal = getInitValueAsConst(adaptor.padding_value());
    Value paddingVal =
        paddingConstVal
            ? rewriter.create<ConstantOp>(loc, paddingConstVal).getResult()
            : rewriter.create<tensor::ExtractOp>(loc, adaptor.padding_value());

    const auto &edgePaddingLow = op.edge_padding_low();
    const auto &edgePaddingHigh = op.edge_padding_high();
    SmallVector<OpFoldResult, 4> low, high;
    for (auto it :
         llvm::enumerate(llvm::zip(edgePaddingLow, edgePaddingHigh))) {
      low.push_back(rewriter.createOrFold<ConstantIndexOp>(
          loc, std::get<0>(it.value()).getZExtValue()));
      high.push_back(rewriter.createOrFold<ConstantIndexOp>(
          loc, std::get<1>(it.value()).getZExtValue()));
    }
    Type resultType = op.getResult().getType();
    auto padTensorOp = linalg::PadTensorOp::createPadScalarOp(
        resultType, adaptor.operand(), paddingVal, low, high, loc, rewriter);
    rewriter.replaceOp(op, padTensorOp.getResult());
    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.conv conversion patterns.
//===----------------------------------------------------------------------===//

namespace {

static bool isDepthwiseConv(mhlo::ConvOp op) {
  auto shape = op.rhs().getType().cast<ShapedType>().getShape();
  auto numGroups =
      shape[op.dimension_numbers().kernel_input_feature_dimension().getInt()];
  return op.feature_group_count() > 1u && op.feature_group_count() == numGroups;
}

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

/// Converts mhlo.conv operation to linalg named op. This only covers normal
/// convolution cases. The op must have canonical dimension numbers. Depthwise
/// convolution and pointwise convolution are not handled in the conversion.
struct NormalConvOpConversion : public OpConversionPattern<mhlo::ConvOp> {
  using OpConversionPattern<mhlo::ConvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    if (!hasCanonicalDimensionNumbers(op.dimension_numbers())) return failure();
    if (isDepthwiseConv(op)) return failure();

    mhlo::ConvOp::Adaptor adaptor(args);
    Location loc = op.getLoc();
    Value input = adaptor.lhs();
    Value filter = adaptor.rhs();
    auto resultType = op.getResult().getType().cast<ShapedType>();
    int rank = resultType.getRank();

    // Check if padding is zero.
    DenseIntElementsAttr padding = op.paddingAttr();
    if (padding &&
        (!padding.isSplat() || padding.getSplatValue<int64_t>() != 0)) {
      return rewriter.notifyMatchFailure(op, "expected no padding");
    }

    // The output shape is N spatial_dims F.
    SmallVector<Value, 8> dynSizes;
    for (int i = 0, e = rank - 1; i < e; ++i) {
      if (!resultType.isDynamicDim(i)) continue;
      dynSizes.push_back(rewriter.create<DimOp>(loc, input, i));
    }
    if (resultType.isDynamicDim(rank - 1)) {
      dynSizes.push_back(rewriter.create<DimOp>(loc, filter, rank - 1));
    }
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, dynSizes, resultType.getShape(), resultType.getElementType());
    auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
    Value zero = rewriter.create<ConstantOp>(loc, zeroAttr);
    Value zeroTensor =
        rewriter.create<linalg::FillOp>(loc, initTensor, zero).getResult(0);
    linalg::LinalgOp res;
    Attribute strides = op.window_stridesAttr();
    // TODO(ataei): Only support dilated kernel right now. We need to consider
    // input dilation for deconvolution cases.
    Attribute dilations = op.rhs_dilationAttr();
    switch (rank) {
      case 3: {
        res = rewriter.create<linalg::ConvInputNWCFilterWCFOp>(
            loc, resultType, ValueRange{input, filter}, ValueRange{zeroTensor},
            dilations, strides);
        break;
      }
      case 4: {
        res = rewriter.create<linalg::ConvInputNHWCFilterHWCFOp>(
            loc, resultType, ValueRange{input, filter}, ValueRange{zeroTensor},
            dilations, strides);
        break;
      }
      case 5: {
        res = rewriter.create<linalg::ConvInputNDHWCFilterDHWCFOp>(
            loc, resultType, ValueRange{input, filter}, ValueRange{zeroTensor},
            dilations, strides);
        break;
      }
      default:
        return rewriter.notifyMatchFailure(op, "expected 1/2/3D conv op");
    }
    rewriter.replaceOp(op, res.getOperation()->getResults());
    return success();
  }
};
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
                  PadOpConversion, NormalConvOpConversion>(context);
}

std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass() {
  return std::make_unique<ConvertHLOToLinalgOnTensorsPass>();
}

static PassRegistration<ConvertHLOToLinalgOnTensorsPass> legalize_pass(
    "iree-codegen-hlo-to-linalg-on-tensors",
    "Convert from XLA-HLO ops to Linalg ops on tensors");

}  // namespace iree_compiler
}  // namespace mlir
