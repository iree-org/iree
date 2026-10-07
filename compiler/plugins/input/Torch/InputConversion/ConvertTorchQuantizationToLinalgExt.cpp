// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_CONVERTTORCHQUANTIZATIONTOLINALGEXTPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace Torch = torch::Torch;
namespace TorchConversion = torch::TorchConversion;

namespace {

// The affine quantization described by a PT2E quantize or dequantize op. It
// does not depend on the direction of the op, and per-tensor and per-channel
// quantization differ only in whether `axis` is set. Per-channel-group
// quantization sets `groupSize` instead.
struct PT2EQuantization {
  // Builtin tensor that is quantized or dequantized.
  Value input;
  // Builtin scale: an f64 scalar for per-tensor quantization or a rank-one
  // float tensor for per-channel quantization. Per-channel-group scales are
  // rank-two, with one value per row and group.
  Value scale;
  // Builtin zero point: an i64 scalar or a rank-one integer tensor; null for
  // symmetric quantization. Per-channel-group zero points are shaped like the
  // scales.
  Value zeroPoint;
  // Whether the zero-point elements are unsigned. Read from the Torch type
  // because builtin integer types are signless.
  bool zeroPointUnsigned = false;
  // Non-negative input dimension indexed by per-channel parameters; nullopt
  // for per-tensor quantization.
  std::optional<int64_t> axis;
  // Number of consecutive last-dimension elements that share parameters in
  // per-channel-group quantization; zero otherwise.
  int64_t groupSize = 0;
  // Inclusive lower bound of the quantized value range.
  int64_t quantMin = 0;
  // Inclusive upper bound of the quantized value range.
  int64_t quantMax = 0;
};

} // namespace

// LinalgExt takes the bounds as attributes, so they must be Torch constants.
static LogicalResult matchConstantBounds(Operation *op, Value quantMin,
                                         Value quantMax,
                                         PT2EQuantization &quantization,
                                         ConversionPatternRewriter &rewriter) {
  if (!matchPattern(quantMin,
                    Torch::m_TorchConstantInt(&quantization.quantMin)) ||
      !matchPattern(quantMax,
                    Torch::m_TorchConstantInt(&quantization.quantMax))) {
    return rewriter.notifyMatchFailure(op,
                                       "expected constant quantization bounds");
  }
  return success();
}

// LinalgExt indexes per-channel parameters with an affine map, so the axis
// must be a Torch constant. Requires `quantization.input` to be set.
static LogicalResult matchConstantAxis(Operation *op, Value axis,
                                       PT2EQuantization &quantization,
                                       ConversionPatternRewriter &rewriter) {
  int64_t dimension;
  if (!matchPattern(axis, Torch::m_TorchConstantInt(&dimension))) {
    return rewriter.notifyMatchFailure(op, "expected a constant axis");
  }
  int64_t rank = cast<RankedTensorType>(quantization.input.getType()).getRank();
  int64_t positiveDimension = Torch::toPositiveDim(dimension, rank);
  if (!Torch::isValidDim(positiveDimension, rank)) {
    return rewriter.notifyMatchFailure(op, "expected an axis within the rank");
  }
  quantization.axis = positiveDimension;
  return success();
}

// LinalgExt indexes per-channel-group parameters on the input with its last
// dimension split into groups, so the group size must be a Torch constant that
// divides the static last dimension of a rank-2 input. Requires
// `quantization.input` to be set.
static LogicalResult
matchConstantGroupSize(Operation *op, Value groupSize,
                       PT2EQuantization &quantization,
                       ConversionPatternRewriter &rewriter) {
  if (!matchPattern(groupSize,
                    Torch::m_TorchConstantInt(&quantization.groupSize)) ||
      quantization.groupSize <= 1) {
    return rewriter.notifyMatchFailure(
        op, "expected a constant group size greater than one");
  }
  auto inputType = cast<RankedTensorType>(quantization.input.getType());
  if (inputType.getRank() != 2 || inputType.isDynamicDim(1)) {
    return rewriter.notifyMatchFailure(
        op, "expected a rank-2 input with a static last dimension");
  }
  if (inputType.getDimSize(1) % quantization.groupSize != 0) {
    return rewriter.notifyMatchFailure(
        op, "expected groups that evenly split the last dimension");
  }
  return success();
}

// How each PT2E op class spells its operands is the only thing that differs
// between the ops of one direction, so each class has its own reader.

static FailureOr<PT2EQuantization> readPT2EQuantization(
    Torch::QuantizedDecomposedQuantizePerTensorOp op,
    Torch::QuantizedDecomposedQuantizePerTensorOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  PT2EQuantization quantization;
  quantization.input = adaptor.getInput();
  quantization.scale = adaptor.getScale();
  quantization.zeroPoint = adaptor.getZeroPoint();
  if (failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter))) {
    return failure();
  }
  return quantization;
}

static FailureOr<PT2EQuantization> readPT2EQuantization(
    Torch::QuantizedDecomposedDequantizePerTensorOp op,
    Torch::QuantizedDecomposedDequantizePerTensorOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  PT2EQuantization quantization;
  quantization.input = adaptor.getInput();
  quantization.scale = adaptor.getScale();
  quantization.zeroPoint = adaptor.getZeroPoint();
  if (failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter))) {
    return failure();
  }
  return quantization;
}

static FailureOr<PT2EQuantization> readPT2EQuantization(
    Torch::QuantizedDecomposedQuantizePerChannelOp op,
    Torch::QuantizedDecomposedQuantizePerChannelOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  PT2EQuantization quantization;
  quantization.input = adaptor.getInput();
  quantization.scale = adaptor.getScales();
  quantization.zeroPoint = adaptor.getZeroPoints();
  quantization.zeroPointUnsigned =
      torch::torch_to_linalg::isUnsignedTorchType(op.getZeroPoints().getType());
  if (failed(matchConstantAxis(op, op.getAxis(), quantization, rewriter)) ||
      failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter))) {
    return failure();
  }
  return quantization;
}

static FailureOr<PT2EQuantization> readPT2EQuantization(
    Torch::QuantizedDecomposedDequantizePerChannelOp op,
    Torch::QuantizedDecomposedDequantizePerChannelOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  PT2EQuantization quantization;
  quantization.input = adaptor.getInput();
  quantization.scale = adaptor.getScales();
  // A None zero point denotes symmetric quantization.
  if (!isa<Torch::NoneType>(op.getZeroPoints().getType())) {
    quantization.zeroPoint = adaptor.getZeroPoints();
    quantization.zeroPointUnsigned =
        torch::torch_to_linalg::isUnsignedTorchType(
            op.getZeroPoints().getType());
  }
  if (failed(matchConstantAxis(op, op.getAxis(), quantization, rewriter)) ||
      failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter))) {
    return failure();
  }
  return quantization;
}

// Returns the type the LinalgExt op works on: the input type, with the last
// dimension split into groups for per-channel-group quantization.
static RankedTensorType getQuantizedType(const PT2EQuantization &quantization) {
  auto inputType = cast<RankedTensorType>(quantization.input.getType());
  if (!quantization.groupSize) {
    return inputType;
  }
  return inputType.clone({inputType.getDimSize(0),
                          inputType.getDimSize(1) / quantization.groupSize,
                          quantization.groupSize});
}

// Checks the per-channel-group operands that LinalgExt cannot express.
// `realType` and `storageType` are the Torch dtypes of the real values and the
// storage. PT2E does not check the bounds, and bool zero points would become
// i1, which LinalgExt sign-extends.
static LogicalResult checkPerChannelGroupOperands(
    Operation *op, Type realType, IntegerType storageType,
    const PT2EQuantization &quantization, ConversionPatternRewriter &rewriter) {
  if (!isa<FloatType>(realType)) {
    return rewriter.notifyMatchFailure(op,
                                       "expected floating-point real values");
  }
  unsigned storageWidth = storageType.getWidth();
  bool boundsFit = storageType.isUnsigned()
                       ? quantization.quantMin >= 0 &&
                             llvm::isUIntN(storageWidth, quantization.quantMax)
                       : llvm::isIntN(storageWidth, quantization.quantMin) &&
                             llvm::isIntN(storageWidth, quantization.quantMax);
  if (quantization.quantMin > quantization.quantMax || !boundsFit) {
    return rewriter.notifyMatchFailure(
        op, "expected ordered bounds within the storage type");
  }
  ArrayRef<int64_t> quantizedShape = getQuantizedType(quantization).getShape();
  SmallVector<int64_t> parameterShape = {quantizedShape[0], quantizedShape[1]};
  auto scaleType = dyn_cast<RankedTensorType>(quantization.scale.getType());
  if (!scaleType || !isa<FloatType>(scaleType.getElementType()) ||
      failed(verifyCompatibleShape(scaleType.getShape(), parameterShape))) {
    return rewriter.notifyMatchFailure(
        op, "expected float scales with one value per row and group");
  }
  if (!quantization.zeroPoint) {
    return success();
  }
  auto zeroPointType =
      dyn_cast<RankedTensorType>(quantization.zeroPoint.getType());
  auto zeroPointElementType =
      zeroPointType ? dyn_cast<IntegerType>(zeroPointType.getElementType())
                    : nullptr;
  if (!zeroPointElementType || zeroPointElementType.getWidth() == 1 ||
      failed(verifyCompatibleShape(zeroPointType.getShape(), parameterShape))) {
    return rewriter.notifyMatchFailure(
        op, "expected non-bool integer zero points shaped like the scales");
  }
  return success();
}

// The per-channel-group readers also check what LinalgExt can express.

static FailureOr<PT2EQuantization> readPT2EQuantization(
    Torch::QuantizedDecomposedQuantizePerChannelGroupOp op,
    Torch::QuantizedDecomposedQuantizePerChannelGroupOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  PT2EQuantization quantization;
  quantization.input = adaptor.getInput();
  quantization.scale = adaptor.getScales();
  quantization.zeroPoint = adaptor.getZeroPoints();
  quantization.zeroPointUnsigned =
      torch::torch_to_linalg::isUnsignedTorchType(op.getZeroPoints().getType());
  if (failed(matchConstantGroupSize(op, op.getGroupSize(), quantization,
                                    rewriter)) ||
      failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter)) ||
      failed(checkPerChannelGroupOperands(
          op, cast<Torch::BaseTensorType>(op.getInput().getType()).getDtype(),
          cast<IntegerType>(
              cast<Torch::BaseTensorType>(op.getType()).getDtype()),
          quantization, rewriter))) {
    return failure();
  }
  // quantize_affine computes in the scale type, whereas the reference computes
  // in the wider of the input and scale types and, in 16 bits, rounds its sum
  // with the zero point.
  auto inputElementType =
      dyn_cast<FloatType>(getElementTypeOrSelf(quantization.input));
  auto scaleElementType =
      dyn_cast<FloatType>(getElementTypeOrSelf(quantization.scale));
  if (inputElementType && scaleElementType &&
      scaleElementType.getWidth() <
          std::max(32u, inputElementType.getWidth())) {
    return rewriter.notifyMatchFailure(
        op,
        "expected scales of at least 32 bits and as wide as the real values");
  }
  return quantization;
}

static FailureOr<PT2EQuantization> readPT2EQuantization(
    Torch::QuantizedDecomposedDequantizePerChannelGroupOp op,
    Torch::QuantizedDecomposedDequantizePerChannelGroupOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  PT2EQuantization quantization;
  quantization.input = adaptor.getInput();
  quantization.scale = adaptor.getScales();
  // A None zero point denotes symmetric quantization. An optional one may be
  // either, so it is left for Torch-to-Linalg.
  Type zeroPointType = op.getZeroPoints().getType();
  if (isa<Torch::OptionalType>(zeroPointType)) {
    return rewriter.notifyMatchFailure(op,
                                       "expected a tensor or None zero point");
  }
  if (!isa<Torch::NoneType>(zeroPointType)) {
    quantization.zeroPoint = adaptor.getZeroPoints();
    quantization.zeroPointUnsigned =
        torch::torch_to_linalg::isUnsignedTorchType(zeroPointType);
  }
  if (failed(matchConstantGroupSize(op, op.getGroupSize(), quantization,
                                    rewriter)) ||
      failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter)) ||
      failed(checkPerChannelGroupOperands(
          op, cast<Torch::BaseTensorType>(op.getType()).getDtype(),
          cast<IntegerType>(
              cast<Torch::BaseTensorType>(op.getInput().getType()).getDtype()),
          quantization, rewriter))) {
    return failure();
  }
  return quantization;
}

// Checks whether LinalgExt can use the scale and zero point, reporting why
// unsupported parameters prevent conversion. Per-tensor scalars need no
// additional checks. `realType` is the element type of the floating-point
// input to quantize or output from dequantize.
static LogicalResult
checkScaleAndZeroPointSupport(Operation *op, Type realType,
                              const PT2EQuantization &quantization,
                              ConversionPatternRewriter &rewriter) {
  if (!quantization.axis && !quantization.groupSize) {
    return success();
  }
  auto scaleType = dyn_cast<RankedTensorType>(quantization.scale.getType());
  if (!scaleType ||
      (quantization.zeroPoint &&
       !isa<RankedTensorType>(quantization.zeroPoint.getType()))) {
    return rewriter.notifyMatchFailure(
        op, "expected per-channel parameters with known dtypes");
  }
  // LinalgExt cannot mix distinct float types of equal width, e.g. a bf16 real
  // type with f16 scales. Per-tensor scales are converted to f32, which no
  // PyTorch real type conflicts with.
  auto scaleElementType = cast<FloatType>(scaleType.getElementType());
  if (scaleElementType != realType &&
      scaleElementType.getWidth() == realType.getIntOrFloatBitWidth()) {
    return rewriter.notifyMatchFailure(
        op, "expected scales and real values of distinct widths or one type");
  }
  return success();
}

// Truncates a per-tensor scale to f32 and returns tensor scales unchanged. A
// per-tensor scale is a Python float, which converts to an f64 scalar.
// Inductor applies it as an f32 constant and rounds once to the result dtype,
// and LinalgExt computes in the element type of the scale, so an f32 scale
// reproduces that arithmetic. Eager PyTorch computes f64 results in f64, which
// this deliberately does not follow.
// https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_inductor/lowering.py#L1960-L1965
static Value truncateScalarScaleToF32(OpBuilder &builder, Location location,
                                      Value scale) {
  Type f32Type = builder.getF32Type();
  if (!isa<FloatType>(scale.getType()) || scale.getType() == f32Type) {
    return scale;
  }
  return arith::TruncFOp::create(builder, location, f32Type, scale);
}

// Returns the LinalgExt indexing maps in operand order: input, scale, zero
// point when present, and output.
static ArrayAttr getIndexingMaps(Builder &builder,
                                 const PT2EQuantization &quantization) {
  int64_t rank = getQuantizedType(quantization).getRank();
  // The scale and zero point are scalars for per-tensor quantization and are
  // indexed by the channel dimension for per-channel quantization.
  AffineMap parameterMap =
      quantization.axis
          ? AffineMap::get(rank, 0,
                           builder.getAffineDimExpr(*quantization.axis))
          : AffineMap::get(rank, 0, builder.getContext());
  // Per-channel-group parameters are indexed by the row and group of the view.
  if (quantization.groupSize) {
    parameterMap = AffineMap::getMultiDimMapWithTargets(rank, {0, 1},
                                                        builder.getContext());
  }
  AffineMap identity = builder.getMultiDimIdentityMap(rank);
  SmallVector<AffineMap> maps = {identity, parameterMap};
  if (quantization.zeroPoint) {
    maps.push_back(parameterMap);
  }
  maps.push_back(identity);
  return builder.getAffineMapArrayAttr(maps);
}

static Value createInit(OpBuilder &builder, Location location,
                        const PT2EQuantization &quantization,
                        RankedTensorType resultType) {
  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(builder, location, quantization.input);
  // The per-channel-group view splits the static last dimension into groups.
  if (quantization.groupSize) {
    ArrayRef<int64_t> viewShape = getQuantizedType(quantization).getShape();
    sizes.back() = builder.getIndexAttr(viewShape[1]);
    sizes.push_back(builder.getIndexAttr(viewShape[2]));
  }
  return tensor::EmptyOp::create(builder, location, sizes,
                                 resultType.getElementType());
}

// The per-channel-group view splits the last dimension of a rank-2 input.
static SmallVector<ReassociationIndices> getGroupReassociation() {
  return {{0}, {1, 2}};
}

// Returns `quantization.input`, expanded into the quantized view for
// per-channel-group quantization.
static Value createQuantizedInput(OpBuilder &builder, Location location,
                                  const PT2EQuantization &quantization) {
  if (!quantization.groupSize) {
    return quantization.input;
  }
  return tensor::ExpandShapeOp::create(
      builder, location, getQuantizedType(quantization), quantization.input,
      getGroupReassociation());
}

// Returns `resultType` on the quantized view for per-channel-group
// quantization.
static RankedTensorType
getQuantizedResultType(const PT2EQuantization &quantization,
                       RankedTensorType resultType) {
  if (!quantization.groupSize) {
    return resultType;
  }
  return resultType.clone(getQuantizedType(quantization).getShape());
}

// Collapses a per-channel-group `result` back into the input shape.
static Value collapseQuantizedResult(OpBuilder &builder, Location location,
                                     Value result,
                                     const PT2EQuantization &quantization) {
  if (!quantization.groupSize) {
    return result;
  }
  return tensor::CollapseShapeOp::create(builder, location, result,
                                         getGroupReassociation());
}

static Value createQuantizeAffine(OpBuilder &builder, Location location,
                                  RankedTensorType resultType,
                                  const PT2EQuantization &quantization,
                                  bool storageUnsigned) {
  // The PT2E reference upcasts f16 and bf16 inputs and computes in f32. It
  // multiplies by `1.0 / scale` rounded to f32, whereas LinalgExt divides by
  // the scale, so results can differ by one ulp before rounding. The
  // per-channel-group reference also adds the zero point before rounding, so
  // ties with odd zero points differ, as in Torch-to-Linalg.
  Value input = createQuantizedInput(builder, location, quantization);
  RankedTensorType quantizedResultType =
      getQuantizedResultType(quantization, resultType);
  Value scale = truncateScalarScaleToF32(builder, location, quantization.scale);
  auto quantize = IREE::LinalgExt::QuantizeAffineOp::create(
      builder, location, quantizedResultType, input, scale,
      quantization.zeroPoint,
      createInit(builder, location, quantization, resultType),
      getIndexingMaps(builder, quantization),
      builder.getI64IntegerAttr(quantization.quantMin),
      builder.getI64IntegerAttr(quantization.quantMax),
      storageUnsigned ? builder.getUnitAttr() : nullptr,
      quantization.zeroPointUnsigned ? builder.getUnitAttr() : nullptr);
  return collapseQuantizedResult(builder, location, quantize->getResult(0),
                                 quantization);
}

static Value createDequantizeAffine(OpBuilder &builder, Location location,
                                    RankedTensorType resultType,
                                    const PT2EQuantization &quantization,
                                    bool inputUnsigned) {
  Value input = createQuantizedInput(builder, location, quantization);
  RankedTensorType quantizedResultType =
      getQuantizedResultType(quantization, resultType);
  Value scale = truncateScalarScaleToF32(builder, location, quantization.scale);
  auto dequantize = IREE::LinalgExt::DequantizeAffineOp::create(
      builder, location, quantizedResultType, input, scale,
      quantization.zeroPoint,
      createInit(builder, location, quantization, resultType),
      getIndexingMaps(builder, quantization),
      builder.getI64IntegerAttr(quantization.quantMin),
      builder.getI64IntegerAttr(quantization.quantMax),
      inputUnsigned ? builder.getUnitAttr() : nullptr,
      quantization.zeroPointUnsigned ? builder.getUnitAttr() : nullptr);
  return collapseQuantizedResult(builder, location, dequantize->getResult(0),
                                 quantization);
}

namespace {

// Converts `quantize_per_tensor`, `quantize_per_channel` or
// `quantize_per_channel_group`, whose input holds the real values and whose
// result holds the quantized storage.
template <typename OpTy>
struct ConvertQuantize final : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultType =
        this->getTypeConverter()->template convertType<RankedTensorType>(
            op.getType());
    if (!inputType || !resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected tensors with known dtypes");
    }
    // PT2E also quantizes to fp8, which LinalgExt storage cannot express.
    if (!isa<IntegerType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "expected integer storage");
    }

    FailureOr<PT2EQuantization> quantization =
        readPT2EQuantization(op, adaptor, rewriter);
    if (failed(quantization)) {
      return failure();
    }
    if (failed(checkScaleAndZeroPointSupport(
            op,
            /*realType=*/inputType.getElementType(), *quantization,
            rewriter))) {
      return failure();
    }

    bool storageUnsigned =
        torch::torch_to_linalg::isUnsignedTorchType(op.getType());
    rewriter.replaceOp(op,
                       createQuantizeAffine(rewriter, op.getLoc(), resultType,
                                            *quantization, storageUnsigned));
    return success();
  }
};

// Converts `dequantize_per_tensor`, `dequantize_per_channel` or
// `dequantize_per_channel_group`, whose input holds the quantized storage and
// whose result holds the real values.
template <typename OpTy>
struct ConvertDequantize final : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultType =
        this->getTypeConverter()->template convertType<RankedTensorType>(
            op.getType());
    if (!inputType || !resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected tensors with known dtypes");
    }
    // PT2E also quantizes to fp8, which LinalgExt storage cannot express.
    if (!isa<IntegerType>(inputType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "expected integer storage");
    }

    FailureOr<PT2EQuantization> quantization =
        readPT2EQuantization(op, adaptor, rewriter);
    if (failed(quantization)) {
      return failure();
    }
    if (failed(checkScaleAndZeroPointSupport(
            op,
            /*realType=*/resultType.getElementType(), *quantization,
            rewriter))) {
      return failure();
    }

    bool inputUnsigned =
        torch::torch_to_linalg::isUnsignedTorchType(op.getInput().getType());
    rewriter.replaceOp(op,
                       createDequantizeAffine(rewriter, op.getLoc(), resultType,
                                              *quantization, inputUnsigned));
    return success();
  }
};

class ConvertTorchQuantizationToLinalgExtPass final
    : public impl::ConvertTorchQuantizationToLinalgExtPassBase<
          ConvertTorchQuantizationToLinalgExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect,
                    TorchConversion::TorchConversionDialect,
                    tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<IREE::LinalgExt::IREELinalgExtDialect,
                           tensor::TensorDialect, arith::ArithDialect>();
    TypeConverter typeConverter;
    // Keeps types the backend conversion does not handle, such as the
    // `!torch.none` zero points of symmetric quantization, so that they do not
    // block operand conversion.
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    patterns.add<
        ConvertQuantize<Torch::QuantizedDecomposedQuantizePerTensorOp>,
        ConvertQuantize<Torch::QuantizedDecomposedQuantizePerChannelOp>,
        ConvertQuantize<Torch::QuantizedDecomposedQuantizePerChannelGroupOp>,
        ConvertDequantize<Torch::QuantizedDecomposedDequantizePerTensorOp>,
        ConvertDequantize<Torch::QuantizedDecomposedDequantizePerChannelOp>,
        ConvertDequantize<
            Torch::QuantizedDecomposedDequantizePerChannelGroupOp>>(
        typeConverter, context);
    // The PT2E ops are deliberately not marked illegal: ops whose metadata
    // LinalgExt cannot represent remain for the Torch-to-Linalg lowering.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::TorchInput
