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
// does not depend on the direction of the op, and the granularities differ only
// in `groupSize` and in the `axes` of the quantized view along which the scale
// and zero point vary.
struct PT2EQuantization {
  // Builtin tensor that is quantized or dequantized.
  Value input;
  // Builtin scale: an f64 scalar for per-tensor quantization or a float tensor
  // with one dimension per entry of `axes` otherwise.
  Value scale;
  // Builtin zero point: an i64 scalar or an integer tensor shaped like the
  // scale; null for symmetric quantization.
  Value zeroPoint;
  // Whether the zero-point elements are unsigned. Read from the Torch type
  // because builtin integer types are signless.
  bool zeroPointUnsigned = false;
  // Number of consecutive elements of the last input dimension that share
  // parameters in per-channel-group quantization; zero otherwise. Determines
  // the quantized view, see `getQuantizedType`.
  int64_t groupSize = 0;
  // Axes of the quantized view along which the scale and zero point vary, in
  // order: none for per-tensor, the quantization axis for per-channel, and the
  // row and group axes for per-channel-group quantization.
  SmallVector<int64_t> axes;
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
  quantization.axes = {positiveDimension};
  return success();
}

// LinalgExt indexes per-channel-group parameters on an input whose last
// dimension is split into groups, so the group size must be a Torch constant
// above one that evenly splits the static last dimension of a rank-2 input.
// Requires `quantization.input` and `quantization.scale` to be set.
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
  int64_t numColumns = inputType.getDimSize(1);
  // PT2E reads a group size beyond the row with one scale per row as a single
  // group spanning the row.
  auto scaleType = dyn_cast<RankedTensorType>(quantization.scale.getType());
  if (quantization.groupSize > numColumns && numColumns > 0 && scaleType &&
      scaleType.getRank() > 0 && scaleType.getShape().back() == 1) {
    quantization.groupSize = numColumns;
  }
  if (numColumns % quantization.groupSize != 0) {
    return rewriter.notifyMatchFailure(
        op, "expected groups that evenly split the last dimension");
  }
  quantization.axes = {0, 1};
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
                                 quantization, rewriter))) {
    return failure();
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
  // A None zero point denotes symmetric quantization.
  if (!isa<Torch::NoneType>(op.getZeroPoints().getType())) {
    quantization.zeroPoint = adaptor.getZeroPoints();
    quantization.zeroPointUnsigned =
        torch::torch_to_linalg::isUnsignedTorchType(
            op.getZeroPoints().getType());
  }
  if (failed(matchConstantGroupSize(op, op.getGroupSize(), quantization,
                                    rewriter)) ||
      failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter))) {
    return failure();
  }
  return quantization;
}

// Returns the type of the view that the LinalgExt op quantizes: the input
// itself, or for per-channel-group quantization the input with its last
// dimension split into groups.
static RankedTensorType getQuantizedType(const PT2EQuantization &quantization) {
  auto inputType = cast<RankedTensorType>(quantization.input.getType());
  if (!quantization.groupSize) {
    return inputType;
  }
  return inputType.clone({inputType.getDimSize(0),
                          inputType.getDimSize(1) / quantization.groupSize,
                          quantization.groupSize});
}

// Checks whether LinalgExt can use the scale and zero point, reporting why
// unsupported parameters prevent conversion. Per-tensor scalars need no
// additional checks. `realType` is the element type of the floating-point
// input to quantize or output from dequantize.
static LogicalResult
checkScaleAndZeroPointSupport(Operation *op, Type realType,
                              const PT2EQuantization &quantization,
                              ConversionPatternRewriter &rewriter) {
  if (quantization.axes.empty()) {
    return success();
  }
  // Tensor parameters have the extents of the quantized view along `axes`.
  ArrayRef<int64_t> quantizedShape = getQuantizedType(quantization).getShape();
  SmallVector<int64_t> parameterShape = llvm::map_to_vector(
      quantization.axes, [&](int64_t axis) { return quantizedShape[axis]; });
  auto scaleType = dyn_cast<RankedTensorType>(quantization.scale.getType());
  auto scaleElementType =
      scaleType ? dyn_cast<FloatType>(scaleType.getElementType()) : nullptr;
  if (!scaleElementType ||
      failed(verifyCompatibleShape(scaleType.getShape(), parameterShape))) {
    return rewriter.notifyMatchFailure(
        op, "expected float scales with the extents of the indexed axes");
  }
  if (quantization.zeroPoint) {
    auto zeroPointType =
        dyn_cast<RankedTensorType>(quantization.zeroPoint.getType());
    auto zeroPointElementType =
        zeroPointType ? dyn_cast<IntegerType>(zeroPointType.getElementType())
                      : nullptr;
    // Torch bool zero points become i1, which LinalgExt would sign-extend to
    // -1 while PT2E reads them as 1.
    if (!zeroPointElementType || zeroPointElementType.getWidth() == 1 ||
        failed(
            verifyCompatibleShape(zeroPointType.getShape(), parameterShape))) {
      return rewriter.notifyMatchFailure(
          op, "expected non-bool integer zero points shaped like the scales");
    }
  }
  // LinalgExt cannot mix distinct float types of equal width, e.g. a bf16 real
  // type with f16 scales. Per-tensor scales are converted to f32, which no
  // PyTorch real type conflicts with.
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
  // The scale and zero point vary along `axes`, which are none for per-tensor
  // quantization.
  AffineMap parameterMap = AffineMap::getMultiDimMapWithTargets(
      rank, llvm::to_vector_of<unsigned>(quantization.axes),
      builder.getContext());
  AffineMap identity = builder.getMultiDimIdentityMap(rank);
  SmallVector<AffineMap> maps = {identity, parameterMap};
  if (quantization.zeroPoint) {
    maps.push_back(parameterMap);
  }
  maps.push_back(identity);
  return builder.getAffineMapArrayAttr(maps);
}

// Creates the init tensor of the op: the sizes of `input` with the element
// type of `resultType`.
static Value createInit(OpBuilder &builder, Location location, Value input,
                        RankedTensorType resultType) {
  return tensor::EmptyOp::create(
      builder, location, tensor::getMixedSizes(builder, location, input),
      resultType.getElementType());
}

// Builds a LinalgExt `quantize_affine` or `dequantize_affine` op, which take
// the same operands, on the quantized view. `storageUnsigned` is the
// signedness of the integer storage: the result of a quantize op or the input
// of a dequantize op.
template <typename OpTy>
static Value createAffineQuantizationOp(OpBuilder &builder, Location location,
                                        RankedTensorType resultType,
                                        const PT2EQuantization &quantization,
                                        bool storageUnsigned) {
  Value input = quantization.input;
  RankedTensorType opResultType = resultType;
  // The quantized view splits the last dimension of a rank-2 input.
  SmallVector<ReassociationIndices> groupReassociation = {{0}, {1, 2}};
  if (quantization.groupSize) {
    RankedTensorType quantizedType = getQuantizedType(quantization);
    input = tensor::ExpandShapeOp::create(builder, location, quantizedType,
                                          input, groupReassociation);
    opResultType = resultType.clone(quantizedType.getShape());
  }
  Value scale = truncateScalarScaleToF32(builder, location, quantization.scale);
  auto affineOp = OpTy::create(
      builder, location, opResultType, input, scale, quantization.zeroPoint,
      createInit(builder, location, input, opResultType),
      getIndexingMaps(builder, quantization),
      builder.getI64IntegerAttr(quantization.quantMin),
      builder.getI64IntegerAttr(quantization.quantMax),
      storageUnsigned ? builder.getUnitAttr() : nullptr,
      quantization.zeroPointUnsigned ? builder.getUnitAttr() : nullptr);
  if (!quantization.groupSize) {
    return affineOp->getResult(0);
  }
  return tensor::CollapseShapeOp::create(
      builder, location, affineOp->getResult(0), groupReassociation);
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
    if (!isa<FloatType>(inputType.getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "expected floating-point real values");
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
    // quantize_affine computes in the element type of the scale, so f16 and
    // bf16 scales would round f32 real values, which the PT2E reference
    // multiplies in f32. Such ops remain for Torch-to-Linalg.
    auto scaleType = dyn_cast<RankedTensorType>(quantization->scale.getType());
    if (scaleType && scaleType.getElementTypeBitWidth() < 32) {
      return rewriter.notifyMatchFailure(op,
                                         "expected scales of at least 32 bits");
    }

    bool storageUnsigned =
        torch::torch_to_linalg::isUnsignedTorchType(op.getType());
    // The PT2E reference upcasts f16 and bf16 inputs and computes in f32. It
    // multiplies by `1.0 / scale` rounded to f32, whereas LinalgExt divides by
    // the scale, so results can differ by one ulp before rounding. The
    // per-channel-group reference also computes f64 inputs in f64 and adds the
    // zero point before rounding, so exact ties round differently for odd zero
    // points, as they do in Torch-to-Linalg.
    rewriter.replaceOp(
        op,
        createAffineQuantizationOp<IREE::LinalgExt::QuantizeAffineOp>(
            rewriter, op.getLoc(), resultType, *quantization, storageUnsigned));
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
    if (!isa<FloatType>(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "expected floating-point real values");
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
    rewriter.replaceOp(
        op,
        createAffineQuantizationOp<IREE::LinalgExt::DequantizeAffineOp>(
            rewriter, op.getLoc(), resultType, *quantization, inputUnsigned));
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
