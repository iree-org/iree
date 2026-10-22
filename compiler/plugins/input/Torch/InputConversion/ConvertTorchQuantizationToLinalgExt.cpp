// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include <cmath>

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_CONVERTTORCHQUANTIZATIONTOLINALGEXTPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace Torch = torch::Torch;
namespace TorchConversion = torch::TorchConversion;

namespace {

// The affine quantization described by a PT2E or torchao quantize or
// dequantize op. It does not depend on the direction of the op, and `blockSize`
// describes its granularity.
struct PT2EQuantization {
  // Builtin tensor that is quantized or dequantized.
  Value input;
  // Builtin scale: an f64 scalar for per-tensor quantization or a float tensor
  // with one value per block.
  Value scale;
  // Builtin zero point: an i64 scalar or an integer tensor shaped like the
  // scale; null for symmetric quantization.
  Value zeroPoint;
  // Whether the zero-point elements are unsigned. Read from the Torch type
  // because builtin integer types are signless.
  bool zeroPointUnsigned = false;
  // Number of consecutive input elements along each dimension that share
  // parameters, as in torchao's `block_size`. A block spanning a dynamic
  // dimension is `ShapedType::kDynamic`.
  SmallVector<int64_t> blockSize;
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
// must be a Torch constant. Sets `blockSize` to one along the axis and to the
// dimension size elsewhere. Requires `quantization.input` to be set.
static LogicalResult matchConstantAxis(Operation *op, Value axis,
                                       PT2EQuantization &quantization,
                                       ConversionPatternRewriter &rewriter) {
  int64_t dimension;
  if (!matchPattern(axis, Torch::m_TorchConstantInt(&dimension))) {
    return rewriter.notifyMatchFailure(op, "expected a constant axis");
  }
  auto inputType = cast<RankedTensorType>(quantization.input.getType());
  int64_t rank = inputType.getRank();
  int64_t positiveDimension = Torch::toPositiveDim(dimension, rank);
  if (!Torch::isValidDim(positiveDimension, rank)) {
    return rewriter.notifyMatchFailure(op, "expected an axis within the rank");
  }
  quantization.blockSize = llvm::to_vector(inputType.getShape());
  quantization.blockSize[positiveDimension] = 1;
  return success();
}

// LinalgExt indexes per-channel-group parameters on the input with its last
// dimension split into groups, so the group size must be a Torch constant that
// divides the static last dimension of a rank-2 input. Sets `blockSize` to one
// along the first dimension and to the group size along the last. Requires
// `quantization.input` to be set.
static LogicalResult
matchConstantGroupSize(Operation *op, Value groupSize,
                       PT2EQuantization &quantization,
                       ConversionPatternRewriter &rewriter) {
  int64_t size;
  if (!matchPattern(groupSize, Torch::m_TorchConstantInt(&size)) || size <= 1) {
    return rewriter.notifyMatchFailure(
        op, "expected a constant group size greater than one");
  }
  auto inputType = cast<RankedTensorType>(quantization.input.getType());
  if (inputType.getRank() != 2 || inputType.isDynamicDim(1)) {
    return rewriter.notifyMatchFailure(
        op, "expected a rank-2 input with a static last dimension");
  }
  if (inputType.getDimSize(1) % size != 0) {
    return rewriter.notifyMatchFailure(
        op, "expected groups that evenly split the last dimension");
  }
  quantization.blockSize = {1, size};
  return success();
}

// Each vector along the last input dimension is a token with its own
// parameters. Sets `blockSize` to the dimension size along the last dimension
// and to one elsewhere. Requires `quantization.input` to be set.
static void setTokenBlockSize(PT2EQuantization &quantization) {
  auto inputType = cast<RankedTensorType>(quantization.input.getType());
  quantization.blockSize = llvm::to_vector(inputType.getShape());
  std::fill_n(quantization.blockSize.begin(), inputType.getRank() - 1, 1);
}

// LinalgExt splits each dimension holding several blocks of several elements,
// which needs a static size. A non-constant block entry, which torch.export
// gives for a per-tensor block of a dynamic dimension, spans its dimension.
// Sets `blockSize` to the torchao block size. Requires `quantization.input` to
// be set.
static LogicalResult matchBlockSize(Operation *op, Value blockSize,
                                    PT2EQuantization &quantization,
                                    ConversionPatternRewriter &rewriter) {
  ArrayRef<int64_t> inputShape =
      cast<RankedTensorType>(quantization.input.getType()).getShape();
  FailureOr<SmallVector<int64_t>> blocks =
      Torch::getTorchaoBlockSize(blockSize, inputShape);
  if (failed(blocks)) {
    return rewriter.notifyMatchFailure(op, "expected a block_size list");
  }
  for (auto [size, block] : llvm::zip_equal(inputShape, *blocks)) {
    if (block != 1 && block != size && ShapedType::isDynamic(size)) {
      return rewriter.notifyMatchFailure(
          op, "expected blocks that split only static dimensions");
    }
  }
  quantization.blockSize = std::move(*blocks);
  return success();
}

// Returns whether the elements of a Torch zero point are unsigned. Bools count
// as unsigned, since LinalgExt would otherwise sign-extend `true` to -1.
static bool isUnsignedZeroPoint(Type zeroPointType) {
  return torch::torch_to_linalg::isUnsignedTorchType(zeroPointType) ||
         cast<Torch::ValueTensorType>(zeroPointType).getDtype().isInteger(1);
}

// Sets the zero point of `quantization` from `torchZeroPoint` and its converted
// value `zeroPoint`. A None zero point and a float one known to be zero leave
// it unset, which denotes symmetric quantization. Other float zero points
// prevent conversion, since LinalgExt zero points are integers.
static LogicalResult matchZeroPoint(Operation *op, Value torchZeroPoint,
                                    Value zeroPoint,
                                    PT2EQuantization &quantization,
                                    ConversionPatternRewriter &rewriter) {
  if (isa<Torch::NoneType>(torchZeroPoint.getType())) {
    return success();
  }
  if (isa<FloatType>(getElementTypeOrSelf(zeroPoint))) {
    if (!IREE::LinalgExt::isKnownZero(torchZeroPoint)) {
      return rewriter.notifyMatchFailure(
          op, "expected integer or all-zero float zero points");
    }
    return success();
  }
  quantization.zeroPoint = zeroPoint;
  quantization.zeroPointUnsigned =
      isUnsignedZeroPoint(torchZeroPoint.getType());
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
  quantization.blockSize = llvm::to_vector(
      cast<RankedTensorType>(quantization.input.getType()).getShape());
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
  quantization.blockSize = llvm::to_vector(
      cast<RankedTensorType>(quantization.input.getType()).getShape());
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
  if (failed(matchConstantAxis(op, op.getAxis(), quantization, rewriter)) ||
      failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter)) ||
      failed(matchZeroPoint(op, op.getZeroPoints(), adaptor.getZeroPoints(),
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
  if (failed(matchConstantAxis(op, op.getAxis(), quantization, rewriter)) ||
      failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter)) ||
      failed(matchZeroPoint(op, op.getZeroPoints(), adaptor.getZeroPoints(),
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
  if (failed(matchConstantGroupSize(op, op.getGroupSize(), quantization,
                                    rewriter)) ||
      failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter)) ||
      failed(matchZeroPoint(op, op.getZeroPoints(), adaptor.getZeroPoints(),
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
  if (failed(matchConstantGroupSize(op, op.getGroupSize(), quantization,
                                    rewriter)) ||
      failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter)) ||
      failed(matchZeroPoint(op, op.getZeroPoints(), adaptor.getZeroPoints(),
                            quantization, rewriter))) {
    return failure();
  }
  return quantization;
}

static FailureOr<PT2EQuantization> readPT2EQuantization(
    Torch::QuantizedDecomposedQuantizePerTokenOp op,
    Torch::QuantizedDecomposedQuantizePerTokenOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  PT2EQuantization quantization;
  quantization.input = adaptor.getInput();
  quantization.scale = adaptor.getScales();
  setTokenBlockSize(quantization);
  if (failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter)) ||
      failed(matchZeroPoint(op, op.getZeroPoints(), adaptor.getZeroPoints(),
                            quantization, rewriter))) {
    return failure();
  }
  return quantization;
}

static FailureOr<PT2EQuantization> readPT2EQuantization(
    Torch::QuantizedDecomposedDequantizePerTokenOp op,
    Torch::QuantizedDecomposedDequantizePerTokenOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  PT2EQuantization quantization;
  quantization.input = adaptor.getInput();
  quantization.scale = adaptor.getScales();
  setTokenBlockSize(quantization);
  if (failed(matchConstantBounds(op, op.getQuantMin(), op.getQuantMax(),
                                 quantization, rewriter)) ||
      failed(matchZeroPoint(op, op.getZeroPoints(), adaptor.getZeroPoints(),
                            quantization, rewriter))) {
    return failure();
  }
  return quantization;
}

// Both torchao directions spell their operands alike. None bounds are the
// bounds of `storageDtype`.
template <typename OpTy>
static FailureOr<PT2EQuantization>
readTorchaoQuantization(OpTy op, typename OpTy::Adaptor adaptor,
                        Value storageDtype,
                        ConversionPatternRewriter &rewriter) {
  PT2EQuantization quantization;
  quantization.input = adaptor.getInput();
  quantization.scale = adaptor.getScale();
  if (failed(matchBlockSize(op, op.getBlockSize(), quantization, rewriter)) ||
      failed(matchZeroPoint(op, op.getZeroPoint(), adaptor.getZeroPoint(),
                            quantization, rewriter))) {
    return failure();
  }
  FailureOr<std::pair<double, double>> bounds = Torch::getTorchaoQuantBounds(
      op.getQuantMin(), op.getQuantMax(), storageDtype);
  if (failed(bounds)) {
    return rewriter.notifyMatchFailure(
        op, "expected constant quantization bounds and dtype");
  }
  // LinalgExt takes integer bounds, so float bounds must be integral. The
  // verifier keeps them within the storage dtype.
  auto [quantMin, quantMax] = *bounds;
  if (quantMin != std::trunc(quantMin) || quantMax != std::trunc(quantMax)) {
    return rewriter.notifyMatchFailure(op,
                                       "expected integral quantization bounds");
  }
  quantization.quantMin = static_cast<int64_t>(quantMin);
  quantization.quantMax = static_cast<int64_t>(quantMax);
  return quantization;
}

// torchao multiplies by `1.0 / scale`, whereas LinalgExt divides by the scale
// like Torch-to-Linalg, so the two can round differently at ties. LinalgExt
// computes in the scale type, torchao in the wider of the scale and real types.
static FailureOr<PT2EQuantization>
readPT2EQuantization(Torch::TorchaoQuantizeAffineOp op,
                     Torch::TorchaoQuantizeAffineOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) {
  return readTorchaoQuantization(op, adaptor, op.getOutputDtype(), rewriter);
}

// torchao subtracts the zero point in the real type, whereas LinalgExt
// subtracts integers, so results can differ for 16-bit real values with
// storage wider than 8 bits. LinalgExt multiplies in the scale type, torchao in
// the wider of the scale and real types.
static FailureOr<PT2EQuantization>
readPT2EQuantization(Torch::TorchaoDequantizeAffineOp op,
                     Torch::TorchaoDequantizeAffineOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) {
  return readTorchaoQuantization(op, adaptor, op.getInputDtype(), rewriter);
}

// Returns, for each input dimension, its dimensions in the blocked type. A
// dimension with several blocks of several elements is split into the block
// and the position within it, so that LinalgExt can index the block.
static SmallVector<ReassociationIndices>
getBlockedReassociation(const PT2EQuantization &quantization) {
  ArrayRef<int64_t> inputShape =
      cast<RankedTensorType>(quantization.input.getType()).getShape();
  SmallVector<ReassociationIndices> reassociation;
  int64_t dim = 0;
  for (auto [size, block] :
       llvm::zip_equal(inputShape, quantization.blockSize)) {
    if (block == 1 || block == size) {
      reassociation.push_back({dim++});
    } else {
      reassociation.push_back({dim, dim + 1});
      dim += 2;
    }
  }
  return reassociation;
}

// Returns the type the LinalgExt op works on: the input type, split as
// `getBlockedReassociation` describes.
static RankedTensorType getBlockedType(const PT2EQuantization &quantization) {
  auto inputType = cast<RankedTensorType>(quantization.input.getType());
  SmallVector<int64_t> shape;
  for (auto [dims, size, block] :
       llvm::zip_equal(getBlockedReassociation(quantization),
                       inputType.getShape(), quantization.blockSize)) {
    if (dims.size() == 1) {
      shape.push_back(size);
    } else {
      shape.append({size / block, block});
    }
  }
  return inputType.clone(shape);
}

// Returns the dimensions of the blocked type that select a block and so index
// the scale and zero point: none for per-tensor quantization, the channel
// dimension for per-channel quantization, the row and group for
// per-channel-group quantization, every dimension but the last for per-token
// quantization, and every dimension with several blocks for torchao
// quantization.
static SmallVector<unsigned>
getParameterDims(const PT2EQuantization &quantization) {
  ArrayRef<int64_t> inputShape =
      cast<RankedTensorType>(quantization.input.getType()).getShape();
  SmallVector<unsigned> parameterDims;
  for (auto [dims, size, block] :
       llvm::zip_equal(getBlockedReassociation(quantization), inputShape,
                       quantization.blockSize)) {
    if (block != size) {
      parameterDims.push_back(dims.front());
    }
  }
  return parameterDims;
}

// Checks whether LinalgExt can use the scale and zero point, reporting why
// unsupported parameters prevent conversion. Per-tensor scalars need no
// additional checks. `realType` is the element type of the floating-point
// input to quantize or output from dequantize.
static LogicalResult
checkScaleAndZeroPointSupport(Operation *op, Type realType,
                              const PT2EQuantization &quantization,
                              ConversionPatternRewriter &rewriter) {
  // Only per-tensor scales are scalars. The block size alone cannot identify
  // per-tensor quantization, since per-channel quantization along a dimension
  // of size one has the same block size.
  if (isa<FloatType>(quantization.scale.getType())) {
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
  int64_t rank = getBlockedType(quantization).getRank();
  AffineMap parameterMap = AffineMap::getMultiDimMapWithTargets(
      rank, getParameterDims(quantization), builder.getContext());
  AffineMap identity = builder.getMultiDimIdentityMap(rank);
  SmallVector<AffineMap> maps = {identity, parameterMap};
  if (quantization.zeroPoint) {
    maps.push_back(parameterMap);
  }
  maps.push_back(identity);
  return builder.getAffineMapArrayAttr(maps);
}

// Returns an empty tensor shaped like `input`, with the element type of
// `resultType`.
static Value createInit(OpBuilder &builder, Location location, Value input,
                        RankedTensorType resultType) {
  return tensor::EmptyOp::create(
      builder, location, tensor::getMixedSizes(builder, location, input),
      resultType.getElementType());
}

// Returns `quantization.input`, expanded into the blocked type.
static Value createBlockedInput(OpBuilder &builder, Location location,
                                const PT2EQuantization &quantization) {
  RankedTensorType blockedType = getBlockedType(quantization);
  if (blockedType == quantization.input.getType()) {
    return quantization.input;
  }
  return tensor::ExpandShapeOp::create(builder, location, blockedType,
                                       quantization.input,
                                       getBlockedReassociation(quantization));
}

// Collapses a `result` with the blocked shape back into the input shape.
static Value collapseBlockedResult(OpBuilder &builder, Location location,
                                   Value result,
                                   const PT2EQuantization &quantization) {
  if (getBlockedType(quantization) == quantization.input.getType()) {
    return result;
  }
  return tensor::CollapseShapeOp::create(builder, location, result,
                                         getBlockedReassociation(quantization));
}

// Drops the static unit dimensions of a tensor `parameter`, since LinalgExt
// does not index them, and returns scalars and an absent zero point unchanged.
static Value collapseParameter(OpBuilder &builder, Location location,
                               Value parameter) {
  auto parameterType =
      parameter ? dyn_cast<RankedTensorType>(parameter.getType()) : nullptr;
  if (!parameterType) {
    return parameter;
  }
  llvm::SmallBitVector unitDims(parameterType.getRank());
  for (auto [dim, size] : llvm::enumerate(parameterType.getShape())) {
    unitDims[dim] = size == 1;
  }
  if (unitDims.none()) {
    return parameter;
  }
  return tensor::dropGivenUnitDims(builder, location, parameter, unitDims);
}

static Value createQuantizeAffine(OpBuilder &builder, Location location,
                                  RankedTensorType resultType,
                                  const PT2EQuantization &quantization,
                                  bool storageUnsigned) {
  // The PT2E reference upcasts f16 and bf16 inputs and computes in f32. It
  // multiplies by `1.0 / scale` rounded to f32, whereas LinalgExt divides by
  // the scale, so results can differ by one ulp before rounding. The
  // per-channel-group and per-token references also add the zero point before
  // rounding, so ties with odd zero points differ. Torch-to-Linalg follows only
  // the per-token one.
  Value input = createBlockedInput(builder, location, quantization);
  Value scale = truncateScalarScaleToF32(builder, location, quantization.scale);
  scale = collapseParameter(builder, location, scale);
  Value zeroPoint =
      collapseParameter(builder, location, quantization.zeroPoint);
  Value init = createInit(builder, location, input, resultType);
  auto quantize = IREE::LinalgExt::QuantizeAffineOp::create(
      builder, location, init.getType(), input, scale, zeroPoint, init,
      getIndexingMaps(builder, quantization),
      builder.getI64IntegerAttr(quantization.quantMin),
      builder.getI64IntegerAttr(quantization.quantMax),
      storageUnsigned ? builder.getUnitAttr() : nullptr,
      quantization.zeroPointUnsigned ? builder.getUnitAttr() : nullptr);
  return collapseBlockedResult(builder, location, quantize->getResult(0),
                               quantization);
}

static Value createDequantizeAffine(OpBuilder &builder, Location location,
                                    RankedTensorType resultType,
                                    const PT2EQuantization &quantization,
                                    bool inputUnsigned) {
  Value input = createBlockedInput(builder, location, quantization);
  Value scale = truncateScalarScaleToF32(builder, location, quantization.scale);
  scale = collapseParameter(builder, location, scale);
  Value zeroPoint =
      collapseParameter(builder, location, quantization.zeroPoint);
  Value init = createInit(builder, location, input, resultType);
  auto dequantize = IREE::LinalgExt::DequantizeAffineOp::create(
      builder, location, init.getType(), input, scale, zeroPoint, init,
      getIndexingMaps(builder, quantization),
      builder.getI64IntegerAttr(quantization.quantMin),
      builder.getI64IntegerAttr(quantization.quantMax),
      inputUnsigned ? builder.getUnitAttr() : nullptr,
      quantization.zeroPointUnsigned ? builder.getUnitAttr() : nullptr);
  return collapseBlockedResult(builder, location, dequantize->getResult(0),
                               quantization);
}

namespace {

// Converts `quantize_per_tensor`, `quantize_per_channel`,
// `quantize_per_channel_group`, `quantize_per_token` or torchao
// `quantize_affine`, whose input holds the real values and whose result holds
// the quantized storage.
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

// Converts `dequantize_per_tensor`, `dequantize_per_channel`,
// `dequantize_per_channel_group`, `dequantize_per_token` or torchao
// `dequantize_affine`, whose input holds the quantized storage and whose result
// holds the real values.
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
        ConvertQuantize<Torch::QuantizedDecomposedQuantizePerTokenOp>,
        ConvertQuantize<Torch::TorchaoQuantizeAffineOp>,
        ConvertDequantize<Torch::QuantizedDecomposedDequantizePerTensorOp>,
        ConvertDequantize<Torch::QuantizedDecomposedDequantizePerChannelOp>,
        ConvertDequantize<
            Torch::QuantizedDecomposedDequantizePerChannelGroupOp>,
        ConvertDequantize<Torch::QuantizedDecomposedDequantizePerTokenOp>,
        ConvertDequantize<Torch::TorchaoDequantizeAffineOp>>(typeConverter,
                                                             context);
    // These ops are deliberately not marked illegal: ops whose metadata
    // LinalgExt cannot represent remain for the Torch-to-Linalg lowering.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::TorchInput
