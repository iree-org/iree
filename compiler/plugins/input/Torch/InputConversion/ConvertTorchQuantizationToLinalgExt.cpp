// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_CONVERTTORCHQUANTIZATIONTOLINALGEXTPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace Torch = torch::Torch;
namespace TorchConversion = torch::TorchConversion;

namespace {

static RankedTensorType getBuiltinTensorType(Value value) {
  auto torchType = dyn_cast<Torch::ValueTensorType>(value.getType());
  if (!torchType || !torchType.hasSizes() || !torchType.hasDtype()) {
    return nullptr;
  }
  auto builtinType = dyn_cast<RankedTensorType>(torchType.toBuiltinTensor());
  if (!builtinType) {
    return nullptr;
  }
  auto integerType = dyn_cast<IntegerType>(builtinType.getElementType());
  if (!integerType || integerType.isSignless()) {
    return builtinType;
  }
  return cast<RankedTensorType>(builtinType.clone(
      IntegerType::get(value.getContext(), integerType.getWidth())));
}

static bool hasUnsignedElementType(Value value) {
  auto torchType = dyn_cast<Torch::ValueTensorType>(value.getType());
  auto integerType =
      torchType ? dyn_cast_or_null<IntegerType>(torchType.getDtype()) : nullptr;
  return integerType && integerType.isUnsigned();
}

template <bool IsQuantize, typename TorchOp>
static FailureOr<std::pair<RankedTensorType, RankedTensorType>>
getQuantizationTensorTypes(TorchOp op, PatternRewriter &rewriter) {
  RankedTensorType inputType = getBuiltinTensorType(op.getInput());
  RankedTensorType resultType = getBuiltinTensorType(op.getResult());
  if (!inputType || !resultType) {
    return rewriter.notifyMatchFailure(
        op, "expected ranked input and result types with known dtypes");
  }
  if (inputType.getShape() != resultType.getShape()) {
    return rewriter.notifyMatchFailure(
        op, "expected input and result shapes to match");
  }

  Type realType =
      IsQuantize ? inputType.getElementType() : resultType.getElementType();
  Type storageType =
      IsQuantize ? resultType.getElementType() : inputType.getElementType();
  if (!isa<FloatType>(realType) || !storageType.isSignlessInteger()) {
    return rewriter.notifyMatchFailure(
        op, "expected a floating-point real type and an integer storage type");
  }
  return std::pair(inputType, resultType);
}

static FailureOr<std::pair<int64_t, int64_t>>
getQuantizationBounds(Value minimum, Value maximum) {
  int64_t minimumValue;
  int64_t maximumValue;
  if (!matchPattern(minimum, Torch::m_TorchConstantInt(&minimumValue)) ||
      !matchPattern(maximum, Torch::m_TorchConstantInt(&maximumValue))) {
    return failure();
  }
  return std::pair(minimumValue, maximumValue);
}

static Value convertTensor(PatternRewriter &rewriter, Location location,
                           Value value, RankedTensorType type) {
  return TorchConversion::ToBuiltinTensorOp::create(rewriter, location, type,
                                                    value);
}

static Value convertScale(PatternRewriter &rewriter, Location location,
                          Value scale) {
  Value scaleF64 = TorchConversion::ToF64Op::create(rewriter, location, scale);
  return convertScalarToDtype(rewriter, location, scaleF64,
                              rewriter.getF32Type(),
                              /*isUnsignedCast=*/false);
}

template <bool IsQuantize, typename TorchOp>
static void replaceWithAffineQuantization(
    TorchOp op, RankedTensorType resultType, Value input, Value scale,
    Value zeroPoint, AffineMap parameterMap, std::pair<int64_t, int64_t> bounds,
    bool storageUnsigned, bool zeroPointUnsigned, PatternRewriter &rewriter) {
  Location location = op.getLoc();
  Value init = tensor::EmptyOp::create(
      rewriter, location, tensor::getMixedSizes(rewriter, location, input),
      resultType.getElementType());
  AffineMap identity = rewriter.getMultiDimIdentityMap(resultType.getRank());
  SmallVector<AffineMap> indexingMaps{identity, parameterMap};
  if (zeroPoint) {
    indexingMaps.push_back(parameterMap);
  }
  indexingMaps.push_back(identity);

  Value result;
  auto indexingMapsAttr = rewriter.getAffineMapArrayAttr(indexingMaps);
  auto minimumAttr = rewriter.getI64IntegerAttr(bounds.first);
  auto maximumAttr = rewriter.getI64IntegerAttr(bounds.second);
  UnitAttr storageUnsignedAttr =
      storageUnsigned ? rewriter.getUnitAttr() : nullptr;
  UnitAttr zeroPointUnsignedAttr =
      zeroPointUnsigned ? rewriter.getUnitAttr() : nullptr;
  if constexpr (IsQuantize) {
    result = IREE::LinalgExt::QuantizeAffineOp::create(
                 rewriter, location, resultType, input, scale, zeroPoint, init,
                 indexingMapsAttr, minimumAttr, maximumAttr,
                 storageUnsignedAttr, zeroPointUnsignedAttr)
                 ->getResult(0);
  } else {
    result = IREE::LinalgExt::DequantizeAffineOp::create(
                 rewriter, location, resultType, input, scale, zeroPoint, init,
                 indexingMapsAttr, minimumAttr, maximumAttr,
                 storageUnsignedAttr, zeroPointUnsignedAttr)
                 ->getResult(0);
  }
  rewriter.replaceOpWithNewOp<TorchConversion::FromBuiltinTensorOp>(
      op, op.getResult().getType(), result);
}

template <typename TorchOp, bool IsQuantize>
struct ConvertPerTensorQuantization : OpRewritePattern<TorchOp> {
  using OpRewritePattern<TorchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TorchOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<std::pair<RankedTensorType, RankedTensorType>> tensorTypes =
        getQuantizationTensorTypes<IsQuantize>(op, rewriter);
    if (failed(tensorTypes)) {
      return failure();
    }
    auto [inputType, resultType] = *tensorTypes;
    FailureOr<std::pair<int64_t, int64_t>> bounds =
        getQuantizationBounds(op.getQuantMin(), op.getQuantMax());
    if (failed(bounds)) {
      return rewriter.notifyMatchFailure(
          op, "expected constant quantization bounds");
    }

    Location location = op.getLoc();
    Value input = convertTensor(rewriter, location, op.getInput(), inputType);
    Value scale = convertScale(rewriter, location, op.getScale());
    Value zeroPoint =
        TorchConversion::ToI64Op::create(rewriter, location, op.getZeroPoint());
    AffineMap scalarMap =
        AffineMap::get(inputType.getRank(), 0, rewriter.getContext());
    bool storageUnsigned =
        hasUnsignedElementType(IsQuantize ? op.getResult() : op.getInput());
    replaceWithAffineQuantization<IsQuantize>(
        op, resultType, input, scale, zeroPoint, scalarMap, *bounds,
        storageUnsigned, /*zeroPointUnsigned=*/false, rewriter);
    return success();
  }
};

template <typename TorchOp, bool IsQuantize>
struct ConvertPerChannelQuantization : OpRewritePattern<TorchOp> {
  using OpRewritePattern<TorchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TorchOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<std::pair<RankedTensorType, RankedTensorType>> tensorTypes =
        getQuantizationTensorTypes<IsQuantize>(op, rewriter);
    if (failed(tensorTypes)) {
      return failure();
    }
    auto [inputType, resultType] = *tensorTypes;
    RankedTensorType scaleType = getBuiltinTensorType(op.getScales());
    if (!scaleType || scaleType.getRank() != 1 ||
        !isa<FloatType>(scaleType.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "expected rank-one floating-point scales");
    }

    int64_t axis;
    if (!matchPattern(op.getAxis(), Torch::m_TorchConstantInt(&axis))) {
      return rewriter.notifyMatchFailure(op, "expected a constant axis");
    }
    axis += axis < 0 ? inputType.getRank() : 0;
    if (axis < 0 || axis >= inputType.getRank()) {
      return rewriter.notifyMatchFailure(op, "axis is out of range");
    }
    FailureOr<std::pair<int64_t, int64_t>> bounds =
        getQuantizationBounds(op.getQuantMin(), op.getQuantMax());
    if (failed(bounds)) {
      return rewriter.notifyMatchFailure(
          op, "expected constant quantization bounds");
    }

    RankedTensorType zeroPointType;
    bool zeroPointUnsigned = false;
    if (!isa<Torch::NoneType>(op.getZeroPoints().getType())) {
      zeroPointType = getBuiltinTensorType(op.getZeroPoints());
      if (!zeroPointType || zeroPointType.getRank() != 1 ||
          !zeroPointType.getElementType().isSignlessInteger()) {
        return rewriter.notifyMatchFailure(
            op, "expected rank-one integer zero points");
      }
      zeroPointUnsigned = hasUnsignedElementType(op.getZeroPoints());
    }

    int64_t channelSize = inputType.getDimSize(axis);
    int64_t scaleSize = scaleType.getDimSize(0);
    if (!ShapedType::isDynamic(channelSize) &&
        !ShapedType::isDynamic(scaleSize) && channelSize != scaleSize) {
      return rewriter.notifyMatchFailure(
          op, "scale length does not match the channel dimension");
    }
    if (zeroPointType) {
      int64_t zeroPointSize = zeroPointType.getDimSize(0);
      if (!ShapedType::isDynamic(zeroPointSize) &&
          !ShapedType::isDynamic(scaleSize) && zeroPointSize != scaleSize) {
        return rewriter.notifyMatchFailure(
            op, "zero-point and scale lengths do not match");
      }
    }

    Location location = op.getLoc();
    Value input = convertTensor(rewriter, location, op.getInput(), inputType);
    Value scale = convertTensor(rewriter, location, op.getScales(), scaleType);
    Value zeroPoint;
    if (zeroPointType) {
      zeroPoint =
          convertTensor(rewriter, location, op.getZeroPoints(), zeroPointType);
    }

    AffineMap channelMap =
        AffineMap::get(inputType.getRank(), 0, rewriter.getAffineDimExpr(axis));
    bool storageUnsigned =
        hasUnsignedElementType(IsQuantize ? op.getResult() : op.getInput());
    replaceWithAffineQuantization<IsQuantize>(
        op, resultType, input, scale, zeroPoint, channelMap, *bounds,
        storageUnsigned, zeroPointUnsigned, rewriter);
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
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertPerTensorQuantization<
                     Torch::QuantizedDecomposedQuantizePerTensorOp, true>,
                 ConvertPerTensorQuantization<
                     Torch::QuantizedDecomposedDequantizePerTensorOp, false>,
                 ConvertPerChannelQuantization<
                     Torch::QuantizedDecomposedQuantizePerChannelOp, true>,
                 ConvertPerChannelQuantization<
                     Torch::QuantizedDecomposedDequantizePerChannelOp, false>>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::TorchInput
