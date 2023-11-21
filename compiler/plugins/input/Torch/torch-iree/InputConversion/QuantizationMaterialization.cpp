//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include <cstdint>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static const StringLiteral subZeroPointMarker = "__quant_zeropoint__";
static const StringLiteral mulScaleMarker = "__quant_scale__";
static const StringLiteral quantizedMarker = "__quantized__";

static inline torch_upstream::ScalarType
materializeQType(torch_upstream::ScalarType t) {
  if (t == torch_upstream::ScalarType::QInt8)
    return torch_upstream::ScalarType::Char;
  if (t == torch_upstream::ScalarType::QUInt8)
    return torch_upstream::ScalarType::Byte;
  if (t == torch_upstream::ScalarType::QInt32)
    return torch_upstream::ScalarType::Int;
  return t;
}

static void transposeOutputChannels(PatternRewriter &rewriter, Location loc,
                                    Operation *target, Value other) {
  Value zero =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
  Value one =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  AtenTransposeIntOp transposed = rewriter.create<AtenTransposeIntOp>(
      loc,
      Torch::ValueTensorType::getWithLeastStaticInformation(
          target->getContext()),
      other, zero, one);
  transposed->moveBefore(target);
  target->replaceUsesOfWith(other, transposed.getResult());
}

static LogicalResult commuteTensorOperand(PatternRewriter &rewriter,
                                          Operation *op, Value tensorOperand,
                                          bool annotate) {
  AtenMulTensorOp mulScale;
  if (!(mulScale = tensorOperand.getDefiningOp<AtenMulTensorOp>()) ||
      !mulScale->hasAttr(mulScaleMarker)) {
    return rewriter.notifyMatchFailure(
        op, "op input isn't from dequantization multiply");
  }

  AtenSubTensorOp subZeroPoint;
  if (!(subZeroPoint = mulScale.getSelf().getDefiningOp<AtenSubTensorOp>()) ||
      !subZeroPoint->hasAttr(subZeroPointMarker)) {
    return rewriter.notifyMatchFailure(
        op, "dequantization scaling doesn't come from sub zero point ");
  }

  auto result = op->getResult(0);
  auto loc = op->getLoc();
  auto newSub = rewriter.create<AtenSubTensorOp>(
      loc, subZeroPoint.getType(), result, subZeroPoint.getOther(),
      subZeroPoint.getAlpha());
  auto newMul = rewriter.create<AtenMulTensorOp>(
      loc, mulScale.getType(), newSub.getResult(), mulScale.getOther());
  if (annotate) {
    newMul->setAttr(mulScaleMarker, rewriter.getUnitAttr());
    newSub->setAttr(subZeroPointMarker, rewriter.getUnitAttr());
    op->setAttr(quantizedMarker, rewriter.getUnitAttr());
  }

  op->replaceUsesOfWith(mulScale.getResult(), subZeroPoint.getSelf());
  result.replaceAllUsesExcept(newMul.getResult(), newSub);
  newSub->moveAfter(op);
  newMul->moveAfter(newSub);
  return success();
}

static LogicalResult passThroughTensorOperand(PatternRewriter &rewriter,
                                              Operation *op,
                                              Value tensorOperand) {
  AtenMulTensorOp mulScale;
  if (!(mulScale = tensorOperand.getDefiningOp<AtenMulTensorOp>()) ||
      !mulScale->hasAttr(mulScaleMarker)) {
    return rewriter.notifyMatchFailure(
        op, "op input isn't from dequantization multiply");
  }

  AtenSubTensorOp subZeroPoint;
  if (!(subZeroPoint = mulScale.getSelf().getDefiningOp<AtenSubTensorOp>()) ||
      !subZeroPoint->hasAttr(subZeroPointMarker)) {
    return rewriter.notifyMatchFailure(
        op, "dequantization scaling doesn't come from sub zero point ");
  }

  op->replaceUsesOfWith(mulScale.getResult(), subZeroPoint.getSelf());
  return success();
}

// =====================================
// Materialize Integer Types
// =====================================

namespace {
class MaterializeQuantizePerTensorOp
    : public OpRewritePattern<AtenQuantizePerTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenQuantizePerTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto result = op.getResult();
    if (!result.hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "quantize per tensor op has multiple uses");

    AtenIntReprOp reprOp;
    if (!(reprOp = dyn_cast<AtenIntReprOp>(*result.getUsers().begin())))
      return rewriter.notifyMatchFailure(
          op, "quantize per tensor op use must be aten.int_repr");

    int64_t dtypeInt;
    if (!matchPattern(op.getDtype(), m_TorchConstantInt(&dtypeInt))) {
      return failure();
    }

    // Annotate dequantization
    AtenSubTensorOp subZeroPoint;
    if (!(subZeroPoint = dyn_cast<AtenSubTensorOp>(
              *reprOp.getResult().getUsers().begin()))) {
      return rewriter.notifyMatchFailure(
          op, "quantize op does not have sub tensor dequantization");
    }

    AtenMulTensorOp mulScale;
    if (!(mulScale = dyn_cast<AtenMulTensorOp>(
              *subZeroPoint.getResult().getUsers().begin()))) {
      return rewriter.notifyMatchFailure(
          op, "quantize op does not have mul tensor dequantization");
    }

    subZeroPoint->setAttr(subZeroPointMarker, rewriter.getUnitAttr());
    mulScale->setAttr(mulScaleMarker, rewriter.getUnitAttr());

    auto scalarDtype = materializeQType((torch_upstream::ScalarType)dtypeInt);
    auto dtypeValue = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr((int64_t)scalarDtype));

    auto resultType = result.getType();
    auto quantized = rewriter.create<AtenQuantizePerTensorOp>(
        loc, resultType, op.getSelf(), op.getScale(), op.getZeroPoint(),
        dtypeValue);

    reprOp.getResult().replaceAllUsesWith(quantized.getResult());
    rewriter.eraseOp(reprOp);
    rewriter.replaceOp(op, quantized.getResult());
    return success();
  }
};
} // end namespace

namespace {
class MaterializeQuantizePerChannelOp
    : public OpRewritePattern<AtenQuantizePerChannelOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenQuantizePerChannelOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto result = op.getResult();
    if (!result.hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "quantize per channel op has multiple uses");

    AtenIntReprOp reprOp;
    if (!(reprOp = dyn_cast<AtenIntReprOp>(*result.getUsers().begin())))
      return rewriter.notifyMatchFailure(
          op, "quantize per channel op use must be aten.int_repr");

    int64_t dtypeInt;
    if (!matchPattern(op.getDtype(), m_TorchConstantInt(&dtypeInt))) {
      return failure();
    }

    auto scalarDtype = materializeQType((torch_upstream::ScalarType)dtypeInt);
    auto dtypeValue = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr((int64_t)scalarDtype));

    auto resultType = result.getType();
    auto quantized = rewriter.create<AtenQuantizePerChannelOp>(
        loc, resultType, op.getSelf(), op.getScales(), op.getZeroPoints(),
        op.getAxis(), dtypeValue);

    reprOp.getResult().replaceAllUsesWith(quantized.getResult());
    rewriter.eraseOp(reprOp);
    rewriter.replaceOp(op, quantized.getResult());
    return success();
  }
};
} // end namespace

// =====================================
// Commute Dequantization
// =====================================

namespace {
template <typename OpTy>
class CommuteUnaryLinearOp : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(quantizedMarker)) {
      return rewriter.notifyMatchFailure(op,
                                         "unary linear op already quantized.");
    }

    auto tensorOperands = llvm::to_vector<6>(
        llvm::make_filter_range(op->getOperands(), [](Value v) {
          return v.getType().isa<Torch::ValueTensorType>();
        }));

    assert(tensorOperands.size() == 1 && "Found non-singular tensor arguments");
    return commuteTensorOperand(rewriter, op, tensorOperands[0],
                                /*annotate=*/true);
  }
};
} // namespace

namespace {
template <typename OpTy>
class PassThroughUnaryOp : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(quantizedMarker)) {
      return rewriter.notifyMatchFailure(op,
                                         "unary linear op already quantized.");
    }

    auto tensorOperands = llvm::to_vector<6>(
        llvm::make_filter_range(op->getOperands(), [](Value v) {
          return v.getType().isa<Torch::ValueTensorType>();
        }));

    assert(tensorOperands.size() == 1 && "Found non-singular tensor arguments");
    return passThroughTensorOperand(rewriter, op, tensorOperands[0]);
  }
};
} // namespace

namespace {
class CommuteAtenCatOp : public OpRewritePattern<AtenCatOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCatOp op,
                                PatternRewriter &rewriter) const override {

    if (op->hasAttr(quantizedMarker)) {
      return rewriter.notifyMatchFailure(op, "cat op already quantized.");
    }

    auto tensorList = op.getTensors();
    SmallVector<Value> tensorVector;
    if (!getListConstructElements(tensorList, tensorVector))
      return op.emitError(
          "unimplemented: the tensor list is not from list construct");

    assert(tensorVector.size() > 0 &&
           "Expecting to have non-zero number of concatenated vectors");

    SmallVector<Value> mulScaleValues;
    SmallVector<Value> subZeroPointValues;
    SmallVector<Value> quantizedSources;
    AtenSubTensorOp subZeroPoint;
    AtenMulTensorOp mulScale;
    for (auto tensor : tensorVector) {
      if (!(mulScale = tensor.getDefiningOp<AtenMulTensorOp>()) ||
          !mulScale->hasAttr(mulScaleMarker)) {
        return rewriter.notifyMatchFailure(
            op, "aten cat input isn't from dequantization multiply");
      }
      mulScaleValues.push_back(mulScale.getOther());

      if (!(subZeroPoint =
                mulScale.getSelf().getDefiningOp<AtenSubTensorOp>()) ||
          !subZeroPoint->hasAttr(subZeroPointMarker)) {
        return rewriter.notifyMatchFailure(op,
                                           "aten cat dequantization scaling "
                                           "doesn't come from sub zero point ");
      }
      subZeroPointValues.push_back(subZeroPoint.getOther());
      quantizedSources.push_back(subZeroPoint.getSelf());
    }

    if (!std::equal(mulScaleValues.begin() + 1, mulScaleValues.end(),
                    mulScaleValues.begin()))
      return op.emitError(
          "unimplemented: concatenated tensors with non-homogenous scale");

    if (!std::equal(subZeroPointValues.begin() + 1, subZeroPointValues.end(),
                    subZeroPointValues.begin()))
      return op.emitError(
          "unimplemented: concatenated tensors with non-homogenous zero point");

    auto loc = op->getLoc();
    auto newTensorList = rewriter.create<PrimListConstructOp>(
        loc, tensorList.getType(), quantizedSources);
    auto newCatOp = rewriter.create<AtenCatOp>(loc, op.getType(), newTensorList,
                                               op.getDim());
    auto newSub = rewriter.create<AtenSubTensorOp>(
        loc, subZeroPoint.getType(), newCatOp.getResult(),
        subZeroPoint.getOther(), subZeroPoint.getAlpha());
    auto newMul = rewriter.create<AtenMulTensorOp>(
        loc, mulScale.getType(), newSub.getResult(), mulScale.getOther());

    newMul->setAttr(mulScaleMarker, rewriter.getUnitAttr());
    newSub->setAttr(subZeroPointMarker, rewriter.getUnitAttr());
    newCatOp->setAttr(quantizedMarker, rewriter.getUnitAttr());
    rewriter.replaceOp(op, newMul.getResult());

    return success();
  }
};
} // end namespace

// =====================================
// Quantize Convolutions
// =====================================

namespace {
class CommuteQuantizePerChannelOp
    : public OpRewritePattern<AtenQuantizePerChannelOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenQuantizePerChannelOp op,
                                PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    if (!result.hasOneUse())
      return rewriter.notifyMatchFailure(op, "quantize op has multiple uses");

    if (auto clampOp = dyn_cast<AtenClampOp>(*result.getUsers().begin())) {
      result = clampOp.getResult();
    }

    AtenSubTensorOp subZeroPoint;
    if (!(subZeroPoint =
              dyn_cast<AtenSubTensorOp>(*result.getUsers().begin()))) {
      return rewriter.notifyMatchFailure(
          op, "quantize op does not have sub tensor as user");
    }
    auto subResult = subZeroPoint.getResult();

    AtenMulTensorOp mulScale;
    if (!(mulScale =
              dyn_cast<AtenMulTensorOp>(*subResult.getUsers().begin()))) {
      return rewriter.notifyMatchFailure(
          op, "quantize op does not have mul tensor in chain");
    }
    auto mulResult = mulScale.getResult();

    Aten_ConvolutionOp conv;
    if (!(conv = dyn_cast<Aten_ConvolutionOp>(*mulResult.getUsers().begin()))) {
      return rewriter.notifyMatchFailure(
          op, "quantize op does not have convolution in chain");
    }

    auto convResult = conv.getResult();
    conv->replaceUsesOfWith(mulResult, result);
    convResult.replaceAllUsesWith(mulResult);
    subZeroPoint->replaceUsesOfWith(result, convResult);
    subZeroPoint->moveAfter(conv);
    mulScale->moveAfter(subZeroPoint);

    Value other;
    if (subZeroPoint.getSelf() == convResult) {
      other = subZeroPoint.getOther();
    } else {
      other = subZeroPoint.getSelf();
    }
    Location otherLoc = conv->getLoc();
    transposeOutputChannels(rewriter, otherLoc, subZeroPoint, other);

    if (mulScale.getSelf() == subResult) {
      other = mulScale.getOther();
    } else {
      other = mulScale.getSelf();
    }
    transposeOutputChannels(rewriter, otherLoc, mulScale, other);

    return success();
  }
};
} // end namespace

namespace {
class QuantizeConvolutionBias : public OpRewritePattern<Aten_ConvolutionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    auto bias = op.getBias();
    if (bias.getType().isa<Torch::NoneType>()) {
      return rewriter.notifyMatchFailure(op, "convolution does not have bias");
    }

    AtenMulTensorOp mulScale;
    if (!(mulScale = bias.getDefiningOp<AtenMulTensorOp>())) {
      return rewriter.notifyMatchFailure(
          op, "convolution bias has no scale multiply");
    }

    AtenSubTensorOp subZeroPoint;
    if (!(subZeroPoint = mulScale.getSelf().getDefiningOp<AtenSubTensorOp>()) &&
        !(subZeroPoint =
              mulScale.getOther().getDefiningOp<AtenSubTensorOp>())) {
      return rewriter.notifyMatchFailure(
          op, "convolution bias has no zero point subtract");
    }

    op->replaceUsesOfWith(bias, subZeroPoint.getSelf());
    return success();
  }
};
} // end namespace

namespace {
class QuantizeConvolutionInput : public OpRewritePattern<Aten_ConvolutionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    return commuteTensorOperand(rewriter, op, op.getInput(),
                                /*annotate=*/false);
  }
};
} // end namespace

namespace {
class MaterializeQuantizationPass
    : public MaterializeQuantizationBase<MaterializeQuantizationPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // Fold away int_repr ops and annotate dequantization ops
    {
      RewritePatternSet patterns(context);

      patterns.add<MaterializeQuantizePerTensorOp>(context);
      patterns.add<MaterializeQuantizePerChannelOp>(context);

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Bubble down the dequantization ops (ideally to convolution)
    {
      RewritePatternSet patterns(context);

      patterns.add<CommuteUnaryLinearOp<AtenMaxPool2dOp>>(context);
      patterns.add<CommuteUnaryLinearOp<AtenUpsampleNearest2dOp>>(context);
      patterns.add<CommuteUnaryLinearOp<AtenSliceTensorOp>>(context);
      patterns.add<CommuteUnaryLinearOp<AtenViewOp>>(context);
      patterns.add<CommuteUnaryLinearOp<AtenPermuteOp>>(context);
      patterns.add<CommuteUnaryLinearOp<AtenContiguousOp>>(context);
      patterns.add<PassThroughUnaryOp<AtenSizeIntOp>>(context);
      patterns.add<CommuteAtenCatOp>(context);

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Quantize convolutions and other endpoint ops
    {
      RewritePatternSet patterns(context);

      patterns.add<CommuteQuantizePerChannelOp>(context);
      patterns.add<QuantizeConvolutionInput>(context);
      patterns.add<QuantizeConvolutionBias>(context);

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createMaterializeQuantizationPass() {
  return std::make_unique<MaterializeQuantizationPass>();
}
