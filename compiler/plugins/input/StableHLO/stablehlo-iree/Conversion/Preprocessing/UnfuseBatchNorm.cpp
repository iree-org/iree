// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h"
#include "stablehlo-iree/Conversion/Preprocessing/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_UNFUSEBATCHNORM
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h.inc"

namespace {
// Broadcasts the 1D value tensor 'value_1d' to the shape of 'result_type'. If
// 'shape_value' is initialized, creates a dynamic broadcast, otherwise creates
// a static broadcast.
Value broadcastToFeatureDim(Location loc, RankedTensorType resultType,
                            Value value1d, Value shapeValue, int64_t featureDim,
                            PatternRewriter &rewriter) {
  auto dimsType = RankedTensorType::get({1}, rewriter.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dimsType, {featureDim});
  if (shapeValue) {
    return rewriter.createOrFold<mlir::stablehlo::DynamicBroadcastInDimOp>(
        loc, resultType, value1d, shapeValue, dims);
  }
  assert(resultType.hasStaticShape());
  return rewriter.create<mlir::stablehlo::BroadcastInDimOp>(loc, resultType,
                                                            value1d, dims);
}

// Get the shape of operand, assuming it is a dynamic shape with static rank.
Value getShapeValue(Location loc, Value operand, PatternRewriter &rewriter) {
  RankedTensorType resultType = cast<RankedTensorType>(operand.getType());
  return rewriter.create<mlir::shape::ShapeOfOp>(
      loc,
      RankedTensorType::get({resultType.getRank()}, rewriter.getIndexType()),
      operand);
}

Value materializeEpsilon(Operation *op, FloatAttr epsilonAttr, FloatType fpType,
                         Value broadcastTo, RankedTensorType broadcastToType,
                         PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  if (epsilonAttr.getType() != fpType) {
    // Need to convert.
    bool losesInfo;
    APFloat epsilonFloat = epsilonAttr.getValue();
    auto status = epsilonFloat.convert(
        fpType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
    if ((status & (~APFloat::opInexact)) != APFloat::opOK) {
      op->emitWarning() << "Could not convert batch_norm epsilon to target fp "
                           "type: opStatus = "
                        << static_cast<int>(status);
      return nullptr;
    }
    if (losesInfo) {
      op->emitWarning("Conversion of epsilon loses precision");
    }
    epsilonAttr = b.getFloatAttr(fpType, epsilonFloat);
  }

  auto scalarType = RankedTensorType::get({}, fpType);
  auto epsilonTensorAttr =
      DenseElementsAttr::get(scalarType, {cast<Attribute>(epsilonAttr)});
  Value epsilon = b.create<mlir::stablehlo::ConstantOp>(epsilonTensorAttr);
  auto dimsType = RankedTensorType::get({0}, b.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
  if (broadcastToType.hasStaticShape()) {
    return b.create<mlir::stablehlo::BroadcastInDimOp>(broadcastToType, epsilon,
                                                       /*broadcast_dims=*/dims);
  }
  Value shapeValue = getShapeValue(op->getLoc(), broadcastTo, rewriter);
  return b.createOrFold<mlir::stablehlo::DynamicBroadcastInDimOp>(
      broadcastToType, epsilon, shapeValue,
      /*broadcast_dims=*/dims);
}

struct UnfuseBatchNormInferencePattern final
    : OpRewritePattern<mlir::stablehlo::BatchNormInferenceOp> {
  using OpRewritePattern ::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BatchNormInferenceOp bnOp,
                                PatternRewriter &rewriter) const override {
    // Enforce type invariants.
    // Note that we deduce the actual element type from the variance,
    // which should not be subject to quantization at a higher level.
    auto inputType = dyn_cast<RankedTensorType>(bnOp.getOperand().getType());
    auto varianceType =
        llvm::dyn_cast<RankedTensorType>(bnOp.getVariance().getType());
    if (!inputType || !varianceType) {
      return failure();
    }
    auto fpType = dyn_cast<FloatType>(varianceType.getElementType());
    if (!fpType) {
      return failure();
    }
    int64_t featureDim = bnOp.getFeatureIndex();

    // Add epsilon to the variance and sqrt to get stddev:
    // stddev = sqrt(variance + epsilon)
    auto epsilon =
        materializeEpsilon(bnOp.getOperation(), bnOp.getEpsilonAttr(), fpType,
                           bnOp.getVariance(), varianceType, rewriter);
    if (!epsilon) {
      return failure();
    }
    Value stddev = rewriter.create<mlir::stablehlo::AddOp>(
        bnOp.getLoc(), bnOp.getVariance(), epsilon);
    stddev = rewriter.create<mlir::stablehlo::SqrtOp>(bnOp.getLoc(), stddev);

    // Broadcast all terms.
    Value shapeValue;
    if (!inputType.hasStaticShape()) {
      shapeValue = getShapeValue(bnOp.getLoc(), bnOp.getOperand(), rewriter);
    }
    auto broadcastScale =
        broadcastToFeatureDim(bnOp.getLoc(), inputType, bnOp.getScale(),
                              shapeValue, featureDim, rewriter);
    auto broadcastOffset =
        broadcastToFeatureDim(bnOp.getLoc(), inputType, bnOp.getOffset(),
                              shapeValue, featureDim, rewriter);
    auto broadcastMean =
        broadcastToFeatureDim(bnOp.getLoc(), inputType, bnOp.getMean(),
                              shapeValue, featureDim, rewriter);
    auto broadcastStddev = broadcastToFeatureDim(
        bnOp.getLoc(), inputType, stddev, shapeValue, featureDim, rewriter);

    // Compute:
    // scale * (input - mean) / stddev + offset
    Value result = rewriter.create<mlir::stablehlo::SubtractOp>(
        bnOp.getLoc(), bnOp.getOperand(), broadcastMean);
    result = rewriter.create<mlir::stablehlo::MulOp>(bnOp.getLoc(), result,
                                                     broadcastScale);
    result = rewriter.create<mlir::stablehlo::DivOp>(bnOp.getLoc(), result,
                                                     broadcastStddev);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::AddOp>(bnOp, result,
                                                        broadcastOffset);

    return success();
  }
};

// Create "stablehlo.reduce", "operand" is reduce input and "zero" is init
// value, reduce sum from operand to operand[feature_index].
Value createReduce(Location loc, Value operand, Value zero,
                   SmallVector<int64_t> &reduceDims, int64_t featureIndex,
                   PatternRewriter &rewriter) {
  auto operandType = cast<RankedTensorType>(operand.getType());
  auto reduceResultType = RankedTensorType::get(
      {operandType.getDimSize(featureIndex)}, operandType.getElementType());
  auto reduce = rewriter.create<mlir::stablehlo::ReduceOp>(
      loc, reduceResultType, operand, zero,
      rewriter.getI64TensorAttr(reduceDims));

  // setup "stablehlo.reduce"'s body
  Region &region = reduce.getBody();
  Block &block = region.emplaceBlock();
  RankedTensorType blockArgumentType =
      RankedTensorType::get({}, operandType.getElementType());
  block.addArgument(blockArgumentType, loc);
  block.addArgument(blockArgumentType, loc);
  auto firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value addResult = rewriter.create<mlir::stablehlo::AddOp>(
        loc, *firstArgument, *secondArgument);
    rewriter.create<mlir::stablehlo::ReturnOp>(loc, addResult);
  }

  return reduce.getResult(0);
}

// Calculate total reduce size, assuming it is a dynamic shape with static rank.
// Reduce from operand to operand[feature_index]/scale
Value calculateReduceSize(Operation *op, Value operand,
                          RankedTensorType operandType, Value scale,
                          RankedTensorType scaleType, int64_t featureIndex,
                          PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Type indexType = b.getIndexType();
  if (!operandType.hasStaticShape()) {
    // the "operand" has dynamic shape with static rank
    Value operandShape = getShapeValue(op->getLoc(), operand, rewriter);
    Value scaleShape = getShapeValue(op->getLoc(), scale, rewriter);
    Value operandTotalSize =
        b.create<shape::NumElementsOp>(indexType, operandShape);
    Value scaleTotalSize =
        b.create<shape::NumElementsOp>(indexType, scaleShape);
    Value reduceSize =
        b.create<shape::DivOp>(indexType, operandTotalSize, scaleTotalSize);
    reduceSize = b.create<arith::IndexCastOp>(b.getI64Type(), reduceSize);
    reduceSize = b.create<tensor::FromElementsOp>(reduceSize);
    reduceSize = b.create<mlir::stablehlo::ConvertOp>(
        RankedTensorType::get({1}, operandType.getElementType()), reduceSize);
    reduceSize = b.create<mlir::stablehlo::ReshapeOp>(
        RankedTensorType::get({}, operandType.getElementType()), reduceSize);
    return b.createOrFold<mlir::stablehlo::DynamicBroadcastInDimOp>(
        scaleType, reduceSize, scaleShape, b.getI64TensorAttr({}));
  }

  // the "operand" has static shape
  int64_t reduceDimsSize = 1;
  for (int64_t i = 0, e = operandType.getRank(); i < e; i++) {
    if (i != featureIndex) {
      reduceDimsSize *= operandType.getDimSize(i);
    }
  }
  llvm::APFloat floatValue(static_cast<double>(reduceDimsSize));
  bool losesInfo;
  floatValue.convert(
      cast<FloatType>(scaleType.getElementType()).getFloatSemantics(),
      APFloat::rmNearestTiesToEven, &losesInfo);
  if (losesInfo) {
    op->emitWarning("Conversion of reduce_dims_size loses precision");
  }
  Value reduceSize = b.create<mlir::stablehlo::ConstantOp>(
      DenseFPElementsAttr::get(scaleType, floatValue));
  return reduceSize;
}

// BatchNormTraining(X, scale, offset) =
//    ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale + offset.
struct UnfuseBatchNormTrainingPattern final
    : OpRewritePattern<mlir::stablehlo::BatchNormTrainingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BatchNormTrainingOp bnOp,
                                PatternRewriter &rewriter) const override {
    auto operandType = dyn_cast<RankedTensorType>(bnOp.getOperand().getType());
    auto scaleType = dyn_cast<RankedTensorType>(bnOp.getScale().getType());
    if (!operandType || !scaleType) {
      return failure();
    }
    auto fpType = dyn_cast<FloatType>(operandType.getElementType());
    if (!fpType) {
      return failure();
    }
    int64_t featureIndex = bnOp.getFeatureIndex();
    SmallVector<int64_t> dimensionsWithoutFeature;
    for (int64_t i = 0, e = operandType.getRank(); i < e; i++) {
      if (i != featureIndex) {
        dimensionsWithoutFeature.push_back(i);
      }
    }

    // zero constant
    Value constZero = rewriter.create<mlir::stablehlo::ConstantOp>(
        bnOp.getLoc(),
        DenseFPElementsAttr::get(RankedTensorType::get({}, fpType),
                                 APFloat::getZero(fpType.getFloatSemantics())));
    // epsilon
    auto epsilon =
        materializeEpsilon(bnOp.getOperation(), bnOp.getEpsilonAttr(), fpType,
                           bnOp.getScale(), scaleType, rewriter);
    if (!epsilon) {
      return failure();
    }
    // reduce size constant
    Value reduceSize =
        calculateReduceSize(bnOp.getOperation(), bnOp.getOperand(), operandType,
                            bnOp.getScale(), scaleType, featureIndex, rewriter);
    if (!reduceSize) {
      return failure();
    }
    // Sum[X]
    Value sum = createReduce(bnOp.getLoc(), bnOp.getOperand(), constZero,
                             dimensionsWithoutFeature, featureIndex, rewriter);
    // X^2
    Value operandSquare = rewriter.create<mlir::stablehlo::MulOp>(
        bnOp.getLoc(), bnOp.getOperand(), bnOp.getOperand());
    // Sum[X^2]
    Value squareSum =
        createReduce(bnOp.getLoc(), operandSquare, constZero,
                     dimensionsWithoutFeature, featureIndex, rewriter);
    // E[X]
    Value mean =
        rewriter.create<mlir::stablehlo::DivOp>(bnOp.getLoc(), sum, reduceSize);
    // E[X^2]
    Value squareMean = rewriter.create<mlir::stablehlo::DivOp>(
        bnOp.getLoc(), squareSum, reduceSize);
    // E^2[X]
    Value meanSquare =
        rewriter.create<mlir::stablehlo::MulOp>(bnOp.getLoc(), mean, mean);
    // Var[X]
    Value var = rewriter.create<mlir::stablehlo::SubtractOp>(
        bnOp.getLoc(), squareMean, meanSquare);
    // Var[X] + epsilon
    Value varAddEpsilon =
        rewriter.create<mlir::stablehlo::AddOp>(bnOp.getLoc(), var, epsilon);
    // Sqrt(Var[X] + epsilon)
    Value sqrtVar =
        rewriter.create<mlir::stablehlo::SqrtOp>(bnOp.getLoc(), varAddEpsilon);

    Value shapeValue;
    if (!operandType.hasStaticShape()) {
      shapeValue = getShapeValue(bnOp.getLoc(), bnOp.getOperand(), rewriter);
    }
    // X - E[X]
    Value meanBroadcast = broadcastToFeatureDim(
        bnOp.getLoc(), operandType, mean, shapeValue, featureIndex, rewriter);
    Value operandMinusMean = rewriter.create<mlir::stablehlo::SubtractOp>(
        bnOp.getLoc(), bnOp.getOperand(), meanBroadcast);
    // (X - E[X]) / Sqrt(Var[X] + epsilon)
    Value sqrtVarBroadcast =
        broadcastToFeatureDim(bnOp.getLoc(), operandType, sqrtVar, shapeValue,
                              featureIndex, rewriter);
    Value normalized = rewriter.create<mlir::stablehlo::DivOp>(
        bnOp.getLoc(), operandMinusMean, sqrtVarBroadcast);

    // ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale
    Value scaleBroadcast =
        broadcastToFeatureDim(bnOp.getLoc(), operandType, bnOp.getScale(),
                              shapeValue, featureIndex, rewriter);
    Value scaledNormalized = rewriter.create<mlir::stablehlo::MulOp>(
        bnOp.getLoc(), normalized, scaleBroadcast);
    // ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale + offset.
    Value offsetBroadcast =
        broadcastToFeatureDim(bnOp.getLoc(), operandType, bnOp.getOffset(),
                              shapeValue, featureIndex, rewriter);
    Value shiftedNormalized = rewriter.create<mlir::stablehlo::AddOp>(
        bnOp.getLoc(), scaledNormalized, offsetBroadcast);

    // results
    SmallVector<Value> results = {shiftedNormalized, mean, var};
    rewriter.replaceOp(bnOp, results);

    return success();
  }
};

struct UnfuseBatchNorm final : impl::UnfuseBatchNormBase<UnfuseBatchNorm> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, shape::ShapeDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePreprocessingUnfuseBatchNormPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

void populatePreprocessingUnfuseBatchNormPatterns(mlir::MLIRContext *context,
                                                  RewritePatternSet *patterns) {
  patterns
      ->add<UnfuseBatchNormInferencePattern, UnfuseBatchNormTrainingPattern>(
          context);
}

} // namespace mlir::iree_compiler::stablehlo
