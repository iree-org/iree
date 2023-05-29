// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering CHLO ops to StableHLO and Shape dialect ops.

#include "iree/compiler/InputConversion/StableHLO/Passes.h"
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Rewriters.h"
#include "iree/compiler/InputConversion/StableHLO/Rewriters.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/BroadcastUtils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LEGALIZECHLO
#include "iree/compiler/InputConversion/StableHLO/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers.
//===----------------------------------------------------------------------===//

template <typename FromOpTy, typename ToOpTy>
struct HloNaryElementwiseAdaptor {
  static ToOpTy createOp(FromOpTy fromOp, Type resultType,
                         ValueRange broadcastedOperands, OpBuilder& builder) {
    return builder.create<ToOpTy>(fromOp.getLoc(), resultType,
                                  broadcastedOperands);
  }
};

static std::optional<mlir::stablehlo::ComparisonDirection>
toStableHloComparisonDirection(mlir::chlo::ComparisonDirection value) {
  switch (value) {
    case mlir::chlo::ComparisonDirection::EQ:
      return mlir::stablehlo::ComparisonDirection::EQ;
    case mlir::chlo::ComparisonDirection::NE:
      return mlir::stablehlo::ComparisonDirection::NE;
    case mlir::chlo::ComparisonDirection::GE:
      return mlir::stablehlo::ComparisonDirection::GE;
    case mlir::chlo::ComparisonDirection::GT:
      return mlir::stablehlo::ComparisonDirection::GT;
    case mlir::chlo::ComparisonDirection::LE:
      return mlir::stablehlo::ComparisonDirection::LE;
    case mlir::chlo::ComparisonDirection::LT:
      return mlir::stablehlo::ComparisonDirection::LT;
  }
  return {};
}

static std::optional<mlir::stablehlo::ComparisonType> toStableHloComparisonType(
    mlir::chlo::ComparisonType value) {
  switch (value) {
    case mlir::chlo::ComparisonType::NOTYPE:
      return mlir::stablehlo::ComparisonType::NOTYPE;
    case mlir::chlo::ComparisonType::FLOAT:
      return mlir::stablehlo::ComparisonType::FLOAT;
    case mlir::chlo::ComparisonType::TOTALORDER:
      return mlir::stablehlo::ComparisonType::TOTALORDER;
    case mlir::chlo::ComparisonType::SIGNED:
      return mlir::stablehlo::ComparisonType::SIGNED;
    case mlir::chlo::ComparisonType::UNSIGNED:
      return mlir::stablehlo::ComparisonType::UNSIGNED;
  }
  return {};
}

struct HloCompareAdaptor {
  static mlir::stablehlo::CompareOp createOp(
      mlir::chlo::BroadcastCompareOp fromOp, Type resultType,
      ValueRange broadcastedOperands, OpBuilder& builder) {
    auto chloDirection = fromOp.getComparisonDirection();
    auto mhloDirection = toStableHloComparisonDirection(chloDirection);
    if (!mhloDirection) return nullptr;
    auto chloType =
        fromOp.getCompareType().value_or(mlir::chlo::ComparisonType::NOTYPE);
    auto mhloType = toStableHloComparisonType(chloType);
    if (!mhloType) return nullptr;
    auto mhloTypeAttr = fromOp.getCompareType()
                            ? mlir::stablehlo::ComparisonTypeAttr::get(
                                  builder.getContext(), *mhloType)
                            : nullptr;
    return builder.create<mlir::stablehlo::CompareOp>(
        fromOp.getLoc(), resultType, broadcastedOperands[0],
        broadcastedOperands[1], *mhloDirection, mhloTypeAttr);
  }
};

// Populate a pattern for each Broadcasting Chlo op. This requires the pattern
// to take a ChloOpTy, NonBroadcastingOpTy, and an Adaptor as templated values.
template <template <typename, typename, typename> typename Pattern,
          typename... ConstructorArgs>
static void populateForBroadcastingBinaryOp(MLIRContext* context,
                                            RewritePatternSet* patterns,
                                            ConstructorArgs&&... args) {
#define POPULATE_BCAST(ChloOp, HloOp)                                          \
  patterns                                                                     \
      ->add<Pattern<ChloOp, HloOp, HloNaryElementwiseAdaptor<ChloOp, HloOp>>>( \
          context, args...);

  POPULATE_BCAST(mlir::chlo::BroadcastAddOp, mlir::stablehlo::AddOp);
  POPULATE_BCAST(mlir::chlo::BroadcastAndOp, mlir::stablehlo::AndOp);
  POPULATE_BCAST(mlir::chlo::BroadcastAtan2Op, mlir::stablehlo::Atan2Op);
  POPULATE_BCAST(mlir::chlo::BroadcastComplexOp, mlir::stablehlo::ComplexOp);
  POPULATE_BCAST(mlir::chlo::BroadcastDivOp, mlir::stablehlo::DivOp);
  POPULATE_BCAST(mlir::chlo::BroadcastMaxOp, mlir::stablehlo::MaxOp);
  POPULATE_BCAST(mlir::chlo::BroadcastMinOp, mlir::stablehlo::MinOp);
  POPULATE_BCAST(mlir::chlo::BroadcastMulOp, mlir::stablehlo::MulOp);
  POPULATE_BCAST(mlir::chlo::BroadcastNextAfterOp, mlir::chlo::NextAfterOp);
  POPULATE_BCAST(mlir::chlo::BroadcastOrOp, mlir::stablehlo::OrOp);
  POPULATE_BCAST(mlir::chlo::BroadcastPolygammaOp, mlir::chlo::PolygammaOp);
  POPULATE_BCAST(mlir::chlo::BroadcastPowOp, mlir::stablehlo::PowOp);
  POPULATE_BCAST(mlir::chlo::BroadcastRemOp, mlir::stablehlo::RemOp);
  POPULATE_BCAST(mlir::chlo::BroadcastShiftLeftOp,
                 mlir::stablehlo::ShiftLeftOp);
  POPULATE_BCAST(mlir::chlo::BroadcastShiftRightArithmeticOp,
                 mlir::stablehlo::ShiftRightArithmeticOp);
  POPULATE_BCAST(mlir::chlo::BroadcastShiftRightLogicalOp,
                 mlir::stablehlo::ShiftRightLogicalOp);
  POPULATE_BCAST(mlir::chlo::BroadcastSubOp, mlir::stablehlo::SubtractOp);
  POPULATE_BCAST(mlir::chlo::BroadcastXorOp, mlir::stablehlo::XorOp);
  POPULATE_BCAST(mlir::chlo::BroadcastZetaOp, mlir::chlo::ZetaOp);

#undef POPULATE_BCAST

  // Broadcasting ops requiring special construction.
  patterns->add<Pattern<mlir::chlo::BroadcastCompareOp,
                        mlir::stablehlo::CompareOp, HloCompareAdaptor>>(
      context, args...);
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns.
//===----------------------------------------------------------------------===//

// Converts binary ops that statically are determined to not broadcast directly
// to the corresponding stablehlo non-broadcasting op.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertTrivialNonBroadcastBinaryOp final
    : OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ChloOpTy op, typename ChloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    auto lhsType = dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(adaptor.getRhs().getType());
    if (!lhsType || !rhsType) return failure();

    // Requires rank broadcast.
    if (lhsType.getRank() != rhsType.getRank()) return failure();

    // Any dynamic dimension may require broadcasting and requires more
    // analysis.
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
      return failure();
    }

    if (!llvm::equal(lhsType.getShape(), rhsType.getShape())) {
      return failure();
    }

    rewriter.replaceOp(
        op, ValueRange{Adaptor::createOp(op, op.getResult().getType(),
                                         adaptor.getOperands(), rewriter)});
    return success();
  }
};

// Converts a binary op with ranked broadcasting operands to explicitly
// broadcast and invoke the corresponding stablehlo non-broadcasting op.
// Note that dynamic broadcasting supported by this pattern is only valid for
// "numpy" broadcasting semantics as defined here:
//   https://docs.scipy.org/doc/numpy/reference/ufuncs.html
// Specifically, this includes the following cases:
//   - Same rank broadcast (operands have the same static rank).
//   - Different-rank broadcast, either without a broadcast_dims attribute or
//     with the broadcast_dims attribute set to map to a prefix padding.
//   - Legal combinations of degenerate (1-dim) implicit broadcasting.
// The restriction on broadcast_dims derives from the definition of the
// `shape.broadcast` op, which only supports prefix-padding.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertRankedDynamicBroadcastBinaryOp final
    : OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ChloOpTy op, typename ChloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Only support ranked operands.
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!lhsType || !rhsType || !resultType) return failure();

    // Check for "numpy"-style rank broadcast.
    auto broadcastDimensions = op.getBroadcastDimensions();
    if (broadcastDimensions && !mlir::hlo::isLegalNumpyRankedBroadcast(
                                   lhs, rhs, *broadcastDimensions)) {
      // Note: It is unclear whether the general specification of explicit
      // broadcast_dimensions on binary ops is a feature we want to carry
      // forward. While it can technically be implemented for ranked-dynamic,
      // it is incompatible with unranked inputs. If this warning is emitted
      // in real programs, it is an indication that the feature should be
      // implemented versus just falling back on the more standard definition
      // of numpy-like prefix-padding.
      op.emitWarning() << "unsupported non prefix-padded dynamic rank "
                       << "broadcast_dimensions = " << *broadcastDimensions;
      return failure();
    }

    // Compute result shape.
    Location loc = op.getLoc();

    // Insert a constraint on the shapes being broadcastable and insert all
    // future code into an assuming block reliant on the constraint.
    Value lhsShape = rewriter.create<shape::ShapeOfOp>(loc, lhs);
    Value rhsShape = rewriter.create<shape::ShapeOfOp>(loc, rhs);
    auto broadcastableCstr =
        rewriter.create<shape::CstrBroadcastableOp>(loc, lhsShape, rhsShape);
    auto assumingOp = rewriter.create<shape::AssumingOp>(
        loc, ArrayRef<Type>{resultType}, broadcastableCstr.getResult());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&assumingOp.getDoRegion());

    int64_t resultRank = std::max(lhsType.getRank(), rhsType.getRank());
    Value resultExtents =
        hlo::computeBinaryElementwiseBroadcastingResultExtents(loc, lhs, rhs,
                                                               rewriter);

    // Note that we unconditionally emit DynamicBroadcastInDim ops and let
    // downstream canonicalizations fold them away if possible. This is
    // because, in the dynamic case, there are many corner cases regarding
    // when it is safe to omit, and some of them require analysis to prove
    // properly.
    auto lhsBroadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(resultRank - lhsType.getRank(), resultRank));
    Value broadcastedLhs =
        rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
            loc,
            RankedTensorType::get(resultType.getShape(),
                                  lhsType.getElementType()),
            lhs, resultExtents,
            rewriter.getI64TensorAttr(lhsBroadcastDimensions));
    auto rhsBroadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(resultRank - rhsType.getRank(), resultRank));
    Value broadcastedRhs =
        rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
            loc,
            RankedTensorType::get(resultType.getShape(),
                                  rhsType.getElementType()),
            rhs, resultExtents,
            rewriter.getI64TensorAttr(rhsBroadcastDimensions));

    // And generate the final non-broadcasted binary op.
    Value finalResult = Adaptor::createOp(
        op, resultType, {broadcastedLhs, broadcastedRhs}, rewriter);
    rewriter.create<shape::AssumingYieldOp>(loc, finalResult);
    rewriter.replaceOp(op, {assumingOp.getResult(0)});
    return success();
  }
};

struct ConvertConstantOp final : OpConversionPattern<mlir::chlo::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, op.getValue());
    return success();
  }
};

struct ConvertConstantLikeOp final
    : OpConversionPattern<mlir::chlo::ConstantLikeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::ConstantLikeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto resultTy = cast<ShapedType>(op.getType());

    // Unranked uses are not supported.
    if (!resultTy.hasRank()) return failure();

    // Lower to HLO constant if statically shaped.
    if (resultTy.hasStaticShape()) {
      auto complexAttr = dyn_cast<mlir::complex::NumberAttr>(op.getValue());
      auto attr = DenseElementsAttr::get(
          resultTy, complexAttr ? complexAttr : op.getValue());
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, attr);
      return success();
    }

    // Lower to broadcasted constant.
    Location loc = op.getLoc();
    Value constant =
        rewriter.create<mlir::stablehlo::ConstantOp>(loc, op.getValue());
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, adaptor.getOperand());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::DynamicBroadcastInDimOp>(
        op, resultTy, constant, shape, rewriter.getI64TensorAttr({}));
    return success();
  }
};

struct ConvertSelectOp final
    : OpConversionPattern<mlir::chlo::BroadcastSelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::BroadcastSelectOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Only support ranked operands.
    Value pred = adaptor.getPred();
    Value onTrue = adaptor.getOnTrue();
    Value onFalse = adaptor.getOnFalse();
    auto predType = dyn_cast<RankedTensorType>(pred.getType());
    auto onTrueType = dyn_cast<RankedTensorType>(onTrue.getType());
    auto onFalseType = dyn_cast<RankedTensorType>(onFalse.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!predType || !onTrueType || !onFalseType || !resultType) {
      return failure();
    }

    Location loc = op.getLoc();
    Value predShape = rewriter.createOrFold<shape::ShapeOfOp>(loc, pred);
    Value onTrueShape = rewriter.createOrFold<shape::ShapeOfOp>(loc, onTrue);
    Value onFalseShape = rewriter.createOrFold<shape::ShapeOfOp>(loc, onFalse);
    int64_t resultRank = std::max(
        {predType.getRank(), onTrueType.getRank(), onFalseType.getRank()});

    Value broadcastableCstr = rewriter.createOrFold<shape::CstrBroadcastableOp>(
        loc, ValueRange{predShape, onTrueShape, onFalseShape});
    auto assumingOp = rewriter.create<shape::AssumingOp>(
        loc, ArrayRef<Type>{resultType}, broadcastableCstr);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&assumingOp.getDoRegion());

    Value resultExtents = rewriter.createOrFold<shape::BroadcastOp>(
        loc, shape::getExtentTensorType(op.getContext()),
        ValueRange{predShape, onTrueShape, onFalseShape},
        /*error=*/nullptr);
    auto shapeType =
        RankedTensorType::get({resultRank}, rewriter.getIndexType());
    resultExtents =
        rewriter.createOrFold<tensor::CastOp>(loc, shapeType, resultExtents);

    Value broadcastedPred = pred;
    // Pred has an implicit broadcast for scalars, so use that when convenient.
    if (predType.getRank() > 0) {
      auto predBroadcastDimensions = llvm::to_vector<4>(
          llvm::seq<int64_t>(resultRank - predType.getRank(), resultRank));
      broadcastedPred =
          rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
              loc,
              RankedTensorType::get(resultType.getShape(),
                                    predType.getElementType()),
              pred, resultExtents,
              rewriter.getI64TensorAttr(predBroadcastDimensions));
    }
    auto onTrueBroadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(resultRank - onTrueType.getRank(), resultRank));
    Value broadcastedOnTrue =
        rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
            loc,
            RankedTensorType::get(resultType.getShape(),
                                  onTrueType.getElementType()),
            onTrue, resultExtents,
            rewriter.getI64TensorAttr(onTrueBroadcastDimensions));
    auto onFalseBroadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(resultRank - onFalseType.getRank(), resultRank));
    Value broadcastedOnFalse =
        rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
            loc,
            RankedTensorType::get(resultType.getShape(),
                                  onFalseType.getElementType()),
            onFalse, resultExtents,
            rewriter.getI64TensorAttr(onFalseBroadcastDimensions));

    // And generate the final non-broadcasted ternary op.
    Value finalResult = rewriter.create<mlir::stablehlo::SelectOp>(
        loc, resultType, broadcastedPred, broadcastedOnTrue,
        broadcastedOnFalse);
    rewriter.create<shape::AssumingYieldOp>(loc, finalResult);
    rewriter.replaceOp(op, {assumingOp.getResult(0)});
    return success();
  }
};

struct ConvertDynamicReshapeOp final
    : OpRewritePattern<mlir::chlo::DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::chlo::DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    TypedValue<TensorType> tensor = op.getOperand();
    TypedValue<RankedTensorType> shape = op.getOutputShape();

    auto shapeTy = cast<ShapedType>(shape.getType());
    auto resultTy = cast<ShapedType>(op.getType());

    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, tensor);
    Value numEls = rewriter.create<shape::NumElementsOp>(loc, inputShape);
    Value cstr =
        rewriter.create<mlir::stablehlo::CstrReshapableOp>(loc, numEls, shape);
    rewriter.replaceOpWithNewOp<shape::AssumingOp>(
        op, cstr, [&](OpBuilder& b, Location l) {
          Value computedShape =
              b.create<mlir::stablehlo::ComputeReshapeShapeOp>(l, shapeTy,
                                                               numEls, shape);
          SmallVector<Value> result;
          result.push_back(b.create<mlir::stablehlo::DynamicReshapeOp>(
              l, resultTy, tensor, computedShape));
          return result;
        });

    return success();
  }
};

struct LegalizeChlo final : impl::LegalizeChloBase<LegalizeChlo> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::shape::ShapeDialect,
                    mlir::stablehlo::StablehloDialect,
                    mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    {
      ConversionTarget conversionTarget(getContext());
      RewritePatternSet conversionPatterns(ctx);
      conversionTarget.addIllegalDialect<chlo::ChloDialect>();
      conversionTarget.addLegalOp<chlo::MinimumBroadcastShapesOp>();
      conversionTarget.addLegalDialect<
          mlir::stablehlo::StablehloDialect, mlir::arith::ArithDialect,
          mlir::func::FuncDialect, mlir::shape::ShapeDialect,
          mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();

      populateLegalizeChloPatterns(ctx, &conversionPatterns);
      if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                        std::move(conversionPatterns)))) {
        return signalPassFailure();
      }
    }

    {
      // Add canonicalization patterns to simplify produced ops from other
      // dialects.
      RewritePatternSet patterns(ctx);
      populateCanonicalizationPatterns(ctx, &patterns);
      mlir::shape::AssumingOp::getCanonicalizationPatterns(patterns, ctx);
      mlir::shape::ShapeOfOp::getCanonicalizationPatterns(patterns, ctx);
      mlir::shape::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      mlir::shape::CstrBroadcastableOp::getCanonicalizationPatterns(patterns,
                                                                    ctx);
      mlir::tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
}  // namespace

void populateLegalizeChloPatterns(MLIRContext* context,
                                  RewritePatternSet* patterns) {
  // Instantiate conversion templates for conforming binary elementwise ops
  // that do not have different dtypes between operands and results and do
  // not have special attributes that need to be preserved.
  populateForBroadcastingBinaryOp<ConvertTrivialNonBroadcastBinaryOp>(
      context, patterns, 10);
  populateForBroadcastingBinaryOp<ConvertRankedDynamicBroadcastBinaryOp>(
      context, patterns, 5);
  patterns->add<ConvertConstantOp, ConvertConstantLikeOp,
                ConvertDynamicReshapeOp, ConvertSelectOp>(context);
}
}  // namespace mlir::iree_compiler::stablehlo
