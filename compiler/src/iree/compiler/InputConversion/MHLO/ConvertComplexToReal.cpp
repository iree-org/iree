// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Rewriters.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

namespace {

inline std::optional<chlo::ComparisonDirection> chloComparisonDirection(
    mhlo::ComparisonDirection value) {
  switch (value) {
    case mhlo::ComparisonDirection::EQ:
      return chlo::ComparisonDirection::EQ;
    case mhlo::ComparisonDirection::NE:
      return chlo::ComparisonDirection::NE;
    case mhlo::ComparisonDirection::GE:
      return chlo::ComparisonDirection::GE;
    case mhlo::ComparisonDirection::GT:
      return chlo::ComparisonDirection::GT;
    case mhlo::ComparisonDirection::LE:
      return chlo::ComparisonDirection::LE;
    case mhlo::ComparisonDirection::LT:
      return chlo::ComparisonDirection::LT;
    default:
      return {};
  }
}

inline std::optional<chlo::ComparisonType> chloComparisonType(
    mhlo::ComparisonType value) {
  switch (value) {
    case mhlo::ComparisonType::NOTYPE:
      return chlo::ComparisonType::NOTYPE;
    case mhlo::ComparisonType::FLOAT:
      return chlo::ComparisonType::FLOAT;
    case mhlo::ComparisonType::TOTALORDER:
      return chlo::ComparisonType::TOTALORDER;
    case mhlo::ComparisonType::SIGNED:
      return chlo::ComparisonType::SIGNED;
    case mhlo::ComparisonType::UNSIGNED:
      return chlo::ComparisonType::UNSIGNED;
    default:
      return {};
  }
}

bool isComplexTensor(Value v) {
  if (auto tt = llvm::dyn_cast<TensorType>(v.getType())) {
    return llvm::isa<ComplexType>(tt.getElementType());
  }
  return false;
}

Type convertComplexTensorTypeToReal(Type complexTensorType) {
  auto newElementType =
      llvm::cast<ComplexType>(
          complexTensorType.cast<TensorType>().getElementType())
          .getElementType();
  if (auto tt = llvm::dyn_cast<RankedTensorType>(complexTensorType)) {
    return RankedTensorType::get(tt.getShape(), newElementType,
                                 tt.getEncoding());
  } else if (auto tt = llvm::dyn_cast<UnrankedTensorType>(complexTensorType)) {
    return UnrankedTensorType::get(newElementType);
  }
  assert(false && "unknown TensorType subclass");
  return Type();
}

// Add and subtraction are elementwise and can be distributed across the real
// and imaginary components.
template <typename OpTy>
struct ConvertAddSubOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  static Value createOp(OpBuilder &b, mhlo::AddOp op, Value lhs, Value rhs) {
    return b.create<mhlo::AddOp>(op.getLoc(), lhs, rhs);
  }
  static Value createOp(OpBuilder &b, mhlo::SubtractOp op, Value lhs,
                        Value rhs) {
    return b.create<mhlo::SubtractOp>(op.getLoc(), lhs, rhs);
  }
  static Value createOp(OpBuilder &b, chlo::BroadcastAddOp op, Value lhs,
                        Value rhs) {
    return b.create<chlo::BroadcastAddOp>(op.getLoc(), lhs, rhs, nullptr);
  }
  static Value createOp(OpBuilder &b, chlo::BroadcastSubOp op, Value lhs,
                        Value rhs) {
    return b.create<chlo::BroadcastSubOp>(op.getLoc(), lhs, rhs, nullptr);
  }

  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (!isComplexTensor(adaptor.getLhs()) ||
        !isComplexTensor(adaptor.getRhs())) {
      return rewriter.notifyMatchFailure(op, "not complex tensor");
    }

    Value real =
        createOp(rewriter, op,
                 rewriter.createOrFold<mhlo::RealOp>(loc, adaptor.getLhs()),
                 rewriter.createOrFold<mhlo::RealOp>(loc, adaptor.getRhs()));
    Value imag =
        createOp(rewriter, op,
                 rewriter.createOrFold<mhlo::ImagOp>(loc, adaptor.getLhs()),
                 rewriter.createOrFold<mhlo::ImagOp>(loc, adaptor.getRhs()));
    Value result = rewriter.create<mhlo::ComplexOp>(loc, real, imag);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Complex multiplication results in a cross product multiplication between the
// real and imaginary components such that:
//   result.real = lhs.real * rhs.real - lhs.imag * rhs.imag
//   result.imag = lhs.imag * rhs.real + lhs.real * rhs.imag
template <typename MulOpTy>
struct ConvertMulOp : public OpConversionPattern<MulOpTy> {
  using OpConversionPattern<MulOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      MulOpTy op, typename MulOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!isComplexTensor(adaptor.getLhs()) ||
        !isComplexTensor(adaptor.getRhs())) {
      return rewriter.notifyMatchFailure(op, "not complex tensor");
    }

    auto lhsReal = rewriter.createOrFold<mhlo::RealOp>(loc, adaptor.getLhs());
    auto lhsImag = rewriter.createOrFold<mhlo::ImagOp>(loc, adaptor.getLhs());
    auto rhsReal = rewriter.createOrFold<mhlo::RealOp>(loc, adaptor.getRhs());
    auto rhsImag = rewriter.createOrFold<mhlo::ImagOp>(loc, adaptor.getRhs());

    auto realComponent = rewriter.create<mhlo::SubtractOp>(
        loc,
        rewriter.create<chlo::BroadcastMulOp>(loc, lhsReal, rhsReal,
                                              /*broadcast_dimensions=*/nullptr),
        rewriter.create<chlo::BroadcastMulOp>(
            loc, lhsImag, rhsImag, /*broadcast_dimensions=*/nullptr));
    auto imagComponent = rewriter.create<mhlo::AddOp>(
        loc,
        rewriter.create<chlo::BroadcastMulOp>(loc, lhsReal, rhsImag,
                                              /*broadcast_dimensions=*/nullptr),
        rewriter.create<chlo::BroadcastMulOp>(
            loc, lhsImag, rhsReal, /*broadcast_dimensions=*/nullptr));
    Value result = rewriter.createOrFold<mhlo::ComplexOp>(loc, realComponent,
                                                          imagComponent);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Division is performed by normalizing the denominator by multiplying by the
// conjugate of the rhs.
//   numerator = lhs * conj(rhs)
//   denominator = rhs * conj(rhs)
template <typename DivOpTy>
struct ConvertDivOp : public OpConversionPattern<DivOpTy> {
  using OpConversionPattern<DivOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      DivOpTy op, typename DivOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!isComplexTensor(adaptor.getLhs()) ||
        !isComplexTensor(adaptor.getRhs())) {
      return rewriter.notifyMatchFailure(op, "not complex tensor");
    }

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto rhsReal = rewriter.createOrFold<mhlo::RealOp>(loc, rhs);
    auto rhsImag = rewriter.createOrFold<mhlo::ImagOp>(loc, rhs);

    Value conj = rewriter.createOrFold<mhlo::ComplexOp>(
        loc, rhsReal, rewriter.create<mhlo::NegOp>(loc, rhsImag));
    Value complexNumerator = rewriter.create<chlo::BroadcastMulOp>(
        loc, lhs, conj, /*broadcast_dimensions=*/nullptr);
    Value denominator = rewriter.create<mhlo::AddOp>(
        loc, rewriter.create<mhlo::MulOp>(loc, rhsReal, rhsReal),
        rewriter.create<mhlo::MulOp>(loc, rhsImag, rhsImag));

    Value realComponent = rewriter.create<chlo::BroadcastDivOp>(
        loc, rewriter.create<mhlo::RealOp>(loc, complexNumerator), denominator,
        /*broadcast_dimensions=*/nullptr);
    Value imagComponent = rewriter.create<chlo::BroadcastDivOp>(
        loc, rewriter.create<mhlo::ImagOp>(loc, complexNumerator), denominator,
        /*broadcast_dimensions=*/nullptr);

    Value result = rewriter.createOrFold<mhlo::ComplexOp>(loc, realComponent,
                                                          imagComponent);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Absolute value is evaluated as:
//   result = sqrt(val.real * val.real + val.imag * val.imag)
struct ConvertAbsOp : public OpConversionPattern<mhlo::AbsOp> {
  using OpConversionPattern<mhlo::AbsOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::AbsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!isComplexTensor(adaptor.getOperand())) {
      return rewriter.notifyMatchFailure(op, "not complex tensor");
    }

    auto operandReal =
        rewriter.createOrFold<mhlo::RealOp>(loc, adaptor.getOperand());
    auto operandImag =
        rewriter.createOrFold<mhlo::ImagOp>(loc, adaptor.getOperand());
    rewriter.replaceOpWithNewOp<mhlo::SqrtOp>(
        op,
        rewriter.create<mhlo::AddOp>(
            loc, rewriter.create<mhlo::MulOp>(loc, operandReal, operandReal),
            rewriter.create<mhlo::MulOp>(loc, operandImag, operandImag)));
    return success();
  }
};

// Exponential can be lowered to an exponential on the real component and a
// sum of sinusoids of the imaginary component, which equates to a normal
// exponential operator multiplied by Euler's formula.
//
// Exp(a + ib) = Exp(a) * Exp(ib) = Exp(a) * Cos(b) + Exp(a) * iSin(b))
struct ConvertExpOp : public OpConversionPattern<mhlo::ExpOp> {
  using OpConversionPattern<mhlo::ExpOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ExpOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!isComplexTensor(adaptor.getOperand())) {
      return rewriter.notifyMatchFailure(op, "not complex tensor");
    }

    auto operandReal = rewriter.create<mhlo::RealOp>(loc, adaptor.getOperand());
    auto operandImag = rewriter.create<mhlo::ImagOp>(loc, adaptor.getOperand());

    Value expReal = rewriter.create<mhlo::ExpOp>(loc, operandReal);
    Value result = rewriter.createOrFold<mhlo::ComplexOp>(
        loc,
        rewriter.create<mhlo::MulOp>(
            loc, rewriter.create<mhlo::CosineOp>(loc, operandImag), expReal),
        rewriter.create<mhlo::MulOp>(
            loc, rewriter.create<mhlo::SineOp>(loc, operandImag), expReal));
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename CompareOpTy, typename ComparatorOpTy>
struct ConvertCHLOCompareOp : public OpConversionPattern<CompareOpTy> {
  using OpConversionPattern<CompareOpTy>::OpConversionPattern;
  ConvertCHLOCompareOp(TypeConverter &typeConverter, MLIRContext *context,
                       chlo::ComparisonDirection direction)
      : OpConversionPattern<CompareOpTy>(typeConverter, context),
        direction(direction) {}

  LogicalResult matchAndRewrite(
      CompareOpTy op, typename CompareOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!isComplexTensor(adaptor.getLhs()) ||
        !isComplexTensor(adaptor.getRhs())) {
      return rewriter.notifyMatchFailure(op, "not complex tensor");
    }
    if (direction != op.getComparisonDirection()) {
      return rewriter.notifyMatchFailure(op, "not matching direction");
    }

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsReal = rewriter.createOrFold<mhlo::RealOp>(loc, lhs);
    auto lhsImag = rewriter.createOrFold<mhlo::ImagOp>(loc, lhs);
    auto rhsReal = rewriter.createOrFold<mhlo::RealOp>(loc, rhs);
    auto rhsImag = rewriter.createOrFold<mhlo::ImagOp>(loc, rhs);

    rewriter.replaceOpWithNewOp<ComparatorOpTy>(
        op,
        rewriter.create<chlo::BroadcastCompareOp>(
            loc, lhsReal, rhsReal,
            /*broadcast_dimensions=*/nullptr,
            adaptor.getComparisonDirectionAttr(), adaptor.getCompareTypeAttr()),
        rewriter.create<chlo::BroadcastCompareOp>(
            loc, lhsImag, rhsImag,
            /*broadcast_dimensions=*/nullptr,
            adaptor.getComparisonDirectionAttr(),
            adaptor.getCompareTypeAttr()));

    return success();
  }

  chlo::ComparisonDirection direction;
};

template <typename CompareOpTy, typename ComparatorOpTy>
struct ConvertMHLOCompareOp : public OpConversionPattern<CompareOpTy> {
  using OpConversionPattern<CompareOpTy>::OpConversionPattern;
  ConvertMHLOCompareOp(TypeConverter &typeConverter, MLIRContext *context,
                       mhlo::ComparisonDirection direction)
      : OpConversionPattern<CompareOpTy>(typeConverter, context),
        direction(direction) {}

  LogicalResult matchAndRewrite(
      CompareOpTy op, typename CompareOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!isComplexTensor(adaptor.getLhs()) ||
        !isComplexTensor(adaptor.getRhs())) {
      return rewriter.notifyMatchFailure(op, "not complex tensor");
    }
    if (direction != op.getComparisonDirection()) {
      return rewriter.notifyMatchFailure(op, "not matching direction");
    }

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsReal = rewriter.createOrFold<mhlo::RealOp>(loc, lhs);
    auto lhsImag = rewriter.createOrFold<mhlo::ImagOp>(loc, lhs);
    auto rhsReal = rewriter.createOrFold<mhlo::RealOp>(loc, rhs);
    auto rhsImag = rewriter.createOrFold<mhlo::ImagOp>(loc, rhs);

    // If the input op is an mhlo op, we need to convert the attributes to the
    // corresponding chlo one..
    chlo::ComparisonDirection chloCmpDirection =
        *chloComparisonDirection(adaptor.getComparisonDirection());

    std::optional<mhlo::ComparisonType> mhloCmpType = adaptor.getCompareType();
    chlo::ComparisonTypeAttr chloCmpType;
    if (mhloCmpType)
      chloCmpType = chlo::ComparisonTypeAttr::get(
          rewriter.getContext(), *chloComparisonType(*mhloCmpType));

    rewriter.replaceOpWithNewOp<ComparatorOpTy>(
        op,
        rewriter.create<chlo::BroadcastCompareOp>(
            loc, lhsReal, rhsReal,
            /*broadcast_dimensions=*/nullptr, chloCmpDirection, chloCmpType),
        rewriter.create<chlo::BroadcastCompareOp>(
            loc, lhsImag, rhsImag,
            /*broadcast_dimensions=*/nullptr, chloCmpDirection, chloCmpType));

    return success();
  }

  mhlo::ComparisonDirection direction;
};

struct ElideComplexPattern : public OpConversionPattern<mhlo::ComplexOp> {
  using OpConversionPattern<mhlo::ComplexOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ComplexOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ElideRealPattern : public OpConversionPattern<mhlo::RealOp> {
  using OpConversionPattern<mhlo::RealOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::RealOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto complexProducer =
        adaptor.getOperands()[0].getDefiningOp<mhlo::ComplexOp>();
    if (complexProducer) {
      rewriter.replaceOp(op, complexProducer.getLhs());
      return success();
    }
    return failure();
  }
};

struct ElideImagPattern : public OpConversionPattern<mhlo::ImagOp> {
  using OpConversionPattern<mhlo::ImagOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ImagOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto complexProducer =
        adaptor.getOperands()[0].getDefiningOp<mhlo::ComplexOp>();
    if (complexProducer) {
      rewriter.replaceOp(op, complexProducer.getRhs());
      return success();
    }
    return failure();
  }
};

}  // namespace

void populateMHLOComplexToRealPatterns(MLIRContext *context,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  // Add an subtract patterns.
  patterns.insert<ConvertAddSubOp<mhlo::AddOp>>(typeConverter, context);
  patterns.insert<ConvertAddSubOp<mhlo::SubtractOp>>(typeConverter, context);
  patterns.insert<ConvertAddSubOp<chlo::BroadcastAddOp>>(typeConverter,
                                                         context);
  patterns.insert<ConvertAddSubOp<chlo::BroadcastSubOp>>(typeConverter,
                                                         context);

  // Mul patterns.
  patterns.insert<ConvertMulOp<mhlo::MulOp>>(typeConverter, context);
  patterns.insert<ConvertMulOp<chlo::BroadcastMulOp>>(typeConverter, context);

  // Div patterns.
  patterns.insert<ConvertDivOp<mhlo::DivOp>>(typeConverter, context);
  patterns.insert<ConvertDivOp<chlo::BroadcastDivOp>>(typeConverter, context);

  // Unary ops.
  patterns.insert<ConvertAbsOp>(typeConverter, context);
  patterns.insert<ConvertExpOp>(typeConverter, context);

  // Compare ops.
  patterns.insert<ConvertMHLOCompareOp<mhlo::CompareOp, mhlo::OrOp>>(
      typeConverter, context, mhlo::ComparisonDirection::NE);
  patterns.insert<ConvertMHLOCompareOp<mhlo::CompareOp, mhlo::AndOp>>(
      typeConverter, context, mhlo::ComparisonDirection::EQ);
  patterns.insert<ConvertCHLOCompareOp<chlo::BroadcastCompareOp, mhlo::OrOp>>(
      typeConverter, context, chlo::ComparisonDirection::NE);
  patterns.insert<ConvertCHLOCompareOp<chlo::BroadcastCompareOp, mhlo::AndOp>>(
      typeConverter, context, chlo::ComparisonDirection::EQ);

  // Complex/Real/Imag conversions should fold away.
  // Note that this is an opinion taken because these patterns are targeted
  // at full conversion scenarios and we would rather know eagerly if
  // conversion is not possible. A more lax conversion would not include the
  // ElideComplexPattern.
  // Doing it this way makes error messages nice because a failure will report
  // which remaining live op is keeping it from being erased.
  patterns.insert<ElideComplexPattern>(typeConverter, context, 0);
  patterns.insert<ElideRealPattern>(typeConverter, context);
  patterns.insert<ElideImagPattern>(typeConverter, context);
}

namespace {

struct TestMHLOConvertComplexToRealPass
    : public TestMHLOConvertComplexToRealBase<
          TestMHLOConvertComplexToRealPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect, chlo::ChloDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    MLIRContext *context = &getContext();
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });

    populateMHLOComplexToRealPatterns(context, typeConverter, patterns);

    ConversionTarget target(*context);
    auto hasNoComplexTypes = [](Operation *op) {
      for (Value operand : op->getOperands()) {
        if (auto st = llvm::dyn_cast<ShapedType>(operand.getType())) {
          if (llvm::isa<ComplexType>(st.getElementType())) {
            return false;
          }
        }
      }
      for (Value result : op->getResults()) {
        if (auto st = llvm::dyn_cast<ShapedType>(result.getType())) {
          if (llvm::isa<ComplexType>(st.getElementType())) {
            return false;
          }
        }
      }
      return true;
    };

    target.addLegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<chlo::ChloDialect>();
    target.addLegalDialect<func::FuncDialect, mlir::arith::ArithDialect>();

    // For the test, require that casts fully convert.
    target.addIllegalOp<mhlo::ComplexOp>();
    target.addIllegalOp<mhlo::ImagOp>();
    target.addIllegalOp<mhlo::RealOp>();

    // Binary elementwise.
    target.addDynamicallyLegalOp<mhlo::AddOp>(hasNoComplexTypes);
    target.addDynamicallyLegalOp<chlo::BroadcastAddOp>(hasNoComplexTypes);
    target.addDynamicallyLegalOp<mhlo::SubtractOp>(hasNoComplexTypes);
    target.addDynamicallyLegalOp<chlo::BroadcastSubOp>(hasNoComplexTypes);
    target.addDynamicallyLegalOp<mhlo::MulOp>(hasNoComplexTypes);
    target.addDynamicallyLegalOp<chlo::BroadcastMulOp>(hasNoComplexTypes);
    target.addDynamicallyLegalOp<mhlo::DivOp>(hasNoComplexTypes);
    target.addDynamicallyLegalOp<chlo::BroadcastDivOp>(hasNoComplexTypes);

    // Unary.
    target.addDynamicallyLegalOp<mhlo::AbsOp>(hasNoComplexTypes);
    target.addDynamicallyLegalOp<mhlo::ExpOp>(hasNoComplexTypes);

    // Compare.
    target.addDynamicallyLegalOp<mhlo::CompareOp>(hasNoComplexTypes);
    target.addDynamicallyLegalOp<chlo::BroadcastCompareOp>(hasNoComplexTypes);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createTestMHLOConvertComplexToRealPass() {
  return std::make_unique<TestMHLOConvertComplexToRealPass>();
}

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir
