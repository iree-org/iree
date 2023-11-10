// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering scalar StableHLO ops to arith dialect.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo-iree/InputConversion/LegalizeToLinalgUtils.h"
#include "stablehlo-iree/InputConversion/Rewriters.h"
#include "stablehlo-iree/InputConversion/TypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

template <typename OpTy>
struct ScalarHloToFuncPatterns final : OpConversionPattern<OpTy> {
  ScalarHloToFuncPatterns(TypeConverter &typeConverter, MLIRContext *context,
                          PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, context, benefit) {}
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<mlir::func::FuncOp>(op->getParentOp())) {
      return rewriter.notifyMatchFailure(op,
                                         "Return must be inside a function");
    }
    mlir::Operation::operand_range operands = op.getOperands();
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, operands);
    return success();
  }
};
template <typename OpTy>
struct ScalarHloToArithmeticPattern final : OpConversionPattern<OpTy> {
  ScalarHloToArithmeticPattern(
      TypeConverter &typeConverter, MLIRContext *context,
      llvm::function_ref<bool(Operation *)> filterFn = nullptr,
      PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, context, benefit),
        filterFn(filterFn) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (filterFn && !filterFn(op))
      return failure();

    auto isScalar = [](Value v) {
      return cast<ShapedType>(v.getType()).getRank() == 0;
    };

    if (!llvm::all_of(adaptor.getOperands(), isScalar))
      return rewriter.notifyMatchFailure(op, "All operands must be scalar.");

    Location loc = op.getLoc();

    auto resultTy = dyn_cast_or_null<ShapedType>(
        this->getTypeConverter()->convertType(op->getResultTypes().front()));
    if (!resultTy)
      return failure();

    SmallVector<Value> operands;
    for (Value operand : adaptor.getOperands()) {
      operands.push_back(
          rewriter.create<tensor::ExtractOp>(loc, operand, ValueRange()));
    }
    Value scalarResult = mlir::stablehlo::StableHloOpToStdScalarOp::mapOp(
        op, resultTy.getElementType(), operands, &rewriter);
    if (!scalarResult)
      return failure();
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy,
                                                        scalarResult);
    return success();
  }

private:
  llvm::function_ref<bool(Operation *)> filterFn;
};

} // namespace

namespace detail {
void populateScalarHloToArithConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns,
    llvm::function_ref<bool(Operation *)> filterFn) {
  // TODO(#12678): Handle the XLA rng op.
  patterns->add<
      ScalarHloToArithmeticPattern<mlir::stablehlo::AbsOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::AddOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::AndOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::Atan2Op>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::BitcastConvertOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::CbrtOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::CeilOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::ClampOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::ClzOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::CompareOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::ComplexOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::ConvertOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::CosineOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::DivOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::ExpOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::Expm1Op>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::FloorOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::ImagOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::IsFiniteOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::Log1pOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::LogOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::LogisticOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::MaxOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::MinOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::MulOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::NegOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::NotOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::OrOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::PopulationCountOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::PowOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::RealOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::ReducePrecisionOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::RemOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::RoundNearestEvenOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::RoundOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::RsqrtOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::SelectOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::ShiftLeftOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::ShiftRightArithmeticOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::ShiftRightLogicalOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::SignOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::SineOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::SqrtOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::SubtractOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::TanhOp>,
      ScalarHloToArithmeticPattern<mlir::stablehlo::XorOp>>(typeConverter,
                                                            context, filterFn);
  patterns->add<ScalarHloToFuncPatterns<mlir::stablehlo::ReturnOp>>(
      typeConverter, context);
}
} // namespace detail

} // namespace mlir::iree_compiler::stablehlo
