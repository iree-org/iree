// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering scalar StableHLO ops to arith dialect.

#include "iree/compiler/InputConversion/StableHLO/LegalizeToLinalgUtils.h"
#include "iree/compiler/InputConversion/StableHLO/Rewriters.h"
#include "iree/compiler/InputConversion/StableHLO/TypeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {
namespace stablehlo = mlir::stablehlo;

template <typename OpTy>
struct ScalarHloToArithmeticPattern final : OpConversionPattern<OpTy> {
  ScalarHloToArithmeticPattern(
      TypeConverter& typeConverter, MLIRContext* context,
      llvm::function_ref<bool(Operation*)> filterFn = nullptr,
      PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, context, benefit),
        filterFn(filterFn) {}

  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (filterFn && !filterFn(op)) return failure();

    auto isScalar = [](Value v) {
      return cast<ShapedType>(v.getType()).getRank() == 0;
    };

    if (!llvm::all_of(adaptor.getOperands(), isScalar))
      return rewriter.notifyMatchFailure(op, "All operands must be scalar.");

    Location loc = op.getLoc();

    auto resultTy = dyn_cast_or_null<ShapedType>(
        this->typeConverter->convertType(op->getResultTypes().front()));
    if (!resultTy) return failure();

    SmallVector<Value> operands;
    for (Value operand : adaptor.getOperands()) {
      operands.push_back(
          rewriter.create<tensor::ExtractOp>(loc, operand, ValueRange()));
    }
    Value scalarResult = stablehlo::StableHloOpToStdScalarOp::mapOp(
        op, resultTy.getElementType(), operands, &rewriter);
    if (!scalarResult) return failure();
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy,
                                                        scalarResult);
    return success();
  }

 private:
  llvm::function_ref<bool(Operation*)> filterFn;
};

}  // namespace

namespace detail {
void populateScalarHloToArithConversionPatterns(
    MLIRContext* context, TypeConverter& typeConverter,
    RewritePatternSet* patterns,
    llvm::function_ref<bool(Operation*)> filterFn) {
  // TODO(#12678): Handle the XLA rng op.
  patterns->add<ScalarHloToArithmeticPattern<stablehlo::AbsOp>,
                ScalarHloToArithmeticPattern<stablehlo::AddOp>,
                ScalarHloToArithmeticPattern<stablehlo::AndOp>,
                ScalarHloToArithmeticPattern<stablehlo::Atan2Op>,
                ScalarHloToArithmeticPattern<stablehlo::BitcastConvertOp>,
                ScalarHloToArithmeticPattern<stablehlo::CbrtOp>,
                ScalarHloToArithmeticPattern<stablehlo::CeilOp>,
                ScalarHloToArithmeticPattern<stablehlo::ClampOp>,
                ScalarHloToArithmeticPattern<stablehlo::ClzOp>,
                ScalarHloToArithmeticPattern<stablehlo::CompareOp>,
                ScalarHloToArithmeticPattern<stablehlo::ComplexOp>,
                ScalarHloToArithmeticPattern<stablehlo::ConvertOp>,
                ScalarHloToArithmeticPattern<stablehlo::CosineOp>,
                ScalarHloToArithmeticPattern<stablehlo::DivOp>,
                ScalarHloToArithmeticPattern<stablehlo::ExpOp>,
                ScalarHloToArithmeticPattern<stablehlo::Expm1Op>,
                ScalarHloToArithmeticPattern<stablehlo::FloorOp>,
                ScalarHloToArithmeticPattern<stablehlo::ImagOp>,
                ScalarHloToArithmeticPattern<stablehlo::IsFiniteOp>,
                ScalarHloToArithmeticPattern<stablehlo::Log1pOp>,
                ScalarHloToArithmeticPattern<stablehlo::LogOp>,
                ScalarHloToArithmeticPattern<stablehlo::LogisticOp>,
                ScalarHloToArithmeticPattern<stablehlo::MaxOp>,
                ScalarHloToArithmeticPattern<stablehlo::MinOp>,
                ScalarHloToArithmeticPattern<stablehlo::MulOp>,
                ScalarHloToArithmeticPattern<stablehlo::NegOp>,
                ScalarHloToArithmeticPattern<stablehlo::NotOp>,
                ScalarHloToArithmeticPattern<stablehlo::OrOp>,
                ScalarHloToArithmeticPattern<stablehlo::PopulationCountOp>,
                ScalarHloToArithmeticPattern<stablehlo::PowOp>,
                ScalarHloToArithmeticPattern<stablehlo::RealOp>,
                ScalarHloToArithmeticPattern<stablehlo::ReducePrecisionOp>,
                ScalarHloToArithmeticPattern<stablehlo::RemOp>,
                ScalarHloToArithmeticPattern<stablehlo::RoundNearestEvenOp>,
                ScalarHloToArithmeticPattern<stablehlo::RoundOp>,
                ScalarHloToArithmeticPattern<stablehlo::RsqrtOp>,
                ScalarHloToArithmeticPattern<stablehlo::SelectOp>,
                ScalarHloToArithmeticPattern<stablehlo::ShiftLeftOp>,
                ScalarHloToArithmeticPattern<stablehlo::ShiftRightArithmeticOp>,
                ScalarHloToArithmeticPattern<stablehlo::ShiftRightLogicalOp>,
                ScalarHloToArithmeticPattern<stablehlo::SignOp>,
                ScalarHloToArithmeticPattern<stablehlo::SineOp>,
                ScalarHloToArithmeticPattern<stablehlo::SqrtOp>,
                ScalarHloToArithmeticPattern<stablehlo::SubtractOp>,
                ScalarHloToArithmeticPattern<stablehlo::TanhOp>,
                ScalarHloToArithmeticPattern<stablehlo::XorOp> >(
      typeConverter, context, filterFn);
}
}  // namespace detail

}  // namespace mlir::iree_compiler::stablehlo
