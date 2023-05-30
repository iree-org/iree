// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/MathToVM/ConvertMathToVM.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class UnaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): support vectors.
    if (llvm::isa<VectorType>(srcOp.getResult().getType())) return failure();

    switch (adaptor.getOperand().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(
            srcOp, adaptor.getOperand().getType(), adaptor.getOperand());
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(
            srcOp, adaptor.getOperand().getType(), adaptor.getOperand());
        break;
      default:
        assert(false && "invalid target type");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class BinaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): support vectors.
    if (llvm::isa<VectorType>(srcOp.getResult().getType())) return failure();

    switch (adaptor.getLhs().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(
            srcOp, adaptor.getLhs().getType(), adaptor.getLhs(),
            adaptor.getRhs());
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(
            srcOp, adaptor.getLhs().getType(), adaptor.getLhs(),
            adaptor.getRhs());
        break;
      default:
        assert(false && "invalid target type");
    }
    return success();
  }
};

}  // namespace

void populateMathToVMPatterns(MLIRContext *context,
                              TypeConverter &typeConverter,
                              RewritePatternSet &patterns) {
  patterns.insert<
      UnaryArithmeticOpConversion<math::AbsFOp, IREE::VM::AbsF32Op,
                                  IREE::VM::AbsF64Op>,
      UnaryArithmeticOpConversion<math::CeilOp, IREE::VM::CeilF32Op,
                                  IREE::VM::CeilF64Op>,
      UnaryArithmeticOpConversion<math::FloorOp, IREE::VM::FloorF32Op,
                                  IREE::VM::FloorF64Op>,
      UnaryArithmeticOpConversion<math::RoundOp, IREE::VM::RoundF32Op,
                                  IREE::VM::RoundF64Op>,
      UnaryArithmeticOpConversion<math::RoundEvenOp, IREE::VM::RoundF32EvenOp,
                                  IREE::VM::RoundF64EvenOp>,
      UnaryArithmeticOpConversion<math::AtanOp, IREE::VM::AtanF32Op,
                                  IREE::VM::AtanF64Op>,
      BinaryArithmeticOpConversion<math::Atan2Op, IREE::VM::Atan2F32Op,
                                   IREE::VM::Atan2F64Op>,
      UnaryArithmeticOpConversion<math::CosOp, IREE::VM::CosF32Op,
                                  IREE::VM::CosF64Op>,
      UnaryArithmeticOpConversion<math::SinOp, IREE::VM::SinF32Op,
                                  IREE::VM::SinF64Op>,
      UnaryArithmeticOpConversion<math::ExpOp, IREE::VM::ExpF32Op,
                                  IREE::VM::ExpF64Op>,
      UnaryArithmeticOpConversion<math::Exp2Op, IREE::VM::Exp2F32Op,
                                  IREE::VM::Exp2F64Op>,
      UnaryArithmeticOpConversion<math::ExpM1Op, IREE::VM::ExpM1F32Op,
                                  IREE::VM::ExpM1F64Op>,
      UnaryArithmeticOpConversion<math::LogOp, IREE::VM::LogF32Op,
                                  IREE::VM::LogF64Op>,
      UnaryArithmeticOpConversion<math::Log10Op, IREE::VM::Log10F32Op,
                                  IREE::VM::Log10F64Op>,
      UnaryArithmeticOpConversion<math::Log1pOp, IREE::VM::Log1pF32Op,
                                  IREE::VM::Log1pF64Op>,
      UnaryArithmeticOpConversion<math::Log2Op, IREE::VM::Log2F32Op,
                                  IREE::VM::Log2F64Op>,
      BinaryArithmeticOpConversion<math::PowFOp, IREE::VM::PowF32Op,
                                   IREE::VM::PowF64Op>,
      UnaryArithmeticOpConversion<math::RsqrtOp, IREE::VM::RsqrtF32Op,
                                  IREE::VM::RsqrtF64Op>,
      UnaryArithmeticOpConversion<math::SqrtOp, IREE::VM::SqrtF32Op,
                                  IREE::VM::SqrtF64Op>,
      UnaryArithmeticOpConversion<math::TanhOp, IREE::VM::TanhF32Op,
                                  IREE::VM::TanhF64Op>,
      UnaryArithmeticOpConversion<math::ErfOp, IREE::VM::ErfF32Op,
                                  IREE::VM::ErfF64Op>,
      UnaryArithmeticOpConversion<math::CountLeadingZerosOp,
                                  IREE::VM::CtlzI32Op, IREE::VM::CtlzI64Op>>(
      typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
