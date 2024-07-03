// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/ArithToVM/Patterns.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  TypeConverter &typeConverter;
  ConstantOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}
  LogicalResult
  matchAndRewrite(arith::ConstantOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto targetType = typeConverter.convertType(srcOp.getType());
    if (!targetType) {
      return srcOp.emitError() << "could not convert type: " << srcOp.getType()
                               << " (check -iree-vm-target-* options)";
    }
    if (llvm::isa<IntegerType>(targetType)) {
      auto integerAttr = llvm::dyn_cast<IntegerAttr>(srcOp.getValue());
      if (!integerAttr) {
        return srcOp.emitRemark() << "unsupported const type for dialect";
      }
      switch (targetType.getIntOrFloatBitWidth()) {
      case 1:
      case 32:
        if (integerAttr.getInt()) {
          rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(
              srcOp,
              integerAttr.getType().isInteger(1) ? 1 : integerAttr.getInt());
        } else {
          rewriter.replaceOpWithNewOp<IREE::VM::ConstI32ZeroOp>(srcOp);
        }
        break;
      case 64:
        if (integerAttr.getInt()) {
          rewriter.replaceOpWithNewOp<IREE::VM::ConstI64Op>(
              srcOp, integerAttr.getInt());
        } else {
          rewriter.replaceOpWithNewOp<IREE::VM::ConstI64ZeroOp>(srcOp);
        }
        break;
      default:
        return srcOp.emitRemark()
               << "unsupported const integer bit width for dialect";
      }
    } else if (llvm::isa<FloatType>(targetType)) {
      auto floatAttr = llvm::dyn_cast<FloatAttr>(srcOp.getValue());
      if (!floatAttr) {
        return srcOp.emitRemark() << "unsupported const type for dialect";
      }
      switch (targetType.getIntOrFloatBitWidth()) {
      case 32:
        if (floatAttr.getValue().isZero()) {
          rewriter.replaceOpWithNewOp<IREE::VM::ConstF32ZeroOp>(srcOp);
        } else {
          rewriter.replaceOpWithNewOp<IREE::VM::ConstF32Op>(srcOp, floatAttr);
        }
        break;
      case 64:
        if (floatAttr.getValue().isZero()) {
          rewriter.replaceOpWithNewOp<IREE::VM::ConstF64ZeroOp>(srcOp);
        } else {
          rewriter.replaceOpWithNewOp<IREE::VM::ConstF64Op>(srcOp, floatAttr);
        }
        break;
      default:
        return srcOp.emitRemark()
               << "unsupported const floating-point bit width for dialect";
      }
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

struct CmpI32OpConversion : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::CmpIOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLhs().getType().isInteger(32))
      return failure();
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpEQI32Op>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::ne:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpNEI32Op>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::slt:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32SOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::sle:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32SOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::sgt:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32SOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::sge:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32SOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::ult:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::ule:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::ugt:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::uge:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    default:
      return failure();
    }
  }
};

struct CmpI64OpConversion : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::CmpIOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLhs().getType().isInteger(64))
      return failure();
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpEQI64Op>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::ne:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpNEI64Op>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::slt:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI64SOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::sle:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI64SOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::sgt:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI64SOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::sge:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI64SOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::ult:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI64UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::ule:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI64UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::ugt:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI64UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case arith::CmpIPredicate::uge:
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI64UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    default:
      return failure();
    }
  }
};

struct CmpF32OpConversion : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::CmpFOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLhs().getType().isF32())
      return failure();
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
    case arith::CmpFPredicate::AlwaysFalse: // 0
      rewriter.replaceOpWithNewOp<IREE::VM::ConstI32ZeroOp>(srcOp);
      break;
    case arith::CmpFPredicate::AlwaysTrue: // 1
      rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(srcOp, 1);
      break;
    case arith::CmpFPredicate::UNO: // isnan(lhs) || isnan(rhs)
      rewriter.replaceOpWithNewOp<IREE::VM::OrI32Op>(
          srcOp, returnType,
          rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
              srcOp.getLoc(), returnType, adaptor.getLhs()),
          rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
              srcOp.getLoc(), returnType, adaptor.getRhs()));
      break;
    case arith::CmpFPredicate::ORD: // !(isnan(lhs) || isnan(rhs))
      rewriter.replaceOpWithNewOp<IREE::VM::XorI32Op>(
          srcOp, returnType,
          rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1),
          rewriter.createOrFold<IREE::VM::AndI32Op>(
              srcOp.getLoc(), returnType,
              rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                  srcOp.getLoc(), returnType, adaptor.getLhs()),
              rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                  srcOp.getLoc(), returnType, adaptor.getRhs())));
      break;
    case arith::CmpFPredicate::OEQ: // ordered and equal
      rewriter.replaceOpWithNewOp<IREE::VM::CmpEQF32OOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::OGT: // ordered and greater than
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTF32OOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::OGE: // ordered and greater than or equal
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEF32OOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::OLT: // ordered and less than
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTF32OOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::OLE: // ordered and less than or equal
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEF32OOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::ONE: // ordered and not equal
      rewriter.replaceOpWithNewOp<IREE::VM::CmpNEF32OOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::UEQ: // unordered or equal
      rewriter.replaceOpWithNewOp<IREE::VM::CmpEQF32UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::UGT: // unordered or greater than
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTF32UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::UGE: // unordered or greater than or equal
      rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEF32UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::ULT: // unordered or less than
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTF32UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::ULE: // unordered or less than or equal
      rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEF32UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case arith::CmpFPredicate::UNE: // unordered or not equal
      rewriter.replaceOpWithNewOp<IREE::VM::CmpNEF32UOp>(
          srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
      break;
    default:
      return rewriter.notifyMatchFailure(srcOp,
                                         "unhandled arith::CmpFPredicate");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
struct UnaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
      return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
struct BinaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    switch (adaptor.getLhs().getType().getIntOrFloatBitWidth()) {
    case 32:
      rewriter.replaceOpWithNewOp<Dst32OpTy>(srcOp, adaptor.getLhs().getType(),
                                             adaptor.getLhs(),
                                             adaptor.getRhs());
      break;
    case 64:
      rewriter.replaceOpWithNewOp<Dst64OpTy>(srcOp, adaptor.getLhs().getType(),
                                             adaptor.getLhs(),
                                             adaptor.getRhs());
      break;
    default:
      return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
struct ShiftArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value amount = adaptor.getRhs();
    if (amount.getType().getIntOrFloatBitWidth() > 32) {
      // Shift amounts are always 32-bit in the VM.
      amount = rewriter.createOrFold<arith::TruncIOp>(
          srcOp.getLoc(), rewriter.getI32Type(), amount);
    }
    switch (adaptor.getLhs().getType().getIntOrFloatBitWidth()) {
    case 32:
      rewriter.replaceOpWithNewOp<Dst32OpTy>(srcOp, rewriter.getI32Type(),
                                             adaptor.getLhs(), amount);
      break;
    case 64:
      rewriter.replaceOpWithNewOp<Dst64OpTy>(srcOp, rewriter.getI64Type(),
                                             adaptor.getLhs(), amount);
      break;
    default:
      return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

template <typename OpTy, typename ExtOpTy>
struct IndexCastOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy srcOp, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(jpienaar): Audit and fix if needed.
    auto srcType = adaptor.getIn().getType();
    auto dstType =
        this->getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType == dstType) {
      rewriter.replaceOp(srcOp, adaptor.getOperands());
    } else if (srcType.getIntOrFloatBitWidth() <
               dstType.getIntOrFloatBitWidth()) {
      rewriter.replaceOpWithNewOp<ExtOpTy>(srcOp, dstType, adaptor.getIn());
    } else {
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(srcOp, dstType,
                                                   adaptor.getIn());
    }
    return success();
  }
};

struct ZeroExtendIOpConversion : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ExtUIOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isInteger(1)) {
      // NOTE: this may not be required - if we know that the i1 is never able
      // to have more than bit 0 manipulated then this is wasted work.
      auto maskedValue = rewriter.createOrFold<IREE::VM::AndI32Op>(
          srcOp.getLoc(), rewriter.getI32Type(), adaptor.getIn(),
          rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1));
      if (dstType.isInteger(32)) {
        rewriter.replaceOp(srcOp, maskedValue);
      } else if (dstType.isInteger(64)) {
        rewriter.replaceOpWithNewOp<IREE::VM::ExtI32I64UOp>(srcOp, dstType,
                                                            maskedValue);
      } else {
        return rewriter.notifyMatchFailure(srcOp,
                                           "unsupported i1 zero extension");
      }
    } else if (srcType.isInteger(8) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I32UOp>(srcOp, dstType,
                                                         adaptor.getIn());
    } else if (srcType.isInteger(8) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I64UOp>(srcOp, dstType,
                                                         adaptor.getIn());
    } else if (srcType.isInteger(16) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI16I32UOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(16) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI16I64UOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI32I64UOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported zero extension");
    }
    return success();
  }
};

struct SignExtendIOpConversion : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ExtSIOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isInteger(1)) {
      if (dstType.isInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::SelectI32Op>(
            srcOp, dstType, adaptor.getIn(),
            rewriter.create<IREE::VM::ConstI32Op>(srcOp.getLoc(), 0xFFFFFFFFu),
            rewriter.create<IREE::VM::ConstI32ZeroOp>(srcOp.getLoc()));
      } else if (dstType.isInteger(64)) {
        rewriter.replaceOpWithNewOp<IREE::VM::SelectI64Op>(
            srcOp, dstType, adaptor.getIn(),
            rewriter.create<IREE::VM::ConstI64Op>(srcOp.getLoc(),
                                                  0xFFFFFFFFFFFFFFFFull),
            rewriter.create<IREE::VM::ConstI64ZeroOp>(srcOp.getLoc()));
      } else {
        return rewriter.notifyMatchFailure(srcOp,
                                           "unsupported i1 sign extension");
      }
    } else if (srcType.isInteger(8) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I32SOp>(srcOp, dstType,
                                                         adaptor.getIn());
    } else if (srcType.isInteger(8) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I64SOp>(srcOp, dstType,
                                                         adaptor.getIn());
    } else if (srcType.isInteger(16) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI16I32SOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(16) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI16I64SOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI32I64SOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported sign extension");
    }
    return success();
  }
};

struct TruncateIOpConversion : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::TruncIOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto resultType = srcOp.getResult().getType();
    auto dstType = getTypeConverter()->convertType(resultType);
    if (resultType.isInteger(1)) {
      // i1 is represented as i32, so just mask off the bit and truncate as
      // normal. Note that if we started as i64 we need to first get that into
      // an i32 that we can work with.
      auto value = adaptor.getIn();
      if (srcType.isInteger(64)) {
        value = rewriter.createOrFold<IREE::VM::TruncI64I32Op>(srcOp.getLoc(),
                                                               dstType, value);
      }
      rewriter.replaceOpWithNewOp<IREE::VM::AndI32Op>(
          srcOp, dstType, value,
          rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1));
    } else if (srcType.isInteger(16) && resultType.isInteger(8)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI16I8Op>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(32) && resultType.isInteger(8)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI32I8Op>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(32) && resultType.isInteger(16)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI32I16Op>(srcOp, dstType,
                                                           adaptor.getIn());
    } else if (srcType.isInteger(64) && resultType.isInteger(8)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI64I8Op>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(64) && resultType.isInteger(16)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI64I16Op>(srcOp, dstType,
                                                           adaptor.getIn());
    } else if (srcType.isInteger(64) && resultType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI64I32Op>(srcOp, dstType,
                                                           adaptor.getIn());
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported truncation");
    }
    return success();
  }
};

template <typename OpTy, typename ExtOpTy, typename CastOpTy>
struct IntToFPOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy srcOp, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = srcOp.getResult().getType();
    if (!dstType.isF32() ||
        !(srcType.isSignedInteger() || srcType.isSignlessInteger())) {
      return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    Value input = srcOp.getIn();
    if (!(srcType.isSignlessInteger(32) || srcType.isSignedInteger(32))) {
      if (srcType.getIntOrFloatBitWidth() < 32) {
        input = rewriter.create<ExtOpTy>(
            srcOp.getLoc(), IntegerType::get(this->getContext(), 32), input);
      } else {
        return rewriter.notifyMatchFailure(srcOp, "unsupported type");
      }
    }

    auto resultType = this->getTypeConverter()->convertType(dstType);
    rewriter.replaceOpWithNewOp<CastOpTy>(srcOp, resultType, input);
    return success();
  }
};

struct FPToSIOpConversion : public OpConversionPattern<arith::FPToSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::FPToSIOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = srcOp.getResult().getType();
    auto resultType = getTypeConverter()->convertType(dstType);
    if (srcType.isF32()) {
      // This uses the resultType rather than dstType as any truncation
      // required will be handled via interpretation by consumer.
      if (resultType.isSignlessInteger(32) || resultType.isSignedInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastF32SI32Op>(srcOp, resultType,
                                                             adaptor.getIn());
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

struct FPToUIOpConversion : public OpConversionPattern<arith::FPToUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::FPToUIOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = srcOp.getResult().getType();
    auto resultType = getTypeConverter()->convertType(dstType);
    if (srcType.isF32()) {
      if (dstType.isSignlessInteger(32) || dstType.isUnsignedInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastF32UI32Op>(srcOp, resultType,
                                                             adaptor.getIn());
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

struct BitcastOpConversion : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::BitcastOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = srcOp.getResult().getType();
    auto resultType =
        getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isF32() && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastF32I32Op>(
          srcOp, resultType, adaptor.getOperands()[0]);
    } else if (srcType.isInteger(32) && dstType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastI32F32Op>(
          srcOp, resultType, adaptor.getOperands()[0]);
    } else if (srcType.isF64() && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastF64I64Op>(
          srcOp, resultType, adaptor.getOperands()[0]);
    } else if (srcType.isInteger(64) && dstType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastI64F64Op>(
          srcOp, resultType, adaptor.getOperands()[0]);
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported bitcast");
    }
    return success();
  }
};

struct SelectOpConversion : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto valueType = adaptor.getTrueValue().getType();
    if (valueType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectI32Op>(
          srcOp, valueType, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    } else if (valueType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectI64Op>(
          srcOp, valueType, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    } else if (valueType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectF32Op>(
          srcOp, valueType, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    } else if (valueType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectF64Op>(
          srcOp, valueType, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    } else if (llvm::isa<IREE::VM::RefType>(valueType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectRefOp>(
          srcOp, valueType, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    } else {
      return rewriter.notifyMatchFailure(srcOp,
                                         "unsupported select element type");
    }
  }
};

} // namespace

void populateArithToVMPatterns(MLIRContext *context,
                               TypeConverter &typeConverter,
                               RewritePatternSet &patterns) {
  // TODO(#2878): figure out how to pass the type converter in a supported way.
  // Right now if we pass the type converter as the first argument - triggering
  // the ConversionPattern stuff - it'll do weird things.
  patterns.insert<ConstantOpConversion>(context, typeConverter);

  // Comparison.
  patterns.insert<CmpI32OpConversion, CmpI64OpConversion, CmpF32OpConversion,
                  SelectOpConversion>(typeConverter, context);

  // Casting and conversion.
  patterns.insert<IndexCastOpConversion<arith::IndexCastOp, arith::ExtSIOp>,
                  IndexCastOpConversion<arith::IndexCastUIOp, arith::ExtUIOp>,
                  ZeroExtendIOpConversion, SignExtendIOpConversion,
                  TruncateIOpConversion>(typeConverter, context);

  // Integer arithmetic ops.
  patterns
      .insert<BinaryArithmeticOpConversion<arith::AddIOp, IREE::VM::AddI32Op,
                                           IREE::VM::AddI64Op>,
              BinaryArithmeticOpConversion<arith::DivSIOp, IREE::VM::DivI32SOp,
                                           IREE::VM::DivI64SOp>,
              BinaryArithmeticOpConversion<arith::DivUIOp, IREE::VM::DivI32UOp,
                                           IREE::VM::DivI64UOp>,
              BinaryArithmeticOpConversion<arith::MulIOp, IREE::VM::MulI32Op,
                                           IREE::VM::MulI64Op>,
              BinaryArithmeticOpConversion<arith::RemSIOp, IREE::VM::RemI32SOp,
                                           IREE::VM::RemI64SOp>,
              BinaryArithmeticOpConversion<arith::RemUIOp, IREE::VM::RemI32UOp,
                                           IREE::VM::RemI64UOp>,
              BinaryArithmeticOpConversion<arith::MinSIOp, IREE::VM::MinI32SOp,
                                           IREE::VM::MinI64SOp>,
              BinaryArithmeticOpConversion<arith::MinUIOp, IREE::VM::MinI32UOp,
                                           IREE::VM::MinI64UOp>,
              BinaryArithmeticOpConversion<arith::MaxSIOp, IREE::VM::MaxI32SOp,
                                           IREE::VM::MaxI64SOp>,
              BinaryArithmeticOpConversion<arith::MaxUIOp, IREE::VM::MaxI32UOp,
                                           IREE::VM::MaxI64UOp>,
              BinaryArithmeticOpConversion<arith::SubIOp, IREE::VM::SubI32Op,
                                           IREE::VM::SubI64Op>,
              BinaryArithmeticOpConversion<arith::AndIOp, IREE::VM::AndI32Op,
                                           IREE::VM::AndI64Op>,
              BinaryArithmeticOpConversion<arith::OrIOp, IREE::VM::OrI32Op,
                                           IREE::VM::OrI64Op>,
              BinaryArithmeticOpConversion<arith::XOrIOp, IREE::VM::XorI32Op,
                                           IREE::VM::XorI64Op>>(typeConverter,
                                                                context);

  // Floating-point arithmetic ops.
  patterns.insert<
      BinaryArithmeticOpConversion<arith::AddFOp, IREE::VM::AddF32Op,
                                   IREE::VM::AddF64Op>,
      BinaryArithmeticOpConversion<arith::DivFOp, IREE::VM::DivF32Op,
                                   IREE::VM::DivF64Op>,
      BinaryArithmeticOpConversion<arith::MulFOp, IREE::VM::MulF32Op,
                                   IREE::VM::MulF64Op>,
      UnaryArithmeticOpConversion<arith::NegFOp, IREE::VM::NegF32Op,
                                  IREE::VM::NegF64Op>,
      BinaryArithmeticOpConversion<arith::RemFOp, IREE::VM::RemF32Op,
                                   IREE::VM::RemF64Op>,
      BinaryArithmeticOpConversion<arith::SubFOp, IREE::VM::SubF32Op,
                                   IREE::VM::SubF64Op>,
      BinaryArithmeticOpConversion<arith::MinimumFOp, IREE::VM::MinF32Op,
                                   IREE::VM::MinF64Op>,
      BinaryArithmeticOpConversion<arith::MinNumFOp, IREE::VM::MinF32Op,
                                   IREE::VM::MinF64Op>,
      BinaryArithmeticOpConversion<arith::MaximumFOp, IREE::VM::MaxF32Op,
                                   IREE::VM::MaxF64Op>,
      BinaryArithmeticOpConversion<arith::MaxNumFOp, IREE::VM::MaxF32Op,
                                   IREE::VM::MaxF64Op>>(typeConverter, context);

  // Floating-point conversion ops.
  patterns.insert<IntToFPOpConversion<arith::SIToFPOp, arith::ExtSIOp,
                                      IREE::VM::CastSI32F32Op>,
                  IntToFPOpConversion<arith::UIToFPOp, arith::ExtUIOp,
                                      IREE::VM::CastUI32F32Op>,
                  FPToSIOpConversion, FPToUIOpConversion, BitcastOpConversion>(
      typeConverter, context);

  // Shift ops.
  patterns
      .insert<ShiftArithmeticOpConversion<arith::ShLIOp, IREE::VM::ShlI32Op,
                                          IREE::VM::ShlI64Op>,
              ShiftArithmeticOpConversion<arith::ShRSIOp, IREE::VM::ShrI32SOp,
                                          IREE::VM::ShrI64SOp>,
              ShiftArithmeticOpConversion<arith::ShRUIOp, IREE::VM::ShrI32UOp,
                                          IREE::VM::ShrI64UOp>>(typeConverter,
                                                                context);
}

} // namespace mlir::iree_compiler
