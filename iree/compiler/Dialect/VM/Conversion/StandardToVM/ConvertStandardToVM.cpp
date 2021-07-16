// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"

#include "iree/base/api.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

class ModuleOpConversion : public OpConversionPattern<ModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ModuleOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Do not attempt to convert the top level module.
    // This mechanism can only support rewriting non top-level modules.
    if (!srcOp->getParentOp() || !isa<ModuleOp>(srcOp->getParentOp())) {
      return failure();
    }

    StringRef name = srcOp.getName() ? *srcOp.getName() : "module";
    auto newModuleOp =
        rewriter.create<IREE::VM::ModuleOp>(srcOp.getLoc(), name);
    assert(!newModuleOp.getBodyRegion().empty());
    Block *firstCreatedBlock = &newModuleOp.getBodyRegion().front();
    rewriter.inlineRegionBefore(srcOp.getBodyRegion(), firstCreatedBlock);
    auto blockRange = llvm::make_range(Region::iterator(firstCreatedBlock),
                                       newModuleOp.getBodyRegion().end());
    for (Block &block : llvm::make_early_inc_range(blockRange)) {
      rewriter.eraseBlock(&block);
    }
    rewriter.replaceOp(srcOp, {});
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(&newModuleOp.getBodyRegion().front());
    rewriter.create<IREE::VM::ModuleTerminatorOp>(srcOp.getLoc());
    return success();
  }
};

// Allowlist of function attributes to retain when converting to vm.func.
constexpr const char *kRetainedAttributes[] = {
    "iree.reflection",
    "sym_visibility",
    "noinline",
};

class FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FuncOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FunctionType srcFuncType = srcOp.getType();
    TypeConverter::SignatureConversion signatureConversion(
        srcOp.getNumArguments());

    // Convert function arguments.
    for (unsigned i = 0, e = srcFuncType.getNumInputs(); i < e; ++i) {
      if (failed(getTypeConverter()->convertSignatureArg(
              i, srcFuncType.getInput(i), signatureConversion))) {
        return rewriter.notifyMatchFailure(srcOp, "argument failed to convert");
      }
    }

    // Convert function results.
    SmallVector<Type, 1> convertedResultTypes;
    if (failed(getTypeConverter()->convertTypes(srcFuncType.getResults(),
                                                convertedResultTypes))) {
      return rewriter.notifyMatchFailure(srcOp, "results failed to convert");
    }

    // Create new function with converted argument and result types.
    // Note that attributes are dropped. Consider preserving some if needed.
    auto newFuncType = mlir::FunctionType::get(
        srcOp.getContext(), signatureConversion.getConvertedTypes(),
        convertedResultTypes);
    auto newFuncOp = rewriter.create<IREE::VM::FuncOp>(
        srcOp.getLoc(), srcOp.getName(), newFuncType);
    rewriter.inlineRegionBefore(srcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Retain function attributes in the allowlist.
    auto retainedAttributes = ArrayRef<const char *>(
        kRetainedAttributes,
        sizeof(kRetainedAttributes) / sizeof(kRetainedAttributes[0]));
    for (auto retainAttrName : retainedAttributes) {
      StringRef attrName(retainAttrName);
      Attribute attr = srcOp->getAttr(attrName);
      if (attr) {
        newFuncOp->setAttr(attrName, attr);
      }
    }

    // Tell the rewriter to convert the region signature.
    TypeConverter &typeConverter = *getTypeConverter();
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    // Also add an export for the "raw" form of this function, which operates
    // on low level VM types and does no verification. A later pass will
    // materialize high level API-friendly wrappers.
    if (srcOp.isPublic()) {
      StringRef exportName = newFuncOp.getName();
      rewriter.create<IREE::VM::ExportOp>(srcOp.getLoc(), newFuncOp,
                                          exportName);
    }

    rewriter.replaceOp(srcOp, llvm::None);
    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<mlir::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::ReturnOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ReturnOp>(srcOp, operands);
    return success();
  }
};

struct ConstantOpConversion : public OpConversionPattern<ConstantOp> {
  ConstantOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  TypeConverter &typeConverter;

  LogicalResult matchAndRewrite(
      ConstantOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto targetType = typeConverter.convertType(srcOp.getType());
    if (targetType.isa<IntegerType>()) {
      auto integerAttr = srcOp.getValue().dyn_cast<IntegerAttr>();
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
    } else if (targetType.isa<FloatType>()) {
      auto floatAttr = srcOp.getValue().dyn_cast<FloatAttr>();
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

class CmpIOpConversion : public OpConversionPattern<CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CmpIOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CmpIOp::Adaptor srcAdaptor(operands);
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
      case CmpIPredicate::eq:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQI32Op>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::ne:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEI32Op>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::slt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32SOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::sle:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32SOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::sgt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32SOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::sge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32SOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::ult:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::ule:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::ugt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::uge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      default:
        return failure();
    }
  }
};

class CmpFOpConversion : public OpConversionPattern<CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CmpFOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CmpFOp::Adaptor srcAdaptor(operands);
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
      case CmpFPredicate::AlwaysFalse:  // 0
        rewriter.replaceOpWithNewOp<IREE::VM::ConstI32ZeroOp>(srcOp);
        break;
      case CmpFPredicate::AlwaysTrue:  // 1
        rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(srcOp, 1);
        break;
      case CmpFPredicate::UNO:  // isnan(lhs) || isnan(rhs)
        rewriter.replaceOpWithNewOp<IREE::VM::OrI32Op>(
            srcOp, returnType,
            rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                srcOp.getLoc(), returnType, srcAdaptor.lhs()),
            rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                srcOp.getLoc(), returnType, srcAdaptor.rhs()));
        break;
      case CmpFPredicate::ORD:  // !(isnan(lhs) || isnan(rhs))
        rewriter.replaceOpWithNewOp<IREE::VM::XorI32Op>(
            srcOp, returnType,
            rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1),
            rewriter.createOrFold<IREE::VM::AndI32Op>(
                srcOp.getLoc(), returnType,
                rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                    srcOp.getLoc(), returnType, srcAdaptor.lhs()),
                rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                    srcOp.getLoc(), returnType, srcAdaptor.rhs())));
        break;
      case CmpFPredicate::OEQ:  // ordered and equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::OGT:  // ordered and greater than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::OGE:  // ordered and greater than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::OLT:  // ordered and less than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::OLE:  // ordered and less than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::ONE:  // ordered and not equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::UEQ:  // unordered or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::UGT:  // unordered or greater than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::UGE:  // unordered or greater than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::ULT:  // unordered or less than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::ULE:  // unordered or less than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::UNE:  // unordered or not equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      default:
        return rewriter.notifyMatchFailure(srcOp, "unhandled CmpFPredicate");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class UnaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdaptor(operands);
    switch (srcAdaptor.operand().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(
            srcOp, srcAdaptor.operand().getType(), srcAdaptor.operand());
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(
            srcOp, srcAdaptor.operand().getType(), srcAdaptor.operand());
        break;
      default:
        return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class BinaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdaptor(operands);
    switch (srcAdaptor.lhs().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(
            srcOp, srcAdaptor.lhs().getType(), srcAdaptor.lhs(),
            srcAdaptor.rhs());
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(
            srcOp, srcAdaptor.lhs().getType(), srcAdaptor.lhs(),
            srcAdaptor.rhs());
        break;
      default:
        return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class ShiftArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdaptor(operands);
    Value amount = srcAdaptor.rhs();
    if (amount.getType().getIntOrFloatBitWidth() > 32) {
      // Shift amounts are always 32-bit in the VM.
      amount = rewriter.createOrFold<TruncateIOp>(
          srcOp.getLoc(), rewriter.getI32Type(), amount);
    }
    switch (srcAdaptor.lhs().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(srcOp, srcOp.getType(),
                                               srcAdaptor.lhs(), amount);
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(srcOp, srcOp.getType(),
                                               srcAdaptor.lhs(), amount);
        break;
      default:
        return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

template <typename StdOp>
class CastingOpConversion : public OpConversionPattern<StdOp> {
  using OpConversionPattern<StdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      StdOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(srcOp, operands);
    return success();
  }
};

class IndexCastOpConversion : public OpConversionPattern<IndexCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IndexCastOp srcOp, ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    IndexCastOpAdaptor operands(rawOperands);
    auto srcType = operands.in().getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType == dstType) {
      rewriter.replaceOp(srcOp, rawOperands);
    } else if (srcType.getIntOrFloatBitWidth() <
               dstType.getIntOrFloatBitWidth()) {
      rewriter.replaceOpWithNewOp<ZeroExtendIOp>(srcOp, dstType, operands.in());
    } else {
      rewriter.replaceOpWithNewOp<TruncateIOp>(srcOp, dstType, operands.in());
    }
    return success();
  }
};

class ZeroExtendIOpConversion : public OpConversionPattern<ZeroExtendIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ZeroExtendIOp srcOp, ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    ZeroExtendIOpAdaptor operands(rawOperands);
    auto srcType = srcOp.value().getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isInteger(1) && dstType.isInteger(32)) {
      // This may not be needed but ensures that the input was treated as a
      // single bit.
      // NOTE: this may not be required - if we know that the i1 is never able
      // to have more than bit 0 manipulated then this is wasted work.
      rewriter.replaceOpWithNewOp<IREE::VM::AndI32Op>(
          srcOp, dstType, operands.value(),
          rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1));
    } else if (srcType.isInteger(8) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I32UOp>(srcOp, dstType,
                                                         operands.value());
    } else if (srcType.isInteger(16) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI16I32UOp>(srcOp, dstType,
                                                          operands.value());
    } else if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI32I64UOp>(srcOp, dstType,
                                                          operands.value());
    } else {
      // TODO(benvanik): we should be building a sequence of extensions for
      // things like i8 -> i64.
      return rewriter.notifyMatchFailure(srcOp, "unsupported zero extension");
    }
    return success();
  }
};

class SignExtendIOpConversion : public OpConversionPattern<SignExtendIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SignExtendIOp srcOp, ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    SignExtendIOpAdaptor operands(rawOperands);
    auto srcType = srcOp.value().getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isInteger(8) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I32SOp>(srcOp, dstType,
                                                         operands.value());
    } else if (srcType.isInteger(16) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI16I32SOp>(srcOp, dstType,
                                                          operands.value());
    } else if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI32I64SOp>(srcOp, dstType,
                                                          operands.value());
    } else {
      // TODO(benvanik): we should be building a sequence of extensions for
      // things like i8 -> i64.
      return rewriter.notifyMatchFailure(srcOp, "unsupported sign extension");
    }
    return success();
  }
};

class TruncateIOpConversion : public OpConversionPattern<TruncateIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TruncateIOp srcOp, ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    TruncateIOpAdaptor operands(rawOperands);
    auto srcType = srcOp.value().getType();
    auto resultType = srcOp.getResult().getType();
    auto dstType = getTypeConverter()->convertType(resultType);
    if (resultType.isInteger(1)) {
      // i1 is represented as i32, so just mask off the bit and truncate as
      // normal. Note that if we started as i64 we need to first get that into
      // an i32 that we can work with.
      auto value = operands.value();
      if (srcType.isInteger(64)) {
        value = rewriter.createOrFold<IREE::VM::TruncI64I32Op>(srcOp.getLoc(),
                                                               dstType, value);
      }
      rewriter.replaceOpWithNewOp<IREE::VM::AndI32Op>(
          srcOp, dstType, value,
          rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1));
    } else if (srcType.isInteger(32) && resultType.isInteger(8)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI32I8Op>(srcOp, dstType,
                                                          operands.value());
    } else if (srcType.isInteger(32) && resultType.isInteger(16)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI32I16Op>(srcOp, dstType,
                                                           operands.value());
    } else if (srcType.isInteger(64) && resultType.isInteger(8)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI64I8Op>(srcOp, dstType,
                                                          operands.value());
    } else if (srcType.isInteger(64) && resultType.isInteger(16)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI64I16Op>(srcOp, dstType,
                                                           operands.value());
    } else if (srcType.isInteger(64) && resultType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI64I32Op>(srcOp, dstType,
                                                           operands.value());
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported truncation");
    }
    return success();
  }
};

class SIToFPOpConversion : public OpConversionPattern<SIToFPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SIToFPOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SIToFPOpAdaptor srcAdaptor(operands);
    auto srcType = operands[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isSignlessInteger(32) || srcType.isSignedInteger(32)) {
      if (dstType.isF32()) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastSI32F32Op>(srcOp, dstType,
                                                             operands[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class UIToFPOpConversion : public OpConversionPattern<UIToFPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UIToFPOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    UIToFPOpAdaptor srcAdaptor(operands);
    auto srcType = operands[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isUnsignedInteger(32)) {
      if (dstType.isF32()) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastUI32F32Op>(srcOp, dstType,
                                                             operands[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class FPToSIOpConversion : public OpConversionPattern<FPToSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FPToSIOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FPToSIOpAdaptor srcAdaptor(operands);
    auto srcType = operands[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isF32()) {
      if (dstType.isSignlessInteger(32) || dstType.isSignedInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastF32SI32Op>(srcOp, dstType,
                                                             operands[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class FPToUIOpConversion : public OpConversionPattern<FPToUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FPToUIOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FPToUIOpAdaptor srcAdaptor(operands);
    auto srcType = operands[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isF32()) {
      if (srcType.isUnsignedInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastF32UI32Op>(srcOp, dstType,
                                                             operands[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class SelectOpConversion : public OpConversionPattern<SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SelectOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SelectOp::Adaptor srcAdaptor(operands);
    auto valueType = srcAdaptor.true_value().getType();
    if (valueType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectI32Op>(
          srcOp, valueType, srcAdaptor.condition(), srcAdaptor.true_value(),
          srcAdaptor.false_value());
      return success();
    } else if (valueType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectI64Op>(
          srcOp, valueType, srcAdaptor.condition(), srcAdaptor.true_value(),
          srcAdaptor.false_value());
      return success();
    } else if (valueType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectF32Op>(
          srcOp, valueType, srcAdaptor.condition(), srcAdaptor.true_value(),
          srcAdaptor.false_value());
      return success();
    } else if (valueType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectF64Op>(
          srcOp, valueType, srcAdaptor.condition(), srcAdaptor.true_value(),
          srcAdaptor.false_value());
      return success();
    } else if (valueType.isa<IREE::VM::RefType>()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectRefOp>(
          srcOp, valueType, srcAdaptor.condition(), srcAdaptor.true_value(),
          srcAdaptor.false_value());
      return success();
    } else {
      return rewriter.notifyMatchFailure(srcOp,
                                         "unsupported select element type");
    }
  }
};

class AssertOpConversion : public OpConversionPattern<AssertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AssertOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    AssertOpAdaptor adaptor(operands);
    Location loc = srcOp.getLoc();

    // Start by splitting the block containing the assert into two. The part
    // before will contain the condition, and the part after will contain
    // the continuation point.
    Block *condBlock = rewriter.getInsertionBlock();
    Block::iterator opPosition = rewriter.getInsertionPoint();
    Block *continuationBlock = rewriter.splitBlock(condBlock, opPosition);

    // Create a new block for the target of the failure.
    Block *failureBlock;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Region *parentRegion = condBlock->getParent();
      failureBlock = rewriter.createBlock(parentRegion, parentRegion->end());
      auto status = rewriter.create<IREE::VM::ConstI32Op>(
          loc, rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                       IREE_STATUS_FAILED_PRECONDITION));
      rewriter.create<IREE::VM::FailOp>(loc, status, srcOp.msgAttr());
    }

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.replaceOpWithNewOp<CondBranchOp>(srcOp, adaptor.arg(),
                                              continuationBlock, failureBlock);
    return success();
  }
};

class BranchOpConversion : public OpConversionPattern<BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BranchOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::BranchOp>(srcOp, srcOp.getDest(),
                                                    operands);
    return success();
  }
};

class CondBranchOpConversion : public OpConversionPattern<CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CondBranchOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Block *trueDest = srcOp.getTrueDest();
    rewriter.replaceOpWithNewOp<IREE::VM::CondBranchOp>(
        srcOp, operands[0], trueDest,
        operands.slice(1, trueDest->getNumArguments()), srcOp.getFalseDest(),
        operands.slice(1 + trueDest->getNumArguments()));
    return success();
  }
};

class CallOpConversion : public OpConversionPattern<CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CallOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CallOp::Adaptor srcAdaptor(operands);
    // Convert function result types. The conversion framework will ensure
    // that the callee has been equivalently converted.
    SmallVector<Type, 4> resultTypes;
    for (auto resultType : srcOp.getResultTypes()) {
      resultType = getTypeConverter()->convertType(resultType);
      if (!resultType) {
        return failure();
      }
      resultTypes.push_back(resultType);
    }
    rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        srcOp, srcOp.getCallee(), resultTypes, srcAdaptor.operands());

    return success();
  }
};

}  // namespace

void populateStandardToVMPatterns(MLIRContext *context,
                                  TypeConverter &typeConverter,
                                  OwningRewritePatternList &patterns) {
  patterns.insert<AssertOpConversion, BranchOpConversion, CallOpConversion,
                  CmpIOpConversion, CmpFOpConversion, CondBranchOpConversion,
                  ModuleOpConversion, FuncOpConversion, ReturnOpConversion,
                  SelectOpConversion>(typeConverter, context);

  // TODO(#2878): figure out how to pass the type converter in a supported way.
  // Right now if we pass the type converter as the first argument - triggering
  // the ConversionPattern stuff - it'll do weird things.
  patterns.insert<ConstantOpConversion>(context, typeConverter);

  patterns.insert<CastingOpConversion<UnrealizedConversionCastOp>,
                  IndexCastOpConversion, ZeroExtendIOpConversion,
                  SignExtendIOpConversion, TruncateIOpConversion>(typeConverter,
                                                                  context);

  // Integer arithmetic ops.
  patterns.insert<
      BinaryArithmeticOpConversion<AddIOp, IREE::VM::AddI32Op,
                                   IREE::VM::AddI64Op>,
      BinaryArithmeticOpConversion<SignedDivIOp, IREE::VM::DivI32SOp,
                                   IREE::VM::DivI64SOp>,
      BinaryArithmeticOpConversion<UnsignedDivIOp, IREE::VM::DivI32UOp,
                                   IREE::VM::DivI64UOp>,
      BinaryArithmeticOpConversion<MulIOp, IREE::VM::MulI32Op,
                                   IREE::VM::MulI64Op>,
      BinaryArithmeticOpConversion<SignedRemIOp, IREE::VM::RemI32SOp,
                                   IREE::VM::RemI64SOp>,
      BinaryArithmeticOpConversion<UnsignedRemIOp, IREE::VM::RemI32UOp,
                                   IREE::VM::RemI64UOp>,
      BinaryArithmeticOpConversion<SubIOp, IREE::VM::SubI32Op,
                                   IREE::VM::SubI64Op>,
      BinaryArithmeticOpConversion<AndOp, IREE::VM::AndI32Op,
                                   IREE::VM::AndI64Op>,
      BinaryArithmeticOpConversion<OrOp, IREE::VM::OrI32Op, IREE::VM::OrI64Op>,
      BinaryArithmeticOpConversion<XOrOp, IREE::VM::XorI32Op,
                                   IREE::VM::XorI64Op>>(typeConverter, context);

  // Floating-point arithmetic ops.
  patterns.insert<UnaryArithmeticOpConversion<AbsFOp, IREE::VM::AbsF32Op,
                                              IREE::VM::AbsF64Op>,
                  BinaryArithmeticOpConversion<AddFOp, IREE::VM::AddF32Op,
                                               IREE::VM::AddF64Op>,
                  UnaryArithmeticOpConversion<CeilFOp, IREE::VM::CeilF32Op,
                                              IREE::VM::CeilF64Op>,
                  UnaryArithmeticOpConversion<FloorFOp, IREE::VM::FloorF32Op,
                                              IREE::VM::FloorF64Op>,
                  BinaryArithmeticOpConversion<DivFOp, IREE::VM::DivF32Op,
                                               IREE::VM::DivF64Op>,
                  BinaryArithmeticOpConversion<MulFOp, IREE::VM::MulF32Op,
                                               IREE::VM::MulF64Op>,
                  UnaryArithmeticOpConversion<NegFOp, IREE::VM::NegF32Op,
                                              IREE::VM::NegF64Op>,
                  BinaryArithmeticOpConversion<RemFOp, IREE::VM::RemF32Op,
                                               IREE::VM::RemF64Op>,
                  BinaryArithmeticOpConversion<SubFOp, IREE::VM::SubF32Op,
                                               IREE::VM::SubF64Op>>(
      typeConverter, context);

  // Floating-point conversion ops.
  patterns.insert<SIToFPOpConversion, UIToFPOpConversion, FPToSIOpConversion,
                  FPToUIOpConversion>(typeConverter, context);

  // Shift ops.
  patterns.insert<
      ShiftArithmeticOpConversion<ShiftLeftOp, IREE::VM::ShlI32Op,
                                  IREE::VM::ShlI64Op>,
      ShiftArithmeticOpConversion<SignedShiftRightOp, IREE::VM::ShrI32SOp,
                                  IREE::VM::ShrI64SOp>,
      ShiftArithmeticOpConversion<UnsignedShiftRightOp, IREE::VM::ShrI32UOp,
                                  IREE::VM::ShrI64UOp>>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
