// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
      ModuleOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Do not attempt to convert the top level module.
    // This mechanism can only support rewriting non top-level modules.
    if (VMConversionTarget::isTopLevelModule(srcOp)) {
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
      FuncOp srcOp, OpAdaptor adaptor,
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
    // VM functions are private by default and exported via the dedicated
    // vm.export ops.
    newFuncOp.setPrivate();

    rewriter.replaceOp(srcOp, llvm::None);
    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<mlir::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::ReturnOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ReturnOp>(srcOp,
                                                    adaptor.getOperands());
    return success();
  }
};

struct ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  ConstantOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  TypeConverter &typeConverter;

  LogicalResult matchAndRewrite(
      arith::ConstantOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto targetType = typeConverter.convertType(srcOp.getType());
    if (!targetType) {
      return srcOp.emitError() << "could not convert type: " << srcOp.getType()
                               << " (check -iree-vm-target-* options)";
    }
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

class CmpIOpConversion : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::CmpIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
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

class CmpFOpConversion : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::CmpFOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
      case arith::CmpFPredicate::AlwaysFalse:  // 0
        rewriter.replaceOpWithNewOp<IREE::VM::ConstI32ZeroOp>(srcOp);
        break;
      case arith::CmpFPredicate::AlwaysTrue:  // 1
        rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(srcOp, 1);
        break;
      case arith::CmpFPredicate::UNO:  // isnan(lhs) || isnan(rhs)
        rewriter.replaceOpWithNewOp<IREE::VM::OrI32Op>(
            srcOp, returnType,
            rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                srcOp.getLoc(), returnType, adaptor.getLhs()),
            rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                srcOp.getLoc(), returnType, adaptor.getRhs()));
        break;
      case arith::CmpFPredicate::ORD:  // !(isnan(lhs) || isnan(rhs))
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
      case arith::CmpFPredicate::OEQ:  // ordered and equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::OGT:  // ordered and greater than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::OGE:  // ordered and greater than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::OLT:  // ordered and less than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::OLE:  // ordered and less than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::ONE:  // ordered and not equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::UEQ:  // unordered or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::UGT:  // unordered or greater than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::UGE:  // unordered or greater than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::ULT:  // unordered or less than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::ULE:  // unordered or less than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::UNE:  // unordered or not equal
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
class UnaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
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
class BinaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
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
        return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class ShiftArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value amount = adaptor.getRhs();
    if (amount.getType().getIntOrFloatBitWidth() > 32) {
      // Shift amounts are always 32-bit in the VM.
      amount = rewriter.createOrFold<arith::TruncIOp>(
          srcOp.getLoc(), rewriter.getI32Type(), amount);
    }
    switch (adaptor.getLhs().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(srcOp, srcOp.getType(),
                                               adaptor.getLhs(), amount);
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(srcOp, srcOp.getType(),
                                               adaptor.getLhs(), amount);
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
      StdOp srcOp, typename StdOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(srcOp, adaptor.getOperands());
    return success();
  }
};

class IndexCastOpConversion : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::IndexCastOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = adaptor.getIn().getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType == dstType) {
      rewriter.replaceOp(srcOp, adaptor.getOperands());
    } else if (srcType.getIntOrFloatBitWidth() <
               dstType.getIntOrFloatBitWidth()) {
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(srcOp, dstType,
                                                  adaptor.getIn());
    } else {
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(srcOp, dstType,
                                                   adaptor.getIn());
    }
    return success();
  }
};

class ZeroExtendIOpConversion : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::ExtUIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isInteger(1) && dstType.isInteger(32)) {
      // This may not be needed but ensures that the input was treated as a
      // single bit.
      // NOTE: this may not be required - if we know that the i1 is never able
      // to have more than bit 0 manipulated then this is wasted work.
      rewriter.replaceOpWithNewOp<IREE::VM::AndI32Op>(
          srcOp, dstType, adaptor.getIn(),
          rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1));
    } else if (srcType.isInteger(8) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I32UOp>(srcOp, dstType,
                                                         adaptor.getIn());
    } else if (srcType.isInteger(16) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI16I32UOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI32I64UOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else {
      // TODO(benvanik): we should be building a sequence of extensions for
      // things like i8 -> i64.
      return rewriter.notifyMatchFailure(srcOp, "unsupported zero extension");
    }
    return success();
  }
};

class SignExtendIOpConversion : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::ExtSIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isInteger(8) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I32SOp>(srcOp, dstType,
                                                         adaptor.getIn());
    } else if (srcType.isInteger(16) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI16I32SOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI32I64SOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else {
      // TODO(benvanik): we should be building a sequence of extensions for
      // things like i8 -> i64.
      return rewriter.notifyMatchFailure(srcOp, "unsupported sign extension");
    }
    return success();
  }
};

class TruncateIOpConversion : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::TruncIOp srcOp, OpAdaptor adaptor,
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

class SIToFPOpConversion : public OpConversionPattern<arith::SIToFPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::SIToFPOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = adaptor.getOperands()[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isSignlessInteger(32) || srcType.isSignedInteger(32)) {
      if (dstType.isF32()) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastSI32F32Op>(
            srcOp, dstType, adaptor.getOperands()[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class UIToFPOpConversion : public OpConversionPattern<arith::UIToFPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::UIToFPOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = adaptor.getOperands()[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isUnsignedInteger(32)) {
      if (dstType.isF32()) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastUI32F32Op>(
            srcOp, dstType, adaptor.getOperands()[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class FPToSIOpConversion : public OpConversionPattern<arith::FPToSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::FPToSIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = adaptor.getOperands()[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isF32()) {
      if (dstType.isSignlessInteger(32) || dstType.isSignedInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastF32SI32Op>(
            srcOp, dstType, adaptor.getOperands()[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class FPToUIOpConversion : public OpConversionPattern<arith::FPToUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::FPToUIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = adaptor.getOperands()[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isF32()) {
      if (srcType.isUnsignedInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastF32UI32Op>(
            srcOp, dstType, adaptor.getOperands()[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class BitcastOpConversion : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::BitcastOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = adaptor.getOperands()[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isF32() && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastF32I32Op>(
          srcOp, dstType, adaptor.getOperands()[0]);
    } else if (srcType.isInteger(32) && dstType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastI32F32Op>(
          srcOp, dstType, adaptor.getOperands()[0]);
    } else if (srcType.isF64() && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastF64I64Op>(
          srcOp, dstType, adaptor.getOperands()[0]);
    } else if (srcType.isInteger(64) && dstType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastI64F64Op>(
          srcOp, dstType, adaptor.getOperands()[0]);
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported bitcast");
    }
    return success();
  }
};

class SelectOpConversion : public OpConversionPattern<SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SelectOp srcOp, OpAdaptor adaptor,
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
    } else if (valueType.isa<IREE::VM::RefType>()) {
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

class AssertOpConversion : public OpConversionPattern<AssertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AssertOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto status = rewriter.create<IREE::VM::ConstI32Op>(
        srcOp.getLoc(),
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(32),
            static_cast<int32_t>(IREE::Util::StatusCode::FailedPrecondition)));
    // TODO(benvanik): invert cond_fail instead.
    auto invertedCondition = rewriter.createOrFold<IREE::VM::XorI32Op>(
        srcOp.getLoc(), adaptor.getArg().getType(), adaptor.getArg(),
        rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1));
    rewriter.replaceOpWithNewOp<IREE::VM::CondFailOp>(
        srcOp, invertedCondition, status, adaptor.getMsg().getValue());
    return success();
  }
};

class BranchOpConversion : public OpConversionPattern<BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BranchOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::BranchOp>(srcOp, srcOp.getDest(),
                                                    adaptor.getOperands());
    return success();
  }
};

class CondBranchOpConversion : public OpConversionPattern<CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CondBranchOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Block *trueDest = srcOp.getTrueDest();
    rewriter.replaceOpWithNewOp<IREE::VM::CondBranchOp>(
        srcOp, adaptor.getCondition(), trueDest, adaptor.getTrueDestOperands(),
        srcOp.getFalseDest(), adaptor.getFalseDestOperands());
    return success();
  }
};

class CallOpConversion : public OpConversionPattern<CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CallOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
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
        srcOp, srcOp.getCallee(), resultTypes, adaptor.operands());

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
  patterns
      .insert<UnaryArithmeticOpConversion<math::AbsOp, IREE::VM::AbsF32Op,
                                          IREE::VM::AbsF64Op>,
              BinaryArithmeticOpConversion<arith::AddFOp, IREE::VM::AddF32Op,
                                           IREE::VM::AddF64Op>,
              UnaryArithmeticOpConversion<math::CeilOp, IREE::VM::CeilF32Op,
                                          IREE::VM::CeilF64Op>,
              UnaryArithmeticOpConversion<math::FloorOp, IREE::VM::FloorF32Op,
                                          IREE::VM::FloorF64Op>,
              BinaryArithmeticOpConversion<arith::DivFOp, IREE::VM::DivF32Op,
                                           IREE::VM::DivF64Op>,
              BinaryArithmeticOpConversion<arith::MulFOp, IREE::VM::MulF32Op,
                                           IREE::VM::MulF64Op>,
              UnaryArithmeticOpConversion<arith::NegFOp, IREE::VM::NegF32Op,
                                          IREE::VM::NegF64Op>,
              BinaryArithmeticOpConversion<arith::RemFOp, IREE::VM::RemF32Op,
                                           IREE::VM::RemF64Op>,
              BinaryArithmeticOpConversion<arith::SubFOp, IREE::VM::SubF32Op,
                                           IREE::VM::SubF64Op>>(typeConverter,
                                                                context);

  // Floating-point conversion ops.
  patterns.insert<SIToFPOpConversion, UIToFPOpConversion, FPToSIOpConversion,
                  FPToUIOpConversion, BitcastOpConversion>(typeConverter,
                                                           context);

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

}  // namespace iree_compiler
}  // namespace mlir
