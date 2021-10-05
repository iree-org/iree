// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/IREE/IREEOps.h"
#include "iree-dialects/Dialect/IREEPyDM/IR/Ops.h"
#include "iree-dialects/Dialect/IREEPyDM/Transforms/ToIREE/Patterns.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::iree_pydm;

namespace arith_d = mlir;
namespace iree_d = mlir::iree;
namespace builtin_d = mlir;
namespace std_d = mlir;
namespace pydm_d = mlir::iree_pydm;

namespace {

enum class ExceptionCode : int {
  Success = 0,
  StopIteration = -1,
  StopAsyncIteration = -2,
  RuntimeError = -3,
  ValueError = -4,
  NotImplementedError = -5,
  KeyError = -6,
  IndexError = -7,
  AttributeError = -8,
  TypeError = -9,
  UnboundLocalError = -10,
};

}  // namespace

static Value getNullValue(Location loc, OpBuilder &builder, Type t) {
  return TypeSwitch<Type, Value>(t)
      .Case<iree_d::ListType>([&](auto t) -> Value {
        return builder.create<iree_d::NullOp>(loc, t);
      })
      .Default([&](Type t) -> Value {
        auto attr = builder.getZeroAttr(t);
        assert(attr && "could not get zero attr for builtin type");
        return builder.create<std_d::ConstantOp>(loc, t, attr);
      });
}

/// Creates a slow path block at the end of the function. The current block
/// will always dominate.
static Block *createSlowPathBlock(OpBuilder &builder) {
  Region *parentRegion = builder.getInsertionBlock()->getParent();
  return builder.createBlock(parentRegion, parentRegion->end());
}

static Type getVariantListType(Builder &builder) {
  return builder.getType<iree_d::ListType>(
      builder.getType<iree_d::VariantType>());
}

static Value getSuccessStatusValue(Location loc, OpBuilder &builder) {
  return builder.create<std_d::ConstantOp>(loc, builder.getI32IntegerAttr(0));
}

static Value getFailureStatusValue(Location loc, OpBuilder &builder,
                                   ExceptionCode code) {
  return builder.create<std_d::ConstantOp>(
      loc, builder.getI32IntegerAttr(static_cast<int>(code)));
}

static Value createUndefObjectList(Location loc, OpBuilder &builder) {
  return builder.create<iree_d::ListCreateOp>(loc, getVariantListType(builder),
                                              /*capacity=*/nullptr);
}

void resetObjectList(Location loc, OpBuilder &builder, Value list, int typeCode,
                     Value data) {
  // Note: The list can record optional runtime state at positions > 1, so
  // to truly reset, we have to resize. Low level optimizations should be able
  // to elide this if it turns out to be unnecessary.
  auto size = builder.create<std_d::ConstantOp>(loc, builder.getIndexAttr(2));
  builder.create<iree_d::ListResizeOp>(loc, list, size);
  auto index0 = builder.create<std_d::ConstantOp>(loc, builder.getIndexAttr(0));
  Value typeCodeValue = builder.create<std_d::ConstantOp>(
      loc, builder.getI32IntegerAttr(typeCode));
  builder.create<iree_d::ListSetOp>(loc, list, index0, typeCodeValue);
  auto index1 = builder.create<std_d::ConstantOp>(loc, builder.getIndexAttr(1));
  builder.create<iree_d::ListSetOp>(loc, list, index1, data);
}

static Value createObjectList(Location loc, OpBuilder &builder, int typeCode,
                              Value data) {
  auto list = createUndefObjectList(loc, builder);
  resetObjectList(loc, builder, list, typeCode, data);
  return list;
}

static Value castIntegerValue(Location loc, Value input,
                              builtin_d::IntegerType resultType,
                              OpBuilder &builder) {
  builtin_d::IntegerType inputType =
      input.getType().cast<builtin_d::IntegerType>();
  if (inputType.getWidth() == resultType.getWidth()) {
    return input;
  } else if (inputType.getWidth() < resultType.getWidth()) {
    return builder.create<arith_d::SignExtendIOp>(loc, resultType, input);
  } else {
    return builder.create<arith_d::TruncateIOp>(loc, resultType, input);
  }
}

static Optional<arith_d::CmpIPredicate> convertIntegerComparePredicate(
    StringAttr dunderName, bool isSigned, Builder &builder) {
  StringRef v = dunderName.getValue();
  if (v == "lt") {
    return isSigned ? arith_d::CmpIPredicate::slt : arith_d::CmpIPredicate::ult;
  } else if (v == "le") {
    return isSigned ? arith_d::CmpIPredicate::sle : arith_d::CmpIPredicate::ule;
  } else if (v == "eq" || v == "is") {
    return arith_d::CmpIPredicate::eq;
  } else if (v == "ne" || v == "isnot") {
    return arith_d::CmpIPredicate::ne;
  } else if (v == "gt") {
    return isSigned ? arith_d::CmpIPredicate::sgt : arith_d::CmpIPredicate::ugt;
  } else if (v == "ge") {
    return isSigned ? arith_d::CmpIPredicate::sge : arith_d::CmpIPredicate::uge;
  }

  return {};
}

static Optional<arith_d::CmpFPredicate> convertFpComparePredicate(
    StringAttr dunderName, Builder &builder) {
  StringRef v = dunderName.getValue();
  if (v == "lt") {
    return arith_d::CmpFPredicate::OLT;
  } else if (v == "le") {
    return arith_d::CmpFPredicate::OLE;
  } else if (v == "eq" || v == "is") {
    return arith_d::CmpFPredicate::OEQ;
  } else if (v == "ne" || v == "isnot") {
    return arith_d::CmpFPredicate::ONE;
  } else if (v == "gt") {
    return arith_d::CmpFPredicate::OGT;
  } else if (v == "ge") {
    return arith_d::CmpFPredicate::OGE;
  }

  return {};
}

namespace {

class AllocFreeVarOpConversion
    : public OpConversionPattern<pydm_d::AllocFreeVarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::AllocFreeVarOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO: We may want to initialize the list structurally in some way.
    // This will fail either way on read from unassigned variable, but we need
    // to see what works better for good runtime error messages.
    auto loc = srcOp.getLoc();
    Value list = createUndefObjectList(loc, rewriter);
    rewriter.replaceOp(srcOp, list);
    return success();
  }
};

class ApplyBinaryNumericConversion
    : public OpConversionPattern<pydm_d::ApplyBinaryOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::ApplyBinaryOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type leftType = adaptor.left().getType();
    Type rightType = adaptor.right().getType();
    Type resultType = typeConverter->convertType(srcOp.result().getType());
    if (!resultType || leftType != rightType || leftType != resultType) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "not same type operands/results");
    }
    if (leftType.isa<builtin_d::IntegerType>()) {
      bool isSigned = true;  // TODO: Unsigned.
      Value converted =
          convertIntegerOp(srcOp.getLoc(), adaptor.dunder_name().getValue(),
                           adaptor.left(), adaptor.right(), isSigned, rewriter);
      if (!converted)
        return rewriter.notifyMatchFailure(srcOp, "unsupported operation");
      rewriter.replaceOp(srcOp, converted);
      return success();
    } else if (leftType.isa<builtin_d::FloatType>()) {
      // TODO: Implement float binary
      return rewriter.notifyMatchFailure(srcOp, "unsupported operation");
    }

    return rewriter.notifyMatchFailure(srcOp, "non numeric type");
  }

  Value convertIntegerOp(Location loc, StringRef dunderName, Value left,
                         Value right, bool isSigned,
                         ConversionPatternRewriter &rewriter) const {
    // TODO: matmul, truediv, floordiv, mod, divmod, pow
    if (dunderName == "add") {
      return rewriter.create<arith_d::AddIOp>(loc, left, right);
    } else if (dunderName == "and") {
      return rewriter.create<arith_d::AndOp>(loc, left, right);
    } else if (dunderName == "mul") {
      return rewriter.create<arith_d::MulIOp>(loc, left, right);
    } else if (dunderName == "lshift") {
      return rewriter.create<arith_d::ShiftLeftOp>(loc, left, right);
    } else if (dunderName == "or") {
      return rewriter.create<arith_d::OrOp>(loc, left, right);
    } else if (dunderName == "rshift") {
      if (isSigned)
        return rewriter.create<arith_d::SignedShiftRightOp>(loc, left, right);
      else
        return rewriter.create<arith_d::UnsignedShiftRightOp>(loc, left, right);
    } else if (dunderName == "sub") {
      return rewriter.create<arith_d::SubIOp>(loc, left, right);
    } else if (dunderName == "xor") {
      return rewriter.create<arith_d::XOrOp>(loc, left, right);
    }
    return nullptr;
  }
};

class ApplyCompareNumericConversion
    : public OpConversionPattern<pydm_d::ApplyCompareOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::ApplyCompareOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type leftType = adaptor.left().getType();
    Type rightType = adaptor.right().getType();
    if (leftType != rightType) {
      return rewriter.notifyMatchFailure(srcOp, "not same type operands");
    }
    if (leftType.isa<builtin_d::IntegerType>()) {
      bool isSigned = true;  // TODO: Unsigned.
      auto predicate = convertIntegerComparePredicate(adaptor.dunder_name(),
                                                      isSigned, rewriter);
      if (!predicate)
        return rewriter.notifyMatchFailure(srcOp, "unsupported predicate");
      rewriter.replaceOpWithNewOp<arith_d::CmpIOp>(
          srcOp, *predicate, adaptor.left(), adaptor.right());
      return success();
    } else if (leftType.isa<builtin_d::FloatType>()) {
      auto predicate =
          convertFpComparePredicate(adaptor.dunder_name(), rewriter);
      if (!predicate)
        return rewriter.notifyMatchFailure(srcOp, "unsupported predicate");
      rewriter.replaceOpWithNewOp<arith_d::CmpFOp>(
          srcOp, *predicate, adaptor.left(), adaptor.right());
      return success();
    }

    return rewriter.notifyMatchFailure(srcOp, "non numeric type");
  }
};

class BoolToPredConversion : public OpConversionPattern<pydm_d::BoolToPredOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::BoolToPredOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    pydm_d::BoolToPredOp::Adaptor adaptor(operands);
    rewriter.replaceOp(srcOp, adaptor.value());
    return success();
  }
};

class BoxOpConversion : public OpConversionPattern<pydm_d::BoxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::BoxOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    auto origType =
        srcOp.primitive().getType().dyn_cast<pydm_d::PythonTypeInterface>();
    if (!origType)
      return rewriter.notifyMatchFailure(srcOp,
                                         "not a PythonTypeInterface type");
    auto typeCode = origType.getTypeCode();
    auto list = createObjectList(loc, rewriter, static_cast<int>(typeCode),
                                 operands[0]);
    rewriter.replaceOp(srcOp, list);
    return success();
  }
};

class CallOpConversion : public OpConversionPattern<pydm_d::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::CallOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    pydm_d::CallOp::Adaptor adaptor(operands);
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(srcOp.getResultTypes(),
                                                resultTypes))) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "result types could not be converted");
    }
    rewriter.replaceOpWithNewOp<std_d::CallOp>(srcOp, srcOp.callee(),
                                               resultTypes, adaptor.operands());
    return success();
  }
};

class ConstantOpConversion : public OpConversionPattern<pydm_d::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::ConstantOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    Type resultType = typeConverter->convertType(srcOp.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          srcOp, "constant type could not be converted");
    Attribute newValue = adaptor.value();
    // Fixup widths of integer types that may be wider/narrower than the
    // stored attribute (which tends to be stored in high precision in pydm
    // constants).
    TypeSwitch<Type>(resultType)
        .Case([&](builtin_d::IntegerType t) {
          APInt intValue =
              newValue.cast<IntegerAttr>().getValue().sextOrTrunc(t.getWidth());
          newValue = rewriter.getIntegerAttr(t, intValue);
        })
        .Case([&](builtin_d::FloatType t) {
          APFloat fpValue = newValue.cast<FloatAttr>().getValue();
          if (APFloat::SemanticsToEnum(fpValue.getSemantics()) !=
              APFloat::SemanticsToEnum(t.getFloatSemantics())) {
            // Convert.
            APFloat newFpValue = fpValue;
            bool losesInfo;
            newFpValue.convert(t.getFloatSemantics(),
                               APFloat::rmNearestTiesToEven, &losesInfo);
            if (losesInfo) {
              emitWarning(loc) << "conversion of " << newValue << " to " << t
                               << " loses information";
            }
            newValue = rewriter.getFloatAttr(t, newFpValue);
          }
        });

    if (!newValue)
      return rewriter.notifyMatchFailure(
          srcOp, "constant cannot be represented as a standard constant");
    rewriter.replaceOpWithNewOp<std_d::ConstantOp>(srcOp, resultType, newValue);
    return success();
  }
};

/// Generates a failure exception code.
/// This is just temporary to allow some libraries to signal exceptions.
class FailureOpConversion : public OpConversionPattern<pydm_d::FailureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::FailureOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type i32 = rewriter.getI32Type();
    // '-3' == RuntimeError
    rewriter.replaceOpWithNewOp<std_d::ConstantOp>(
        srcOp, i32, rewriter.getIntegerAttr(i32, -3));
    return success();
  }
};

class FuncOpConversion : public OpConversionPattern<pydm_d::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::FuncOp srcOp, ArrayRef<Value> operands,
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
    auto newFuncOp = rewriter.create<mlir::FuncOp>(
        srcOp.getLoc(), srcOp.getName(), newFuncType);
    newFuncOp.setVisibility(srcOp.getVisibility());
    rewriter.inlineRegionBefore(srcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Tell the rewriter to convert the region signature.
    TypeConverter &typeConverter = *getTypeConverter();
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    rewriter.replaceOp(srcOp, llvm::None);
    return success();
  }
};

class GetTypeCodeConversion
    : public OpConversionPattern<pydm_d::GetTypeCodeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::GetTypeCodeOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    // Gets the 0'th element of the object list, optionally casting it to the
    // converted integer type.
    Type resultType = typeConverter->convertType(srcOp.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(srcOp,
                                         "result type could not be converted");
    Type i32Type = rewriter.getIntegerType(32);
    Value index0 =
        rewriter.create<std_d::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value typeCode = rewriter.create<iree_d::ListGetOp>(
        loc, i32Type, adaptor.value(), index0);
    rewriter.replaceOp(
        srcOp,
        castIntegerValue(loc, typeCode,
                         resultType.cast<builtin_d::IntegerType>(), rewriter));
    return success();
  }
};

class LoadVarOpConversion : public OpConversionPattern<pydm_d::LoadVarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::LoadVarOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    auto resultType =
        getTypeConverter()->convertType(srcOp.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          srcOp, "could not convert load_var result type");
    auto list = operands[0];
    auto index1 =
        rewriter.create<std_d::ConstantOp>(loc, rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<iree_d::ListGetOp>(srcOp, resultType, list,
                                                   index1);
    return success();
  }
};

/// Converts a `none` operation to a `constant 0 : i32`.
/// See also the type conversion rule for `NoneType` which must align.
/// TODO: What we are really reaching for is a zero width type.
class NoneOpConversion : public OpConversionPattern<pydm_d::NoneOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::NoneOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Type i32 = rewriter.getI32Type();
    rewriter.replaceOpWithNewOp<std_d::ConstantOp>(
        srcOp, i32, rewriter.getIntegerAttr(i32, 0));
    return success();
  }
};

/// Raises an excpetion (failing status) on failure.
/// This pattern matches raise_on_failure ops at function scope. Those nested
/// within exception blocks are different.
class RaiseOnFailureOpConversion
    : public OpConversionPattern<pydm_d::RaiseOnFailureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::RaiseOnFailureOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();

    Value status = operands[0];
    // Get the containing function return type so that we can create a
    // suitable null return value.
    auto parentFunc = srcOp->getParentOfType<builtin_d::FuncOp>();
    if (!parentFunc)
      return rewriter.notifyMatchFailure(srcOp, "not contained by a func");
    Type convertedReturnType = parentFunc.getType().getResult(1);

    // Split the entry block.
    Block *entryBlock = rewriter.getInsertionBlock();
    Block *continuationBlock = rewriter.splitBlock(
        rewriter.getInsertionBlock(), rewriter.getInsertionPoint());
    Block *raiseAndReturnBlock = createSlowPathBlock(rewriter);

    // Branch on success conditional.
    rewriter.setInsertionPointToEnd(entryBlock);
    Value successValue = getSuccessStatusValue(loc, rewriter);
    Value isSuccess = rewriter.create<std_d::CmpIOp>(
        loc, std_d::CmpIPredicate::eq, successValue, status);
    rewriter.create<std_d::CondBranchOp>(loc, isSuccess, continuationBlock,
                                         raiseAndReturnBlock);
    rewriter.eraseOp(srcOp);

    // Raise and return block.
    rewriter.setInsertionPointToEnd(raiseAndReturnBlock);
    auto nullReturnValue = getNullValue(loc, rewriter, convertedReturnType);
    rewriter.create<std_d::ReturnOp>(loc, ValueRange{status, nullReturnValue});
    return success();
  }
};

/// Converts to a successful return (0 exception result and actual value).
class ReturnOpConversion : public OpConversionPattern<pydm_d::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::ReturnOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    auto zeroResult =
        rewriter.create<std_d::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<std_d::ReturnOp>(
        srcOp, ValueRange{zeroResult, operands[0]});
    return success();
  }
};

class StoreVarOpConversion : public OpConversionPattern<pydm_d::StoreVarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::StoreVarOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();

    auto origStoreType =
        srcOp.value().getType().dyn_cast<pydm_d::PythonTypeInterface>();
    if (!origStoreType)
      return rewriter.notifyMatchFailure(srcOp,
                                         "not a python type for value()");
    int typeCode = static_cast<int>(origStoreType.getTypeCode());

    auto list = operands[0];
    auto newValue = operands[1];
    resetObjectList(loc, rewriter, list, typeCode, newValue);
    rewriter.eraseOp(srcOp);
    return success();
  }
};

class UnboxOpConversion : public OpConversionPattern<pydm_d::UnboxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pydm_d::UnboxOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    auto list = operands[0];

    // Target exception result type.
    Type statusType = getTypeConverter()->convertType(srcOp.status().getType());
    // Target unboxed type.
    Type targetUnboxedType =
        getTypeConverter()->convertType(srcOp.primitive().getType());
    if (!targetUnboxedType || !statusType)
      return rewriter.notifyMatchFailure(
          srcOp, "could not convert unbox result types");

    // Compute the target type code.
    auto origUnboxedType =
        srcOp.primitive().getType().dyn_cast<pydm_d::PythonTypeInterface>();
    if (!origUnboxedType)
      return rewriter.notifyMatchFailure(
          srcOp, "not a python type for primitive() unboxed result");
    int typeCode = static_cast<int>(origUnboxedType.getTypeCode());

    // Split the entry block.
    Block *entryBlock = rewriter.getInsertionBlock();
    Block *continuationBlock = rewriter.splitBlock(
        rewriter.getInsertionBlock(), rewriter.getInsertionPoint());
    Block *typesMatchBlock = rewriter.createBlock(continuationBlock);
    Block *slowPathMismatchBlock = createSlowPathBlock(rewriter);
    continuationBlock->addArguments({statusType, targetUnboxedType});
    rewriter.replaceOp(srcOp, continuationBlock->getArguments());

    // Type code extraction and comparison.
    {
      rewriter.setInsertionPointToEnd(entryBlock);
      auto index0 =
          rewriter.create<std_d::ConstantOp>(loc, rewriter.getIndexAttr(0));
      Value requiredTypeCodeValue = rewriter.create<std_d::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(typeCode));
      Value actualTypeCodeValue = rewriter.create<iree_d::ListGetOp>(
          loc, rewriter.getI32Type(), list, index0);
      Value typeCodeEqual = rewriter.create<std_d::CmpIOp>(
          loc, std_d::CmpIPredicate::eq, requiredTypeCodeValue,
          actualTypeCodeValue);
      rewriter.create<std_d::CondBranchOp>(loc, typeCodeEqual, typesMatchBlock,
                                           slowPathMismatchBlock);
    }

    // Fast path types match block.
    {
      rewriter.setInsertionPointToEnd(typesMatchBlock);
      auto index1 =
          rewriter.create<std_d::ConstantOp>(loc, rewriter.getIndexAttr(1));
      Value successResult = getSuccessStatusValue(loc, rewriter);
      Value unboxedValue = rewriter.create<iree_d::ListGetOp>(
          loc, targetUnboxedType, list, index1);
      rewriter.create<std_d::BranchOp>(loc, continuationBlock,
                                       ValueRange{successResult, unboxedValue});
    }

    // Slow path coercion on mismatch.
    // TODO: Currently just fails - should emit a runtime call.
    {
      rewriter.setInsertionPointToEnd(slowPathMismatchBlock);
      Value failureResult =
          getFailureStatusValue(loc, rewriter, ExceptionCode::ValueError);
      Value nullResult = getNullValue(loc, rewriter, targetUnboxedType);
      rewriter.create<std_d::BranchOp>(loc, continuationBlock,
                                       ValueRange{failureResult, nullResult});
    }

    return success();
  }
};

//------------------------------------------------------------------------------
// Outside pydm op conversions
// These are largely identity conversions for CFG related standard ops, and
// those that can be emitted as part of canonicalizations.
//------------------------------------------------------------------------------

class BuiltinBranchConversion : public OpConversionPattern<std_d::BranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      std_d::BranchOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<std_d::BranchOp>(srcOp, srcOp.dest(),
                                                 adaptor.destOperands());
    return success();
  }
};

class BuiltinCondBranchConversion
    : public OpConversionPattern<std_d::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      std_d::CondBranchOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<std_d::CondBranchOp>(
        srcOp, adaptor.condition(), srcOp.trueDest(),
        adaptor.trueDestOperands(), srcOp.falseDest(),
        adaptor.falseDestOperands());
    return success();
  }
};

class BuiltinSelectConversion : public OpConversionPattern<std_d::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      std_d::SelectOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<std_d::SelectOp>(srcOp, adaptor.condition(),
                                                 adaptor.true_value(),
                                                 adaptor.false_value());
    return success();
  }
};

}  // namespace

void mlir::iree_pydm::populatePyDMToIREELoweringPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  // PyDM conversions.
  patterns.insert<AllocFreeVarOpConversion, ApplyBinaryNumericConversion,
                  ApplyCompareNumericConversion, BoolToPredConversion,
                  BoxOpConversion, CallOpConversion, ConstantOpConversion,
                  FailureOpConversion, FuncOpConversion, GetTypeCodeConversion,
                  LoadVarOpConversion, RaiseOnFailureOpConversion,
                  ReturnOpConversion, StoreVarOpConversion, UnboxOpConversion>(
      typeConverter, context);

  // External CFG ops.
  patterns.insert<BuiltinBranchConversion, BuiltinCondBranchConversion,
                  BuiltinSelectConversion>(typeConverter, context);

  // Constants and constructors.
  patterns.insert<NoneOpConversion>(typeConverter, context);
}
