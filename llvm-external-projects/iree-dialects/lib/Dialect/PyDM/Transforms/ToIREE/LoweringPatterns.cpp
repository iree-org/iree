// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputOps.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.h"
#include "iree-dialects/Dialect/PyDM/Transforms/ToIREE/Patterns.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

using llvm::enumerate;
using namespace mlir;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
namespace Input = mlir::iree_compiler::IREE::Input;
using namespace PYDM;

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

static Type getVariantListType(Builder &builder) {
  return builder.getType<Input::ListType>(
      builder.getType<Input::VariantType>());
}

static Value getNullValue(Location loc, OpBuilder &builder, Type t) {
  return TypeSwitch<Type, Value>(t)
      .Case<Input::ListType>([&](auto t) -> Value {
        // TODO: If it becomes important to optimize this, come up with a way
        // to return an empty list without creating one.
        return builder.create<Input::ListCreateOp>(
            loc, getVariantListType(builder), /*capacity=*/nullptr);
      })
      .Default([&](Type t) -> Value {
        auto attr = builder.getZeroAttr(t);
        assert(attr && "could not get zero attr for builtin type");
        return builder.create<arith::ConstantOp>(loc, t, attr);
      });
}

/// Creates a slow path block at the end of the function. The current block
/// will always dominate.
static Block *createSlowPathBlock(OpBuilder &builder) {
  Region *parentRegion = builder.getInsertionBlock()->getParent();
  return builder.createBlock(parentRegion, parentRegion->end());
}

static Value getSuccessStatusValue(Location loc, OpBuilder &builder) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(0));
}

static Value getFailureStatusValue(Location loc, OpBuilder &builder,
                                   ExceptionCode code) {
  return builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(static_cast<int>(code)));
}

static Value createUndefObjectList(Location loc, OpBuilder &builder) {
  return builder.create<Input::ListCreateOp>(loc, getVariantListType(builder),
                                             /*capacity=*/nullptr);
}

void resetObjectList(Location loc, OpBuilder &builder, Value list, int typeCode,
                     Value data) {
  // Note: The list can record optional runtime state at positions > 1, so
  // to truly reset, we have to resize. Low level optimizations should be able
  // to elide this if it turns out to be unnecessary.
  auto size = builder.create<arith::ConstantIndexOp>(loc, 2);
  builder.create<Input::ListResizeOp>(loc, list, size);
  auto index0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value typeCodeValue = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(typeCode));
  builder.create<Input::ListSetOp>(loc, list, index0, typeCodeValue);
  auto index1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<Input::ListSetOp>(loc, list, index1, data);
}

static Value createObjectList(Location loc, OpBuilder &builder, int typeCode,
                              Value data) {
  auto list = createUndefObjectList(loc, builder);
  resetObjectList(loc, builder, list, typeCode, data);
  return list;
}

static Value castIntegerValue(Location loc, Value input,
                              mlir::IntegerType resultType,
                              OpBuilder &builder) {
  mlir::IntegerType inputType = input.getType().cast<mlir::IntegerType>();
  if (inputType.getWidth() == resultType.getWidth()) {
    return input;
  } else if (inputType.getWidth() < resultType.getWidth()) {
    return builder.create<arith::ExtSIOp>(loc, resultType, input);
  } else {
    return builder.create<arith::TruncIOp>(loc, resultType, input);
  }
}

static Optional<arith::CmpIPredicate> convertIntegerComparePredicate(
    StringAttr dunderName, bool isSigned, Builder &builder) {
  StringRef v = dunderName.getValue();
  if (v == "lt") {
    return isSigned ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult;
  } else if (v == "le") {
    return isSigned ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule;
  } else if (v == "eq" || v == "is") {
    return arith::CmpIPredicate::eq;
  } else if (v == "ne" || v == "isnot") {
    return arith::CmpIPredicate::ne;
  } else if (v == "gt") {
    return isSigned ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ugt;
  } else if (v == "ge") {
    return isSigned ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge;
  }

  return {};
}

static Optional<arith::CmpFPredicate> convertFpComparePredicate(
    StringAttr dunderName, Builder &builder) {
  StringRef v = dunderName.getValue();
  if (v == "lt") {
    return arith::CmpFPredicate::OLT;
  } else if (v == "le") {
    return arith::CmpFPredicate::OLE;
  } else if (v == "eq" || v == "is") {
    return arith::CmpFPredicate::OEQ;
  } else if (v == "ne" || v == "isnot") {
    return arith::CmpFPredicate::ONE;
  } else if (v == "gt") {
    return arith::CmpFPredicate::OGT;
  } else if (v == "ge") {
    return arith::CmpFPredicate::OGE;
  }

  return {};
}

/// Does a low-level boxing operation on the given `convertedValue`, which
/// has already been subject to type conversion. This is based on the original
/// `pythonType` which must implement PythonTypeInterface.
/// If `pythonType` is already boxed, then this does nothing.
/// Returns nullptr for unsupported cases, not emitting diagnostics.
static Value boxConvertedValue(Location loc, Type pythonType,
                               Value convertedValue, OpBuilder &builder) {
  if (pythonType.isa<PYDM::ObjectType>()) return convertedValue;
  auto ptiType = pythonType.dyn_cast<PYDM::PythonTypeInterface>();
  if (!ptiType) return {};
  auto typeCode = ptiType.getTypeCode();
  auto list = createObjectList(loc, builder, static_cast<int>(typeCode),
                               convertedValue);
  return list;
}

namespace {

class AllocFreeVarOpConversion
    : public OpConversionPattern<PYDM::AllocFreeVarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::AllocFreeVarOp srcOp, OpAdaptor adaptor,
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
    : public OpConversionPattern<PYDM::ApplyBinaryOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::ApplyBinaryOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type pyLeftType = srcOp.left().getType();
    Type pyRightType = srcOp.right().getType();
    Type leftType = adaptor.left().getType();
    Type rightType = adaptor.right().getType();
    Type resultType = typeConverter->convertType(srcOp.result().getType());
    if (!resultType || pyLeftType != pyRightType || leftType != rightType ||
        leftType != resultType) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "not same type operands/results");
    }
    if (auto pyIntegerType = pyLeftType.dyn_cast<PYDM::IntegerType>()) {
      bool isSigned = pyIntegerType.isSigned();
      Value converted =
          convertIntegerOp(srcOp.getLoc(), adaptor.dunder_name(),
                           adaptor.left(), adaptor.right(), isSigned, rewriter);
      if (!converted)
        return rewriter.notifyMatchFailure(srcOp, "unsupported operation");
      rewriter.replaceOp(srcOp, converted);
      return success();
    } else if (leftType.isa<mlir::FloatType>()) {
      Value converted =
          convertFloatOp(srcOp.getLoc(), adaptor.dunder_name(), adaptor.left(),
                         adaptor.right(), rewriter);
      if (!converted)
        return rewriter.notifyMatchFailure(srcOp, "unsupported operation");
      rewriter.replaceOp(srcOp, converted);
      return success();
    }

    return rewriter.notifyMatchFailure(srcOp, "non numeric type");
  }

  Value convertIntegerOp(Location loc, StringRef dunderName, Value left,
                         Value right, bool isSigned,
                         ConversionPatternRewriter &rewriter) const {
    // TODO: matmul, truediv, floordiv, mod, divmod, pow
    if (dunderName == "add") {
      return rewriter.create<arith::AddIOp>(loc, left, right);
    } else if (dunderName == "and") {
      return rewriter.create<arith::AndIOp>(loc, left, right);
    } else if (dunderName == "mul") {
      return rewriter.create<arith::MulIOp>(loc, left, right);
    } else if (dunderName == "lshift") {
      return rewriter.create<arith::ShLIOp>(loc, left, right);
    } else if (dunderName == "or") {
      return rewriter.create<arith::OrIOp>(loc, left, right);
    } else if (dunderName == "rshift") {
      if (isSigned)
        return rewriter.create<arith::ShRSIOp>(loc, left, right);
      else
        return rewriter.create<arith::ShRUIOp>(loc, left, right);
    } else if (dunderName == "sub") {
      return rewriter.create<arith::SubIOp>(loc, left, right);
    } else if (dunderName == "xor") {
      return rewriter.create<arith::XOrIOp>(loc, left, right);
    }
    return nullptr;
  }

  Value convertFloatOp(Location loc, StringRef dunderName, Value left,
                       Value right, ConversionPatternRewriter &rewriter) const {
    // TODO: matmul, truediv, floordiv, mod, divmod, pow
    if (dunderName == "add") {
      return rewriter.create<arith::AddFOp>(loc, left, right);
    } else if (dunderName == "mul") {
      return rewriter.create<arith::MulFOp>(loc, left, right);
    } else if (dunderName == "sub") {
      return rewriter.create<arith::SubFOp>(loc, left, right);
    }
    return nullptr;
  }
};

class ApplyCompareNumericConversion
    : public OpConversionPattern<PYDM::ApplyCompareOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::ApplyCompareOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type leftType = adaptor.left().getType();
    Type rightType = adaptor.right().getType();
    if (leftType != rightType) {
      return rewriter.notifyMatchFailure(srcOp, "not same type operands");
    }
    if (leftType.isa<mlir::IntegerType>()) {
      bool isSigned = true;  // TODO: Unsigned.
      auto predicate = convertIntegerComparePredicate(adaptor.dunder_nameAttr(),
                                                      isSigned, rewriter);
      if (!predicate)
        return rewriter.notifyMatchFailure(srcOp, "unsupported predicate");
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(
          srcOp, *predicate, adaptor.left(), adaptor.right());
      return success();
    } else if (leftType.isa<mlir::FloatType>()) {
      auto predicate =
          convertFpComparePredicate(adaptor.dunder_nameAttr(), rewriter);
      if (!predicate)
        return rewriter.notifyMatchFailure(srcOp, "unsupported predicate");
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(
          srcOp, *predicate, adaptor.left(), adaptor.right());
      return success();
    }

    return rewriter.notifyMatchFailure(srcOp, "non numeric type");
  }
};

class AssignSubscriptListConversion
    : public OpConversionPattern<PYDM::AssignSubscriptOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      PYDM::AssignSubscriptOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto pySequence = srcOp.lhs();
    if (!pySequence.getType().isa<PYDM::ListType>())
      return rewriter.notifyMatchFailure(srcOp, "not builtin sequence");
    auto pySlice = srcOp.slice();
    if (!pySlice.getType().isa<PYDM::IntegerType>())
      return rewriter.notifyMatchFailure(srcOp,
                                         "slice is not static integer type");

    auto loc = srcOp.getLoc();
    auto sequence = adaptor.lhs();
    auto slice = adaptor.slice();
    auto indexType = rewriter.getType<IndexType>();
    Type statusType =
        getTypeConverter()->convertType(srcOp.exc_result().getType());
    Value valueToSet =
        boxIfNecessary(loc, pySequence.getType().cast<PYDM::ListType>(),
                       srcOp.rhs().getType(), adaptor.rhs(), rewriter);
    if (!valueToSet) {
      return rewriter.notifyMatchFailure(
          srcOp, "unsupported list assignment boxing mode");
    }

    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, slice.getType());
    Value listSizeIndex =
        rewriter.create<Input::ListSizeOp>(loc, indexType, sequence);
    Value listSizeInteger = rewriter.create<arith::IndexCastOp>(
        loc, slice.getType(), listSizeIndex);
    Block *entryBlock = rewriter.getInsertionBlock();
    Block *continuationBlock = rewriter.splitBlock(
        rewriter.getInsertionBlock(), rewriter.getInsertionPoint());
    Block *indexLtZeroBlock = rewriter.createBlock(continuationBlock);
    Block *indexCheckBlock = rewriter.createBlock(continuationBlock);
    indexCheckBlock->addArgument(indexType, loc);
    Block *setElementBlock = rewriter.createBlock(continuationBlock);
    setElementBlock->addArgument(indexType, loc);
    Block *failureBlock = createSlowPathBlock(rewriter);
    continuationBlock->addArgument(statusType, loc);
    rewriter.replaceOp(srcOp, continuationBlock->getArguments());

    // Comparison index < 0.
    {
      rewriter.setInsertionPointToEnd(entryBlock);
      Value ltZero = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, slice, zero);
      auto sliceIndex =
          rewriter.create<arith::IndexCastOp>(loc, indexType, slice);
      rewriter.create<mlir::CondBranchOp>(loc, ltZero, indexLtZeroBlock,
                                          indexCheckBlock,
                                          ValueRange{sliceIndex});
    }

    // Handle index < 0.
    {
      rewriter.setInsertionPointToEnd(indexLtZeroBlock);
      Value positiveSlice =
          rewriter.create<arith::AddIOp>(loc, slice, listSizeInteger);
      Value positiveSliceIndex =
          rewriter.create<arith::IndexCastOp>(loc, indexType, positiveSlice);
      rewriter.create<mlir::BranchOp>(loc, ValueRange{positiveSliceIndex},
                                      indexCheckBlock);
    }

    // Index check.
    {
      rewriter.setInsertionPointToEnd(indexCheckBlock);
      Value sliceIndex = indexCheckBlock->getArgument(0);
      Value ltSize = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, sliceIndex, listSizeIndex);
      rewriter.create<mlir::CondBranchOp>(loc, ltSize, setElementBlock,
                                          ValueRange{sliceIndex}, failureBlock,
                                          ValueRange{});
    }

    // Set element.
    {
      rewriter.setInsertionPointToEnd(setElementBlock);
      Value successResult = getSuccessStatusValue(loc, rewriter);
      rewriter.create<Input::ListSetOp>(
          loc, sequence, setElementBlock->getArgument(0), valueToSet);
      rewriter.create<mlir::BranchOp>(loc, continuationBlock,
                                      ValueRange{successResult});
    }

    // Failure.
    {
      rewriter.setInsertionPointToEnd(failureBlock);
      Value failureResult =
          getFailureStatusValue(loc, rewriter, ExceptionCode::IndexError);
      rewriter.create<mlir::BranchOp>(loc, continuationBlock,
                                      ValueRange{failureResult});
    }

    return success();
  }

  Value boxIfNecessary(Location loc, PYDM::ListType listType, Type origRhsType,
                       Value rhs, ConversionPatternRewriter &rewriter) const {
    switch (listType.getStorageClass()) {
      case CollectionStorageClass::Boxed:
      case CollectionStorageClass::Empty: {
        return boxConvertedValue(loc, origRhsType, rhs, rewriter);
        break;
      }
      case CollectionStorageClass::Unboxed:
        // TODO: Implement.
        return nullptr;
    }
  }
};

class BoolToPredConversion : public OpConversionPattern<PYDM::BoolToPredOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::BoolToPredOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(srcOp, adaptor.value());
    return success();
  }
};

class BoxOpConversion : public OpConversionPattern<PYDM::BoxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::BoxOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    Value boxedValue = boxConvertedValue(loc, srcOp.primitive().getType(),
                                         adaptor.primitive(), rewriter);
    if (!boxedValue)
      return rewriter.notifyMatchFailure(srcOp,
                                         "not a supported type for boxing");
    rewriter.replaceOp(srcOp, boxedValue);
    return success();
  }
};

class CallOpConversion : public OpConversionPattern<PYDM::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::CallOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(srcOp.getResultTypes(),
                                                resultTypes))) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "result types could not be converted");
    }
    rewriter.replaceOpWithNewOp<mlir::CallOp>(srcOp, srcOp.callee(),
                                              resultTypes, adaptor.operands());
    return success();
  }
};

class ConstantOpConversion : public OpConversionPattern<PYDM::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::ConstantOp srcOp, OpAdaptor adaptor,
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
        .Case([&](mlir::IntegerType t) {
          APInt intValue =
              newValue.cast<IntegerAttr>().getValue().sextOrTrunc(t.getWidth());
          newValue = rewriter.getIntegerAttr(t, intValue);
        })
        .Case([&](mlir::FloatType t) {
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
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(srcOp, resultType, newValue);
    return success();
  }
};

/// Expands dynamic unpacking of a tuple or list by taking advantage that they
/// both are just variant lists. A size check is emitted, with a branch to
/// a failure block. The success block will just get each element.
class DynamicUnpackOpConversion
    : public OpConversionPattern<PYDM::DynamicUnpackOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::DynamicUnpackOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    // Convert types.
    Type excResultType =
        getTypeConverter()->convertType(srcOp.exc_result().getType());
    if (!excResultType)
      return rewriter.notifyMatchFailure(srcOp,
                                         "could not convert exc_result type");
    int arity = srcOp.slots().size();
    SmallVector<Type> slotTypes;
    slotTypes.reserve(arity);
    for (auto slot : srcOp.slots()) {
      Type slotType = getTypeConverter()->convertType(slot.getType());
      if (!slotType)
        return rewriter.notifyMatchFailure(
            srcOp, "could not convert result slot type");
      slotTypes.push_back(slotType);
    }

    // Split the entry block.
    Block *entryBlock = rewriter.getInsertionBlock();
    Block *continuationBlock = rewriter.splitBlock(
        rewriter.getInsertionBlock(), rewriter.getInsertionPoint());
    Block *arityMatchBlock = rewriter.createBlock(continuationBlock);
    Block *errorBlock = createSlowPathBlock(rewriter);
    continuationBlock->addArgument(excResultType, loc);
    for (auto slotType : slotTypes) {
      continuationBlock->addArgument(slotType, loc);
    }
    rewriter.replaceOp(srcOp, continuationBlock->getArguments());

    // Entry block - check arity.
    {
      rewriter.setInsertionPointToEnd(entryBlock);
      auto arityValue =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(arity));
      Value listSize = rewriter.create<Input::ListSizeOp>(
          loc, rewriter.getIndexType(), adaptor.sequence());
      Value arityMatch = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, arityValue, listSize);
      rewriter.create<mlir::CondBranchOp>(loc, arityMatch, arityMatchBlock,
                                          errorBlock);
    }

    // Arity match.
    {
      rewriter.setInsertionPointToEnd(arityMatchBlock);
      SmallVector<Value> branchArgs;
      branchArgs.push_back(getSuccessStatusValue(loc, rewriter));
      for (auto it : enumerate(slotTypes)) {
        Value index = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(it.index()));
        Value slotValue = rewriter.create<Input::ListGetOp>(
            loc, it.value(), adaptor.sequence(), index);
        branchArgs.push_back(slotValue);
      }
      rewriter.create<mlir::BranchOp>(loc, continuationBlock, branchArgs);
    }

    // Error block.
    {
      rewriter.setInsertionPointToEnd(errorBlock);
      SmallVector<Value> branchArgs;
      branchArgs.push_back(
          getFailureStatusValue(loc, rewriter, ExceptionCode::ValueError));
      for (Type slotType : slotTypes) {
        branchArgs.push_back(getNullValue(loc, rewriter, slotType));
      }
      rewriter.create<mlir::BranchOp>(loc, continuationBlock, branchArgs);
    }

    return success();
  }
};

/// If at this phase, there is nothing to do with a static info cast.
/// Just drop it.
class ElideStaticInfoCast : public OpConversionPattern<PYDM::StaticInfoCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::StaticInfoCastOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(srcOp, srcOp.value());
    return success();
  }
};

/// Generates a failure exception code.
/// This is just temporary to allow some libraries to signal exceptions.
class FailureOpConversion : public OpConversionPattern<PYDM::FailureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::FailureOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type i32 = rewriter.getI32Type();
    // '-3' == RuntimeError
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        srcOp, i32, rewriter.getIntegerAttr(i32, -3));
    return success();
  }
};

class FuncOpConversion : public OpConversionPattern<PYDM::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::FuncOp srcOp, OpAdaptor adaptor,
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

class GetTypeCodeConversion : public OpConversionPattern<PYDM::GetTypeCodeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::GetTypeCodeOp srcOp, OpAdaptor adaptor,
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
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value typeCode = rewriter.create<Input::ListGetOp>(loc, i32Type,
                                                       adaptor.value(), index0);
    rewriter.replaceOp(
        srcOp,
        castIntegerValue(loc, typeCode, resultType.cast<mlir::IntegerType>(),
                         rewriter));
    return success();
  }
};

class LoadVarOpConversion : public OpConversionPattern<PYDM::LoadVarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::LoadVarOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    auto resultType =
        getTypeConverter()->convertType(srcOp.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          srcOp, "could not convert load_var result type");
    auto list = adaptor.getOperands()[0];
    auto index1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<Input::ListGetOp>(srcOp, resultType, list,
                                                  index1);
    return success();
  }
};

class MakeListOpBoxedConversion : public OpConversionPattern<PYDM::MakeListOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::MakeListOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    auto listType = srcOp.list().getType().cast<PYDM::ListType>();
    if (listType.getStorageClass() != CollectionStorageClass::Boxed ||
        listType.getStorageClass() == CollectionStorageClass::Empty)
      return rewriter.notifyMatchFailure(srcOp, "unboxed list");
    auto resultType = getTypeConverter()->convertType(listType);
    if (!resultType)
      return rewriter.notifyMatchFailure(srcOp,
                                         "could not convert result type");

    auto size = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(adaptor.elements().size()));
    auto list =
        rewriter.create<Input::ListCreateOp>(loc, getVariantListType(rewriter),
                                             /*capacity=*/size);
    rewriter.create<Input::ListResizeOp>(loc, list, size);
    for (auto it : enumerate(adaptor.elements())) {
      auto index = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(it.index()));
      rewriter.create<Input::ListSetOp>(loc, list, index, it.value());
    }

    rewriter.replaceOp(srcOp, ValueRange{list});
    return success();
  }
};

class MakeTupleOpConversion : public OpConversionPattern<PYDM::MakeTupleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::MakeTupleOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    auto resultType = getTypeConverter()->convertType(srcOp.tuple().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(srcOp,
                                         "could not convert result type");

    auto size = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(adaptor.slots().size()));
    auto list =
        rewriter.create<Input::ListCreateOp>(loc, getVariantListType(rewriter),
                                             /*capacity=*/size);
    rewriter.create<Input::ListResizeOp>(loc, list, size);
    for (auto it : enumerate(adaptor.slots())) {
      auto index = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(it.index()));
      rewriter.create<Input::ListSetOp>(loc, list, index, it.value());
    }

    rewriter.replaceOp(srcOp, ValueRange{list});
    return success();
  }
};

/// Converts the `neg` op on integer operand/result to a corresponding sub.
class NegIntegerOpConversion : public OpConversionPattern<PYDM::NegOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      PYDM::NegOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type valueType = adaptor.value().getType();
    Type resultType = getTypeConverter()->convertType(srcOp.result().getType());
    if (!valueType.isa<mlir::IntegerType>() || valueType != resultType)
      return rewriter.notifyMatchFailure(srcOp, "not an integer neg");
    Location loc = srcOp.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, resultType);
    rewriter.replaceOpWithNewOp<arith::SubIOp>(srcOp, zero, adaptor.value());
    return success();
  }
};

/// Converts a `none` operation to a `constant 0 : i32`.
/// See also the type conversion rule for `NoneType` which must align.
/// TODO: What we are really reaching for is a zero width type.
class NoneOpConversion : public OpConversionPattern<PYDM::NoneOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::NoneOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type i32 = rewriter.getI32Type();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        srcOp, i32, rewriter.getIntegerAttr(i32, 0));
    return success();
  }
};

/// Raises an excpetion (failing status) on failure.
/// This pattern matches raise_on_failure ops at function scope. Those nested
/// within exception blocks are different.
class RaiseOnFailureOpConversion
    : public OpConversionPattern<PYDM::RaiseOnFailureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::RaiseOnFailureOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();

    Value status = adaptor.getOperands()[0];
    // Get the containing function return type so that we can create a
    // suitable null return value.
    auto parentFunc = srcOp->getParentOfType<mlir::FuncOp>();
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
    Value isSuccess = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, successValue, status);
    rewriter.create<mlir::CondBranchOp>(loc, isSuccess, continuationBlock,
                                        raiseAndReturnBlock);
    rewriter.eraseOp(srcOp);

    // Raise and return block.
    rewriter.setInsertionPointToEnd(raiseAndReturnBlock);
    auto nullReturnValue = getNullValue(loc, rewriter, convertedReturnType);
    rewriter.create<mlir::ReturnOp>(loc, ValueRange{status, nullReturnValue});
    return success();
  }
};

/// Converts to a successful return (0 exception result and actual value).
class ReturnOpConversion : public OpConversionPattern<PYDM::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::ReturnOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    auto zeroResult =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(
        srcOp, ValueRange{zeroResult, adaptor.getOperands()[0]});
    return success();
  }
};

/// Implements sequence duplication over built-in list, tuple types.
class SequenceCloneBuiltinConversion
    : public OpConversionPattern<PYDM::SequenceCloneOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::SequenceCloneOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type origListType = srcOp.sequence().getType();
    if (!isSupportedList(origListType)) return failure();
    if (origListType != srcOp.getResult().getType()) return failure();
    Type resultType = typeConverter->convertType(srcOp.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(srcOp, "cannot convert result type");
    }
    Type listElementType =
        typeConverter->convertType(getElementAccessType(origListType));
    if (!listElementType) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "cannot convert list element type");
    }

    Value listOperand = adaptor.sequence();
    Value countOperand = adaptor.count();
    auto loc = srcOp.getLoc();
    // Compute the new size, clamping count to >= 0 and construct list.
    Type indexType = rewriter.getType<IndexType>();
    Type listType = listOperand.getType();
    Value subListSize =
        rewriter.create<Input::ListSizeOp>(loc, indexType, listOperand);
    Value countInteger = countOperand;
    Value countIndex =
        rewriter.create<arith::IndexCastOp>(loc, indexType, countOperand);
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIndex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value zeroInteger =
        rewriter.create<arith::ConstantIntOp>(loc, 0, countInteger.getType());
    Value countClampsToZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, countInteger, zeroInteger);
    Value clampedCountIndex = rewriter.create<mlir::arith::SelectOp>(
        loc, countClampsToZero, zeroIndex, countIndex);
    Value newListSize =
        rewriter.create<arith::MulIOp>(loc, subListSize, clampedCountIndex);
    Value newList =
        rewriter.create<Input::ListCreateOp>(loc, listType, clampedCountIndex);
    rewriter.create<Input::ListResizeOp>(loc, newList, newListSize);

    // Split blocks to loop.
    // TODO: Use a new list.copy op instead of an inner loop.
    // OuterCond: (newListIt : index)
    // InnerCond: (newListIt : index, subListIt: index)
    // InnerBody: (newListIt : index, subListIt: index)
    Block *entryBlock = rewriter.getInsertionBlock();
    Block *continuationBlock = rewriter.splitBlock(
        rewriter.getInsertionBlock(), rewriter.getInsertionPoint());
    Block *outerCond = rewriter.createBlock(continuationBlock);
    outerCond->addArgument(indexType, loc);
    Block *innerCond = rewriter.createBlock(continuationBlock);
    innerCond->addArguments({indexType, indexType}, {loc, loc});
    Block *innerBody = rewriter.createBlock(continuationBlock);
    innerBody->addArguments({indexType, indexType}, {loc, loc});

    // Entry block.
    {
      rewriter.setInsertionPointToEnd(entryBlock);
      rewriter.create<BranchOp>(loc, outerCond, ValueRange{zeroIndex});
    }

    // Outer cond.
    {
      rewriter.setInsertionPointToEnd(outerCond);
      Value newListIt = outerCond->getArgument(0);
      Value inBounds = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, newListIt, newListSize);
      rewriter.create<CondBranchOp>(loc, inBounds, innerCond,
                                    ValueRange{newListIt, zeroIndex},
                                    continuationBlock, ValueRange{});
    }

    // Inner cond.
    {
      rewriter.setInsertionPointToEnd(innerCond);
      Value newListIt = innerCond->getArgument(0);
      Value subListIt = innerCond->getArgument(1);
      Value inBounds = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, subListIt, subListSize);
      rewriter.create<CondBranchOp>(loc, inBounds, innerBody,
                                    ValueRange{newListIt, subListIt}, outerCond,
                                    ValueRange{newListIt});
    }

    // Inner body.
    {
      rewriter.setInsertionPointToEnd(innerBody);
      Value newListIt = innerBody->getArgument(0);
      Value subListIt = innerBody->getArgument(1);

      Value elementValue = rewriter.create<Input::ListGetOp>(
          loc, listElementType, listOperand, subListIt);
      rewriter.create<Input::ListSetOp>(loc, newList, newListIt, elementValue);

      newListIt = rewriter.create<arith::AddIOp>(loc, newListIt, oneIndex);
      subListIt = rewriter.create<arith::AddIOp>(loc, subListIt, oneIndex);
      rewriter.create<BranchOp>(loc, innerCond,
                                ValueRange{newListIt, subListIt});
    }

    // Continuation.
    {
      rewriter.setInsertionPointToEnd(continuationBlock);
      rewriter.replaceOp(srcOp, {newList});
    }

    return success();
  }

  bool isSupportedList(Type t) const {
    // Both lists and tuples have the same physical representation and can
    // be supported interchangeably here.
    return t.isa<PYDM::ListType>() || t.isa<PYDM::TupleType>();
  }

  Type getElementAccessType(Type t) const {
    if (auto listType = t.dyn_cast<PYDM::ListType>()) {
      return listType.getElementStorageType();
    } else if (auto tupleType = t.dyn_cast<PYDM::TupleType>()) {
      return tupleType.getElementStorageType();
    }

    llvm_unreachable("unsupported list type");
  }
};

class StoreVarOpConversion : public OpConversionPattern<PYDM::StoreVarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::StoreVarOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();

    auto origStoreType =
        srcOp.value().getType().dyn_cast<PYDM::PythonTypeInterface>();
    if (!origStoreType)
      return rewriter.notifyMatchFailure(srcOp,
                                         "not a python type for value()");
    int typeCode = static_cast<int>(origStoreType.getTypeCode());

    auto list = adaptor.getOperands()[0];
    auto newValue = adaptor.getOperands()[1];
    resetObjectList(loc, rewriter, list, typeCode, newValue);
    rewriter.eraseOp(srcOp);
    return success();
  }
};

/// Lowers the subscript operator on builtin sequence types (list, tuple)
/// based on a statically determined scalar slice (which can be positive or
/// negative).
class SubscriptOpBuiltinSequenceConversion
    : public OpConversionPattern<PYDM::SubscriptOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::SubscriptOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto pySequence = srcOp.value();
    if (!pySequence.getType().isa<PYDM::ListType, PYDM::TupleType>())
      return rewriter.notifyMatchFailure(srcOp, "not builtin sequence");
    auto pySlice = srcOp.slice();
    if (!pySlice.getType().isa<PYDM::IntegerType>())
      return rewriter.notifyMatchFailure(srcOp,
                                         "slice is not static integer type");
    Type resultType = getTypeConverter()->convertType(srcOp.result().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(srcOp,
                                         "could not convert result type");

    auto loc = srcOp.getLoc();
    auto slice = adaptor.slice();
    auto indexType = rewriter.getType<IndexType>();
    Type statusType =
        getTypeConverter()->convertType(srcOp.exc_result().getType());

    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, slice.getType());
    Value listSizeIndex =
        rewriter.create<Input::ListSizeOp>(loc, indexType, adaptor.value());
    Value listSizeInteger = rewriter.create<arith::IndexCastOp>(
        loc, slice.getType(), listSizeIndex);

    // Split blocks:
    //   indexLtZeroBlock
    //   indexCheckBlock(sliceIndex : IndexType)
    //   getElementBlock(sliceIndex : IndexType)
    //   continuationBlock(exc_result, result)
    //   failureBlock
    Block *entryBlock = rewriter.getInsertionBlock();
    Block *continuationBlock = rewriter.splitBlock(
        rewriter.getInsertionBlock(), rewriter.getInsertionPoint());
    Block *indexLtZeroBlock = rewriter.createBlock(continuationBlock);
    Block *indexCheckBlock = rewriter.createBlock(continuationBlock);
    indexCheckBlock->addArgument(indexType, loc);
    Block *getElementBlock = rewriter.createBlock(continuationBlock);
    getElementBlock->addArgument(indexType, loc);
    Block *failureBlock = createSlowPathBlock(rewriter);
    continuationBlock->addArguments({statusType, resultType}, {loc, loc});
    rewriter.replaceOp(srcOp, continuationBlock->getArguments());

    // Comparison index < 0.
    {
      rewriter.setInsertionPointToEnd(entryBlock);
      Value ltZero = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, slice, zero);
      auto sliceIndex =
          rewriter.create<arith::IndexCastOp>(loc, indexType, slice);
      rewriter.create<mlir::CondBranchOp>(loc, ltZero, indexLtZeroBlock,
                                          indexCheckBlock,
                                          ValueRange{sliceIndex});
    }

    // Handle index < 0.
    {
      rewriter.setInsertionPointToEnd(indexLtZeroBlock);
      Value positiveSlice =
          rewriter.create<arith::AddIOp>(loc, slice, listSizeInteger);
      Value positiveSliceIndex =
          rewriter.create<arith::IndexCastOp>(loc, indexType, positiveSlice);
      rewriter.create<mlir::BranchOp>(loc, ValueRange{positiveSliceIndex},
                                      indexCheckBlock);
    }

    // Index check.
    {
      rewriter.setInsertionPointToEnd(indexCheckBlock);
      Value sliceIndex = indexCheckBlock->getArgument(0);
      Value ltSize = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, sliceIndex, listSizeIndex);
      rewriter.create<mlir::CondBranchOp>(loc, ltSize, getElementBlock,
                                          ValueRange{sliceIndex}, failureBlock,
                                          ValueRange{});
    }

    // Get element.
    {
      rewriter.setInsertionPointToEnd(getElementBlock);
      Value successResult = getSuccessStatusValue(loc, rewriter);
      Value resultValue = rewriter.create<Input::ListGetOp>(
          loc, resultType, adaptor.value(), getElementBlock->getArgument(0));
      rewriter.create<mlir::BranchOp>(loc, continuationBlock,
                                      ValueRange{successResult, resultValue});
    }

    // Failure.
    {
      rewriter.setInsertionPointToEnd(failureBlock);
      Value failureResult =
          getFailureStatusValue(loc, rewriter, ExceptionCode::IndexError);
      Value nullResult = getNullValue(loc, rewriter, resultType);
      rewriter.create<mlir::BranchOp>(loc, continuationBlock,
                                      ValueRange{failureResult, nullResult});
    }

    return success();
  }
};

class UnboxOpConversion : public OpConversionPattern<PYDM::UnboxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PYDM::UnboxOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    auto list = adaptor.getOperands()[0];

    // Target exception result type.
    Type statusType =
        getTypeConverter()->convertType(srcOp.exc_result().getType());
    // Target unboxed type.
    Type targetUnboxedType =
        getTypeConverter()->convertType(srcOp.primitive().getType());
    if (!targetUnboxedType || !statusType)
      return rewriter.notifyMatchFailure(
          srcOp, "could not convert unbox result types");

    // Compute the target type code.
    auto origUnboxedType =
        srcOp.primitive().getType().dyn_cast<PYDM::PythonTypeInterface>();
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
    continuationBlock->addArguments({statusType, targetUnboxedType},
                                    {loc, loc});
    rewriter.replaceOp(srcOp, continuationBlock->getArguments());

    // Type code extraction and comparison.
    {
      rewriter.setInsertionPointToEnd(entryBlock);
      auto index0 =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      Value requiredTypeCodeValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(typeCode));
      Value actualTypeCodeValue = rewriter.create<Input::ListGetOp>(
          loc, rewriter.getI32Type(), list, index0);
      Value typeCodeEqual = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, requiredTypeCodeValue,
          actualTypeCodeValue);
      rewriter.create<mlir::CondBranchOp>(loc, typeCodeEqual, typesMatchBlock,
                                          slowPathMismatchBlock);
    }

    // Fast path types match block.
    {
      rewriter.setInsertionPointToEnd(typesMatchBlock);
      auto index1 =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
      Value successResult = getSuccessStatusValue(loc, rewriter);
      Value unboxedValue = rewriter.create<Input::ListGetOp>(
          loc, targetUnboxedType, list, index1);
      rewriter.create<mlir::BranchOp>(loc, continuationBlock,
                                      ValueRange{successResult, unboxedValue});
    }

    // Slow path coercion on mismatch.
    // TODO: Currently just fails - should emit a runtime call.
    {
      rewriter.setInsertionPointToEnd(slowPathMismatchBlock);
      Value failureResult =
          getFailureStatusValue(loc, rewriter, ExceptionCode::ValueError);
      Value nullResult = getNullValue(loc, rewriter, targetUnboxedType);
      rewriter.create<mlir::BranchOp>(loc, continuationBlock,
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

class BuiltinBranchConversion : public OpConversionPattern<mlir::BranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::BranchOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::BranchOp>(srcOp, srcOp.getDest(),
                                                adaptor.getDestOperands());
    return success();
  }
};

class BuiltinCondBranchConversion
    : public OpConversionPattern<mlir::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::CondBranchOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(
        srcOp, adaptor.getCondition(), srcOp.getTrueDest(),
        adaptor.getTrueDestOperands(), srcOp.getFalseDest(),
        adaptor.getFalseDestOperands());
    return success();
  }
};

class BuiltinSelectConversion
    : public OpConversionPattern<mlir::arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::arith::SelectOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        srcOp, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return success();
  }
};

}  // namespace

void PYDM::populatePyDMToIREELoweringPatterns(MLIRContext *context,
                                              TypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
  // PyDM conversions.
  patterns.insert<
      AllocFreeVarOpConversion, ApplyBinaryNumericConversion,
      ApplyCompareNumericConversion, AssignSubscriptListConversion,
      BoolToPredConversion, BoxOpConversion, MakeListOpBoxedConversion,
      CallOpConversion, ConstantOpConversion, DynamicUnpackOpConversion,
      ElideStaticInfoCast, FailureOpConversion, FuncOpConversion,
      GetTypeCodeConversion, LoadVarOpConversion, MakeTupleOpConversion,
      NegIntegerOpConversion, RaiseOnFailureOpConversion, ReturnOpConversion,
      SequenceCloneBuiltinConversion, StoreVarOpConversion,
      SubscriptOpBuiltinSequenceConversion, UnboxOpConversion>(typeConverter,
                                                               context);

  // External CFG ops.
  patterns.insert<BuiltinBranchConversion, BuiltinCondBranchConversion,
                  BuiltinSelectConversion>(typeConverter, context);

  // Constants and constructors.
  patterns.insert<NoneOpConversion>(typeConverter, context);
}
