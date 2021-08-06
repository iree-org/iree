// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/IREEToVM/ConvertIREEToVM.h"

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// util.null
//===----------------------------------------------------------------------===//

class NullOpConversion : public OpConversionPattern<IREE::Util::NullOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::Util::NullOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ConstRefZeroOp>(
        op, IREE::VM::RefType::get(op.getType()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// util.byte_buffer.*
//===----------------------------------------------------------------------===//

class ByteBufferConstantOpConversion
    : public OpConversionPattern<IREE::Util::ByteBufferConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::Util::ByteBufferConstantOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::RodataInlineOp>(
        op,
        IREE::VM::RefType::get(
            IREE::VM::BufferType::get(rewriter.getContext())),
        op.value());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Compiler hints
//===----------------------------------------------------------------------===//

class UnreachableOpConversion
    : public OpConversionPattern<IREE::Util::UnreachableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::Util::UnreachableOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::FailOp>(
        srcOp,
        rewriter.createOrFold<IREE::VM::ConstI32Op>(
            srcOp.getLoc(),
            static_cast<int32_t>(IREE::Util::StatusCode::Unknown)),
        srcOp.message());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lists
//===----------------------------------------------------------------------===//

class ListCreateOpConversion
    : public OpConversionPattern<IREE::Util::ListCreateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::ListCreateOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Util::ListCreateOpAdaptor srcOperands(operands);
    rewriter.replaceOpWithNewOp<IREE::VM::ListAllocOp>(
        srcOp, typeConverter->convertType(srcOp.result().getType()),
        srcOperands.initial_capacity());
    return success();
  }
};

class ListSizeOpConversion
    : public OpConversionPattern<IREE::Util::ListSizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::ListSizeOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Util::ListSizeOpAdaptor srcOperands(operands);
    rewriter.replaceOpWithNewOp<IREE::VM::ListSizeOp>(
        srcOp, typeConverter->convertType(srcOp.result().getType()),
        srcOperands.list());
    return success();
  }
};

class ListResizeOpConversion
    : public OpConversionPattern<IREE::Util::ListResizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::ListResizeOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Util::ListResizeOpAdaptor srcOperands(operands);
    rewriter.replaceOpWithNewOp<IREE::VM::ListResizeOp>(
        srcOp, srcOperands.list(), srcOperands.new_size());
    return success();
  }
};

class ListGetOpConversion : public OpConversionPattern<IREE::Util::ListGetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::ListGetOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Util::ListGetOpAdaptor srcOperands(operands);
    auto resultType = typeConverter->convertType(srcOp.result().getType());
    if (resultType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetI32Op>(
          srcOp, resultType, srcOperands.list(), srcOperands.index());
    } else if (resultType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetI64Op>(
          srcOp, resultType, srcOperands.list(), srcOperands.index());
    } else if (resultType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetF32Op>(
          srcOp, resultType, srcOperands.list(), srcOperands.index());
    } else if (resultType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetF64Op>(
          srcOp, resultType, srcOperands.list(), srcOperands.index());
    } else if (!resultType.isIntOrIndexOrFloat()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetRefOp>(
          srcOp, resultType, srcOperands.list(), srcOperands.index());
    } else {
      return srcOp.emitError() << "unsupported list element type in the VM";
    }
    return success();
  }
};

class ListSetOpConversion : public OpConversionPattern<IREE::Util::ListSetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::ListSetOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Util::ListSetOpAdaptor srcOperands(operands);
    auto valueType = srcOperands.value().getType();
    if (valueType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetI32Op>(
          srcOp, srcOperands.list(), srcOperands.index(), srcOperands.value());
    } else if (valueType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetI64Op>(
          srcOp, srcOperands.list(), srcOperands.index(), srcOperands.value());
    } else if (valueType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetF32Op>(
          srcOp, srcOperands.list(), srcOperands.index(), srcOperands.value());
    } else if (valueType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetF64Op>(
          srcOp, srcOperands.list(), srcOperands.index(), srcOperands.value());
    } else if (!valueType.isIntOrIndexOrFloat()) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetRefOp>(
          srcOp, srcOperands.list(), srcOperands.index(), srcOperands.value());
    } else {
      return srcOp.emitError() << "unsupported list element type in the VM";
    }
    return success();
  }
};

}  // namespace

void populateIREEToVMPatterns(MLIRContext *context,
                              TypeConverter &typeConverter,
                              OwningRewritePatternList &patterns) {
  patterns.insert<NullOpConversion>(typeConverter, context);
  patterns.insert<ByteBufferConstantOpConversion>(typeConverter, context);
  patterns.insert<UnreachableOpConversion>(typeConverter, context);

  typeConverter.addConversion(
      [](IREE::Util::ByteBufferType type) -> Optional<Type> {
        return IREE::VM::RefType::get(
            IREE::VM::BufferType::get(type.getContext()));
      });
  typeConverter.addConversion(
      [](IREE::Util::MutableByteBufferType type) -> Optional<Type> {
        return IREE::VM::RefType::get(
            IREE::VM::BufferType::get(type.getContext()));
      });

  typeConverter.addConversion(
      [&typeConverter](IREE::Util::ListType type) -> Optional<Type> {
        Type elementType;
        if (type.getElementType().isa<IREE::Util::VariantType>()) {
          elementType = IREE::VM::OpaqueType::get(type.getContext());
        } else {
          elementType = typeConverter.convertType(type.getElementType());
        }
        if (!elementType) return llvm::None;
        return IREE::VM::RefType::get(IREE::VM::ListType::get(elementType));
      });
  patterns
      .insert<ListCreateOpConversion, ListSizeOpConversion,
              ListResizeOpConversion, ListGetOpConversion, ListSetOpConversion>(
          typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
