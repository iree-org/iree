// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
// iree.null
//===----------------------------------------------------------------------===//

class NullOpConversion : public OpConversionPattern<IREE::NullOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::NullOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ConstRefZeroOp>(
        op, IREE::VM::RefType::get(op.getType()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// iree.byte_buffer.*
//===----------------------------------------------------------------------===//

class ByteBufferConstantOpConversion
    : public OpConversionPattern<IREE::ByteBufferConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::ByteBufferConstantOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::RodataInlineOp>(
        op, IREE::VM::RefType::get(op.getType()), op.value());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Compiler hints
//===----------------------------------------------------------------------===//

class UnreachableOpConversion
    : public OpConversionPattern<IREE::UnreachableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::UnreachableOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::FailOp>(
        srcOp,
        rewriter.createOrFold<mlir::ConstantIntOp>(
            srcOp.getLoc(), static_cast<int32_t>(IREE::StatusCode::Unknown),
            32),
        "unreachable location reached");
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lists
//===----------------------------------------------------------------------===//

class ListSizeOpConversion : public OpConversionPattern<IREE::ListSizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::ListSizeOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::ListSizeOpAdaptor srcOperands(operands);
    rewriter.replaceOpWithNewOp<IREE::VM::ListSizeOp>(
        srcOp, typeConverter->convertType(srcOp.result().getType()),
        srcOperands.list());
    return success();
  }
};

class ListResizeOpConversion : public OpConversionPattern<IREE::ListResizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::ListResizeOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::ListResizeOpAdaptor srcOperands(operands);
    rewriter.replaceOpWithNewOp<IREE::VM::ListResizeOp>(
        srcOp, srcOperands.list(), srcOperands.new_size());
    return success();
  }
};

class ListGetOpConversion : public OpConversionPattern<IREE::ListGetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::ListGetOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::ListGetOpAdaptor srcOperands(operands);
    auto resultType = typeConverter->convertType(srcOp.result().getType());
    if (resultType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetI32Op>(
          srcOp, resultType, srcOperands.list(), srcOperands.index());
    } else if (resultType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListGetI64Op>(
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

class ListSetOpConversion : public OpConversionPattern<IREE::ListSetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::ListSetOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::ListSetOpAdaptor srcOperands(operands);
    auto valueType = srcOperands.value().getType();
    if (valueType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetI32Op>(
          srcOp, srcOperands.list(), srcOperands.index(), srcOperands.value());
    } else if (valueType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ListSetI64Op>(
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
      [&typeConverter](IREE::ListType type) -> Optional<Type> {
        auto elementType = typeConverter.convertType(type.getElementType());
        if (!elementType) return llvm::None;
        return IREE::VM::RefType::get(IREE::VM::ListType::get(elementType));
      });
  patterns.insert<ListSizeOpConversion, ListResizeOpConversion,
                  ListGetOpConversion, ListSetOpConversion>(typeConverter,
                                                            context);
}

}  // namespace iree_compiler
}  // namespace mlir
