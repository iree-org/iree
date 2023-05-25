// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/Conversion/UtilToVM/ConvertUtilToVM.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// util.align
//===----------------------------------------------------------------------===//

template <typename CONST_OP, typename SUB_OP, typename ADD_OP, typename NOT_OP,
          typename AND_OP>
void insertAlignOps(IREE::Util::AlignOp srcOp,
                    ConversionPatternRewriter &rewriter,
                    IREE::Util::AlignOpAdaptor adaptor,
                    IntegerType integerType) {
  // Aligns |value| up to the given power-of-two |alignment| if required.
  // (value + (alignment - 1)) & ~(alignment - 1)
  // https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
  auto oneConstant = rewriter.createOrFold<CONST_OP>(srcOp.getLoc(), 1);
  // (alignment - 1)
  auto alignmentValue = rewriter.createOrFold<SUB_OP>(
      srcOp.getLoc(), integerType, adaptor.getAlignment(), oneConstant);
  // value + (alignment - 1)
  auto valueAddedValue = rewriter.createOrFold<ADD_OP>(
      srcOp.getLoc(), integerType, adaptor.getValue(), alignmentValue);
  // ~(alignment - 1)
  auto notAlignmentValue = rewriter.createOrFold<NOT_OP>(
      srcOp.getLoc(), integerType, alignmentValue);
  // (value + (alignment - 1)) & ~(alignment - 1)
  auto andValue = rewriter.createOrFold<AND_OP>(
      srcOp.getLoc(), integerType, valueAddedValue, notAlignmentValue);
  rewriter.replaceOp(srcOp, andValue);
}

struct AlignOpConversion : public OpConversionPattern<IREE::Util::AlignOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::AlignOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type valueType = adaptor.getValue().getType();
    if (valueType.isInteger(32)) {
      insertAlignOps<IREE::VM::ConstI32Op, IREE::VM::SubI32Op,
                     IREE::VM::AddI32Op, IREE::VM::NotI32Op,
                     IREE::VM::AndI32Op>(srcOp, rewriter, adaptor,
                                         rewriter.getI32Type());
    } else if (valueType.isInteger(64)) {
      insertAlignOps<IREE::VM::ConstI64Op, IREE::VM::SubI64Op,
                     IREE::VM::AddI64Op, IREE::VM::NotI64Op,
                     IREE::VM::AndI64Op>(srcOp, rewriter, adaptor,
                                         rewriter.getI64Type());
    } else {
      return srcOp.emitError()
             << "unsupported value type for util.align: " << valueType;
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// util.sizeof
//===----------------------------------------------------------------------===//

/// For a `sizeof index` operation, invokes the type converter to derive the
/// concrete type for index and rewrites to that. This allows us to do late
/// resolution of the size of the index type at the point of conversion to VM
/// where it is known.
struct FixateIndexSizeofConversion
    : public OpConversionPattern<IREE::Util::SizeOfOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::SizeOfOp sizeofOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type sizedType = sizeofOp.getSizedType();
    if (llvm::isa<IndexType>(sizedType)) {
      Type converted = getTypeConverter()->convertType(sizedType);
      if (converted) {
        Value newSizeof = rewriter.createOrFold<IREE::Util::SizeOfOp>(
            sizeofOp.getLoc(), converted);
        rewriter.replaceOp(sizeofOp, newSizeof);
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

void populateUtilAlignmentToVMPatterns(MLIRContext *context,
                                       ConversionTarget &conversionTarget,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  conversionTarget.addIllegalOp<IREE::Util::AlignOp>();
  conversionTarget.addIllegalOp<IREE::Util::SizeOfOp>();

  patterns.insert<AlignOpConversion, FixateIndexSizeofConversion>(typeConverter,
                                                                  context);
}

}  // namespace iree_compiler
}  // namespace mlir
