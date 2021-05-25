// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/TensorList/Conversion/ConversionPatterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace TensorList {

namespace {

Value getBufferView(Operation *srcOp, Value srcOperand, Value dstOperand,
                    ConversionPatternRewriter &rewriter) {
  auto operand = IREE::HAL::TensorRewriteAdaptor::getChecked(
      srcOp->getLoc(), srcOperand, dstOperand, rewriter);
  if (!operand.hasValue()) {
    srcOp->emitOpError() << "unable to create adaptor for operand";
    return nullptr;
  }
  auto bufferView = operand->getBufferView();
  if (!bufferView) {
    srcOp->emitOpError() << "unable to get buffer view for operand";
    return nullptr;
  }

  return bufferView;
}

class ReserveOpConversion
    : public OpConversionPattern<IREE::TensorList::ReserveTensor> {
 public:
  ReserveOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      IREE::TensorList::ReserveTensor reserveOp,
      llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    auto elementTy = reserveOp.element_type();
    auto element_value = IREE::HAL::getElementTypeValue(elementTy).getValue();

    auto operand0 = getBufferView(reserveOp, reserveOp.getOperand(0),
                                  newOperands[0], rewriter);
    auto operand1 = getBufferView(reserveOp, reserveOp.getOperand(1),
                                  newOperands[1], rewriter);

    if (!operand0 || !operand1) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::TensorList::Reserve>(
        reserveOp,
        IREE::TensorList::TensorListType::get(reserveOp.getContext()), operand0,
        operand1, rewriter.getI32IntegerAttr(element_value));
    return success();
  }
};

class ConcatOpConversion
    : public OpConversionPattern<IREE::TensorList::ConcatTensor> {
 public:
  ConcatOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      IREE::TensorList::ConcatTensor concatOp,
      llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    auto device =
        rewriter.createOrFold<IREE::HAL::ExSharedDeviceOp>(concatOp.getLoc());
    auto allocator =
        rewriter.create<IREE::HAL::DeviceAllocatorOp>(concatOp.getLoc(), device)
            .getResult();

    auto newConcatOp = rewriter.createOrFold<IREE::TensorList::Concat>(
        concatOp.getLoc(),
        IREE::HAL::BufferViewType::get(rewriter.getContext()), allocator,
        newOperands[0]);

    auto bufferOp = rewriter.createOrFold<IREE::HAL::BufferViewBufferOp>(
        newConcatOp.getLoc(), IREE::HAL::BufferType::get(rewriter.getContext()),
        newConcatOp);

    rewriter.replaceOp(concatOp, bufferOp);
    return success();
  }
};

class StackOpConversion
    : public OpConversionPattern<IREE::TensorList::StackTensor> {
 public:
  StackOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      IREE::TensorList::StackTensor stackOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    auto device =
        rewriter.createOrFold<IREE::HAL::ExSharedDeviceOp>(stackOp.getLoc());
    auto allocator =
        rewriter.create<IREE::HAL::DeviceAllocatorOp>(stackOp.getLoc(), device)
            .getResult();

    auto operand1 =
        getBufferView(stackOp, stackOp.getOperand(1), newOperands[1], rewriter);
    if (!operand1) return failure();

    auto newStackOp = rewriter.createOrFold<IREE::TensorList::Stack>(
        stackOp.getLoc(), IREE::HAL::BufferViewType::get(rewriter.getContext()),
        allocator, newOperands[0], operand1);

    auto bufferOp = rewriter.createOrFold<IREE::HAL::BufferViewBufferOp>(
        stackOp.getLoc(), IREE::HAL::BufferType::get(rewriter.getContext()),
        newStackOp);

    rewriter.replaceOp(stackOp, bufferOp);
    return success();
  }
};

}  // namespace

void populateTensorListToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &typeConverter) {
  patterns.insert<HALOpConversion<IREE::TensorList::FromTensor,
                                  IREE::TensorList::FromTensor>>(context,
                                                                 typeConverter);
  patterns.insert<
      HALOpConversion<IREE::TensorList::GetItem, IREE::TensorList::GetItem>>(
      context, typeConverter);
  patterns.insert<
      HALOpConversion<IREE::TensorList::SetItem, IREE::TensorList::SetItem>>(
      context, typeConverter);
  patterns.insert<ConcatOpConversion>(context, typeConverter);
  patterns.insert<ReserveOpConversion>(context, typeConverter);
  patterns.insert<StackOpConversion>(context, typeConverter);
}

void populateTensorListToVMPatterns(MLIRContext *context,
                                    SymbolTable &importSymbols,
                                    OwningRewritePatternList &patterns,
                                    TypeConverter &typeConverter) {
  patterns.insert<VMImportOpConversion<IREE::TensorList::Reserve>>(
      context, importSymbols, typeConverter, "tensorlist.reserve");
  patterns.insert<VMImportOpConversion<IREE::TensorList::GetItem>>(
      context, importSymbols, typeConverter, "tensorlist.get_item");
  patterns.insert<VMImportOpConversion<IREE::TensorList::SetItem>>(
      context, importSymbols, typeConverter, "tensorlist.set_item");
  patterns.insert<VMImportOpConversion<IREE::TensorList::FromTensor>>(
      context, importSymbols, typeConverter, "tensorlist.from_tensor");
  patterns.insert<VMImportOpConversion<IREE::TensorList::Concat>>(
      context, importSymbols, typeConverter, "tensorlist.concat");
  patterns.insert<VMImportOpConversion<IREE::TensorList::Stack>>(
      context, importSymbols, typeConverter, "tensorlist.stack");
}

}  // namespace TensorList
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
