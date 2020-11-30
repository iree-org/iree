// Copyright 2020 Google LLC
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

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/conversion/convert_flow_to_hal.h"

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

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

class ReserveOpConversion : public OpConversionPattern<tf_tensorlist::Reserve> {
 public:
  ReserveOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      tf_tensorlist::Reserve reserveOp, llvm::ArrayRef<Value> newOperands,
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

class ConcatOpConversion : public OpConversionPattern<tf_tensorlist::Concat> {
 public:
  ConcatOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      tf_tensorlist::Concat concatOp, llvm::ArrayRef<Value> newOperands,
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
        newConcatOp.getLoc(), newConcatOp);

    rewriter.replaceOp(concatOp, bufferOp);
    return success();
  }
};

class StackOpConversion : public OpConversionPattern<tf_tensorlist::Stack> {
 public:
  StackOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      tf_tensorlist::Stack stackOp, llvm::ArrayRef<Value> newOperands,
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
        stackOp.getLoc(), newStackOp);

    rewriter.replaceOp(stackOp, bufferOp);
    return success();
  }
};

}  // namespace

void populateTensorListToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &typeConverter) {
  // We can use the HAL conversion handler for this tensor->buffer conversion
  // as we just want the simple form. If we wanted to perform additional
  // verification or have a specific use case (such as a place where only the
  // buffer is required and the shape is not) we could add our own.
  patterns.insert<
      HALOpConversion<tf_tensorlist::GetItem, IREE::TensorList::GetItem>>(
      context, typeConverter);
  patterns.insert<
      HALOpConversion<tf_tensorlist::SetItem, IREE::TensorList::SetItem>>(
      context, typeConverter);
  patterns.insert<
      HALOpConversion<tf_tensorlist::FromTensor, IREE::TensorList::FromTensor>>(
      context, typeConverter);

  patterns.insert<ConcatOpConversion>(context, typeConverter);
  patterns.insert<ReserveOpConversion>(context, typeConverter);
  patterns.insert<StackOpConversion>(context, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
