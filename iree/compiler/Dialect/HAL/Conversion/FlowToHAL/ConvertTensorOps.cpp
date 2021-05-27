// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class TensorLoadOpConversion
    : public OpConversionPattern<IREE::Flow::TensorLoadOp> {
 public:
  TensorLoadOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::TensorLoadOp loadOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorLoadOp::Adaptor operands(newOperands,
                                               loadOp->getAttrDictionary());
    auto source = IREE::HAL::TensorRewriteAdaptor::getChecked(
        loadOp.getLoc(), loadOp.source(), operands.source(), rewriter);
    if (!source.hasValue()) {
      return loadOp.emitOpError() << "cannot create adaptor for source";
    }

    auto sourceOffset = source->computeOffset(operands.indices());
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferLoadOp>(
        loadOp, converter.convertType(loadOp.result().getType()),
        source->getBuffer(), sourceOffset);
    return success();
  }

 private:
  TypeConverter &converter;
};

class TensorStoreOpConversion
    : public OpConversionPattern<IREE::Flow::TensorStoreOp> {
 public:
  TensorStoreOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::TensorStoreOp storeOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorStoreOp::Adaptor operands(newOperands,
                                                storeOp->getAttrDictionary());
    auto target = IREE::HAL::TensorRewriteAdaptor::getChecked(
        storeOp.getLoc(), storeOp.target(), operands.target(), rewriter);

    if (!target.hasValue()) {
      return storeOp.emitOpError() << "cannot create adaptor for target";
    }

    auto targetOffset = target->computeOffset(operands.indices());
    rewriter.create<IREE::HAL::BufferStoreOp>(
        storeOp.getLoc(), operands.value(), target->getBuffer(), targetOffset);
    rewriter.replaceOp(storeOp, {operands.value()});
    return success();
  }
};

class TensorTraceOpConversion
    : public OpConversionPattern<IREE::Flow::TensorTraceOp> {
 public:
  TensorTraceOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      IREE::Flow::TensorTraceOp traceOp, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = traceOp.getLoc();
    SmallVector<Value, 4> bufferViews;
    for (auto operand : llvm::enumerate(rawOperands)) {
      auto adaptor = IREE::HAL::TensorRewriteAdaptor::get(
          loc, traceOp.getOperand(operand.index()), operand.value(), rewriter);
      bufferViews.emplace_back(adaptor.getBufferView());
    }
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewTraceOp>(
        traceOp, traceOp.keyAttr(), bufferViews);
    return success();
  }
};

}  // namespace

void populateFlowTensorToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &converter) {
  patterns.insert<TensorLoadOpConversion, TensorStoreOpConversion,
                  TensorTraceOpConversion>(context, converter);
}

}  // namespace iree_compiler
}  // namespace mlir
