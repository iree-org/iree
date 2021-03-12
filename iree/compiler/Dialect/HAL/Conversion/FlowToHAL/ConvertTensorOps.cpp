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

class ConstantTensorOpConversion
    : public OpConversionPattern<mlir::ConstantOp> {
 public:
  ConstantTensorOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      mlir::ConstantOp constantOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    if (!constantOp.getType().isa<TensorType>()) return failure();

    auto device =
        rewriter.createOrFold<IREE::HAL::ExSharedDeviceOp>(constantOp.getLoc());
    auto allocator = rewriter.createOrFold<IREE::HAL::DeviceAllocatorOp>(
        constantOp.getLoc(), device);

    // TODO(benvanik): compute from SSA use-def chain uses.
    IREE::HAL::MemoryTypeBitfield memoryTypes =
        IREE::HAL::MemoryTypeBitfield::DeviceLocal |
        IREE::HAL::MemoryTypeBitfield::HostVisible;
    IREE::HAL::BufferUsageBitfield bufferUsage =
        IREE::HAL::BufferUsageBitfield::All |
        IREE::HAL::BufferUsageBitfield::Constant;

    auto elementsAttr = constantOp.getValue().cast<ElementsAttr>();
    auto elementsTy = elementsAttr.getType().cast<ShapedType>();

    // Expand boolean elements to the minimum bit widht supported by the HAL
    // (8-bits).
    // To improve memory bandwidth and increase computae we should prefer to
    // pack 1-bit tensors into wider storage before this lossy conversion. For
    // example bitwise ops on 8x32xi1 can be converted to ops on tensor<8xi32>.
    if (elementsTy.getElementType().isInteger(1)) {
      elementsAttr =
          elementsAttr.mapValues(rewriter.getIntegerType(8),
                                 llvm::function_ref<APInt(const APInt &val)>(
                                     [](const APInt &val) -> APInt {
                                       return APInt(8, val.getBoolValue());
                                     }));
    }

    auto buffer = rewriter.createOrFold<IREE::HAL::AllocatorAllocateConstOp>(
        constantOp.getLoc(), allocator, memoryTypes, bufferUsage, elementsAttr);

    rewriter.replaceOp(constantOp, {buffer});
    return success();
  }
};

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
  patterns.insert<ConstantTensorOpConversion, TensorLoadOpConversion,
                  TensorStoreOpConversion, TensorTraceOpConversion>(context,
                                                                    converter);
}

}  // namespace iree_compiler
}  // namespace mlir
