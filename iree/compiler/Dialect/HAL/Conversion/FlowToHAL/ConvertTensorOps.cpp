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
#include "iree/compiler/Dialect/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class ConstantTensorOpConversion
    : public OpConversionPattern<mlir::ConstantOp> {
 public:
  ConstantTensorOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  PatternMatchResult matchAndRewrite(
      mlir::ConstantOp constantOp, llvm::ArrayRef<Value *> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    if (!constantOp.getType().isa<TensorType>()) return matchFailure();

    auto *device =
        rewriter.createOrFold<IREE::HAL::ExSharedDeviceOp>(constantOp.getLoc());
    auto *allocator = rewriter.createOrFold<IREE::HAL::DeviceAllocatorOp>(
        constantOp.getLoc(), device);

    // TODO(benvanik): compute from SSA use-def chain uses.
    IREE::HAL::MemoryTypeBitfield memoryTypes =
        IREE::HAL::MemoryTypeBitfield::DeviceLocal |
        IREE::HAL::MemoryTypeBitfield::HostVisible;
    IREE::HAL::BufferUsageBitfield bufferUsage =
        IREE::HAL::BufferUsageBitfield::All |
        IREE::HAL::BufferUsageBitfield::Constant;

    auto *buffer = rewriter.createOrFold<IREE::HAL::AllocatorAllocateConstOp>(
        constantOp.getLoc(), allocator, memoryTypes, bufferUsage,
        constantOp.getValue().cast<ElementsAttr>());

    rewriter.replaceOp(constantOp, {buffer});
    return matchSuccess();
  }
};

class TensorLoadOpConversion
    : public OpConversionPattern<IREE::Flow::TensorLoadOp> {
 public:
  TensorLoadOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  PatternMatchResult matchAndRewrite(
      IREE::Flow::TensorLoadOp loadOp, llvm::ArrayRef<Value *> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorLoadOpOperandAdaptor operands(newOperands);
    auto sourceType = loadOp.source()->getType().cast<ShapedType>();
    auto sourceShape = IREE::HAL::getShapeDims(loadOp.source(), rewriter);
    auto *sourceOffset =
        rewriter.createOrFold<IREE::HAL::BufferViewComputeOffsetOp>(
            loadOp.getLoc(), rewriter.getIntegerType(32), operands.source(),
            sourceShape, operands.indices(),
            APInt(32, IREE::HAL::getRoundedElementByteWidth(
                          sourceType.getElementType())));
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferLoadOp>(
        loadOp, converter.convertType(loadOp.result()->getType()),
        operands.source(), sourceOffset);
    return matchSuccess();
  }

 private:
  TypeConverter &converter;
};

class TensorStoreOpConversion
    : public OpConversionPattern<IREE::Flow::TensorStoreOp> {
 public:
  TensorStoreOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  PatternMatchResult matchAndRewrite(
      IREE::Flow::TensorStoreOp storeOp, llvm::ArrayRef<Value *> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorStoreOpOperandAdaptor operands(newOperands);
    auto targetType = storeOp.target()->getType().cast<ShapedType>();
    auto targetShape = IREE::HAL::getShapeDims(storeOp.target(), rewriter);
    auto *targetOffset =
        rewriter.createOrFold<IREE::HAL::BufferViewComputeOffsetOp>(
            storeOp.getLoc(), rewriter.getIntegerType(32), operands.target(),
            targetShape, operands.indices(),
            APInt(32, IREE::HAL::getRoundedElementByteWidth(
                          targetType.getElementType())));
    rewriter.create<IREE::HAL::BufferStoreOp>(
        storeOp.getLoc(), operands.value(), operands.target(), targetOffset);
    rewriter.replaceOp(storeOp, {operands.value()});
    return matchSuccess();
  }
};

}  // namespace

void populateFlowTensorToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &converter) {
  patterns.insert<ConstantTensorOpConversion, TensorLoadOpConversion,
                  TensorStoreOpConversion>(context, converter);
}

}  // namespace iree_compiler
}  // namespace mlir
