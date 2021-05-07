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

#include "iree/compiler/Dialect/HAL/Conversion/IREEToHAL/ConvertIREEToHAL.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

class DynamicShapeConstantOpConversion
    : public OpConversionPattern<IREE::DynamicShapeConstantOp> {
 public:
  using OpConversionPattern<IREE::DynamicShapeConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::DynamicShapeConstantOp constantOp,
      llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    assert(newOperands.empty() && "dynamic_shape_constant takes no operands");
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

    auto shapedType = constantOp.value().getType();
    auto elementType =
        IREE::HAL::getElementTypeValue(shapedType.getElementType());
    if (!elementType.hasValue()) {
      return rewriter.notifyMatchFailure(constantOp, "unhandled element type");
    }

    auto buffer = rewriter.createOrFold<IREE::HAL::AllocatorConstantOp>(
        constantOp.getLoc(), IREE::HAL::BufferType::get(rewriter.getContext()),
        allocator, memoryTypes, bufferUsage, constantOp.value());

    SmallVector<Value, 4> shape;
    if (shapedType.getRank() >= 1) {
      for (auto dim : shapedType.getShape()) {
        shape.push_back(rewriter.createOrFold<mlir::ConstantIndexOp>(
            constantOp.getLoc(), dim));
      }
    }

    auto view = rewriter.createOrFold<IREE::HAL::BufferViewCreateOp>(
        constantOp.getLoc(), buffer, elementType.getValue(), shape);

    rewriter.replaceOpWithNewOp<IREE::DoNotOptimizeOp>(constantOp, view);
    return success();
  }
};

template <typename T>
class GenericConvertTypesConversion : public OpConversionPattern<T> {
 public:
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      T op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newTypes;
    bool anyChanged = false;
    for (auto oldNew : llvm::zip(op->getOperands(), newOperands)) {
      auto oldValue = std::get<0>(oldNew);
      auto newValue = std::get<1>(oldNew);
      if (oldValue.getType() != newValue.getType()) {
        anyChanged = true;
        break;
      }
    }
    for (auto oldType : op.getOperation()->getResultTypes()) {
      auto newType = this->getTypeConverter()->convertType(oldType);
      if (oldType != newType) anyChanged = true;
      newTypes.push_back(newType);
    }
    if (!anyChanged) return failure();
    rewriter.replaceOpWithNewOp<T>(op, newTypes, newOperands, op->getAttrs());
    return success();
  }
};

}  // namespace

void populateIREEToHALPatterns(MLIRContext *context, ConversionTarget &target,
                               TypeConverter &typeConverter,
                               OwningRewritePatternList &patterns) {
  target.addIllegalOp<IREE::DynamicShapeConstantOp>();
  patterns.insert<DynamicShapeConstantOpConversion>(context);

  typeConverter.addConversion([&](IREE::ListType type) {
    auto elementType = typeConverter.convertType(type.getElementType());
    return IREE::ListType::get(elementType);
  });

  target.addDynamicallyLegalOp<IREE::ListCreateOp>([&](IREE::ListCreateOp op) {
    return typeConverter.isLegal(op.getType());
  });
  target.addDynamicallyLegalOp<IREE::ListGetOp>(
      [&](IREE::ListGetOp op) { return typeConverter.isLegal(op.getType()); });
  target.addDynamicallyLegalOp<IREE::ListSetOp>([&](IREE::ListSetOp op) {
    return typeConverter.isLegal(op.value().getType());
  });
  patterns.insert<GenericConvertTypesConversion<IREE::ListCreateOp>,
                  GenericConvertTypesConversion<IREE::ListGetOp>,
                  GenericConvertTypesConversion<IREE::ListSetOp>>(typeConverter,
                                                                  context);
}

}  // namespace iree_compiler
}  // namespace mlir
