// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/StandardToHAL/ConvertStandardToHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
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
      elementsAttr = elementsAttr.cast<DenseIntOrFPElementsAttr>().mapValues(
          rewriter.getIntegerType(8),
          llvm::function_ref<APInt(const APInt &val)>(
              [](const APInt &val) -> APInt {
                return APInt(8, val.getBoolValue());
              }));
    }

    auto buffer = rewriter.createOrFold<IREE::HAL::AllocatorConstantOp>(
        constantOp.getLoc(), IREE::HAL::BufferType::get(rewriter.getContext()),
        allocator, memoryTypes, bufferUsage, elementsAttr);

    rewriter.replaceOp(constantOp, {buffer});
    return success();
  }
};

}  // namespace

void populateStandardConstantToHALPatterns(MLIRContext *context,
                                           OwningRewritePatternList &patterns,
                                           TypeConverter &converter) {
  patterns.insert<ConstantTensorOpConversion>(context, converter);
}

}  // namespace iree_compiler
}  // namespace mlir
