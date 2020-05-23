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

#include "iree/compiler/Dialect/VMLA/Conversion/HALToVMLA/ConvertHALToVMLA.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VMLA/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLADialect.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct InterfaceOpEraser : public OpConversionPattern<IREE::HAL::InterfaceOp> {
  using OpConversionPattern<IREE::HAL::InterfaceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceOp interfaceOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(interfaceOp);
    return success();
  }
};

struct InterfaceLoadConstantOpConversion
    : public OpConversionPattern<IREE::HAL::InterfaceLoadConstantOp> {
  InterfaceLoadConstantOpConversion(MLIRContext *context,
                                    TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceLoadConstantOp loadOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Find the vmla.interface argument to the function.
    auto interfaceArg = loadOp.getParentOfType<FuncOp>().getArgument(0);
    assert(interfaceArg &&
           interfaceArg.getType().isa<IREE::VMLA::InterfaceType>() &&
           "exported VMLA functions require vmla.interface ops as their only "
           "argument");

    IREE::HAL::InterfaceLoadConstantOp::Adaptor newOperands(operands);
    rewriter.replaceOpWithNewOp<IREE::VMLA::InterfaceConstOp>(
        loadOp, typeConverter.convertType(loadOp.getResult().getType()),
        interfaceArg, loadOp.offsetAttr());
    return success();
  }

  TypeConverter &typeConverter;
};

struct InterfaceLoadTensorOpConversion
    : public OpConversionPattern<IREE::HAL::InterfaceLoadTensorOp> {
  InterfaceLoadTensorOpConversion(MLIRContext *context,
                                  TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceLoadTensorOp loadOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Find the vmla.interface argument to the function.
    auto interfaceArg = loadOp.getParentOfType<FuncOp>().getArgument(0);
    assert(interfaceArg &&
           interfaceArg.getType().isa<IREE::VMLA::InterfaceType>() &&
           "exported VMLA functions require vmla.interface ops as their only "
           "argument");
    auto bindingOp = loadOp.queryBindingOp();

    IREE::HAL::InterfaceLoadTensorOp::Adaptor newOperands(operands);
    auto bufferOp = rewriter.create<IREE::VMLA::InterfaceBindingOp>(
        loadOp.getLoc(), IREE::VMLA::BufferType::get(loadOp.getContext()),
        interfaceArg, bindingOp.set(), bindingOp.binding());
    auto byteLengthValue = VMLAConversionTarget::getBufferLength(
        loadOp.getLoc(), loadOp.result(), typeConverter, rewriter);
    if (!byteLengthValue) return failure();
    rewriter.replaceOpWithNewOp<IREE::VMLA::BufferViewOp>(
        loadOp, IREE::VMLA::BufferType::get(loadOp.getContext()),
        bufferOp.result(), newOperands.offset(), byteLengthValue);
    return success();
  }

  TypeConverter &typeConverter;
};

struct InterfaceStoreTensorOpConversion
    : public OpConversionPattern<IREE::HAL::InterfaceStoreTensorOp> {
  InterfaceStoreTensorOpConversion(MLIRContext *context,
                                   TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceStoreTensorOp storeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Find the vmla.interface argument to the function.
    auto interfaceArg = storeOp.getParentOfType<FuncOp>().getArgument(0);
    assert(interfaceArg.getType().isa<IREE::VMLA::InterfaceType>() &&
           "exported VMLA functions require vmla.interface ops as their only "
           "argument");
    auto bindingOp = storeOp.queryBindingOp();

    IREE::HAL::InterfaceStoreTensorOp::Adaptor newOperands(operands);
    auto bufferOp = rewriter.create<IREE::VMLA::InterfaceBindingOp>(
        storeOp.getLoc(), IREE::VMLA::BufferType::get(storeOp.getContext()),
        interfaceArg, bindingOp.set(), bindingOp.binding());

    auto zeroValue =
        rewriter.createOrFold<mlir::ConstantIndexOp>(storeOp.getLoc(), 0);
    auto byteLengthValue = VMLAConversionTarget::getBufferLength(
        storeOp.getLoc(), storeOp.operand(), typeConverter, rewriter);
    rewriter.create<IREE::VMLA::BufferCopyOp>(
        storeOp.getLoc(), newOperands.operand(), zeroValue, bufferOp,
        newOperands.offset(), byteLengthValue);
    rewriter.replaceOp(storeOp, {});
    return success();
  }

  TypeConverter &typeConverter;
};

}  // namespace

void populateHALToVMLAPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns,
                               TypeConverter &typeConverter) {
  patterns.insert<InterfaceOpEraser>(context);
  patterns.insert<InterfaceLoadConstantOpConversion>(context, typeConverter);
  patterns.insert<InterfaceLoadTensorOpConversion>(context, typeConverter);
  patterns.insert<InterfaceStoreTensorOpConversion>(context, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
