// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class CommandBufferFillBufferOpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferFillBufferOp> {
 public:
  CommandBufferFillBufferOpConversion(MLIRContext *context,
                                      SymbolTable &importSymbols,
                                      TypeConverter &typeConverter,
                                      StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::CommandBufferFillBufferOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getType();
    IREE::HAL::CommandBufferFillBufferOp::Adaptor newOperands(operands);

    SmallVector<Value, 8> callOperands = {
        newOperands.command_buffer(),
        newOperands.target_buffer(),
        newOperands.target_offset(),
        newOperands.length(),
    };

    // The pattern will be promoted as needed to a 32 bit type, but we still
    // need to bitcast from float to int and track the original pattern length.
    auto originalPatternType = op.pattern().getType();
    auto patternBitWidth = originalPatternType.getIntOrFloatBitWidth();
    auto patternLength = rewriter.createOrFold<mlir::arith::ConstantIntOp>(
        op.getLoc(), patternBitWidth / 8, 32);
    Value pattern;
    if (originalPatternType.isInteger(8) || originalPatternType.isInteger(16) ||
        originalPatternType.isInteger(32)) {
      pattern = op.pattern();
    } else if (originalPatternType.isF32()) {
      pattern = rewriter.createOrFold<arith::BitcastOp>(
          op.getLoc(), rewriter.getIntegerType(patternBitWidth), op.pattern());
    } else {
      // Note: f16 in particular would need special handling since a promotion
      // to f32 changes the bit representation.
      return op.emitOpError()
             << "unhandled fill buffer type: " << originalPatternType;
    }
    callOperands.push_back(pattern);
    callOperands.push_back(patternLength);

    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, SymbolRefAttr::get(importOp), importType.getResults(),
        callOperands);

    copyImportAttrs(importOp, callOp);
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

class CommandBufferPushDescriptorSetOpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferPushDescriptorSetOp> {
 public:
  CommandBufferPushDescriptorSetOpConversion(MLIRContext *context,
                                             SymbolTable &importSymbols,
                                             TypeConverter &typeConverter,
                                             StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::CommandBufferPushDescriptorSetOp op,
      llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getType();
    IREE::HAL::CommandBufferPushDescriptorSetOp::Adaptor newOperands(operands);

    SmallVector<Value, 8> callOperands = {
        newOperands.command_buffer(),
        newOperands.executable_layout(),
        newOperands.set(),
    };
    SmallVector<int16_t, 5> segmentSizes = {
        /*command_buffer=*/-1,
        /*executable_layout=*/-1,
        /*set=*/-1,
        /*bindings=*/
        static_cast<int16_t>(newOperands.binding_ordinals().size()),
    };
    for (size_t i = 0; i < newOperands.binding_ordinals().size(); ++i) {
      callOperands.push_back(newOperands.binding_ordinals()[i]);
      callOperands.push_back(newOperands.binding_buffers()[i]);
      callOperands.push_back(newOperands.binding_offsets()[i]);
      callOperands.push_back(newOperands.binding_lengths()[i]);
    }

    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallVariadicOp>(
        op, SymbolRefAttr::get(importOp), importType.getResults(), segmentSizes,
        importType.getInputs(), callOperands);
    copyImportAttrs(importOp, callOp);
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

}  // namespace

void populateHALCommandBufferToVMPatterns(MLIRContext *context,
                                          SymbolTable &importSymbols,
                                          TypeConverter &typeConverter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferCreateOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferBeginOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.begin");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferEndOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.end");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferBeginDebugGroupOp>>(
          context, importSymbols, typeConverter,
          "hal.command_buffer.begin_debug_group");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferEndDebugGroupOp>>(
          context, importSymbols, typeConverter,
          "hal.command_buffer.end_debug_group");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferExecutionBarrierOp>>(
          context, importSymbols, typeConverter,
          "hal.command_buffer.execution_barrier");
  patterns.insert<CommandBufferFillBufferOpConversion>(
      context, importSymbols, typeConverter, "hal.command_buffer.fill_buffer");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferCopyBufferOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.copy_buffer");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferPushConstantsOp>>(
          context, importSymbols, typeConverter,
          "hal.command_buffer.push_constants");
  patterns.insert<CommandBufferPushDescriptorSetOpConversion>(
      context, importSymbols, typeConverter,
      "hal.command_buffer.push_descriptor_set");
  patterns.insert<
      VMImportOpConversion<IREE::HAL::CommandBufferBindDescriptorSetOp>>(
      context, importSymbols, typeConverter,
      "hal.command_buffer.bind_descriptor_set");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferDispatchOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.dispatch");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferDispatchIndirectOp>>(
          context, importSymbols, typeConverter,
          "hal.command_buffer.dispatch.indirect");
}

}  // namespace iree_compiler
}  // namespace mlir
