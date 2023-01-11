// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

// TODO(benvanik): import op handling of optional values.
// It'd be nice if the Optional<Index>:$binding_capacity could be emitted as 0
// when not present; today it'll be omitted entirely (as it's not in the operand
// set) but we need it for the fixed call signature.
class CommandBufferCreateOpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferCreateOp> {
 public:
  CommandBufferCreateOpConversion(MLIRContext *context,
                                  SymbolTable &importSymbols,
                                  TypeConverter &typeConverter,
                                  StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::CommandBufferCreateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    SmallVector<Value, 8> callOperands = {
        adaptor.getDevice(),
    };
    auto modesValue = detail::rewriteAttrToOperands(
        op.getLoc(), adaptor.getModesAttr(), rewriter.getI32Type(), rewriter);
    if (!modesValue.has_value()) return failure();
    callOperands.append(modesValue.value());
    auto categoriesValue = detail::rewriteAttrToOperands(
        op.getLoc(), adaptor.getCommandCategoriesAttr(), rewriter.getI32Type(),
        rewriter);
    if (!categoriesValue.has_value()) return failure();
    callOperands.append(categoriesValue.value());
    if (adaptor.getBindingCapacity()) {
      callOperands.push_back(castToImportType(adaptor.getBindingCapacity(),
                                              rewriter.getI32Type(), rewriter));
    } else {
      callOperands.push_back(
          rewriter.create<IREE::VM::ConstI32ZeroOp>(op.getLoc()));
    }

    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, SymbolRefAttr::get(importOp), importType.getResults(),
        callOperands);

    copyImportAttrs(importOp, callOp);
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

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
      IREE::HAL::CommandBufferFillBufferOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    SmallVector<Value, 8> callOperands = {
        adaptor.getCommandBuffer(),
        adaptor.getTargetBuffer(),
        castToImportType(adaptor.getTargetOffset(), rewriter.getI64Type(),
                         rewriter),
        castToImportType(adaptor.getLength(), rewriter.getI64Type(), rewriter),
    };

    // Record the original pattern length then extend it to a 32 bit integer.
    auto originalPatternType = op.getPattern().getType();
    auto patternBitWidth = originalPatternType.getIntOrFloatBitWidth();
    // The pattern length (in bytes) will be used at runtime to issue the fill
    // command. While the pattern itself will be stored in a 32 bit integer,
    // the fill operation will use this length to slice a potentially smaller
    // range of bits from the full pattern.
    auto patternLengthBytes =
        IREE::Util::getRoundedElementByteWidth(originalPatternType);
    auto patternLengthConst = rewriter.createOrFold<mlir::arith::ConstantIntOp>(
        op.getLoc(), patternLengthBytes, 32);
    Value pattern = op.getPattern();
    if (patternBitWidth < 32) {
      pattern = rewriter.createOrFold<arith::ExtUIOp>(
          op.getLoc(), rewriter.getIntegerType(32), pattern);
    }
    callOperands.push_back(pattern);
    callOperands.push_back(patternLengthConst);

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
      IREE::HAL::CommandBufferPushDescriptorSetOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    // Memoize zeros/nulls ala IndexSet.
    // Since there are usually hundreds to thousands of these push ops and each
    // one can have 5-10 of these this saves us a tremendous amount of time
    // creating/verifying/pattern matching/folding/CSE'ing.
    // We could extend IndexSet into a ConstantSet that could use these custom
    // VM ops instead of just arith.constant in order to make this more
    // reusable.
    Value zero;
    auto getI32Zero = [&]() {
      if (!zero) {
        zero = rewriter.create<IREE::VM::ConstI32ZeroOp>(op.getLoc());
      }
      return zero;
    };
    Value null;
    auto getNull = [&]() {
      if (!null) {
        null = rewriter.create<IREE::VM::ConstRefZeroOp>(
            op.getLoc(),
            IREE::VM::RefType::get(rewriter.getType<IREE::HAL::BufferType>()));
      }
      return null;
    };

    SmallVector<Value, 8> callOperands = {
        adaptor.getCommandBuffer(),
        adaptor.getPipelineLayout(),
        adaptor.getSet(),
    };
    SmallVector<int16_t, 5> segmentSizes = {
        /*command_buffer=*/-1,
        /*pipeline_layout=*/-1,
        /*set=*/-1,
        /*bindings=*/
        static_cast<int16_t>(adaptor.getBindingOrdinals().size()),
    };
    for (size_t i = 0; i < adaptor.getBindingOrdinals().size(); ++i) {
      callOperands.push_back(adaptor.getBindingOrdinals()[i]);
      auto bindingBuffer = adaptor.getBindingBuffers()[i];
      if (bindingBuffer.getType().isa<IREE::VM::RefType>()) {
        // Buffer binding; pass 0 for table slot.
        callOperands.push_back(getI32Zero());
        callOperands.push_back(bindingBuffer);
      } else {
        // Binding table reference; pass null for the buffer.
        callOperands.push_back(bindingBuffer);
        callOperands.push_back(getNull());
      }
      callOperands.push_back(castToImportType(adaptor.getBindingOffsets()[i],
                                              rewriter.getI64Type(), rewriter));
      callOperands.push_back(castToImportType(adaptor.getBindingLengths()[i],
                                              rewriter.getI64Type(), rewriter));
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
                                          RewritePatternSet &patterns) {
  patterns.insert<CommandBufferCreateOpConversion>(
      context, importSymbols, typeConverter, "hal.command_buffer.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferFinalizeOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.finalize");
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
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferDispatchOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.dispatch");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferDispatchIndirectOp>>(
          context, importSymbols, typeConverter,
          "hal.command_buffer.dispatch.indirect");
}

}  // namespace iree_compiler
}  // namespace mlir
