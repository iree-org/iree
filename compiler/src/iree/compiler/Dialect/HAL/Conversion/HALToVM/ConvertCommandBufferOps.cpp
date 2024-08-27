// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

// Returns a slot value and a buffer ref value.
// |bufferOrSlot| is intended to be a `AnyTypeOf<[Index, HAL_BufferType]>` in
// the op definition.
static std::tuple<Value, Value>
splitBufferSlot(Location loc, Value bufferOrSlot, OpBuilder &builder) {
  if (!bufferOrSlot) {
    return std::make_tuple(
        builder.create<IREE::VM::ConstI32ZeroOp>(loc),
        builder.create<IREE::VM::ConstRefZeroOp>(
            loc,
            IREE::VM::RefType::get(builder.getType<IREE::HAL::BufferType>())));
  } else if (isa<IREE::VM::RefType>(bufferOrSlot.getType())) {
    // Direct buffer binding; pass 0 for table slot.
    return std::make_tuple(builder.create<IREE::VM::ConstI32ZeroOp>(loc),
                           bufferOrSlot);
  } else {
    // Indirect binding table reference; pass null for the buffer.
    return std::make_tuple(
        castToImportType(bufferOrSlot, builder.getI32Type(), builder),
        builder.create<IREE::VM::ConstRefZeroOp>(
            loc,
            IREE::VM::RefType::get(builder.getType<IREE::HAL::BufferType>())));
  }
}

// TODO(benvanik): import op handling of optional values.
// It'd be nice if the std::optional<Index>:$binding_capacity could be emitted
// as 0 when not present; today it'll be omitted entirely (as it's not in the
// operand set) but we need it for the fixed call signature.
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

  LogicalResult
  matchAndRewrite(IREE::HAL::CommandBufferCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    SmallVector<Value, 8> callOperands = {
        adaptor.getDevice(),
    };
    auto modesValue = detail::rewriteAttrToOperands(
        op.getLoc(), adaptor.getModesAttr(), rewriter.getI32Type(), rewriter);
    if (!modesValue.has_value())
      return failure();
    callOperands.append(modesValue.value());
    auto categoriesValue = detail::rewriteAttrToOperands(
        op.getLoc(), adaptor.getCommandCategoriesAttr(), rewriter.getI32Type(),
        rewriter);
    if (!categoriesValue.has_value())
      return failure();
    callOperands.append(categoriesValue.value());
    callOperands.push_back(adaptor.getQueueAffinity());
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

  LogicalResult
  matchAndRewrite(IREE::HAL::CommandBufferFillBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    auto [targetBufferSlot, targetBuffer] =
        splitBufferSlot(op.getLoc(), adaptor.getTargetBuffer(), rewriter);
    SmallVector<Value, 8> callOperands = {
        adaptor.getCommandBuffer(),
        targetBuffer,
        castToImportType(adaptor.getTargetOffset(), rewriter.getI64Type(),
                         rewriter),
        castToImportType(adaptor.getLength(), rewriter.getI64Type(), rewriter),
        targetBufferSlot,
    };

    // Record the original pattern length then extend it to a 32 bit integer.
    auto originalPatternType = op.getPattern().getType();
    unsigned patternBitWidth = IREE::Util::getTypeBitWidth(originalPatternType);
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

class CommandBufferUpdateBufferOpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferUpdateBufferOp> {
public:
  CommandBufferUpdateBufferOpConversion(MLIRContext *context,
                                        SymbolTable &importSymbols,
                                        TypeConverter &typeConverter,
                                        StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::CommandBufferUpdateBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();
    auto [targetBufferSlot, targetBuffer] =
        splitBufferSlot(op.getLoc(), adaptor.getTargetBuffer(), rewriter);
    SmallVector<Value, 8> callOperands = {
        adaptor.getCommandBuffer(),
        adaptor.getSourceBuffer(),
        castToImportType(adaptor.getSourceOffset(), rewriter.getI64Type(),
                         rewriter),
        targetBuffer,
        castToImportType(adaptor.getTargetOffset(), rewriter.getI64Type(),
                         rewriter),
        castToImportType(adaptor.getLength(), rewriter.getI64Type(), rewriter),
        targetBufferSlot};
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, SymbolRefAttr::get(importOp), importType.getResults(),
        callOperands);
    copyImportAttrs(importOp, callOp);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

class CommandBufferCopyBufferOpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferCopyBufferOp> {
public:
  CommandBufferCopyBufferOpConversion(MLIRContext *context,
                                      SymbolTable &importSymbols,
                                      TypeConverter &typeConverter,
                                      StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::CommandBufferCopyBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();
    auto [sourceBufferSlot, sourceBuffer] =
        splitBufferSlot(op.getLoc(), adaptor.getSourceBuffer(), rewriter);
    auto [targetBufferSlot, targetBuffer] =
        splitBufferSlot(op.getLoc(), adaptor.getTargetBuffer(), rewriter);
    SmallVector<Value, 8> callOperands = {
        adaptor.getCommandBuffer(),
        sourceBufferSlot,
        targetBufferSlot,
        sourceBuffer,
        castToImportType(adaptor.getSourceOffset(), rewriter.getI64Type(),
                         rewriter),
        targetBuffer,
        castToImportType(adaptor.getTargetOffset(), rewriter.getI64Type(),
                         rewriter),
        castToImportType(adaptor.getLength(), rewriter.getI64Type(), rewriter),
    };
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, SymbolRefAttr::get(importOp), importType.getResults(),
        callOperands);
    copyImportAttrs(importOp, callOp);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

class CommandBufferCollectiveOpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferCollectiveOp> {
public:
  CommandBufferCollectiveOpConversion(MLIRContext *context,
                                      SymbolTable &importSymbols,
                                      TypeConverter &typeConverter,
                                      StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::CommandBufferCollectiveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    Value zeroI64;
    auto getZeroI64 = [&]() {
      if (!zeroI64) {
        zeroI64 = rewriter.create<IREE::VM::ConstI64ZeroOp>(op.getLoc());
      }
      return zeroI64;
    };

    // %command_buffer : !vm.ref<!hal.command_buffer>,
    // %channel : !vm.ref<!hal.channel>,
    // %op : i32,
    // %param : i32,
    // %send_buffer_slot : i32,
    // %recv_buffer_slot : i32,
    // %send_buffer : !vm.ref<!hal.buffer>,
    // %recv_buffer : !vm.ref<!hal.buffer>,
    // %send_offset : i64,
    // %send_length : i64,
    // %recv_offset : i64,
    // %recv_length : i64,
    // %element_count : i64
    SmallVector<Value> callOperands;
    callOperands.push_back(adaptor.getCommandBuffer());
    callOperands.push_back(adaptor.getChannel());
    callOperands.push_back(rewriter.create<IREE::VM::ConstI32Op>(
        op.getLoc(), adaptor.getOp().getEncodedValue()));
    if (auto paramValue = adaptor.getParam()) {
      callOperands.push_back(paramValue);
    } else {
      callOperands.push_back(
          rewriter.create<IREE::VM::ConstI32ZeroOp>(op.getLoc()));
    }

    auto [sendBufferSlot, sendBuffer] =
        splitBufferSlot(op.getLoc(), adaptor.getSendBuffer(), rewriter);
    auto [recvBufferSlot, recvBuffer] =
        splitBufferSlot(op.getLoc(), adaptor.getRecvBuffer(), rewriter);
    callOperands.push_back(sendBufferSlot);
    callOperands.push_back(recvBufferSlot);
    callOperands.push_back(sendBuffer);
    callOperands.push_back(recvBuffer);
    callOperands.push_back(adaptor.getSendOffset() ? adaptor.getSendOffset()
                                                   : getZeroI64());
    callOperands.push_back(adaptor.getSendLength() ? adaptor.getSendLength()
                                                   : getZeroI64());
    callOperands.push_back(adaptor.getRecvOffset() ? adaptor.getRecvOffset()
                                                   : getZeroI64());
    callOperands.push_back(adaptor.getRecvLength() ? adaptor.getRecvLength()
                                                   : getZeroI64());

    callOperands.push_back(castToImportType(adaptor.getElementCount(),
                                            rewriter.getI64Type(), rewriter));

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

  LogicalResult
  matchAndRewrite(IREE::HAL::CommandBufferPushDescriptorSetOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();

    SmallVector<Value, 8> callOperands = {
        adaptor.getCommandBuffer(),
        adaptor.getPipelineLayout(),
        castToImportType(adaptor.getSet(), i32Type, rewriter),
    };
    SmallVector<int16_t, 5> segmentSizes = {
        /*command_buffer=*/-1,
        /*pipeline_layout=*/-1,
        /*set=*/-1,
        /*bindings=*/
        static_cast<int16_t>(adaptor.getBindingOrdinals().size()),
    };
    for (size_t i = 0; i < adaptor.getBindingOrdinals().size(); ++i) {
      callOperands.push_back(
          castToImportType(adaptor.getBindingOrdinals()[i], i32Type, rewriter));
      auto [bindingBufferSlot, bindingBuffer] = splitBufferSlot(
          op.getLoc(), adaptor.getBindingBuffers()[i], rewriter);
      callOperands.push_back(bindingBufferSlot);
      callOperands.push_back(bindingBuffer);
      callOperands.push_back(
          castToImportType(adaptor.getBindingOffsets()[i], i64Type, rewriter));
      callOperands.push_back(
          castToImportType(adaptor.getBindingLengths()[i], i64Type, rewriter));
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

class CommandBufferDispatchIndirectOpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferDispatchIndirectOp> {
public:
  CommandBufferDispatchIndirectOpConversion(MLIRContext *context,
                                            SymbolTable &importSymbols,
                                            TypeConverter &typeConverter,
                                            StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::CommandBufferDispatchIndirectOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();
    auto [workgroupsBufferSlot, workgroupsBuffer] =
        splitBufferSlot(op.getLoc(), adaptor.getWorkgroupsBuffer(), rewriter);
    auto flags = adaptor.getFlagsAttr()
                     ? rewriter
                           .create<IREE::VM::ConstI64Op>(
                               op.getLoc(), adaptor.getFlagsAttr().getInt())
                           .getResult()
                     : rewriter.create<IREE::VM::ConstI64ZeroOp>(op.getLoc())
                           .getResult();
    SmallVector<Value, 8> callOperands = {
        adaptor.getCommandBuffer(),
        adaptor.getExecutable(),
        castToImportType(adaptor.getEntryPoint(), rewriter.getI32Type(),
                         rewriter),
        workgroupsBufferSlot,
        workgroupsBuffer,
        castToImportType(adaptor.getWorkgroupsOffset(), rewriter.getI64Type(),
                         rewriter),
        flags,
    };
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, SymbolRefAttr::get(importOp), importType.getResults(),
        callOperands);
    copyImportAttrs(importOp, callOp);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

class CommandBufferDispatch2OpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferDispatch2Op> {
public:
  CommandBufferDispatch2OpConversion(MLIRContext *context,
                                     SymbolTable &importSymbols,
                                     TypeConverter &typeConverter,
                                     StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::CommandBufferDispatch2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    Value zeroI32 = rewriter.create<IREE::VM::ConstI32ZeroOp>(op.getLoc());

    auto flags = adaptor.getFlagsAttr()
                     ? rewriter
                           .create<IREE::VM::ConstI64Op>(
                               op.getLoc(), adaptor.getFlagsAttr().getInt())
                           .getResult()
                     : rewriter.create<IREE::VM::ConstI64ZeroOp>(op.getLoc())
                           .getResult();
    SmallVector<Value, 8> callOperands = {
        adaptor.getCommandBuffer(),
        adaptor.getExecutable(),
        castToImportType(adaptor.getEntryPoint(), i32Type, rewriter),
        castToImportType(adaptor.getWorkgroupX(), i32Type, rewriter),
        castToImportType(adaptor.getWorkgroupY(), i32Type, rewriter),
        castToImportType(adaptor.getWorkgroupZ(), i32Type, rewriter),
        flags,
    };
    SmallVector<int16_t, 5> segmentSizes = {
        /*command_buffer=*/-1,
        /*executable=*/-1,
        /*entry_point=*/-1,
        /*workgroup_x=*/-1,
        /*workgroup_y=*/-1,
        /*workgroup_z=*/-1,
        /*flags=*/-1,
        /*constants=*/static_cast<int16_t>(adaptor.getConstants().size()),
        /*bindings=*/
        static_cast<int16_t>(adaptor.getBindingBuffers().size()),
    };
    llvm::append_range(callOperands, adaptor.getConstants());
    for (auto [bindingBufferOrSlot, bindingOffset, bindingLength] :
         llvm::zip_equal(adaptor.getBindingBuffers(),
                         adaptor.getBindingOffsets(),
                         adaptor.getBindingLengths())) {
      callOperands.push_back(zeroI32);
      auto [bindingBufferSlot, bindingBuffer] =
          splitBufferSlot(op.getLoc(), bindingBufferOrSlot, rewriter);
      callOperands.push_back(bindingBufferSlot);
      callOperands.push_back(bindingBuffer);
      callOperands.push_back(
          castToImportType(bindingOffset, i64Type, rewriter));
      callOperands.push_back(
          castToImportType(bindingLength, i64Type, rewriter));
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

class CommandBufferDispatch2IndirectOpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferDispatch2IndirectOp> {
public:
  CommandBufferDispatch2IndirectOpConversion(MLIRContext *context,
                                             SymbolTable &importSymbols,
                                             TypeConverter &typeConverter,
                                             StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::CommandBufferDispatch2IndirectOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto importType = importOp.getFunctionType();

    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    Value zeroI32 = rewriter.create<IREE::VM::ConstI32ZeroOp>(op.getLoc());

    auto [workgroupsBufferSlot, workgroupsBuffer] =
        splitBufferSlot(op.getLoc(), adaptor.getWorkgroupsBuffer(), rewriter);
    auto flags = adaptor.getFlagsAttr()
                     ? rewriter
                           .create<IREE::VM::ConstI64Op>(
                               op.getLoc(), adaptor.getFlagsAttr().getInt())
                           .getResult()
                     : rewriter.create<IREE::VM::ConstI64ZeroOp>(op.getLoc())
                           .getResult();
    SmallVector<Value, 8> callOperands = {
        adaptor.getCommandBuffer(),
        adaptor.getExecutable(),
        castToImportType(adaptor.getEntryPoint(), i32Type, rewriter),
        workgroupsBufferSlot,
        workgroupsBuffer,
        castToImportType(adaptor.getWorkgroupsOffset(), i64Type, rewriter),
        flags,
    };
    SmallVector<int16_t, 5> segmentSizes = {
        /*command_buffer=*/-1,
        /*executable=*/-1,
        /*entry_point=*/-1,
        /*workgroups_buffer_slot=*/-1,
        /*workgroups_buffer=*/-1,
        /*workgroups_offset=*/-1,
        /*flags=*/-1,
        /*constants=*/static_cast<int16_t>(adaptor.getConstants().size()),
        /*bindings=*/
        static_cast<int16_t>(adaptor.getBindingBuffers().size()),
    };
    llvm::append_range(callOperands, adaptor.getConstants());
    for (auto [bindingBufferOrSlot, bindingOffset, bindingLength] :
         llvm::zip_equal(adaptor.getBindingBuffers(),
                         adaptor.getBindingOffsets(),
                         adaptor.getBindingLengths())) {
      callOperands.push_back(zeroI32);
      auto [bindingBufferSlot, bindingBuffer] =
          splitBufferSlot(op.getLoc(), bindingBufferOrSlot, rewriter);
      callOperands.push_back(bindingBufferSlot);
      callOperands.push_back(bindingBuffer);
      callOperands.push_back(
          castToImportType(bindingOffset, i64Type, rewriter));
      callOperands.push_back(
          castToImportType(bindingLength, i64Type, rewriter));
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

} // namespace

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
  patterns.insert<CommandBufferUpdateBufferOpConversion>(
      context, importSymbols, typeConverter,
      "hal.command_buffer.update_buffer");
  patterns.insert<CommandBufferCopyBufferOpConversion>(
      context, importSymbols, typeConverter, "hal.command_buffer.copy_buffer");
  patterns.insert<CommandBufferCollectiveOpConversion>(
      context, importSymbols, typeConverter, "hal.command_buffer.collective");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferPushConstantsOp>>(
          context, importSymbols, typeConverter,
          "hal.command_buffer.push_constants");
  patterns.insert<CommandBufferPushDescriptorSetOpConversion>(
      context, importSymbols, typeConverter,
      "hal.command_buffer.push_descriptor_set");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferDispatchOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.dispatch");
  patterns.insert<CommandBufferDispatchIndirectOpConversion>(
      context, importSymbols, typeConverter,
      "hal.command_buffer.dispatch.indirect");
  patterns.insert<CommandBufferDispatch2OpConversion>(
      context, importSymbols, typeConverter, "hal.command_buffer.dispatch2");
  patterns.insert<CommandBufferDispatch2IndirectOpConversion>(
      context, importSymbols, typeConverter,
      "hal.command_buffer.dispatch2.indirect");
}

} // namespace mlir::iree_compiler
