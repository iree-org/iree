// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/IO/Parameters/Conversion/ParamsToVM/Patterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/Patterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

// Returns the scope buffer for a parameter operation. If the scope operand is
// null (no scope), returns a zero ref. Otherwise the value is already a
// !vm.ref<!vm.buffer> from the type converter and is passed through.
static Value getScopeBuffer(Location loc, Value scopeBuffer,
                            OpBuilder &builder) {
  if (!scopeBuffer) {
    return IREE::VM::ConstRefZeroOp::create(
        builder, loc,
        IREE::VM::RefType::get(builder.getType<IREE::VM::BufferType>()));
  }
  return scopeBuffer;
}

// Builds a key_table + key_data pair from the given key buffer operands.
// When all keys originate from vm.rodata.inline ops (the common case for
// compile-time constant parameter names), we use the efficient
// vm.rodata.table.inline op to pack them at compile time. For dynamic keys
// (e.g., from util.string.format), we construct the table at runtime using
// VM buffer ops.
//
// The runtime ABI expects:
//   key_table: buffer of {uint32_t offset, uint32_t length} entries
//   key_data:  buffer of concatenated key strings
static std::pair<Value, Value> buildKeyTable(Location loc, ValueRange keys,
                                             OpBuilder &builder) {
  // Try the fast path: all keys are compile-time constants.
  SmallVector<Attribute> keyStringAttrs;
  bool allConstant = true;
  for (Value key : keys) {
    auto rodataOp = key.getDefiningOp<IREE::VM::RodataInlineOp>();
    if (!rodataOp) {
      allConstant = false;
      break;
    }
    keyStringAttrs.push_back(rodataOp.getValue());
  }
  if (allConstant) {
    auto tableOp = IREE::VM::RodataTableInlineOp::create(
        builder, loc, builder.getIntegerType(32),
        builder.getArrayAttr(keyStringAttrs));
    return {tableOp.getTableResult(), tableOp.getDataResult()};
  }

  // Dynamic path: construct key_table and key_data at runtime.
  // Each table entry is {uint32_t offset, uint32_t length} = 8 bytes.
  auto i32Type = builder.getIntegerType(32);
  auto i64Type = builder.getIntegerType(64);
  auto bufferRefType =
      IREE::VM::RefType::get(builder.getType<IREE::VM::BufferType>());
  int64_t entrySize = 8; // sizeof(iree_io_parameters_string_entry_t)
  int64_t tableSize = keys.size() * entrySize;
  auto alignment = IREE::VM::ConstI32Op::create(builder, loc, sizeof(uint32_t));

  // Get each key's length.
  SmallVector<Value> keyLengths;
  for (Value key : keys) {
    keyLengths.push_back(
        IREE::VM::BufferLengthOp::create(builder, loc, i64Type, key));
  }

  // Compute total data size and allocate the concatenated key_data buffer.
  Value totalDataSize = keyLengths[0];
  for (size_t i = 1; i < keyLengths.size(); ++i) {
    totalDataSize = IREE::VM::AddI64Op::create(builder, loc, i64Type,
                                               totalDataSize, keyLengths[i]);
  }
  Value keyData = IREE::VM::BufferAllocOp::create(builder, loc, bufferRefType,
                                                  totalDataSize, alignment);

  // Allocate the key_table buffer.
  Value tableSizeValue = IREE::VM::ConstI64Op::create(builder, loc, tableSize);
  Value keyTable = IREE::VM::BufferAllocOp::create(builder, loc, bufferRefType,
                                                   tableSizeValue, alignment);

  // Copy each key into key_data and write the table entry.
  Value currentOffset = IREE::VM::ConstI64Op::create(builder, loc, 0);
  auto zero = IREE::VM::ConstI64Op::create(builder, loc, 0);
  for (size_t i = 0; i < keys.size(); ++i) {
    // Copy key bytes into key_data at currentOffset.
    IREE::VM::BufferCopyOp::create(builder, loc, keys[i], zero, keyData,
                                   currentOffset, keyLengths[i]);

    // Write table entry: {offset, length} as two i32 values.
    // BufferStoreI32Op takes element indices (not byte offsets); each entry
    // has two uint32_t fields so entry i starts at element index i * 2.
    Value entryOffsetElement =
        IREE::VM::ConstI64Op::create(builder, loc, i * 2);
    Value entryLengthElement =
        IREE::VM::ConstI64Op::create(builder, loc, i * 2 + 1);
    Value offsetI32 =
        IREE::VM::TruncI64I32Op::create(builder, loc, i32Type, currentOffset);
    Value lengthI32 =
        IREE::VM::TruncI64I32Op::create(builder, loc, i32Type, keyLengths[i]);
    IREE::VM::BufferStoreI32Op::create(builder, loc, keyTable,
                                       entryOffsetElement, offsetI32);
    IREE::VM::BufferStoreI32Op::create(builder, loc, keyTable,
                                       entryLengthElement, lengthI32);

    // Advance offset for next key.
    currentOffset = IREE::VM::AddI64Op::create(builder, loc, i64Type,
                                               currentOffset, keyLengths[i]);
  }

  return {keyTable, keyData};
}

static Value buildIndirectSpans(Location loc, ValueRange parameterOffsets,
                                ValueRange bufferOffsets,
                                ValueRange bufferLengths, OpBuilder &builder) {
  // Build the rodata containing all constant values and the list of dynamic
  // updates we'll need to perform. We assume that 95-100% of values are
  // constant and optimize for that - if this changes we can make this more
  // sophisticated to reduce binary size and runtime overhead.
  SmallVector<std::pair<size_t, Value>> dynamicUpdates;
  SmallVector<int64_t> values;
  auto recordValue = [&](Value value) {
    APInt constantValue;
    if (matchPattern(value, m_ConstantInt(&constantValue))) {
      values.push_back(constantValue.getZExtValue());
    } else {
      values.push_back(0);
      dynamicUpdates.push_back(std::make_pair(values.size(), value));
    }
  };
  for (auto [parameterOffset, bufferOffset, bufferLength] :
       llvm::zip_equal(parameterOffsets, bufferOffsets, bufferLengths)) {
    recordValue(parameterOffset);
    recordValue(bufferOffset);
    recordValue(bufferLength);
  }
  Value rodataBuffer = IREE::VM::RodataInlineOp::create(
      builder, loc,
      IREE::VM::RefType::get(builder.getType<IREE::VM::BufferType>()),
      builder.getI64VectorAttr(values));
  if (dynamicUpdates.empty()) {
    // Fast-path for all constant data.
    return rodataBuffer;
  }

  // Clone the rodata so we can mutate it.
  Value rodataSize = IREE::VM::BufferLengthOp::create(
      builder, loc, builder.getI64Type(), rodataBuffer);
  Value clonedBuffer = IREE::VM::BufferCloneOp::create(
      builder, loc,
      IREE::VM::RefType::get(builder.getType<IREE::VM::BufferType>()),
      rodataBuffer, IREE::VM::ConstI32ZeroOp::create(builder, loc), rodataSize,
      IREE::VM::ConstI32Op::create(builder, loc, sizeof(uint32_t)));

  // Perform all updates.
  for (auto [index, value] : dynamicUpdates) {
    IREE::VM::BufferStoreI64Op::create(
        builder, loc, clonedBuffer,
        IREE::VM::ConstI64Op::create(builder, loc, index), value);
  }

  return clonedBuffer;
}

struct LoadOpConversion
    : public OpConversionPattern<IREE::IO::Parameters::LoadOp> {
  LoadOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                   TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }
  LogicalResult
  matchAndRewrite(IREE::IO::Parameters::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [keyTable, keyData] =
        buildKeyTable(loadOp.getLoc(), adaptor.getSourceKeys(), rewriter);
    SmallVector<Value> targetOffsets(
        adaptor.getSourceOffsets().size(),
        IREE::VM::ConstI64Op::create(rewriter, loadOp.getLoc(), 0));
    auto spans =
        buildIndirectSpans(loadOp.getLoc(), adaptor.getSourceOffsets(),
                           targetOffsets, adaptor.getLengths(), rewriter);
    auto bufferType =
        IREE::VM::RefType::get(rewriter.getType<IREE::HAL::BufferType>());
    auto listType = IREE::VM::RefType::get(IREE::VM::ListType::get(bufferType));
    auto callOp = IREE::VM::CallOp::create(
        rewriter, loadOp.getLoc(), importOp.getSymNameAttr(),
        TypeRange{
            listType,
        },
        ValueRange{
            adaptor.getDevice(),
            adaptor.getQueueAffinity(),
            adaptor.getWaitFence(),
            adaptor.getSignalFence(),
            getScopeBuffer(loadOp.getLoc(), adaptor.getSourceScope(), rewriter),
            adaptor.getQueueAffinity(),
            castToImportType(adaptor.getMemoryTypes(), rewriter.getI32Type(),
                             rewriter),
            castToImportType(adaptor.getBufferUsage(), rewriter.getI32Type(),
                             rewriter),
            keyTable,
            keyData,
            spans,
        });
    copyImportAttrs(importOp, callOp);
    SmallVector<Value> buffers;
    buffers.reserve(targetOffsets.size());
    for (size_t i = 0; i < targetOffsets.size(); ++i) {
      buffers.push_back(IREE::VM::ListGetRefOp::create(
          rewriter, loadOp.getLoc(), bufferType, callOp.getResult(0),
          IREE::VM::ConstI32Op::create(rewriter, loadOp.getLoc(), (int32_t)i)));
    }
    rewriter.replaceOp(loadOp, buffers);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

struct GatherOpConversion
    : public OpConversionPattern<IREE::IO::Parameters::GatherOp> {
  GatherOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                     TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }
  LogicalResult
  matchAndRewrite(IREE::IO::Parameters::GatherOp gatherOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [keyTable, keyData] =
        buildKeyTable(gatherOp.getLoc(), adaptor.getSourceKeys(), rewriter);
    auto spans = buildIndirectSpans(
        gatherOp.getLoc(), adaptor.getSourceOffsets(),
        adaptor.getTargetOffsets(), adaptor.getTargetLengths(), rewriter);
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        gatherOp, importOp.getSymNameAttr(), TypeRange{},
        ValueRange{
            adaptor.getDevice(),
            adaptor.getQueueAffinity(),
            adaptor.getWaitFence(),
            adaptor.getSignalFence(),
            getScopeBuffer(gatherOp.getLoc(), adaptor.getSourceScope(),
                           rewriter),
            adaptor.getTargetBuffer(),
            keyTable,
            keyData,
            spans,
        });
    copyImportAttrs(importOp, callOp);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

struct ScatterOpConversion
    : public OpConversionPattern<IREE::IO::Parameters::ScatterOp> {
  ScatterOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                      TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }
  LogicalResult
  matchAndRewrite(IREE::IO::Parameters::ScatterOp scatterOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [keyTable, keyData] =
        buildKeyTable(scatterOp.getLoc(), adaptor.getTargetKeys(), rewriter);
    auto spans = buildIndirectSpans(
        scatterOp.getLoc(), adaptor.getTargetOffsets(),
        adaptor.getSourceOffsets(), adaptor.getSourceLengths(), rewriter);
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        scatterOp, importOp.getSymNameAttr(), TypeRange{},
        ValueRange{
            adaptor.getDevice(),
            adaptor.getQueueAffinity(),
            adaptor.getWaitFence(),
            adaptor.getSignalFence(),
            adaptor.getSourceBuffer(),
            getScopeBuffer(scatterOp.getLoc(), adaptor.getTargetScope(),
                           rewriter),
            keyTable,
            keyData,
            spans,
        });
    copyImportAttrs(importOp, callOp);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

} // namespace

void populateIOParametersToVMPatterns(MLIRContext *context,
                                      ConversionTarget &conversionTarget,
                                      TypeConverter &typeConverter,
                                      SymbolTable &importSymbols,
                                      RewritePatternSet &patterns) {
  patterns.insert<LoadOpConversion>(context, importSymbols, typeConverter,
                                    "io_parameters.load");
  patterns.insert<GatherOpConversion>(context, importSymbols, typeConverter,
                                      "io_parameters.gather");
  patterns.insert<ScatterOpConversion>(context, importSymbols, typeConverter,
                                       "io_parameters.scatter");
}

} // namespace mlir::iree_compiler
