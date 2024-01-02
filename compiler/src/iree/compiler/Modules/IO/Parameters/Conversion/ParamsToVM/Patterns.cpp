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

static Value getStringRodata(Location loc, StringAttr attr,
                             OpBuilder &builder) {
  if (!attr) {
    return builder.create<IREE::VM::ConstRefZeroOp>(
        loc, IREE::VM::RefType::get(builder.getType<IREE::VM::BufferType>()));
  }
  return builder.create<IREE::VM::RodataInlineOp>(loc, attr);
}

static std::pair<Value, Value> buildKeyTable(Location loc, ArrayAttr keysAttr,
                                             OpBuilder &builder) {
  auto tableOp = builder.create<IREE::VM::RodataTableInlineOp>(
      loc, builder.getIntegerType(32), keysAttr);
  return {tableOp.getTableResult(), tableOp.getDataResult()};
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
  Value rodataBuffer = builder.create<IREE::VM::RodataInlineOp>(
      loc, IREE::VM::RefType::get(builder.getType<IREE::VM::BufferType>()),
      builder.getI64VectorAttr(values));
  if (dynamicUpdates.empty()) {
    // Fast-path for all constant data.
    return rodataBuffer;
  }

  // Clone the rodata so we can mutate it.
  Value rodataSize = builder.create<IREE::VM::BufferLengthOp>(
      loc, builder.getI64Type(), rodataBuffer);
  Value clonedBuffer = builder.create<IREE::VM::BufferCloneOp>(
      loc, IREE::VM::RefType::get(builder.getType<IREE::VM::BufferType>()),
      rodataBuffer, builder.create<IREE::VM::ConstI32ZeroOp>(loc), rodataSize,
      builder.create<IREE::VM::ConstI32Op>(loc, sizeof(uint32_t)));

  // Perform all updates.
  for (auto [index, value] : dynamicUpdates) {
    builder.create<IREE::VM::BufferStoreI64Op>(
        loc, clonedBuffer, builder.create<IREE::VM::ConstI64Op>(loc, index),
        value);
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
        buildKeyTable(loadOp.getLoc(), adaptor.getSourceKeysAttr(), rewriter);
    SmallVector<Value> targetOffsets(
        adaptor.getSourceOffsets().size(),
        rewriter.create<IREE::VM::ConstI64Op>(loadOp.getLoc(), 0));
    auto spans =
        buildIndirectSpans(loadOp.getLoc(), adaptor.getSourceOffsets(),
                           targetOffsets, adaptor.getLengths(), rewriter);
    auto bufferType =
        IREE::VM::RefType::get(rewriter.getType<IREE::HAL::BufferType>());
    auto listType = IREE::VM::RefType::get(IREE::VM::ListType::get(bufferType));
    auto callOp = rewriter.create<IREE::VM::CallOp>(
        loadOp.getLoc(), importOp.getSymNameAttr(),
        TypeRange{
            listType,
        },
        ValueRange{
            adaptor.getDevice(),
            adaptor.getQueueAffinity(),
            adaptor.getWaitFence(),
            adaptor.getSignalFence(),
            getStringRodata(loadOp.getLoc(), adaptor.getSourceScopeAttr(),
                            rewriter),
            adaptor.getQueueAffinity(),
            rewriter.create<IREE::VM::ConstI32Op>(
                loadOp.getLoc(), (uint32_t)adaptor.getMemoryTypes()),
            rewriter.create<IREE::VM::ConstI32Op>(
                loadOp.getLoc(), (uint32_t)adaptor.getBufferUsage()),
            keyTable,
            keyData,
            spans,
        });
    copyImportAttrs(importOp, callOp);
    SmallVector<Value> buffers;
    buffers.reserve(targetOffsets.size());
    for (size_t i = 0; i < targetOffsets.size(); ++i) {
      buffers.push_back(rewriter.create<IREE::VM::ListGetRefOp>(
          loadOp.getLoc(), bufferType, callOp.getResult(0),
          rewriter.create<IREE::VM::ConstI32Op>(loadOp.getLoc(), (int32_t)i)));
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
        buildKeyTable(gatherOp.getLoc(), adaptor.getSourceKeysAttr(), rewriter);
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
            getStringRodata(gatherOp.getLoc(), adaptor.getSourceScopeAttr(),
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
    auto [keyTable, keyData] = buildKeyTable(
        scatterOp.getLoc(), adaptor.getTargetKeysAttr(), rewriter);
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
            getStringRodata(scatterOp.getLoc(), adaptor.getTargetScopeAttr(),
                            rewriter),
            adaptor.getSourceBuffer(),
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
