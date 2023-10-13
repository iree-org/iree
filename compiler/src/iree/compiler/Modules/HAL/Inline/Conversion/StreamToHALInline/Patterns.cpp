// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Inline/Conversion/StreamToHALInline/Patterns.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineDialect.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

static Value getResourceSize(Location loc, Value resource, OpBuilder &builder) {
  if (llvm::isa<IREE::HAL::BufferType>(resource.getType())) {
    return builder.createOrFold<IREE::HAL::Inline::BufferLengthOp>(
        loc, builder.getIndexType(), resource);
  }
  return builder.createOrFold<IREE::Util::BufferSizeOp>(
      loc, builder.getIndexType(), resource);
}

struct Storage {
  // Underlying storage buffer.
  Value buffer;
  // Total size of the storage buffer in bytes.
  Value bufferSize;
};

static Storage getResourceStorage(Location loc, Value resource,
                                  Value resourceSize, OpBuilder &builder) {
  if (llvm::isa<IREE::HAL::BufferType>(resource.getType())) {
    // Get the storage of the buffer; the returned buffer is already a subspan.
    auto storageBuffer =
        builder.createOrFold<IREE::HAL::Inline::BufferStorageOp>(loc, resource);
    auto storageSize = getResourceSize(loc, resource, builder);
    return {
        storageBuffer,
        storageSize,
    };
  }
  return {
      resource,
      resourceSize,
  };
}

struct ResourceAllocOpPattern
    : public OpConversionPattern<IREE::Stream::ResourceAllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceAllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto deviceBufferType = rewriter.getType<IREE::HAL::BufferType>();
    auto hostBufferType = rewriter.getType<IREE::Util::BufferType>();

    // For now we don't have this information and assume something conservative.
    Value minAlignment =
        rewriter.create<arith::ConstantIndexOp>(allocOp.getLoc(), 64);

    SmallVector<Value> results;
    for (auto [resourceResult, storageSize] :
         llvm::zip_equal(allocOp.getResults(), allocOp.getStorageSizes())) {
      auto allocateOp = rewriter.create<IREE::HAL::Inline::BufferAllocateOp>(
          allocOp.getLoc(), deviceBufferType, hostBufferType, minAlignment,
          storageSize);
      results.push_back(allocateOp.getResult());
    }

    rewriter.replaceOp(allocOp, results);
    return success();
  }
};

struct ResourceAllocaOpPattern
    : public OpConversionPattern<IREE::Stream::ResourceAllocaOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceAllocaOp allocaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto deviceBufferType = rewriter.getType<IREE::HAL::BufferType>();
    auto hostBufferType = rewriter.getType<IREE::Util::BufferType>();

    // For now we don't have this information and assume something conservative.
    Value minAlignment =
        rewriter.create<arith::ConstantIndexOp>(allocaOp.getLoc(), 64);
    auto allocateOp = rewriter.create<IREE::HAL::Inline::BufferAllocateOp>(
        allocaOp.getLoc(), deviceBufferType, hostBufferType, minAlignment,
        adaptor.getStorageSize());

    auto resolvedTimepoint =
        rewriter.create<arith::ConstantIntOp>(allocaOp.getLoc(), 0, 64)
            .getResult();

    rewriter.replaceOp(allocaOp, {allocateOp.getResult(), resolvedTimepoint});
    return success();
  }
};

struct ResourceDeallocaOpPattern
    : public OpConversionPattern<IREE::Stream::ResourceDeallocaOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceDeallocaOp deallocaOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): discard op?
    auto resolvedTimepoint =
        rewriter.create<arith::ConstantIntOp>(deallocaOp.getLoc(), 0, 64)
            .getResult();
    rewriter.replaceOp(deallocaOp, {resolvedTimepoint});
    return success();
  }
};

struct ResourceSizeOpPattern
    : public OpConversionPattern<IREE::Stream::ResourceSizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceSizeOp sizeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(sizeOp, getResourceSize(sizeOp.getLoc(),
                                               adaptor.getOperand(), rewriter));
    return success();
  }
};

// The constant buffer returned from this is always a !util.buffer.
// We can thus directly pass along the input buffer that's being mapped
// (after taking a subspan for the defined range).
struct ResourceTryMapOpPattern
    : public OpConversionPattern<IREE::Stream::ResourceTryMapOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceTryMapOp tryMapOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value subspan = rewriter.create<IREE::Util::BufferSubspanOp>(
        tryMapOp.getLoc(), adaptor.getSource(),
        getResourceSize(tryMapOp.getLoc(), adaptor.getSource(), rewriter),
        adaptor.getSourceOffset(), adaptor.getResultSize());
    Value didMap =
        rewriter.create<arith::ConstantIntOp>(tryMapOp.getLoc(), 1, 1);
    rewriter.replaceOp(tryMapOp, {didMap, subspan});
    return success();
  }
};

struct ResourceLoadOpPattern
    : public OpConversionPattern<IREE::Stream::ResourceLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto storage = getResourceStorage(loc, adaptor.getSource(),
                                      adaptor.getSourceSize(), rewriter);
    auto loadType =
        getTypeConverter()->convertType(loadOp.getResult().getType());
    auto elementSize =
        rewriter.createOrFold<IREE::Util::SizeOfOp>(loc, loadType);
    rewriter.replaceOpWithNewOp<IREE::Util::BufferLoadOp>(
        loadOp, loadType, storage.buffer, storage.bufferSize,
        adaptor.getSourceOffset(), elementSize);
    return success();
  }
};

struct ResourceStoreOpPattern
    : public OpConversionPattern<IREE::Stream::ResourceStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    auto storage = getResourceStorage(loc, adaptor.getTarget(),
                                      adaptor.getTargetSize(), rewriter);
    auto elementSize = rewriter.createOrFold<IREE::Util::SizeOfOp>(
        loc, adaptor.getValue().getType());
    rewriter.replaceOpWithNewOp<IREE::Util::BufferStoreOp>(
        storeOp, adaptor.getValue(), storage.buffer, storage.bufferSize,
        adaptor.getTargetOffset(), elementSize);
    return success();
  }
};

struct ResourceSubviewOpPattern
    : public OpConversionPattern<IREE::Stream::ResourceSubviewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ResourceSubviewOp subviewOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (llvm::isa<IREE::HAL::BufferType>(adaptor.getSource().getType())) {
      auto bufferType = rewriter.getType<IREE::HAL::BufferType>();
      // NOTE: this aliases! We assume at this point all useful alias analysis
      // has been performed and it's fine to lose the tie information here.
      rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferSubspanOp>(
          subviewOp, bufferType, adaptor.getSource(), adaptor.getSourceOffset(),
          adaptor.getResultSize());
    } else {
      rewriter.replaceOpWithNewOp<IREE::Util::BufferSubspanOp>(
          subviewOp, adaptor.getSource(), adaptor.getSourceSize(),
          adaptor.getSourceOffset(), adaptor.getResultSize());
    }
    return success();
  }
};

struct FileConstantOpPattern
    : public OpConversionPattern<IREE::Stream::FileConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::FileConstantOp constantOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Util::BufferSubspanOp>(
        constantOp, constantOp.getSource(), constantOp.getSourceSize(),
        constantOp.getSourceOffset(), constantOp.getSourceLength());
    return success();
  }
};

struct FileReadOpPattern
    : public OpConversionPattern<IREE::Stream::FileReadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::FileReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value sourceSize = rewriter.create<IREE::Util::BufferSizeOp>(
        readOp.getLoc(), adaptor.getSource());
    rewriter.create<IREE::Util::BufferCopyOp>(
        readOp.getLoc(), adaptor.getSource(), sourceSize,
        rewriter.createOrFold<arith::IndexCastOp>(readOp.getLoc(),
                                                  rewriter.getIndexType(),
                                                  adaptor.getSourceOffset()),
        adaptor.getTarget(), adaptor.getTargetSize(), adaptor.getTargetOffset(),
        adaptor.getLength());
    auto resolvedTimepoint =
        rewriter.create<arith::ConstantIntOp>(readOp.getLoc(), 0, 64)
            .getResult();
    rewriter.replaceOp(readOp, resolvedTimepoint);
    return success();
  }
};

struct FileWriteOpPattern
    : public OpConversionPattern<IREE::Stream::FileWriteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::FileWriteOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value targetSize = rewriter.create<IREE::Util::BufferSizeOp>(
        writeOp.getLoc(), adaptor.getTarget());
    rewriter.create<IREE::Util::BufferCopyOp>(
        writeOp.getLoc(), adaptor.getSource(), adaptor.getSourceSize(),
        adaptor.getSourceOffset(), adaptor.getTarget(), targetSize,
        rewriter.createOrFold<arith::IndexCastOp>(writeOp.getLoc(),
                                                  rewriter.getIndexType(),
                                                  adaptor.getTargetOffset()),
        adaptor.getLength());
    auto resolvedTimepoint =
        rewriter.create<arith::ConstantIntOp>(writeOp.getLoc(), 0, 64)
            .getResult();
    rewriter.replaceOp(writeOp, resolvedTimepoint);
    return success();
  }
};

struct TensorImportBufferOpPattern
    : public OpConversionPattern<IREE::Stream::TensorImportOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TensorImportOp importOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::isa<IREE::HAL::BufferType>(importOp.getSource().getType())) {
      return failure();
    }

    // Directly use the buffer.
    auto buffer = adaptor.getSource();
    rewriter.replaceOp(importOp, buffer);
    return success();
  }
};

struct TensorImportBufferViewOpPattern
    : public OpConversionPattern<IREE::Stream::TensorImportOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TensorImportOp importOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = importOp.getSource().getType();
    if (!llvm::isa<IREE::HAL::BufferViewType>(sourceType) &&
        !llvm::isa<TensorType>(sourceType)) {
      return failure();
    }

    auto bufferView = adaptor.getSource();
    auto bufferType = rewriter.getType<IREE::HAL::BufferType>();
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewBufferOp>(
        importOp, bufferType, bufferView);
    return success();
  }
};

struct TensorExportBufferOpPattern
    : public OpConversionPattern<IREE::Stream::TensorExportOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TensorExportOp exportOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::isa<IREE::HAL::BufferType>(exportOp.getResult().getType())) {
      return failure();
    }
    rewriter.replaceOp(exportOp, adaptor.getSource());
    return success();
  }
};

struct TensorExportBufferViewOpPattern
    : public OpConversionPattern<IREE::Stream::TensorExportOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TensorExportOp exportOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto targetType = exportOp.getResult().getType();
    if (!llvm::isa<IREE::HAL::BufferViewType>(targetType) &&
        !llvm::isa<TensorType>(targetType)) {
      return failure();
    }

    auto loc = exportOp.getLoc();
    auto tensorType = llvm::cast<RankedTensorType>(adaptor.getSourceEncoding());
    auto dynamicDims = adaptor.getSourceEncodingDims();

    // NOTE: we should have verified supported encodings/types at entry into the
    // HAL pipeline.
    auto encodingType =
        IREE::HAL::getEncodingTypeValue(tensorType.getEncoding());
    assert(encodingType.has_value() && "invalid tensor encoding");
    auto elementType =
        IREE::HAL::getElementTypeValue(tensorType.getElementType());
    assert(elementType.has_value() && "invalid tensor element type");

    // Flatten static + dynamic shape dimensions.
    SmallVector<Value> dims;
    unsigned dynamicIdx = 0;
    for (int64_t idx = 0; idx < tensorType.getRank(); ++idx) {
      if (tensorType.isDynamicDim(idx)) {
        dims.push_back(dynamicDims[dynamicIdx++]);
      } else {
        dims.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, tensorType.getDimSize(idx)));
      }
    }

    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewCreateOp>(
        exportOp, adaptor.getSource(),
        rewriter.create<arith::ConstantIndexOp>(loc, 0),
        adaptor.getSourceSize(), elementType.value(), encodingType.value(),
        dims);
    return success();
  }
};

struct TensorTraceOpPattern
    : public OpConversionPattern<IREE::Stream::TensorTraceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TensorTraceOp traceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::Inline::BufferViewTraceOp>(
        traceOp, traceOp.getKeyAttr(), adaptor.getOperands());
    return success();
  }
};

struct CmdFlushOpPattern
    : public OpConversionPattern<IREE::Stream::CmdFlushOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdFlushOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdInvalidateOpPattern
    : public OpConversionPattern<IREE::Stream::CmdInvalidateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdInvalidateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdDiscardOpPattern
    : public OpConversionPattern<IREE::Stream::CmdDiscardOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdDiscardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct CmdFillOpPattern : public OpConversionPattern<IREE::Stream::CmdFillOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdFillOp fillOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = fillOp.getLoc();
    auto storage = getResourceStorage(loc, adaptor.getTarget(),
                                      adaptor.getTargetSize(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Util::BufferFillOp>(
        fillOp, adaptor.getValue(), storage.buffer, storage.bufferSize,
        adaptor.getTargetOffset(), adaptor.getTargetLength());
    return success();
  }
};

struct CmdCopyOpPattern : public OpConversionPattern<IREE::Stream::CmdCopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdCopyOp copyOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = copyOp.getLoc();
    auto sourceStorage = getResourceStorage(loc, adaptor.getSource(),
                                            adaptor.getSourceSize(), rewriter);
    auto targetStorage = getResourceStorage(loc, adaptor.getTarget(),
                                            adaptor.getTargetSize(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Util::BufferCopyOp>(
        copyOp, sourceStorage.buffer, sourceStorage.bufferSize,
        adaptor.getSourceOffset(), targetStorage.buffer,
        targetStorage.bufferSize, adaptor.getTargetOffset(),
        adaptor.getLength());
    return success();
  }
};

struct CmdDispatchOpPattern
    : public OpConversionPattern<IREE::Stream::CmdDispatchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdDispatchOp dispatchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = dispatchOp.getLoc();

    auto callee = dispatchOp->getAttrOfType<SymbolRefAttr>("hal_inline.target");
    if (!callee) {
      return rewriter.notifyMatchFailure(
          dispatchOp, "missing hal_inline.target annotation from the "
                      "--iree-hal-inline-executables pass");
    }

    // The InlineExecutables pass has already done the hard work here; we just
    // need to make a function call to the annotated target function with all
    // operands/bindings.
    SmallVector<Value> callArgs;
    llvm::append_range(callArgs, adaptor.getWorkload());
    llvm::append_range(callArgs, adaptor.getUniformOperands());
    SmallVector<Value> bindingBuffers;
    SmallVector<Value> bindingOffsets;
    for (auto [resource, resourceSize, resourceOffset] :
         llvm::zip_equal(adaptor.getResources(), adaptor.getResourceSizes(),
                         adaptor.getResourceOffsets())) {
      auto storage = getResourceStorage(loc, resource, resourceSize, rewriter);
      bindingBuffers.push_back(storage.buffer);
      bindingOffsets.push_back(resourceOffset);
    }
    llvm::append_range(callArgs, bindingBuffers);
    llvm::append_range(callArgs, bindingOffsets);
    llvm::append_range(callArgs, adaptor.getResourceLengths());
    rewriter.replaceOpWithNewOp<func::CallOp>(dispatchOp, callee, TypeRange{},
                                              callArgs);
    return success();
  }
};

struct CmdFuncOpPattern : public OpConversionPattern<IREE::Stream::CmdFuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdFuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newArgTypes;
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(funcOp.getArgumentTypes(),
                                                newArgTypes)) ||
        failed(getTypeConverter()->convertTypes(funcOp.getResultTypes(),
                                                newResultTypes))) {
      return rewriter.notifyMatchFailure(funcOp, "failed to convert types");
    }
    auto newOp = rewriter.replaceOpWithNewOp<func::FuncOp>(
        funcOp, funcOp.getName(),
        rewriter.getFunctionType(newArgTypes, newResultTypes),
        funcOp.getSymVisibilityAttr(), funcOp.getAllArgAttrs(),
        funcOp.getAllResultAttrs());
    newOp->setDialectAttrs(funcOp->getDialectAttrs());
    return success();
  }
};

struct CmdCallOpPattern : public OpConversionPattern<IREE::Stream::CmdCallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdCallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    size_t resourceIndex = 0;
    for (auto [originalOperand, convertedOperand] : llvm::zip_equal(
             callOp.getResourceOperands(), adaptor.getResourceOperands())) {
      if (llvm::isa<IREE::Stream::ResourceType>(originalOperand.getType())) {
        // Resource type, add offset/length.
        auto resourceSize = adaptor.getResourceOperandSizes()[resourceIndex];
        auto storage = getResourceStorage(callOp.getLoc(), convertedOperand,
                                          resourceSize, rewriter);
        operands.push_back(storage.buffer);
        operands.push_back(adaptor.getResourceOperandOffsets()[resourceIndex]);
        operands.push_back(adaptor.getResourceOperandLengths()[resourceIndex]);
        ++resourceIndex;
      } else {
        // Primitive/custom type.
        operands.push_back(convertedOperand);
      }
    }

    SmallVector<Type> resultTypes;
    for (auto result : callOp.getResults()) {
      SmallVector<Type> convertedTypes;
      if (failed(getTypeConverter()->convertType(result.getType(),
                                                 convertedTypes))) {
        return rewriter.notifyMatchFailure(callOp.getLoc(),
                                           "unconvertable result type");
      }
      llvm::append_range(resultTypes, convertedTypes);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(callOp, callOp.getCalleeAttr(),
                                              resultTypes, operands);
    return success();
  }
};

struct CmdExecuteOpPattern
    : public OpConversionPattern<IREE::Stream::CmdExecuteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdExecuteOp executeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Inline the serial execution region.
    rewriter.inlineBlockBefore(&executeOp.getBody().front(), executeOp,
                               adaptor.getResourceOperands());
    // Immediately resolve the timepoint.
    auto resolvedTimepoint =
        rewriter.create<arith::ConstantIntOp>(executeOp.getLoc(), 0, 64)
            .getResult();
    rewriter.replaceOp(executeOp, resolvedTimepoint);
    return success();
  }
};

struct CmdSerialOpPattern
    : public OpConversionPattern<IREE::Stream::CmdSerialOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdSerialOp serialOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Inline the serial execution region.
    rewriter.inlineBlockBefore(&serialOp.getBody().front(), serialOp);
    rewriter.eraseOp(serialOp);
    return success();
  }
};

struct CmdConcurrentOpPattern
    : public OpConversionPattern<IREE::Stream::CmdConcurrentOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdConcurrentOp concurrentOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Inline the concurrent execution region.
    rewriter.inlineBlockBefore(&concurrentOp.getBody().front(), concurrentOp);
    rewriter.eraseOp(concurrentOp);
    return success();
  }
};

// Annoying we have to have this here, but there's no attribute converter
// equivalent we have access to so that we could do it in a generic way.
struct GlobalTimepointConversionPattern
    : public OpConversionPattern<IREE::Util::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto initialValue = op.getInitialValue();
    if (!initialValue.has_value())
      return failure();
    if (!llvm::isa<IREE::Stream::TimepointAttr>(*initialValue))
      return failure();
    rewriter.updateRootInPlace(
        op, [&]() { op.setInitialValueAttr(rewriter.getI64IntegerAttr(0)); });
    return success();
  }
};

struct TimepointImmediateOpPattern
    : public OpConversionPattern<IREE::Stream::TimepointImmediateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointImmediateOp immediateOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(immediateOp, 0, 64);
    return success();
  }
};

struct TimepointImportOpPattern
    : public OpConversionPattern<IREE::Stream::TimepointImportOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointImportOp importOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        importOp,
        "timepoints are not supported across the ABI with inline execution");
  }
};

struct TimepointExportOpPattern
    : public OpConversionPattern<IREE::Stream::TimepointExportOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointExportOp exportOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        exportOp,
        "timepoints are not supported across the ABI with inline execution");
  }
};

struct TimepointChainExternalOpPattern
    : public OpConversionPattern<IREE::Stream::TimepointChainExternalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointChainExternalOp exportOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        exportOp,
        "timepoints are not supported across the ABI with inline execution");
  }
};

struct TimepointJoinOpPattern
    : public OpConversionPattern<IREE::Stream::TimepointJoinOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointJoinOp joinOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(joinOp, 0, 64);
    return success();
  }
};

struct TimepointBarrierOpPattern
    : public OpConversionPattern<IREE::Stream::TimepointBarrierOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointBarrierOp barrierOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(barrierOp, {
                                      adaptor.getResource(),
                                      rewriter.create<arith::ConstantIntOp>(
                                          barrierOp.getLoc(), 0, 64),
                                  });
    return success();
  }
};

struct TimepointAwaitOpPattern
    : public OpConversionPattern<IREE::Stream::TimepointAwaitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::TimepointAwaitOp awaitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(awaitOp, adaptor.getResourceOperands());
    return success();
  }
};

struct ElideYieldOpPattern : public OpConversionPattern<IREE::Stream::YieldOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::YieldOp yieldOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(yieldOp);
    return success();
  }
};

} // namespace

void populateStreamToHALInlinePatterns(MLIRContext *context,
                                       ConversionTarget &conversionTarget,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  // Resources are just buffers (no shape/encoding/etc).
  // We use !hal.buffer when going across the external ABI boundary but
  // otherwise use our host buffer type.
  typeConverter.addConversion(
      [=](IREE::Stream::ResourceType type, SmallVectorImpl<Type> &results) {
        if (type.getLifetime() == IREE::Stream::Lifetime::External) {
          results.push_back(IREE::HAL::BufferType::get(context));
        } else {
          results.push_back(IREE::Util::BufferType::get(context));
        }
        return success();
      });

  // Today files all originate from host buffers and we just treat them the
  // same. Note that file initialization from buffers may require subviews.
  typeConverter.addConversion(
      [=](IREE::Stream::FileType type, SmallVectorImpl<Type> &results) {
        results.push_back(IREE::Util::BufferType::get(context));
        return success();
      });

  // Timepoints and files are both no-oped in the inline HAL.
  typeConverter.addConversion(
      [=](IREE::Stream::TimepointType type, SmallVectorImpl<Type> &results) {
        results.push_back(IntegerType::get(context, 64));
        return success();
      });

  patterns.insert<ResourceAllocOpPattern, ResourceAllocaOpPattern,
                  ResourceDeallocaOpPattern, ResourceSizeOpPattern,
                  ResourceTryMapOpPattern, ResourceLoadOpPattern,
                  ResourceStoreOpPattern, ResourceSubviewOpPattern>(
      typeConverter, context);

  patterns.insert<FileConstantOpPattern, FileReadOpPattern, FileWriteOpPattern>(
      typeConverter, context);

  patterns.insert<TensorImportBufferOpPattern, TensorImportBufferViewOpPattern,
                  TensorExportBufferOpPattern, TensorExportBufferViewOpPattern,
                  TensorTraceOpPattern>(typeConverter, context);

  patterns
      .insert<CmdFlushOpPattern, CmdInvalidateOpPattern, CmdDiscardOpPattern,
              CmdFillOpPattern, CmdCopyOpPattern, CmdDispatchOpPattern,
              CmdFuncOpPattern, CmdCallOpPattern, CmdExecuteOpPattern,
              CmdSerialOpPattern, CmdConcurrentOpPattern>(typeConverter,
                                                          context);

  patterns.insert<GlobalTimepointConversionPattern>(typeConverter, context);
  patterns.insert<TimepointImmediateOpPattern, TimepointImportOpPattern,
                  TimepointExportOpPattern, TimepointChainExternalOpPattern,
                  TimepointJoinOpPattern, TimepointBarrierOpPattern,
                  TimepointAwaitOpPattern>(typeConverter, context);

  patterns.insert<ElideYieldOpPattern>(typeConverter, context);
}

} // namespace iree_compiler
} // namespace mlir
