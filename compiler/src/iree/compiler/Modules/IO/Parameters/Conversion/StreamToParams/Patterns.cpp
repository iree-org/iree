// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/IO/Parameters/Conversion/StreamToParams/Patterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Modules/IO/Parameters/IR/IOParametersOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct ParameterLoadOpPattern
    : public OpConversionPattern<IREE::Stream::ParameterLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ParameterLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(loadOp, rewriter);

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, loadOp.getResultTimepoint(), rewriter);

    // Derive the allocation requirements.
    auto resourceType =
        llvm::cast<IREE::Stream::ResourceType>(loadOp.getResult().getType());
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
    if (failed(deriveAllowedResourceBufferBits(loc, resourceType, memoryTypes,
                                               bufferUsage))) {
      return failure();
    }

    // Queue operation, which acts like an allocation.
    Value result = rewriter.create<IREE::IO::Parameters::LoadOp>(
        loc, rewriter.getType<IREE::HAL::BufferType>(), device, queueAffinity,
        waitFence, signalFence, adaptor.getSourceScopeAttr(),
        adaptor.getSourceKeyAttr(), adaptor.getSourceOffset(), memoryTypes,
        bufferUsage, adaptor.getResultSize());

    rewriter.replaceOp(loadOp, {result, signalFence});
    return success();
  }
};

struct ParameterReadOpPattern
    : public OpConversionPattern<IREE::Stream::ParameterReadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ParameterReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = readOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(readOp, rewriter);

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, readOp.getResultTimepoint(), rewriter);

    // Queue operation.
    rewriter.create<IREE::IO::Parameters::ReadOp>(
        loc, device, queueAffinity, waitFence, signalFence,
        adaptor.getSourceScopeAttr(), adaptor.getSourceKeyAttr(),
        adaptor.getSourceOffset(), adaptor.getTarget(),
        adaptor.getTargetOffset(), adaptor.getTargetLength());

    rewriter.replaceOp(readOp, {signalFence});
    return success();
  }
};

struct ParameterWriteOpPattern
    : public OpConversionPattern<IREE::Stream::ParameterWriteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ParameterWriteOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = writeOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(writeOp, rewriter);

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, writeOp.getResultTimepoint(), rewriter);

    // Queue operation.
    rewriter.create<IREE::IO::Parameters::WriteOp>(
        loc, device, queueAffinity, waitFence, signalFence, adaptor.getSource(),
        adaptor.getSourceOffset(), adaptor.getTargetScopeAttr(),
        adaptor.getTargetKeyAttr(), adaptor.getTargetOffset(),
        adaptor.getSourceLength());

    rewriter.replaceOp(writeOp, {signalFence});
    return success();
  }
};

struct ParameterGatherOpPattern
    : public OpConversionPattern<IREE::Stream::ParameterGatherOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ParameterGatherOp gatherOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = gatherOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(gatherOp, rewriter);

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, gatherOp.getResultTimepoint(), rewriter);

    // Queue operation.
    rewriter.create<IREE::IO::Parameters::GatherOp>(
        loc, device, queueAffinity, waitFence, signalFence,
        adaptor.getSourceScopeAttr(), adaptor.getSourceKeysAttr(),
        adaptor.getSourceOffsets(), adaptor.getTarget(),
        adaptor.getTargetOffsets(), adaptor.getTargetLengths());

    rewriter.replaceOp(gatherOp, {signalFence});
    return success();
  }
};

struct ParameterScatterOpPattern
    : public OpConversionPattern<IREE::Stream::ParameterScatterOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Stream::ParameterScatterOp scatterOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = scatterOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(scatterOp, rewriter);

    // Scatter wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, scatterOp.getResultTimepoint(), rewriter);

    // Queue operation.
    rewriter.create<IREE::IO::Parameters::ScatterOp>(
        loc, device, queueAffinity, waitFence, signalFence, adaptor.getSource(),
        adaptor.getSourceOffsets(), adaptor.getSourceLengths(),
        adaptor.getTargetScopeAttr(), adaptor.getTargetKeysAttr(),
        adaptor.getTargetOffsets());

    rewriter.replaceOp(scatterOp, {signalFence});
    return success();
  }
};

} // namespace

void populateStreamToIOParametersPatterns(MLIRContext *context,
                                          ConversionTarget &conversionTarget,
                                          TypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  patterns.insert<ParameterLoadOpPattern, ParameterReadOpPattern,
                  ParameterWriteOpPattern, ParameterGatherOpPattern,
                  ParameterScatterOpPattern>(typeConverter, context);
}

} // namespace mlir::iree_compiler
