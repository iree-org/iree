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
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct CmdParameterLoadOpPattern
    : public OpConversionPattern<IREE::Stream::CmdParameterLoadOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdParameterLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();

    // Derive the allocation requirements.
    auto resourceType =
        cast<IREE::Stream::ResourceType>(loadOp.getResults().front().getType());

    auto resolveOp = IREE::HAL::AllocatorResolveMemoryPropertiesOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32Type(),
        IREE::Stream::AffinityAttr::lookupOrDefault(loadOp),
        static_cast<IREE::HAL::Lifetime>(resourceType.getLifetime()));

    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(loadOp, resolveOp.getMemoryTypes(),
                                        resolveOp.getBufferUsage(), rewriter);

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, loadOp.getResultTimepoint(), rewriter);

    // Queue operation, which acts like an allocation.
    SmallVector<Type> newResultTypes(loadOp.getResults().size(),
                                     rewriter.getType<IREE::HAL::BufferType>());
    auto newOp = IREE::IO::Parameters::LoadOp::create(
        rewriter, loc, newResultTypes, device, queueAffinity, waitFence,
        signalFence, adaptor.getSourceScope(), adaptor.getSourceKeys(),
        adaptor.getSourceOffsets(), resolveOp.getMemoryTypes(),
        resolveOp.getBufferUsage(), adaptor.getResultSizes());

    SmallVector<Value> resultReplacements;
    llvm::append_range(resultReplacements, newOp.getResults());
    resultReplacements.push_back(signalFence);
    rewriter.replaceOp(loadOp, resultReplacements);
    return success();
  }
};

struct CmdParameterReadOpPattern
    : public OpConversionPattern<IREE::Stream::CmdParameterReadOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdParameterReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = readOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(readOp, rewriter);

    // Gather wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, readOp.getResultTimepoint(), rewriter);

    // Queue operation (a read is just a gather with a single span).
    IREE::IO::Parameters::GatherOp::create(
        rewriter, loc, device, queueAffinity, waitFence, signalFence,
        adaptor.getSourceScope(), ValueRange{adaptor.getSourceKey()},
        ValueRange{adaptor.getSourceOffset()}, adaptor.getTarget(),
        ValueRange{adaptor.getTargetOffset()},
        ValueRange{adaptor.getTargetLength()});

    rewriter.replaceOp(readOp, {signalFence});
    return success();
  }
};

struct CmdParameterWriteOpPattern
    : public OpConversionPattern<IREE::Stream::CmdParameterWriteOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdParameterWriteOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = writeOp.getLoc();
    auto [device, queueAffinity] =
        lookupDeviceAndQueueAffinityFor(writeOp, rewriter);

    // Scatter wait/signal fence, which are optional.
    Value waitFence =
        getOrCreateWaitFence(loc, adaptor.getAwaitTimepoint(), rewriter);
    Value signalFence = getOrCreateSignalFence(
        loc, device, writeOp.getResultTimepoint(), rewriter);

    // Queue operation (a write is just a scatter with a single span).
    IREE::IO::Parameters::ScatterOp::create(
        rewriter, loc, device, queueAffinity, waitFence, signalFence,
        adaptor.getSource(), ValueRange{adaptor.getSourceOffset()},
        ValueRange{adaptor.getSourceLength()}, adaptor.getTargetScope(),
        ValueRange{adaptor.getTargetKey()},
        ValueRange{adaptor.getTargetOffset()});

    rewriter.replaceOp(writeOp, {signalFence});
    return success();
  }
};

struct CmdParameterGatherOpPattern
    : public OpConversionPattern<IREE::Stream::CmdParameterGatherOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdParameterGatherOp gatherOp,
                  OpAdaptor adaptor,
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
    IREE::IO::Parameters::GatherOp::create(
        rewriter, loc, device, queueAffinity, waitFence, signalFence,
        adaptor.getSourceScope(), adaptor.getSourceKeys(),
        adaptor.getSourceOffsets(), adaptor.getTarget(),
        adaptor.getTargetOffsets(), adaptor.getTargetLengths());

    rewriter.replaceOp(gatherOp, {signalFence});
    return success();
  }
};

struct CmdParameterScatterOpPattern
    : public OpConversionPattern<IREE::Stream::CmdParameterScatterOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::Stream::CmdParameterScatterOp scatterOp,
                  OpAdaptor adaptor,
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
    IREE::IO::Parameters::ScatterOp::create(
        rewriter, loc, device, queueAffinity, waitFence, signalFence,
        adaptor.getSource(), adaptor.getSourceOffsets(),
        adaptor.getSourceLengths(), adaptor.getTargetScope(),
        adaptor.getTargetKeys(), adaptor.getTargetOffsets());

    rewriter.replaceOp(scatterOp, {signalFence});
    return success();
  }
};

} // namespace

void populateStreamToIOParametersPatterns(MLIRContext *context,
                                          ConversionTarget &conversionTarget,
                                          TypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  patterns.insert<CmdParameterLoadOpPattern, CmdParameterReadOpPattern,
                  CmdParameterWriteOpPattern, CmdParameterGatherOpPattern,
                  CmdParameterScatterOpPattern>(typeConverter, context);
}

} // namespace mlir::iree_compiler
