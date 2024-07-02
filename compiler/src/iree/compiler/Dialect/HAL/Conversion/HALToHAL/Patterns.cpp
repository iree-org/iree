// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/HALToHAL/Patterns.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct ConvertDeviceResolveAnyOp
    : public OpConversionPattern<IREE::HAL::DeviceResolveOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::DeviceResolveOp resolveOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getAffinity()) {
      return rewriter.notifyMatchFailure(
          resolveOp, "only resolving unspecified affinities to any device");
    }

    auto deviceType = rewriter.getType<IREE::HAL::DeviceType>();
    Value device;
    auto resolveDevice = [&]() {
      if (!device) {
        device = rewriter.create<IREE::HAL::DevicesGetOp>(
            resolveOp.getLoc(), deviceType,
            rewriter.create<arith::ConstantIndexOp>(resolveOp.getLoc(), 0));
      }
      return device;
    };

    SmallVector<Value> results;
    for (auto resultType : resolveOp.getResultTypes()) {
      if (isa<IREE::HAL::DeviceType>(resultType)) {
        results.push_back(resolveDevice());
      } else if (isa<IREE::HAL::AllocatorType>(resultType)) {
        results.push_back(rewriter.create<IREE::HAL::DeviceAllocatorOp>(
            resolveOp.getLoc(), resolveDevice()));
      } else if (isa<IntegerType>(resultType)) {
        results.push_back(rewriter.create<arith::ConstantIntOp>(
            resolveOp.getLoc(), -1ll, 64));
      }
    }

    rewriter.replaceOp(resolveOp, results);
    return success();
  }
};

struct ConvertDeviceResolveAffinityOp
    : public OpConversionPattern<IREE::HAL::DeviceResolveOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::DeviceResolveOp resolveOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto affinityAttr = adaptor.getAffinityAttr();
    if (!affinityAttr) {
      return rewriter.notifyMatchFailure(
          resolveOp, "only resolving fully specified affinities");
    }
    auto flatDeviceAttr = dyn_cast<FlatSymbolRefAttr>(affinityAttr.getDevice());
    if (!flatDeviceAttr) {
      return rewriter.notifyMatchFailure(
          resolveOp, "nested device references not yet supported");
    }

    auto deviceType = rewriter.getType<IREE::HAL::DeviceType>();
    Value device;
    auto resolveDevice = [&]() {
      if (!device) {
        device = rewriter.create<IREE::Util::GlobalLoadOp>(
            resolveOp.getLoc(), deviceType, flatDeviceAttr.getValue(),
            /*is_immutable=*/true);
      }
      return device;
    };

    SmallVector<Value> results;
    for (auto resultType : resolveOp.getResultTypes()) {
      if (isa<IREE::HAL::DeviceType>(resultType)) {
        results.push_back(resolveDevice());
      } else if (isa<IREE::HAL::AllocatorType>(resultType)) {
        results.push_back(rewriter.create<IREE::HAL::DeviceAllocatorOp>(
            resolveOp.getLoc(), resolveDevice()));
      } else if (isa<IntegerType>(resultType)) {
        results.push_back(rewriter.create<arith::ConstantIntOp>(
            resolveOp.getLoc(), affinityAttr.getQueueMask(), 64));
      }
    }

    rewriter.replaceOp(resolveOp, results);
    return success();
  }
};

struct ConvertExecutableCalculateWorkgroupsOp
    : public OpConversionPattern<IREE::HAL::ExecutableCalculateWorkgroupsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::ExecutableCalculateWorkgroupsOp calculateOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto exportOp =
        SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
            calculateOp, calculateOp.getEntryPoint());
    if (!exportOp) {
      return rewriter.notifyMatchFailure(calculateOp,
                                         "target entry point not found");
    }
    auto workgroupCount = exportOp.calculateWorkgroupCount(
        calculateOp.getLoc(), adaptor.getDevice(), adaptor.getWorkload(),
        rewriter);
    rewriter.replaceOp(calculateOp, workgroupCount);
    return success();
  }
};

} // namespace

void populateHALToHALPatterns(MLIRContext *context,
                              ConversionTarget &conversionTarget,
                              TypeConverter &typeConverter,
                              RewritePatternSet &patterns) {
  conversionTarget.addIllegalOp<IREE::HAL::DeviceResolveOp>();
  patterns.insert<ConvertDeviceResolveAnyOp>(typeConverter, context);
  patterns.insert<ConvertDeviceResolveAffinityOp>(typeConverter, context);

  conversionTarget.addIllegalOp<IREE::HAL::ExecutableCalculateWorkgroupsOp>();
  patterns.insert<ConvertExecutableCalculateWorkgroupsOp>(typeConverter,
                                                          context);
}

} // namespace mlir::iree_compiler
