// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_RESOLVETOPOLOGYQUERIESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

// Checks if the given device has transparent access to all other devices
static bool hasTransparentAccessToAll(IREE::HAL::DeviceTopologyAttr topology,
                                      IREE::Stream::AffinityAttr source,
                                      DeviceOptimalAttr optimalAttr) {
  if (!topology)
    return false;

  bool allHaveTransparentAccess = true;
  for (auto affinity : optimalAttr.getAffinities()) {
    if (!topology.hasTransparentAccess(source, affinity)) {
      allHaveTransparentAccess = false;
      break;
    }
  }
  return allHaveTransparentAccess;
}

static bool tryAddSharedUsageBits(IREE::HAL::DeviceTopologyAttr topology,
                                  IREE::HAL::DeviceOptimalAttr optimalAttr,
                                  IREE::HAL::BufferUsageBitfield &bufferUsage,
                                  IREE::HAL::MemoryTypeBitfield &memoryTypes) {
  for (auto affinity : optimalAttr.getAffinities()) {
    if (hasTransparentAccessToAll(topology, affinity, optimalAttr)) {
      bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Mapping;
      return true;
    }
  }
  return false;
}

struct ResolveMemoryPropertiesPattern
    : public OpRewritePattern<AllocatorResolveMemoryPropertiesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocatorResolveMemoryPropertiesOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the affinity has device.optimal attribute
    auto affinity = op.getAffinity();
    if (!affinity) {
      // should be handled by canonicalizer
      return failure();
    }

    auto optimalAttr = dyn_cast<IREE::HAL::DeviceOptimalAttr>(*affinity);
    if (!optimalAttr) {
      // should be handled by canonicalizer
      return failure();
    }

    // Get the default memory types and buffer usage based on lifetime
    auto loc = op.getLoc();
    auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
    auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
    if (failed(deriveAllowedResourceBufferBits(loc, op.getResourceLifetime(),
                                               memoryTypes, bufferUsage))) {
      return failure();
    }

    // Get the module to access the topology attribute
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return failure();
    }

    // Check if we have a topology attribute
    auto topologyAttr = moduleOp->getAttrOfType<IREE::HAL::DeviceTopologyAttr>(
        "stream.topology");
    if (!topologyAttr) {
      return failure();
    }

    // Get the device affinities from the optimal attribute
    auto affinities = optimalAttr.getAffinities();
    if (affinities.size() < 2) {
      // should be handled by canonicalizer
      return failure();
    }

    // Try to resolve shared usage bits if possible
    if (!tryAddSharedUsageBits(topologyAttr, optimalAttr, bufferUsage,
                               memoryTypes)) {
      return failure();
    }
    // Create the resolved memory type and buffer usage ops
    auto memoryTypeOp =
        rewriter.create<IREE::HAL::MemoryTypeOp>(loc, memoryTypes);
    auto bufferUsageOp =
        rewriter.create<IREE::HAL::BufferUsageOp>(loc, bufferUsage);
    rewriter.replaceOp(op, {memoryTypeOp, bufferUsageOp});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// --iree-hal-resolve-topology-queries
//===----------------------------------------------------------------------===//

struct ResolveTopologyQueriesPass
    : public impl::ResolveTopologyQueriesPassBase<ResolveTopologyQueriesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Apply the pattern to resolve memory properties
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ResolveMemoryPropertiesPattern>(context);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
