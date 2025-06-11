// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hal-resolve-topology-queries"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_RESOLVETOPOLOGYQUERIESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

// Check if all affinities of the optimal attribute refer to the same device ID
static bool allReferToSameDevice(DeviceOptimalAttr optimalAttr,
                                 DeviceAnalysis &deviceAnalysis,
                                 Operation *fromOp) {
  SetVector<IREE::HAL::DeviceTargetAttr> allPossibleTargets;
  for (auto affinity : optimalAttr.getAffinities()) {
    deviceAnalysis.gatherDeviceAffinityTargets(affinity, fromOp,
                                               allPossibleTargets);
  }

  auto mappedRange = llvm::map_range(
      allPossibleTargets,
      [&](IREE::HAL::DeviceTargetAttr target) { return target.getDeviceID(); });
  SetVector<StringAttr> allDeviceIDs(mappedRange.begin(), mappedRange.end());

  return allDeviceIDs.size() == 1;
}

// Checks if the given device has transparent access to all other devices
// This includes both topology-based transparent access and same-backend access
static bool hasTransparentAccessToAll(IREE::HAL::DeviceTopologyAttr topology,
                                      IREE::Stream::AffinityAttr source,
                                      DeviceOptimalAttr optimalAttr) {
  LLVM_DEBUG(llvm::dbgs() << "[resolve_topology_queries] Checking if " << source
                          << " has transparent access to all " << optimalAttr
                          << "\n");
  if (!topology) {
    return false;
  }

  for (auto targetAffinity : optimalAttr.getAffinities()) {
    if (!topology.hasTransparentAccess(source, targetAffinity)) {
      LLVM_DEBUG(llvm::dbgs() << "  ->  " << source
                              << " does not have transparent access to "
                              << targetAffinity << "\n");
      return false;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "  -> " << source
                          << " has transparent access to all devices\n");
  return true;
}

static bool tryAddSharedUsageBits(IREE::HAL::DeviceTopologyAttr topology,
                                  IREE::HAL::DeviceOptimalAttr optimalAttr,
                                  IREE::HAL::BufferUsageBitfield &bufferUsage,
                                  IREE::HAL::MemoryTypeBitfield &memoryTypes,
                                  DeviceAnalysis &deviceAnalysis,
                                  Operation *fromOp) {
  // if all affinities refer to the same device, we dont need to add any
  // extra usage bits
  if (allReferToSameDevice(optimalAttr, deviceAnalysis, fromOp)) {
    return true;
  }
  // check if any device has transparent access to all other devices
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
  DeviceAnalysis &deviceAnalysis;

  ResolveMemoryPropertiesPattern(MLIRContext *context,
                                 DeviceAnalysis &deviceAnalysis)
      : OpRewritePattern(context), deviceAnalysis(deviceAnalysis) {}

  LogicalResult matchAndRewrite(AllocatorResolveMemoryPropertiesOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[resolve_topology_queries] Op: " << op << "\n");

    // Check if the affinity has device.optimal attribute
    auto affinity = op.getAffinity();
    if (!affinity) {
      LLVM_DEBUG(llvm::dbgs() << "  -> No affinity found\n");
      // should be handled by canonicalizer
      return failure();
    }

    auto optimalAttr = dyn_cast<IREE::HAL::DeviceOptimalAttr>(*affinity);
    if (!optimalAttr) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Affinity is not device.optimal\n");
      // should be handled by canonicalizer
      return failure();
    }

    // Get the device affinities from the optimal attribute
    auto affinities = optimalAttr.getAffinities();
    if (affinities.size() < 2) {
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

    auto topologyAttr = moduleOp->getAttrOfType<IREE::HAL::DeviceTopologyAttr>(
        "stream.topology");

    LLVM_DEBUG(llvm::dbgs() << "  -> Topology attr: " << topologyAttr << "\n");

    // Try to resolve shared usage bits if possible
    if (!tryAddSharedUsageBits(topologyAttr, optimalAttr, bufferUsage,
                               memoryTypes, deviceAnalysis,
                               op.getOperation())) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Failed to add shared usage bits\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  -> Successfully resolved memory properties "
                               "with shared usage bits\n");
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

    // Run device analysis to get concrete device targets
    DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    // Apply the pattern to resolve memory properties
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ResolveMemoryPropertiesPattern>(context, deviceAnalysis);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
