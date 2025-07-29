// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-hal-resolve-topology-queries"

static llvm::cl::opt<bool> clExternalResourcesMappable(
    "iree-hal-external-resources-mappable",
    llvm::cl::desc("Allocates external resources as host-visible and mappable. "
                   "This can degrade performance and introduce allocation "
                   "overhead and staging buffers for readback on the host "
                   "should be managed by the calling application instead."),
    llvm::cl::init(false));

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_RESOLVETOPOLOGYQUERIESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

// Maps a resource type to the corresponding HAL memory types and buffer usage.
// This will fail if the resource type is not directly mappable to HAL bits.
// The bits set here represent the superset of required and allowed bits and
// are useful for providing buffers back to users via the ABI that may need to
// be used for more than just what the internal program requires.
static LogicalResult
deriveAllowedResourceBufferBits(Location loc, IREE::HAL::Lifetime lifetime,
                                IREE::HAL::MemoryTypeBitfield &memoryTypes,
                                IREE::HAL::BufferUsageBitfield &bufferUsage) {
  memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
  bufferUsage = IREE::HAL::BufferUsageBitfield::None;
  if (failed(deriveRequiredResourceBufferBits(loc, lifetime, memoryTypes,
                                              bufferUsage))) {
    return failure();
  }
  switch (lifetime) {
  default:
    break;
  case IREE::HAL::Lifetime::External:
    if (clExternalResourcesMappable) {
      // #yolo; these come from/go to outside the program.
      // Today we assume they are device-local|host-visible just for
      // practical purposes but that does not have to be true. We really
      // want this to be something we analyze and handle on the edges
      // (transferring devices/etc if needed).
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal |
                    IREE::HAL::MemoryTypeBitfield::HostVisible;
      // NOTE: we may not map it but users may after they get them back.
      // Another reason we should annotate this - having a buffer be
      // mappable is potentially expensive (may get a 2nd copy in memory!).
      bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Mapping;
    } else {
      memoryTypes = memoryTypes | IREE::HAL::MemoryTypeBitfield::DeviceLocal;
    }
    break;
  }
  return success();
}

// Returns true if all affinities of the optimal attribute refer to the same
// device ID.
static bool allReferToSameDevice(DeviceOptimalAttr optimalAttr,
                                 DeviceAnalysis &deviceAnalysis,
                                 Operation *fromOp) {
  SetVector<IREE::HAL::DeviceTargetAttr> allPossibleTargets;
  for (auto affinity : optimalAttr.getAffinities()) {
    deviceAnalysis.gatherDeviceAffinityTargets(affinity, fromOp,
                                               allPossibleTargets);
  }
  if (allPossibleTargets.empty()) {
    return false;
  }
  StringAttr firstDeviceId = allPossibleTargets.front().getDeviceID();
  for (IREE::HAL::DeviceTargetAttr target : allPossibleTargets) {
    if (firstDeviceId != target.getDeviceID()) {
      return false;
    }
  }
  return true;
}

// Returns true if the given device has transparent access to all other devices.
static bool hasTransparentAccessToAll(IREE::HAL::DeviceTopologyAttr topology,
                                      IREE::Stream::AffinityAttr source,
                                      DeviceOptimalAttr optimalAttr) {
  LLVM_DEBUG(llvm::dbgs() << "[resolve-topology-queries] checking if " << source
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
  // If all affinities refer to the same device, we don't need to add any
  // extra usage bits.
  if (allReferToSameDevice(optimalAttr, deviceAnalysis, fromOp)) {
    return true;
  }
  // Check if any device has transparent access to all other devices.
  for (auto affinity : optimalAttr.getAffinities()) {
    if (hasTransparentAccessToAll(topology, affinity, optimalAttr)) {
      bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::Mapping;
      return true;
    }
  }
  return false;
}

static LogicalResult
resolveMemoryPropertiesOp(AllocatorResolveMemoryPropertiesOp op,
                          DeviceAnalysis &deviceAnalysis) {
  LLVM_DEBUG(llvm::dbgs() << "[resolve-topology-queries] Op: " << op << "\n");

  OpBuilder builder(op);
  auto loc = op.getLoc();

  // Get the default memory types and buffer usage based on lifetime.
  auto memoryTypes = IREE::HAL::MemoryTypeBitfield::None;
  auto bufferUsage = IREE::HAL::BufferUsageBitfield::None;
  if (failed(deriveAllowedResourceBufferBits(loc, op.getLifetime(), memoryTypes,
                                             bufferUsage))) {
    return failure();
  }

  auto optimalAttr = dyn_cast_if_present<IREE::HAL::DeviceOptimalAttr>(
      op.getAffinity().value_or(nullptr));
  if (!optimalAttr) {
    // If we don't have an optimal attribute, we just set the default
    // memory types and buffer usage.
    auto memoryTypeOp =
        builder.create<IREE::HAL::MemoryTypeOp>(loc, memoryTypes);
    auto bufferUsageOp =
        builder.create<IREE::HAL::BufferUsageOp>(loc, bufferUsage);
    op.replaceAllUsesWith(ValueRange{memoryTypeOp, bufferUsageOp});
    op.erase();
    LLVM_DEBUG(llvm::dbgs()
               << "  -> successfully resolved memory properties\n");
    return success();
  }

  // Get the module to access the topology attribute.
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp) {
    return failure();
  }

  auto topologyAttr =
      moduleOp->getAttrOfType<IREE::HAL::DeviceTopologyAttr>("stream.topology");

  LLVM_DEBUG(llvm::dbgs() << "  -> topology attr: " << topologyAttr << "\n");

  // Try to resolve shared usage bits if possible.
  if (!tryAddSharedUsageBits(topologyAttr, optimalAttr, bufferUsage,
                             memoryTypes, deviceAnalysis, op.getOperation())) {
    LLVM_DEBUG(llvm::dbgs() << "  -> failed to add shared usage bits\n");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "  -> successfully resolved memory properties "
                             "with shared usage bits\n");

  // Create the resolved memory type and buffer usage ops.
  auto memoryTypeOp = builder.create<IREE::HAL::MemoryTypeOp>(loc, memoryTypes);
  auto bufferUsageOp =
      builder.create<IREE::HAL::BufferUsageOp>(loc, bufferUsage);
  op.replaceAllUsesWith(ValueRange{memoryTypeOp, bufferUsageOp});
  op.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// --iree-hal-resolve-topology-queries
//===----------------------------------------------------------------------===//

struct ResolveTopologyQueriesPass
    : public impl::ResolveTopologyQueriesPassBase<ResolveTopologyQueriesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    for (auto funcOp : moduleOp.getOps<CallableOpInterface>()) {
      if (auto *region = funcOp.getCallableRegion()) {
        region->walk([&](IREE::HAL::AllocatorResolveMemoryPropertiesOp op) {
          (void)resolveMemoryPropertiesOp(op, deviceAnalysis);
        });
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
