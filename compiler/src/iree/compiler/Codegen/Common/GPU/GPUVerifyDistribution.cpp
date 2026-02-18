// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUVERIFYDISTRIBUTIONPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Pass to verify that writes only happen in distributed contexts.
///
/// The approach is to walk the function in pre-order, skipping into regions
/// that are known to be distributed (thread/lane-mapped scf.forall,
/// lane-scoped pcf.generic/pcf.loop). Any write memory effect encountered
/// outside these skipped regions is an error, since it would execute uniformly
/// across all threads.
struct GPUVerifyDistributionPass final
    : impl::GPUVerifyDistributionPassBase<GPUVerifyDistributionPass> {

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    if (!workgroupSize) {
      funcOp->emitOpError("requires a workgroup size attribute.");
      return signalPassFailure();
    }

    if (llvm::all_of(*workgroupSize, [](int64_t size) { return size == 1; })) {
      return;
    }

    auto privateAddressSpace = gpu::AddressSpaceAttr::get(
        &getContext(), gpu::GPUDialect::getPrivateAddressSpace());

    WalkResult res =
        funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          // PCF generic/loop ops: check the scope attribute to determine
          // if this is a lane-distributed context (skip) or subgroup-level
          // context (advance into body to verify nested writes).
          auto checkPCFScope = [](Attribute scope) -> WalkResult {
            if (isa<IREE::GPU::LaneScopeAttr>(scope)) {
              return WalkResult::skip();
            }
            // Subgroup scope and other scopes are not fully distributed
            // to threads — continue walking into their bodies.
            return WalkResult::advance();
          };
          if (auto genericOp = dyn_cast<IREE::PCF::GenericOp>(op)) {
            return checkPCFScope(genericOp.getScope());
          }
          if (auto loopOp = dyn_cast<IREE::PCF::LoopOp>(op)) {
            return checkPCFScope(loopOp.getScope());
          }

          if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
            std::optional<ArrayAttr> mapping = forallOp.getMapping();
            if (!mapping || mapping.value().empty()) {
              forallOp->emitOpError("requires a mapping attribute.");
              return WalkResult::interrupt();
            }

            Attribute firstMapping = *mapping.value().begin();

            // Lane-mapped foralls must have a subgroup-distributed parent.
            if (isa<IREE::GPU::LaneIdAttr>(firstMapping) &&
                !operationHasParentForallOfMappingType<
                    mlir::gpu::GPUWarpMappingAttr>(forallOp)) {
              forallOp->emitOpError("lane distributed scf.forall must have a "
                                    "parent subgroup distributed loop.");
              return WalkResult::interrupt();
            }

            // Thread-mapped and lane-mapped foralls are distributed contexts.
            // Skip their bodies — writes inside are fine.
            if (isa<mlir::gpu::GPUThreadMappingAttr, IREE::GPU::LaneIdAttr>(
                    firstMapping)) {
              return WalkResult::skip();
            }

            // Other mappings (e.g. warp mapping) — continue walking into
            // their bodies since they aren't fully distributed to threads.
            return WalkResult::advance();
          }

          // Any write memory effect in undistributed context is an error.
          auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
          if (!memoryEffectOp) {
            return WalkResult::advance();
          }

          for (Value operand : memoryEffectOp->getOperands()) {
            auto type = dyn_cast<MemRefType>(operand.getType());
            if (!type || !memoryEffectOp.getEffectOnValue<MemoryEffects::Write>(
                             operand)) {
              continue;
            }

            // Writes to private memory are fine.
            if (type.getMemorySpace() == privateAddressSpace) {
              continue;
            }

            // Allow DMA copies.
            if (isa<linalg::CopyOp>(op) &&
                getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(op)) {
              continue;
            }

            op->emitOpError(
                "write affecting operations on shared resources are "
                "restricted to lane or thread distributed contexts.");
            return WalkResult::interrupt();
          }

          return WalkResult::advance();
        });

    if (res.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
