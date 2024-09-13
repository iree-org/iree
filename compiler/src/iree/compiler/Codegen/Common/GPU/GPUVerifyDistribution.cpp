// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUVERIFYDISTRIBUTIONPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Pass to verify that writes only happen in distributed contexts. Code in
/// shared contexts are executed uniformly across all threads after resolution
/// of distributed contexts (i.e. scf.forall), thus operations with write
/// memory effects are inherently
struct GPUVerifyDistributionPass final
    : impl::GPUVerifyDistributionPassBase<GPUVerifyDistributionPass> {

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    auto privateAddressSpace = gpu::AddressSpaceAttr::get(
        &getContext(), gpu::GPUDialect::getPrivateAddressSpace());

    WalkResult res = funcOp.walk([&](Operation *op) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
        std::optional<ArrayAttr> mapping = forallOp.getMapping();
        if (!mapping || mapping.value().empty()) {
          forallOp->emitOpError("requires a mapping attribute.");
          return WalkResult::interrupt();
        }

        if (isa<IREE::GPU::LaneIdAttr>(*mapping.value().begin()) &&
            !operationHasParentForallOfMappingType<
                mlir::gpu::GPUWarpMappingAttr>(forallOp)) {
          forallOp->emitOpError("lane distributed scf.forall must have a "
                                "parent subgroup distributed loop.");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op)) {
        if (!operationHasParentForallOfMappingType<
                mlir::gpu::GPUThreadMappingAttr, IREE::GPU::LaneIdAttr>(op)) {
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

            op->emitOpError(
                "write affecting operations on shared resources are restricted "
                "to lane or thread distributed contexts.");
            return WalkResult::interrupt();
          }
        }
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
