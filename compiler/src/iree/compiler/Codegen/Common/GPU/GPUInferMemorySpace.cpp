// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUINFERMEMORYSPACEPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Pass to infer the memory spaces of unmarked `bufferization.alloc_tensor`
/// ops. Inferring the memory space during bufferization (in the allocation
/// function) is infeasible due to some limited analysis of surrounding loop
/// structures needed. After this pass, any unexpected allocations are then
/// treated as a compiler failure indicating something went wrong during
/// bufferization.
struct GPUInferMemorySpacePass final
    : impl::GPUInferMemorySpacePassBase<GPUInferMemorySpacePass> {

  void runOnOperation() override;
};

bool isDefinitelyShared(bufferization::AllocTensorOp alloc) {
  // An allocation can be inferred as shared if it is the destination of a
  // thread distributed `scf.forall` op. All other shared allocations are
  // expected to be properly indicated in advance.
  for (auto user : alloc->getUsers()) {
    if (isa<linalg::CopyOp>(user) &&
        getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(user)) {
      continue;
    }

    auto forallOp = dyn_cast<scf::ForallOp>(user);
    if (!forallOp ||
        !forallOpHasMappingType<gpu::GPUThreadMappingAttr,
                                gpu::GPUWarpMappingAttr>(forallOp)) {
      return false;
    }
  }
  return true;
}

void GPUInferMemorySpacePass::runOnOperation() {
  MLIRContext *context = &getContext();
  FunctionOpInterface funcOp = getOperation();

  gpu::AddressSpaceAttr privateAddressSpace = gpu::AddressSpaceAttr::get(
      context, gpu::GPUDialect::getPrivateAddressSpace());
  gpu::AddressSpaceAttr sharedAddressSpace = gpu::AddressSpaceAttr::get(
      context, gpu::GPUDialect::getWorkgroupAddressSpace());

  WalkResult res = funcOp.walk([&](bufferization::AllocTensorOp alloc) {
    // Continue if the allocation already has a valid memory space.
    std::optional<Attribute> currentMemSpace = alloc.getMemorySpace();
    if (currentMemSpace.has_value()) {
      if (currentMemSpace.value() == privateAddressSpace ||
          currentMemSpace.value() == sharedAddressSpace) {
        return WalkResult::advance();
      }
      alloc.emitOpError(
          "unexpected gpu memory space must be private or workgroup.");
      return WalkResult::interrupt();
    }

    /// Determining GPU memory spaces must be trivial by the time of this pass.
    /// Because this pass runs immediately before bufferization, input IR is
    /// expected to mix (thread) distributed and shared contexts. Because after
    /// bufferization distributed loops (scf.forall) ops are expected to be
    /// inlined as-is with no further tiling occurring, all tensors at this
    /// point in the IR are assumed to be thread-local unless it is explicitly
    /// marked as shared. This gives the following invariants:
    ///
    /// 1. If the alloc_tensor is annotated with `#gpu.address_space<private>`
    ///    already, or if it is used as the immediate destination of a thread
    ///    or warp distributed `scf.forall` op, then the allocation must be
    ///    shared memory.
    /// 2. All other allocations are thread local.
    ///
    /// Any allocation that is not explicitly marked as shared memory that is
    /// supposed to be indicates a bug in earlier passes/lowerings.
    if (isDefinitelyShared(alloc)) {
      alloc.setMemorySpaceAttr(sharedAddressSpace);
    } else {
      alloc.setMemorySpaceAttr(privateAddressSpace);
    }
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) {
    funcOp->emitOpError("failed to set the gpu memory space for all "
                        "`bufferization.alloc_tensor` ops");
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler
