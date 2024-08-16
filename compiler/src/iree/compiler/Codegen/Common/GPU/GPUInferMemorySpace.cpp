// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
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
/// treated as a compiler failure indicating something went wrong during or
/// near bufferization.
struct GPUInferMemorySpacePass final
    : impl::GPUInferMemorySpacePassBase<GPUInferMemorySpacePass> {

  void runOnOperation() override;
};

bool isDefinitelyThreadLocal(bufferization::AllocTensorOp alloc) {
  ArrayRef<int64_t> allocShape = alloc.getType().getShape();
  // Give up on dynamic shapes because we can't easily verify that
  // the destination is overwritten.
  if (ShapedType::isDynamicShape(allocShape)) {
    return false;
  }

  // An allocation can safely be allocated as private if from within a
  // distributed context all threads overwrite the whole allocation. The logic
  // below checks for a limited version of this by only looking for
  // `vector.transfer_write` ops that fully overwrite the tensor.
  for (auto user : alloc->getUsers()) {
    if (!operationHasParentForallOfMappingType<gpu::GPUThreadMappingAttr,
                                               IREE::GPU::LaneIdAttr>(user)) {
      return false;
    }
    auto write = dyn_cast<vector::TransferWriteOp>(user);
    // TODO: look through reshapes and linalg op destinations if necessary.
    if (!write) {
      return false;
    }

    ArrayRef<int64_t> sourceVecShape = write.getVectorType().getShape();
    if (!llvm::all_of_zip(allocShape, sourceVecShape,
                          [](int64_t l, int64_t r) { return l == r; })) {
      return false;
    }

    if (!llvm::all_of(write.getIndices(), [](Value value) {
          return getConstantIntValue(value) == static_cast<int64_t>(0);
        })) {
      return false;
    }
  }
  return true;
}

bool isDefinitelyShared(bufferization::AllocTensorOp alloc) {
  // An allocation can be inferred as shared if it is the destination of a
  // thread distributed `scf.forall` op. All other shared allocations are
  // expected to be properly indicated in advance.
  for (auto user : alloc->getUsers()) {
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
    // Continue if the allocation already has a memory space.
    if (alloc.getMemorySpace().has_value()) {
      return WalkResult::advance();
    }

    if (isDefinitelyThreadLocal(alloc)) {
      alloc.setMemorySpaceAttr(privateAddressSpace);
      return WalkResult::advance();
    }

    if (isDefinitelyShared(alloc)) {
      alloc.setMemorySpaceAttr(sharedAddressSpace);
      return WalkResult::advance();
    }

    alloc->emitOpError("failed to infer missing memory space.");
    return WalkResult::interrupt();
  });

  if (res.wasInterrupted()) {
    funcOp->emitOpError("failed to set the gpu memory space for all "
                        "`bufferization.alloc_tensor` ops");
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler
