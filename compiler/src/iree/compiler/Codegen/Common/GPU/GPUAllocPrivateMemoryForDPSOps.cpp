// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUALLOCPRIVATEMEMORYFORDPSOPSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

static constexpr int64_t kAllocationSizeInBytes = 32;

namespace {
/// This pass is a hack to get around other work that can be done to improve
/// the bufferization algorithm.
/// For such cases, need to make Flow preserve the unused result as a
/// result of the dispatch.
struct GPUAllocPrivateMemoryForDPSOpsPass final
    : impl::GPUAllocPrivateMemoryForDPSOpsPassBase<
          GPUAllocPrivateMemoryForDPSOpsPass> {
  void runOnOperation() override;
};

void GPUAllocPrivateMemoryForDPSOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FunctionOpInterface funcOp = getOperation();

  // An arbitrary choice of too big.
  auto isAllocSizeTooBig = [](Type type) -> bool {
    auto shapedType = cast<ShapedType>(type);
    int64_t allocSize = 1;
    for (auto dimSize : shapedType.getShape()) {
      if (ShapedType::isDynamic(dimSize)) {
        continue;
      }
      allocSize *= dimSize;
    }
    return allocSize >= kAllocationSizeInBytes;
  };

  // Iterate over all DPS ops and their inits, collecting ops to add a
  // tensor_alloc for.
  // There are 2 conditions of interest:
  // 1) The result associated to the current init is unused.
  // 2) The allocation size if appropriately small (ignoring dynamic
  // dimensions).
  SmallVector<OpOperand *> worklist;
  funcOp.walk([&](DestinationStyleOpInterface dpsOp) {
    if (!dpsOp.hasPureTensorSemantics()) {
      return;
    }
    for (int idx = 0; idx < dpsOp.getNumDpsInits(); ++idx) {
      OpOperand *value = dpsOp.getDpsInitOperand(idx);
      if (!dpsOp.getTiedOpResult(value).use_empty()) {
        continue;
      }
      if (isAllocSizeTooBig(value->get().getType())) {
        continue;
      }
      worklist.push_back(value);
    }
  });

  // Create alloc for each dps op and set it as the new in-place result.
  auto privSpace = gpu::AddressSpaceAttr::get(
      context, gpu::GPUDialect::getPrivateAddressSpace());
  bufferization::BufferizationOptions options;
  bufferization::BufferizationState bufferizationState;
  IRRewriter rewriter(context);
  for (auto value : worklist) {
    Location loc = value->getOwner()->getLoc();
    rewriter.setInsertionPoint(value->getOwner());
    FailureOr<Value> copy =
        allocateTensorForShapedValue(rewriter, loc, value->get(), options,
                                     bufferizationState, /*copy=*/true);
    if (failed(copy)) {
      funcOp.emitError("Could not allocate a tensor of the required type");
      return signalPassFailure();
    }
    auto allocTensor =
        cast<bufferization::AllocTensorOp>(copy->getDefiningOp());
    allocTensor.setMemorySpaceAttr(privSpace);
    value->set(allocTensor);
  }
}

} // namespace
} // namespace mlir::iree_compiler
