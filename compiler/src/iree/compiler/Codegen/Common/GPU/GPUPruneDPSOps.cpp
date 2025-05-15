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

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPRUNEDPSOPSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
/// Pass to create a `bufferization.alloc_tensor` in private space for all DPS
/// ops with unused results that can't be removed.
/// These unused results, if originating from loads from global memory, trigger
/// allocations in global memory space during bufferization, which will fail.
/// So, the allocations must be made earlier to avoid failed bufferization.
/// This pass is a hack to get around other work that can be done to improve
/// the bufferization algorithm.
/// For such cases, need to make Flow preserve the unused result as a
// result of the dispatch
struct GPUPruneDPSOpsPass final
    : impl::GPUPruneDPSOpsPassBase<GPUPruneDPSOpsPass> {
  void runOnOperation() override;
};

void GPUPruneDPSOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FunctionOpInterface funcOp = getOperation();

  auto privSpace = gpu::AddressSpaceAttr::get(
      context, gpu::GPUDialect::getPrivateAddressSpace());

  bufferization::BufferizationOptions options;
  options.defaultMemorySpaceFn =
      [&p = privSpace](TensorType t) -> std::optional<Attribute> { return p; };
  bufferization::AnalysisState analysisState(options);

  // Check if this is an op that doesn't change the address space
  // and is neither a consumer nor producer.
  auto isPassThroughOp = [](Operation *op) -> bool {
    return isa<linalg::PackOp, tensor::ExtractSliceOp, tensor::DimOp,
               tensor::PadOp>(op);
  };
  auto isGlobalLoadOp = [](Operation *op) -> bool {
    return isa<IREE::TensorExt::DispatchTensorLoadOp>(op);
  };
  // An arbitrary choice of too big.
  auto isAllocSizeTooBig = [](Type typ) -> bool {
    auto shapedType = cast<ShapedType>(typ);
    int allocSize = 1;
    for (auto dimSize : shapedType.getShape()) {
      if (ShapedType::isDynamic(dimSize))
        continue;
      allocSize *= dimSize;
    }
    return allocSize >= 32;
  };

  // Iterate over all DPS ops and their inits, collecting ops to add a
  // tensor_alloc for.
  // There are 3 conditions of interest:
  // 1) The result associated to the current init is unused
  // 2) The allocation size if appropriately small
  // 3) The reverse use-def chain of this op is a global load followed by ops
  //    that do not change the address space.
  // The 3rd condition is not strictly necessary but this is just a precation
  // given the hacky nature of this fix.
  SmallVector<std::pair<DestinationStyleOpInterface, int>> worklist;
  funcOp.walk([&](DestinationStyleOpInterface dpsOp) {
    for (int idx = 0; idx < dpsOp.getNumDpsInits(); ++idx) {
      OpOperand *value = dpsOp.getDpsInitOperand(idx);
      // If the associated result is used, do nothing.
      if (!dpsOp->getResult(idx).use_empty()) {
        continue;
      }

      // If the buffer allocation is too large for private space, do nothing.
      if (isAllocSizeTooBig(value->get().getType())) {
        continue;
      }

      // Traverse use-def chain, returning all ops that aren't subviews or
      // slices of a global load (ops which don't change the addr space from
      // global).
      auto stoppingConditionFn = [&](Value a) -> bool {
        auto op = a.getDefiningOp();
        return !isPassThroughOp(op);
      };
      SetVector<Value> producers = analysisState.findValueInReverseUseDefChain(
          value, stoppingConditionFn);

      // If producer of this OpOperand is some value that isn't a subview or
      // slice of a global load, do nothing.
      auto arePartOfGlobalChain = [&](Value a) -> bool {
        auto op = a.getDefiningOp();
        return isPassThroughOp(op) || isGlobalLoadOp(op);
      };
      bool globalProducer = all_of(producers, arePartOfGlobalChain);
      if (!globalProducer)
        continue;
      worklist.push_back({dpsOp, idx});
    }
  });
  // Create alloc in private space for each dps op and set it as the new
  // in-place result.
  for (auto [dpsOp, idx] : worklist) {
    OpOperand *value = dpsOp.getDpsInitOperand(idx);
    PatternRewriter rewriter(dpsOp->getContext());
    rewriter.setInsertionPoint(dpsOp);
    FailureOr<Value> copy = allocateTensorForShapedValue(
        rewriter, dpsOp->getLoc(), value->get(), options, true);
    if (failed(copy))
      return signalPassFailure();
    auto alloc = cast<bufferization::AllocTensorOp>(copy->getDefiningOp());
    alloc.setMemorySpaceAttr(privSpace);
    rewriter.modifyOpInPlace(dpsOp, [&, &dpsOp = dpsOp, &idx = idx]() {
      dpsOp.setDpsInitOperand(idx, *copy);
    });
  }
}

} // namespace
} // namespace mlir::iree_compiler
