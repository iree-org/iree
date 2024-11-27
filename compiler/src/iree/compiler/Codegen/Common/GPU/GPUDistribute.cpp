// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUDISTRIBUTEPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
static constexpr int64_t kCudaWarpSize = 32;

static void
replaceUnitMappingIdsHelper(RewriterBase &rewriter, Location loc, Block *parent,
                            Value replacement,
                            ArrayRef<int64_t> availableMappingSizes) {
  parent->walk([&](gpu::ThreadIdOp idOp) {
    if (availableMappingSizes[static_cast<int64_t>(idOp.getDimension())] == 1)
      rewriter.replaceAllUsesWith(idOp.getResult(), replacement);
  });
}

// This is an upstream method adapted from
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/GPU/TransformOps/GPUTransformOps.cpp#L846
// to fix the ASAN error.
DiagnosedSilenceableFailure static mapNestedForallToThreadsImpl(
    RewriterBase &rewriter, Operation *target, ArrayRef<int64_t> blockDims,
    int64_t warpSize, bool syncAfterDistribute) {

  if (blockDims.size() != 3) {
    return emitDefiniteFailure(target, "requires size-3 thread mapping");
  }

  Block *parentBlock = target->getBlock();

  // Create an early zero index value for replacements.
  Location loc = target->getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
  WalkResult walkResult = target->walk([&](scf::ForallOp forallOp) {
    diag = mlir::transform::gpu::mapOneForallToThreadsImpl(
        rewriter, std::nullopt, forallOp, blockDims, warpSize,
        syncAfterDistribute);
    if (diag.isDefiniteFailure())
      return WalkResult::interrupt();
    if (diag.succeeded())
      return WalkResult::skip();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return diag;

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  replaceUnitMappingIdsHelper(rewriter, loc, parentBlock, zero, blockDims);
  return DiagnosedSilenceableFailure::success();
}

struct GPUDistributePass final
    : impl::GPUDistributePassBase<GPUDistributePass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp->getContext());

    // First map all lane level forall loops to lanes.
    IREE::GPU::mapLaneForalls(rewriter, funcOp, /*insertBarrier=*/false);

    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    if (!workgroupSize) {
      return;
    }

    // TODO: Thread through subgroup size everywhere.
    std::optional<int64_t> maybeSubgroupSize = getSubgroupSize(funcOp);
    // TODO: Don't hard code kCudaWarpSize here.
    int64_t subgroupSize = maybeSubgroupSize.value_or(kCudaWarpSize);

    rewriter.setInsertionPointToStart(&funcOp.front());

    DiagnosedSilenceableFailure result = DiagnosedSilenceableFailure::success();
    WalkResult walkResult = funcOp->walk([&](scf::ForallOp forallOp) {
      bool hasWorkgroupMapping =
          llvm::any_of(forallOp.getMapping().value(),
                       llvm::IsaPred<IREE::Codegen::WorkgroupMappingAttr>);
      if (!hasWorkgroupMapping) {
        result = mapNestedForallToThreadsImpl(
            rewriter, forallOp, workgroupSize.value(), subgroupSize, false);
        if (result.isDefiniteFailure())
          return WalkResult::interrupt();
        if (result.succeeded())
          return WalkResult::skip();
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace
} // namespace mlir::iree_compiler
