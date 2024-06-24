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
    DiagnosedSilenceableFailure result =
        mlir::transform::gpu::mapNestedForallToThreadsImpl(
            rewriter, std::nullopt, funcOp, workgroupSize.value(), subgroupSize,
            false);
    if (!result.succeeded())
      return signalPassFailure();
  }
};
} // namespace
} // namespace mlir::iree_compiler
