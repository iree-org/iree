// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Support/MathExtras.h"

#define DEBUG_TYPE "iree-codegen-gpu-distribute"

namespace mlir::iree_compiler {

static constexpr int64_t kCudaWarpSize = 32;

namespace {
struct GPUDistributePass : public GPUDistributeBase<GPUDistributePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    if (!workgroupSize) {
      return;
    }

    // TODO: Thread through subgroup size everywhere.
    std::optional<int64_t> maybeSubgroupSize = getSubgroupSize(funcOp);
    // TODO: Don't hard code kCudaWarpSize here.
    int64_t subgroupSize = maybeSubgroupSize.value_or(kCudaWarpSize);

    IRRewriter rewriter(funcOp->getContext());
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

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUDistribute() {
  return std::make_unique<GPUDistributePass>();
}

} // namespace mlir::iree_compiler
