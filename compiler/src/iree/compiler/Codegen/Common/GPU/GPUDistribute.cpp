// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
    if (!isEntryPoint(funcOp))
      return;

    auto workgroupSize = llvm::map_to_vector(
        getEntryPoint(funcOp)->getWorkgroupSize().value(),
        [&](Attribute attr) { return llvm::cast<IntegerAttr>(attr).getInt(); });

    // TODO: Thread through subgroup size everywhere.
    std::optional<llvm::APInt> maybeSubgroupSize =
        getEntryPoint(funcOp)->getSubgroupSize();
    // TODO: Don't hard code kCudaWarpSize here.
    int64_t subgroupSize =
        maybeSubgroupSize ? maybeSubgroupSize->getSExtValue() : kCudaWarpSize;

    IRRewriter rewriter(funcOp->getContext());
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    DiagnosedSilenceableFailure result =
        mlir::transform::gpu::mapNestedForallToThreadsImpl(
            rewriter, std::nullopt, funcOp, workgroupSize, subgroupSize, false);
    if (!result.succeeded())
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGPUDistribute() {
  return std::make_unique<GPUDistributePass>();
}

} // namespace mlir::iree_compiler
