// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CombineLayoutTransformation.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#define DEBUG_TYPE "iree-codegen-gpu-combine-layout-transformation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCOMBINELAYOUTTRANSFORMATIONPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

/// TODO(Max191): Improve heuristic for tile size selection.
static SmallVector<DistributionConfig>
gpuPadDistributionConfigFn(ArrayRef<int64_t> iterationBounds,
                           MLIRContext *ctx) {
  // First level of distribution (workgroup).
  DistributionConfig workgroupDistributionConfig;
  workgroupDistributionConfig.tileSizes.assign(iterationBounds.size(), 1);
  workgroupDistributionConfig.tileSizes.back() = 64;
  workgroupDistributionConfig.mapping = llvm::map_to_vector(
      llvm::seq<int64_t>(iterationBounds.size()),
      [&](int64_t dim) -> Attribute {
        switch (dim) {
        case 0:
        case 1:
        case 2:
          return IREE::Codegen::WorkgroupMappingAttr::get(
              ctx, IREE::Codegen::symbolizeWorkgroupId(dim).value());
        default:
          return IREE::Codegen::WorkgroupMappingAttr::get(
              ctx, IREE::Codegen::WorkgroupId::IdZ, dim - 2);
        }
      });
  // Second level of distribution (thread).
  DistributionConfig threadDistributionConfig;
  threadDistributionConfig.tileSizes.assign(iterationBounds.size(), 1);
  threadDistributionConfig.mapping = llvm::map_to_vector(
      llvm::reverse(llvm::seq<int64_t>(iterationBounds.size())),
      [&](int64_t idx) -> Attribute {
        unsigned mappingId =
            static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx;
        return gpu::GPUThreadMappingAttr::get(
            ctx, static_cast<gpu::MappingId>(mappingId));
      });
  return {workgroupDistributionConfig, threadDistributionConfig};
}

namespace {

struct GPUCombineLayoutTransformationPass final
    : impl::GPUCombineLayoutTransformationPassBase<
          GPUCombineLayoutTransformationPass> {
  using impl::GPUCombineLayoutTransformationPassBase<
      GPUCombineLayoutTransformationPass>::
      GPUCombineLayoutTransformationPassBase;

  void runOnOperation() override {
    if (failed(combineLayoutTransformation(&getContext(), getOperation(),
                                           gpuPadDistributionConfigFn))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
