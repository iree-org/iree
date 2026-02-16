// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CombineLayoutTransformation.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#define DEBUG_TYPE "iree-codegen-gpu-combine-layout-transformation"

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

/// The control function for when to fold a relayout op chain into a map_store
/// op. Only combine complex relayout sequences at the end of a dispatch that
/// are difficult to handle in GPU Codegen. Any of the following list of
/// conditions are considered to indicate a complex relayout sequence, and will
/// cause the control function to return true:
/// 1. There are pack or unpack ops in the chain.
/// 2. There are reshape ops in the chain.
///    - We expect reshape ops to be folded
///      into the output buffer at this point, if possible, so any leftover
///      reshape ops will be difficult to handle in during code generation.
/// 3. There are pad ops in the chain.
///    - We want to be able to pull out the pad and write padding values
///      directly to the output buffer.
///
/// The GPUCombineLayoutTransformationPass is expected to run early on during
/// the configuration phase of Codegen, so the conditions above are based on the
/// assumption that the relayout op chain represents the tail end of a dispatch
/// before any tiling or distribution.
static bool gpuRelayoutCombinationControlFn(OpResult leaf) {
  if (leaf.getNumUses() != 1) {
    return false;
  }
  if (!isa<IREE::Codegen::StoreToBufferOp>(*leaf.getUsers().begin())) {
    return false;
  }
  llvm::SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.filter = isSupportedSingleInputRelayoutOp;
  options.inclusive = true;
  LogicalResult result = getBackwardSlice(leaf, &slice, options);
  if (failed(result)) {
    return false;
  }
  return llvm::any_of(
      slice,
      llvm::IsaPred<linalg::PackOp, linalg::UnPackOp, tensor::ExpandShapeOp,
                    tensor::CollapseShapeOp, tensor::PadOp>);
}

namespace {

struct GPUCombineLayoutTransformationPass final
    : impl::GPUCombineLayoutTransformationPassBase<
          GPUCombineLayoutTransformationPass> {
  using impl::GPUCombineLayoutTransformationPassBase<
      GPUCombineLayoutTransformationPass>::
      GPUCombineLayoutTransformationPassBase;

  void runOnOperation() override {
    if (failed(combineLayoutTransformation(
            &getContext(), getOperation(), gpuPadDistributionConfigFn,
            /*doReshapeByExpansion=*/true, gpuRelayoutCombinationControlFn))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
