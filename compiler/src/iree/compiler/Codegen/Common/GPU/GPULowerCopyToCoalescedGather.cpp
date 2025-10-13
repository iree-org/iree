// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-lower-copy-to-coalesced-gather"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPULOWERCOPYTOCOALESCEDGATHERPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

struct LowerCopyToGatherPattern : public OpRewritePattern<linalg::CopyOp> {
  LowerCopyToGatherPattern(MLIRContext *context, FunctionOpInterface funcOp)
      : OpRewritePattern<linalg::CopyOp>(context), funcOp(funcOp) {}

  LogicalResult matchAndRewrite(linalg::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    auto loweringConfig =
        getLoweringConfig<IREE::Codegen::LoweringConfigAttrInterface>(copy);
    // Check for lowering_config attribute (e.g., use_global_load_dma)
    if (!loweringConfig) {
      return failure();
    }

    // Get the source and destination tensors
    Value source = copy.getOperand(0);
    Value dest = copy.getOutputs().front();

    auto sourceType = dyn_cast<RankedTensorType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(copy,
                                         "source must be a ranked tensor");
    }

    if (sourceType.getRank() == 0) {
      return rewriter.notifyMatchFailure(
          copy, "Cannot convert scalar copy to gather");
    }

    // Create dimension_map attribute - for a copy, we gather along dimension 0
    SmallVector<int64_t> dimensionMap = {0};
    auto dimensionMapAttr = rewriter.getDenseI64ArrayAttr(dimensionMap);

    // Create the gather operation with only source (copy mode)
    auto gatherOp = rewriter.create<IREE::LinalgExt::GatherOp>(
        copy.getLoc(), dest.getType(), /*inputs=*/ValueRange{source},
        /*outputs=*/ValueRange{dest}, dimensionMapAttr);

    // Create a proper lowering config with tile sizes from translation info
    // Get subgroup size from translation info
    std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
    if (!subgroupSize) {
      // Fall back to just transferring the existing config
      setLoweringConfig(gatherOp, loweringConfig);
    } else {
      // Build tile sizes based on the tensor rank and subgroup size
      int64_t rank = sourceType.getRank();

      // Subgroup level: tile innermost dimension with subgroup size
      SmallVector<int64_t> subgroupTiles(rank, 1);
      subgroupTiles[rank - 1] = *subgroupSize;

      // Thread level: tile innermost dimension with 1 (each thread handles 1
      // element)
      SmallVector<int64_t> threadTiles(rank, 1);

      // Create the lowering config with subgroup and thread tile sizes
      SmallVector<NamedAttribute> fields;
      fields.push_back(rewriter.getNamedAttr(
          "subgroup", rewriter.getI64ArrayAttr(subgroupTiles)));
      fields.push_back(rewriter.getNamedAttr(
          "thread", rewriter.getI64ArrayAttr(threadTiles)));

      auto dictAttr = rewriter.getDictionaryAttr(fields);
      auto newConfig =
          IREE::GPU::LoweringConfigAttr::get(rewriter.getContext(), dictAttr);
      setLoweringConfig(gatherOp, newConfig);
    }

    rewriter.replaceOp(copy, gatherOp.getResults());
    return success();
  }

private:
  FunctionOpInterface funcOp;
};

namespace {
struct GPULowerCopyToCoalescedGatherPass final
    : impl::GPULowerCopyToCoalescedGatherPassBase<
          GPULowerCopyToCoalescedGatherPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::FunctionOpInterface funcOp = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<LowerCopyToGatherPattern>(context, funcOp);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }
};
} // namespace

} // namespace mlir::iree_compiler
