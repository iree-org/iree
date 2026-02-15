// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUFUSESUBGROUPCONSUMERSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

using namespace IREE::GPU;
using namespace IREE::PCF;

namespace {

/// Pattern that fuses tilable consumers into subgroup-scoped pcf.generic ops.
struct FuseConsumerIntoSubgroupGenericOp final
    : public OpRewritePattern<GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Only operate on subgroup-scoped pcf.generic ops.
    if (!isa<SubgroupScopeAttr>(genericOp.getScope())) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "not a subgroup-scoped pcf.generic");
    }

    // Find a tilable consumer to fuse.
    ConsumerFusionParams params;
    TilingInterface fusionTarget;
    for (Operation *user : genericOp->getUsers()) {
      fusionTarget = dyn_cast<TilingInterface>(user);
      ConsumerFusionParams tempParams;
      if (fusionTarget && succeeded(matchTilableConsumer(
                              rewriter, genericOp, fusionTarget, tempParams))) {
        std::swap(params, tempParams);
        break;
      }
      fusionTarget = TilingInterface();
    }
    if (!fusionTarget) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "no fusible tilable consumer found");
    }
    fuseTilableConsumer(rewriter, genericOp, fusionTarget, params);
    return success();
  }
};

/// Pattern that fuses tensor.extract_slice consumers into subgroup-scoped
/// pcf.generic ops.
struct FuseExtractSliceIntoSubgroupGenericOp final
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    GenericOp genericOp = extractSliceOp.getSource().getDefiningOp<GenericOp>();
    if (!genericOp) {
      return rewriter.notifyMatchFailure(extractSliceOp,
                                         "no pcf.generic producer");
    }

    // Only operate on subgroup-scoped pcf.generic ops.
    if (!isa<SubgroupScopeAttr>(genericOp.getScope())) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "not a subgroup-scoped pcf.generic");
    }

    if (failed(fuseExtractSliceIntoProducerGeneric(rewriter, genericOp,
                                                   extractSliceOp))) {
      return rewriter.notifyMatchFailure(
          extractSliceOp, "failed to fuse extract_slice into pcf.generic");
    }
    return success();
  }
};

struct GPUFuseSubgroupConsumersPass final
    : public impl::GPUFuseSubgroupConsumersPassBase<
          GPUFuseSubgroupConsumersPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseConsumerIntoSubgroupGenericOp,
                 FuseExtractSliceIntoSubgroupGenericOp>(&getContext());
    populatePCFDropUnusedResultPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
