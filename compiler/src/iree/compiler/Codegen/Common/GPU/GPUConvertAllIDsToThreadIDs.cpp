// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCONVERTALLIDSTOTHREADIDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Compute the linear thread ID from 3D thread IDs using
/// affine.linearize_index. The generated gpu.block_dim ops are folded to
/// constants by PropagateDispatchSizeBounds later in the pipeline.
///   linear = linearize_index [tid.z, tid.y, tid.x] by (bdim.z, bdim.y, bdim.x)
static Value createLinearThreadId(OpBuilder &builder, Location loc) {
  Value tidX = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value tidY = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::y);
  Value tidZ = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::z);
  Value dimX = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::x);
  Value dimY = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::y);
  Value dimZ = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::z);
  return affine::AffineLinearizeIndexOp::create(
      builder, loc, ValueRange{tidZ, tidY, tidX}, ValueRange{dimZ, dimY, dimX},
      /*disjoint=*/true);
}

//===----------------------------------------------------------------------===//
// Rewrite gpu.subgroup_id using thread_id linearization.
//===----------------------------------------------------------------------===//

struct RewriteSubgroupId final : OpRewritePattern<gpu::SubgroupIdOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SubgroupIdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value linear = createLinearThreadId(rewriter, loc);
    Value sgSize =
        gpu::SubgroupSizeOp::create(rewriter, loc, rewriter.getIndexType(),
                                    /*upper_bound=*/nullptr);
    rewriter.replaceOpWithNewOp<arith::DivUIOp>(op, linear, sgSize);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Rewrite gpu.lane_id using thread_id linearization.
//===----------------------------------------------------------------------===//

struct RewriteLaneId final : OpRewritePattern<gpu::LaneIdOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaneIdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value linear = createLinearThreadId(rewriter, loc);
    Value sgSize =
        gpu::SubgroupSizeOp::create(rewriter, loc, rewriter.getIndexType(),
                                    /*upper_bound=*/nullptr);
    rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, linear, sgSize);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation.
//===----------------------------------------------------------------------===//

struct GPUConvertAllIDsToThreadIDsPass final
    : impl::GPUConvertAllIDsToThreadIDsPassBase<
          GPUConvertAllIDsToThreadIDsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<RewriteSubgroupId, RewriteLaneId>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
