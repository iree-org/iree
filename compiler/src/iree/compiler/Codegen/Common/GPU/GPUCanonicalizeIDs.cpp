// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCANONICALIZEIDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Get workgroup size, trying translation info first, then
/// hal.executable.export fallback if it's already been set on the export.
static std::optional<SmallVector<int64_t>>
getWorkgroupSizeWithFallback(FunctionOpInterface funcOp) {
  // Try translation info first.
  std::optional<SmallVector<int64_t>> result = getWorkgroupSize(funcOp);
  if (result) {
    return result;
  }
  // Fall back to hal.executable.export attributes.
  std::optional<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (!exportOp) {
    return std::nullopt;
  }
  std::optional<ArrayAttr> wgSize = exportOp->getWorkgroupSize();
  if (!wgSize || !(*wgSize)) {
    return std::nullopt;
  }
  SmallVector<int64_t> sizes;
  for (Attribute attr : *wgSize) {
    sizes.push_back(cast<IntegerAttr>(attr).getInt());
  }
  return sizes;
}

/// Get subgroup size, trying translation info first, then
/// hal.executable.export.
static std::optional<int64_t>
getSubgroupSizeWithFallback(FunctionOpInterface funcOp) {
  // Try translation info first.
  std::optional<int64_t> result = getSubgroupSize(funcOp);
  if (result) {
    return result;
  }
  // Fall back to hal.executable.export attributes.
  std::optional<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (!exportOp) {
    return std::nullopt;
  }
  std::optional<IntegerAttr> sgSize = exportOp->getSubgroupSizeAttr();
  if (!sgSize || !(*sgSize)) {
    return std::nullopt;
  }
  return sgSize->getInt();
}

/// Compute the linear thread ID from 3D thread IDs. Generates gpu.block_dim
/// ops for the workgroup sizes and relies on the FoldBlockDim pattern to fold
/// them to constants.
///   linear = tid.x + block_dim.x * (tid.y + block_dim.y * tid.z)
static Value createLinearThreadId(OpBuilder &builder, Location loc) {
  Value tidX = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value tidY = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::y);
  Value tidZ = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::z);
  Value dimX = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::x);
  Value dimY = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::y);
  // linear = tid.x + dimX * (tid.y + dimY * tid.z)
  Value inner = arith::MulIOp::create(builder, loc, tidZ, dimY);
  inner = arith::AddIOp::create(builder, loc, tidY, inner);
  inner = arith::MulIOp::create(builder, loc, inner, dimX);
  return arith::AddIOp::create(builder, loc, tidX, inner);
}

//===----------------------------------------------------------------------===//
// Fold gpu.subgroup_size to a constant.
//===----------------------------------------------------------------------===//

struct FoldSubgroupSize final : OpRewritePattern<gpu::SubgroupSizeOp> {
  FoldSubgroupSize(MLIRContext *ctx, int64_t subgroupSize)
      : OpRewritePattern(ctx), subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(gpu::SubgroupSizeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, subgroupSize);
    return success();
  }

  int64_t subgroupSize;
};

//===----------------------------------------------------------------------===//
// Fold gpu.block_dim to a constant.
//===----------------------------------------------------------------------===//

struct FoldBlockDim final : OpRewritePattern<gpu::BlockDimOp> {
  FoldBlockDim(MLIRContext *ctx, ArrayRef<int64_t> workgroupSize)
      : OpRewritePattern(ctx), workgroupSize(workgroupSize) {}

  LogicalResult matchAndRewrite(gpu::BlockDimOp op,
                                PatternRewriter &rewriter) const override {
    int64_t dimIndex = static_cast<int64_t>(op.getDimension());
    if (dimIndex >= static_cast<int64_t>(workgroupSize.size())) {
      return rewriter.notifyMatchFailure(op, "dimension out of range");
    }
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
        op, workgroupSize[dimIndex]);
    return success();
  }

  SmallVector<int64_t> workgroupSize;
};

//===----------------------------------------------------------------------===//
// Fold gpu.thread_id to zero when the dimension has size 1.
//===----------------------------------------------------------------------===//

struct FoldThreadIdToZero final : OpRewritePattern<gpu::ThreadIdOp> {
  FoldThreadIdToZero(MLIRContext *ctx, ArrayRef<int64_t> workgroupSize)
      : OpRewritePattern(ctx), workgroupSize(workgroupSize) {}

  LogicalResult matchAndRewrite(gpu::ThreadIdOp op,
                                PatternRewriter &rewriter) const override {
    int64_t dimIndex = static_cast<int64_t>(op.getDimension());
    if (dimIndex >= static_cast<int64_t>(workgroupSize.size())) {
      return rewriter.notifyMatchFailure(op, "dimension out of range");
    }
    if (workgroupSize[dimIndex] != 1) {
      return rewriter.notifyMatchFailure(op, "workgroup dimension is not 1");
    }
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
    return success();
  }

  SmallVector<int64_t> workgroupSize;
};

//===----------------------------------------------------------------------===//
// Rewrite gpu.subgroup_id using thread_id linearization.
//===----------------------------------------------------------------------===//

struct RewriteSubgroupId final : OpRewritePattern<gpu::SubgroupIdOp> {
  RewriteSubgroupId(MLIRContext *ctx, int64_t subgroupSize)
      : OpRewritePattern(ctx), subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(gpu::SubgroupIdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value linear = createLinearThreadId(rewriter, loc);
    Value sgSize = arith::ConstantIndexOp::create(rewriter, loc, subgroupSize);
    rewriter.replaceOpWithNewOp<arith::DivUIOp>(op, linear, sgSize);
    return success();
  }

  int64_t subgroupSize;
};

//===----------------------------------------------------------------------===//
// Rewrite gpu.lane_id using thread_id linearization.
//===----------------------------------------------------------------------===//

struct RewriteLaneId final : OpRewritePattern<gpu::LaneIdOp> {
  RewriteLaneId(MLIRContext *ctx, int64_t subgroupSize)
      : OpRewritePattern(ctx), subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(gpu::LaneIdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value linear = createLinearThreadId(rewriter, loc);
    Value sgSize = arith::ConstantIndexOp::create(rewriter, loc, subgroupSize);
    rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, linear, sgSize);
    return success();
  }

  int64_t subgroupSize;
};

//===----------------------------------------------------------------------===//
// Fold gpu.num_subgroups to a constant.
//===----------------------------------------------------------------------===//

struct FoldNumSubgroups final : OpRewritePattern<gpu::NumSubgroupsOp> {
  FoldNumSubgroups(MLIRContext *ctx, ArrayRef<int64_t> workgroupSize,
                   int64_t subgroupSize)
      : OpRewritePattern(ctx), workgroupSize(workgroupSize),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(gpu::NumSubgroupsOp op,
                                PatternRewriter &rewriter) const override {
    int64_t totalThreads = 1;
    for (int64_t dim : workgroupSize) {
      totalThreads *= dim;
    }
    int64_t numSubgroups = totalThreads / subgroupSize;
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, numSubgroups);
    return success();
  }

  SmallVector<int64_t> workgroupSize;
  int64_t subgroupSize;
};

//===----------------------------------------------------------------------===//
// Pass implementation.
//===----------------------------------------------------------------------===//

struct GPUCanonicalizeIDsPass final
    : impl::GPUCanonicalizeIDsPassBase<GPUCanonicalizeIDsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSizeWithFallback(funcOp);
    std::optional<int64_t> subgroupSize = getSubgroupSizeWithFallback(funcOp);

    // Bail if neither is available (not an error).
    if (!workgroupSize && !subgroupSize) {
      return;
    }

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    if (workgroupSize) {
      patterns.add<FoldBlockDim>(ctx, *workgroupSize);
      patterns.add<FoldThreadIdToZero>(ctx, *workgroupSize);
    }
    if (subgroupSize) {
      patterns.add<FoldSubgroupSize>(ctx, *subgroupSize);
      patterns.add<RewriteSubgroupId>(ctx, *subgroupSize);
      patterns.add<RewriteLaneId>(ctx, *subgroupSize);
    }
    if (workgroupSize && subgroupSize) {
      patterns.add<FoldNumSubgroups>(ctx, *workgroupSize, *subgroupSize);
    }

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
