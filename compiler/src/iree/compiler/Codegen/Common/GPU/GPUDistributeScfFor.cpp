// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- GPUDistributeScfFor.cpp ----------------------------------------===//
//
// This pass distributes tiled loop nests with `iree.gpu.distribute_dim`
// attributes to invocations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-distribute-scf-for"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUDISTRIBUTESCFFORPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct DistributeLoop final : OpRewritePattern<scf::ForOp> {
  using Base::Base;

  DistributeLoop(MLIRContext *context, bool useBD, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), useBlockDims(useBD) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only distribute if we see the marker attribute.
    auto numDimAttr =
        forOp->getAttrOfType<IntegerAttr>(getGPUDistributeAttrName());
    if (!numDimAttr)
      return failure();

    // Get workgroup sizes if not using gpu.block_dim
    SmallVector<int64_t> workgroupSize;
    if (!useBlockDims) {
      auto funcOp = forOp->getParentOfType<FunctionOpInterface>();
      if (!funcOp) {
        return failure();
      }
      std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
          getWorkgroupSize(funcOp);
      if (!maybeWorkgroupSize) {
        return failure();
      }
      workgroupSize = maybeWorkgroupSize.value();
    }

    Location loc = forOp.getLoc();
    auto indexType = rewriter.getIndexType();
    const std::array<gpu::Dimension, 3> symDims = {
        gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
    gpu::Dimension symDim = symDims[numDimAttr.getInt()];
    auto idOp = gpu::ThreadIdOp::create(rewriter, loc, indexType, symDim);
    Value count =
        useBlockDims ? gpu::BlockDimOp::create(rewriter, loc, indexType, symDim)
                           .getResult()
                     : arith::ConstantIndexOp::create(
                           rewriter, loc, workgroupSize[numDimAttr.getInt()])
                           .getResult();

    MLIRContext *context = getContext();
    AffineExpr sym0, sym1, sym2;
    bindSymbols(context, sym0, sym1, sym2);
    auto mulAddMap = AffineMap::get(0, 3, {sym0 * sym1 + sym2}, context);
    auto mulMap = AffineMap::get(0, 2, {sym0 * sym1}, context);

    auto newLb = affine::AffineApplyOp::create(
        rewriter, loc, mulAddMap,
        ValueRange{idOp, forOp.getStep(), forOp.getLowerBound()});
    auto newStep = affine::AffineApplyOp::create(
        rewriter, loc, mulMap, ValueRange{count, forOp.getStep()});

    forOp.getLowerBoundMutable().assign(newLb);
    forOp.getStepMutable().assign(newStep);
    // Remove the attribute to avoid endless recursion.
    forOp->removeAttr(getGPUDistributeAttrName());
    return success();
  }

private:
  bool useBlockDims;
};

struct GPUDistributeScfForPass final
    : impl::GPUDistributeScfForPassBase<GPUDistributeScfForPass> {
  using GPUDistributeScfForPassBase::GPUDistributeScfForPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DistributeLoop>(context, useBlockDims);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
