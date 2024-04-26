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

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-distribute-scf-for"

namespace mlir::iree_compiler {

namespace {

struct DistributeLoop final : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  DistributeLoop(MLIRContext *context, bool useBD, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), useBlockDims(useBD) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only distribute if we see the marker attribute.
    auto numDimAttr =
        forOp->getAttrOfType<IntegerAttr>(getGPUDistributeAttrName());
    if (!numDimAttr)
      return failure();

    auto funcOp = forOp->getParentOfType<FunctionOpInterface>();
    if (!funcOp) {
      return failure();
    }
    std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
        getWorkgroupSize(funcOp);
    if (!maybeWorkgroupSize) {
      return failure();
    }
    auto workgroupSize = maybeWorkgroupSize.value();

    Location loc = forOp.getLoc();
    auto indexType = rewriter.getIndexType();
    const std::array<gpu::Dimension, 3> symDims = {
        gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
    gpu::Dimension symDim = symDims[numDimAttr.getInt()];
    auto idOp = rewriter.create<gpu::ThreadIdOp>(loc, indexType, symDim);
    Value count = useBlockDims
                      ? rewriter.create<gpu::BlockDimOp>(loc, indexType, symDim)
                            .getResult()
                      : rewriter
                            .create<arith::ConstantIndexOp>(
                                loc, workgroupSize[numDimAttr.getInt()])
                            .getResult();

    MLIRContext *context = getContext();
    AffineExpr sym0, sym1, sym2;
    bindSymbols(context, sym0, sym1, sym2);
    auto mulAddMap = AffineMap::get(0, 3, {sym0 * sym1 + sym2}, context);
    auto mulMap = AffineMap::get(0, 2, {sym0 * sym1}, context);

    auto newLb = rewriter.create<affine::AffineApplyOp>(
        loc, mulAddMap,
        ValueRange{idOp, forOp.getStep(), forOp.getLowerBound()});
    auto newStep = rewriter.create<affine::AffineApplyOp>(
        loc, mulMap, ValueRange{count, forOp.getStep()});

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
    : public GPUDistributeScfForBase<GPUDistributeScfForPass> {
public:
  GPUDistributeScfForPass(bool useBlockDims) {
    this->useBlockDims = useBlockDims;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DistributeLoop>(context, useBlockDims);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUDistributeScfForPass(bool useBlockDims) {
  return std::make_unique<GPUDistributeScfForPass>(useBlockDims);
}

} // namespace mlir::iree_compiler
