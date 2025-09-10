// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/ExtractAddressComputation.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "extract-address-computation-gpu"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_EXTRACTADDRESSCOMPUTATIONGPUPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper functions for the `load base[off0...]`
//  => `load (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

// Matches getSrcMemRef specs for LdMatrixOp.
// \see LoadLikeOpRewriter.
static Value getLdMatrixOpSrcMemRef(nvgpu::LdMatrixOp ldMatrixOp) {
  return ldMatrixOp.getSrcMemref();
}

// Matches rebuildOpFromAddressAndIndices specs for LdMatrixOp.
// \see LoadLikeOpRewriter.
static nvgpu::LdMatrixOp rebuildLdMatrixOp(RewriterBase &rewriter,
                                           nvgpu::LdMatrixOp ldMatrixOp,
                                           Value srcMemRef,
                                           ArrayRef<Value> indices) {
  Location loc = ldMatrixOp.getLoc();
  return nvgpu::LdMatrixOp::create(
      rewriter, loc, ldMatrixOp.getResult().getType(), srcMemRef, indices,
      ldMatrixOp.getTranspose(), ldMatrixOp.getNumTiles());
}

SmallVector<OpFoldResult>
getLdMatrixOpViewSizeForEachDim(RewriterBase &rewriter,
                                nvgpu::LdMatrixOp ldMatrixOp) {
  Location loc = ldMatrixOp.getLoc();
  auto extractStridedMetadataOp = memref::ExtractStridedMetadataOp::create(
      rewriter, loc, ldMatrixOp.getSrcMemref());
  SmallVector<OpFoldResult> srcSizes =
      extractStridedMetadataOp.getConstifiedMixedSizes();
  SmallVector<OpFoldResult> indices =
      getAsOpFoldResult(ldMatrixOp.getIndices());
  SmallVector<OpFoldResult> finalSizes;

  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineExpr s1 = rewriter.getAffineSymbolExpr(1);

  for (auto [srcSize, indice] : llvm::zip(srcSizes, indices)) {
    finalSizes.push_back(affine::makeComposedFoldedAffineApply(
        rewriter, loc, s0 - s1, {srcSize, indice}));
  }
  return finalSizes;
}

static void
populateExtractAddressComputationGPUPatterns(RewritePatternSet &patterns) {
  populateExtractAddressComputationPatterns(patterns);
  patterns.add<StoreLoadLikeOpRewriter<
      nvgpu::LdMatrixOp,
      /*getSrcMemRef=*/getLdMatrixOpSrcMemRef,
      /*rebuildOpFromAddressAndIndices=*/rebuildLdMatrixOp,
      /*getViewSizeForEachDim=*/getLdMatrixOpViewSizeForEachDim>>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//
namespace {
struct ExtractAddressComputationGPUPass final
    : impl::ExtractAddressComputationGPUPassBase<
          ExtractAddressComputationGPUPass> {
  void runOnOperation() override;
};
} // namespace

void ExtractAddressComputationGPUPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateExtractAddressComputationGPUPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
