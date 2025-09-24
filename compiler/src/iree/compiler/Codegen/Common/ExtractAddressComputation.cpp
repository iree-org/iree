// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/ExtractAddressComputation.h"

#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-extract-address-computation"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_EXTRACTADDRESSCOMPUTATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper functions for the `load base[off0...]`
//  => `load (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

// Matches getSrcMemRef specs for LoadOp.
// \see LoadLikeOpRewriter.
static Value getLoadOpSrcMemRef(memref::LoadOp loadOp) {
  return loadOp.getMemRef();
}

// Matches rebuildOpFromAddressAndIndices specs for LoadOp.
// \see LoadLikeOpRewriter.
static memref::LoadOp rebuildLoadOp(RewriterBase &rewriter,
                                    memref::LoadOp loadOp, Value srcMemRef,
                                    ArrayRef<Value> indices) {
  Location loc = loadOp.getLoc();
  return memref::LoadOp::create(rewriter, loc, srcMemRef, indices,
                                loadOp.getNontemporal());
}

SmallVector<OpFoldResult> getLoadOpViewSizeForEachDim(RewriterBase &rewriter,
                                                      memref::LoadOp loadOp) {
  MemRefType ldTy = loadOp.getMemRefType();
  unsigned loadRank = ldTy.getRank();
  return SmallVector<OpFoldResult>(loadRank, rewriter.getIndexAttr(1));
}

//===----------------------------------------------------------------------===//
// Helper functions for the `store val, base[off0...]`
//  => `store val, (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

// Matches getSrcMemRef specs for StoreOp.
// \see LoadStoreLikeOpRewriter.
static Value getStoreOpSrcMemRef(memref::StoreOp storeOp) {
  return storeOp.getMemRef();
}

// Matches rebuildOpFromAddressAndIndices specs for StoreOp.
// \see LoadStoreLikeOpRewriter.
static memref::StoreOp rebuildStoreOp(RewriterBase &rewriter,
                                      memref::StoreOp storeOp, Value srcMemRef,
                                      ArrayRef<Value> indices) {
  Location loc = storeOp.getLoc();
  return memref::StoreOp::create(rewriter, loc, storeOp.getValueToStore(),
                                 srcMemRef, indices, storeOp.getNontemporal());
}

SmallVector<OpFoldResult>
getStoreOpViewSizeForEachDim(RewriterBase &rewriter, memref::StoreOp storeOp) {
  MemRefType ldTy = storeOp.getMemRefType();
  unsigned loadRank = ldTy.getRank();
  return SmallVector<OpFoldResult>(loadRank, rewriter.getIndexAttr(1));
}

void populateExtractAddressComputationPatterns(RewritePatternSet &patterns) {
  patterns.add<StoreLoadLikeOpRewriter<
                   memref::LoadOp,
                   /*getSrcMemRef=*/getLoadOpSrcMemRef,
                   /*rebuildOpFromAddressAndIndices=*/rebuildLoadOp,
                   /*getViewSizeForEachDim=*/getLoadOpViewSizeForEachDim>,
               StoreLoadLikeOpRewriter<
                   memref::StoreOp,
                   /*getSrcMemRef=*/getStoreOpSrcMemRef,
                   /*rebuildOpFromAddressAndIndices=*/rebuildStoreOp,
                   /*getViewSizeForEachDim=*/getStoreOpViewSizeForEachDim>>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//
namespace {

struct ExtractAddressComputationPass final
    : impl::ExtractAddressComputationPassBase<ExtractAddressComputationPass> {
  void runOnOperation() override;
};
} // namespace

void ExtractAddressComputationPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateExtractAddressComputationPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler
