// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_RESOLVESWIZZLEHINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct ResolveSwizzleHintsPass final
    : public impl::ResolveSwizzleHintsPassBase<ResolveSwizzleHintsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

/// Swizzles vector.load(iree_codegen.swizzle_hint, offset). The
/// SwizzleInterfaceAttr exposes two methods:
///   1. getAccessElementCount -> int64_t
///        - Gives the number of contiguous elements in the swizzling pattern.
///   2. swizzleOffset(src: memref<N x !eltype>, offset: index) -> index
///        - Swizzles the |offset| into |src|, returning the new offset.
///
/// For a 1-d load of type `vector<C x !eltype>`, the load is unrolled into
/// loads of size `k = getAccessElementCount()` and individually swizzled.
///
/// For example, if `C = 16` and `k = 4`, this produces:
///
/// %0 = vector.load %src[swizzleOffset(%src, %offset)] : vector<4>
/// %1 = vector.load %src[swizzleOffset(%src, %offset + 4)] : vector<4>
/// %2 = vector.load %src[swizzleOffset(%src, %offset + 8)] : vector<4>
/// %3 = vector.load %src[swizzleOffset(%src, %offset + 12)] : vector<4>
/// %load = concat[%0, %1, %2, %3] : vector<16>
static void swizzleLoad(RewriterBase &rewriter, vector::LoadOp load,
                        IREE::Codegen::SwizzleHintOp hintOp) {
  Location hintLoc = hintOp.getLoc();
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  VectorType type = load.getVectorType();
  int64_t loadWidth = type.getShape()[0];
  Value memrefOffset = load.getIndices()[0];
  VectorType swizzledLoadType =
      VectorType::get({accessWidth}, type.getElementType());

  AffineExpr s0, s1;
  bindSymbols(rewriter.getContext(), s0, s1);
  AffineMap sum = AffineMap::get(0, 2, s0 + s1);

  // ~ vector.undef, overwritten by unrolling.
  Value replacement = rewriter.create<arith::ConstantOp>(
      hintLoc, type, rewriter.getZeroAttr(type));

  // Load type = vector<C>, k = accessWidth
  // i = 0 -> C += k is the offset into the vector of a contiguous group of
  // swizzled elements.
  for (int64_t i = 0; i < loadWidth; i += accessWidth) {
    auto vecOffset = rewriter.getIndexAttr(i);
    auto newBaseOffset = affine::makeComposedFoldedAffineApply(
        rewriter, hintLoc, sum, {memrefOffset, vecOffset});

    Value newOffset = getValueOrCreateConstantIndexOp(
        rewriter, hintLoc,
        hintOp.getSwizzle().swizzleOffset(rewriter, hintOp.getLoc(),
                                          newBaseOffset, hintOp.getOperand()));
    auto subLoad = rewriter.create<vector::LoadOp>(
        load.getLoc(), swizzledLoadType, load.getBase(), newOffset);

    replacement = rewriter.create<vector::InsertStridedSliceOp>(
        load.getLoc(), subLoad, replacement, ArrayRef<int64_t>{i},
        ArrayRef<int64_t>{1});
  }
  rewriter.replaceOp(load, replacement);
}

/// Swizzles vector.store(iree_codegen.swizzle_hint, offset).
///
/// For a 1-d store of type `vector<C x !eltype>`, the store is unrolled into
/// stores of size `k = getAccessElementCount()` and individually swizzled.
///
/// For example, if `C = 16` and `k = 4`, this produces:
///
/// %0, %1, %2, %3 = split[%value_to_store] : vector<16>
/// vector.store %0, %src[swizzleOffset(%src, %offset)] : vector<4>
/// vector.store %1, %src[swizzleOffset(%src, %offset + 4)] : vector<4>
/// vector.store %2, %src[swizzleOffset(%src, %offset + 8)] : vector<4>
/// vector.store %3, %src[swizzleOffset(%src, %offset + 12)] : vector<4>
static void swizzleStore(RewriterBase &rewriter, vector::StoreOp store,
                         IREE::Codegen::SwizzleHintOp hintOp) {
  Location hintLoc = hintOp.getLoc();
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  VectorType type = store.getVectorType();
  int64_t storeWidth = type.getShape()[0];
  Value memrefOffset = store.getIndices()[0];

  AffineExpr s0, s1;
  bindSymbols(rewriter.getContext(), s0, s1);
  AffineMap sum = AffineMap::get(0, 2, s0 + s1);

  // Store type = vector<C>, k = accessWidth
  // i = 0 -> C += k is the offset into the vector of a contiguous group of
  // swizzled elements.
  for (int64_t i = 0; i < storeWidth; i += accessWidth) {
    Value subVec = rewriter.create<vector::ExtractStridedSliceOp>(
        store.getLoc(), store.getValueToStore(), ArrayRef<int64_t>{i},
        ArrayRef<int64_t>{accessWidth}, ArrayRef<int64_t>{1});
    auto vecOffset = rewriter.getIndexAttr(i);
    auto newBaseOffset = affine::makeComposedFoldedAffineApply(
        rewriter, hintLoc, sum, {memrefOffset, vecOffset});

    Value newOffset = getValueOrCreateConstantIndexOp(
        rewriter, hintLoc,
        hintOp.getSwizzle().swizzleOffset(rewriter, hintOp.getLoc(),
                                          newBaseOffset, hintOp.getOperand()));
    rewriter.create<vector::StoreOp>(store.getLoc(), subVec, store.getBase(),
                                     newOffset);
  }
  rewriter.eraseOp(store);
}

/// Resolves all hints. Walks all direct users and splits them into loads and
/// stores. If any user is not a swizzle-able load or store, bail out and
/// silently drop the optimization hint.
static void resolveHintOp(RewriterBase &rewriter,
                          IREE::Codegen::SwizzleHintOp hintOp) {
  SmallVector<vector::LoadOp> loads;
  SmallVector<vector::StoreOp> stores;
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  for (Operation *user : hintOp->getUsers()) {
    if (auto load = dyn_cast<vector::LoadOp>(user)) {
      VectorType loadType = load.getVectorType();
      // Guard on zero rank loads and loads not divisible by the access width.
      if (loadType.getRank() != 1 ||
          loadType.getShape()[0] % accessWidth != 0) {
        return;
      }
      loads.push_back(load);
      continue;
    }
    if (auto store = dyn_cast<vector::StoreOp>(user)) {
      VectorType storeType = store.getVectorType();
      // Guard on zero rank stores and stores not divisible by the access width.
      if (storeType.getRank() != 1 ||
          storeType.getShape()[0] % accessWidth != 0) {
        return;
      }
      stores.push_back(store);
      continue;
    }
    // Bail out if we can't rewrite all users.
    return;
  }

  for (vector::LoadOp load : loads) {
    rewriter.setInsertionPoint(load);
    swizzleLoad(rewriter, load, hintOp);
  }
  for (vector::StoreOp store : stores) {
    rewriter.setInsertionPoint(store);
    swizzleStore(rewriter, store, hintOp);
  }
}

void ResolveSwizzleHintsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  // Collect all hint ops.
  SmallVector<IREE::Codegen::SwizzleHintOp> hintOps;
  funcOp.walk(
      [&](IREE::Codegen::SwizzleHintOp hint) { hintOps.push_back(hint); });

  // Swizzle all load/store uses of the hint ops if possible. If we can't
  // guarantee all accesses of a particular hint are swizzled, this will
  // silently pass through for that hint.
  IRRewriter rewriter(funcOp->getContext());
  for (IREE::Codegen::SwizzleHintOp hintOp : hintOps) {
    resolveHintOp(rewriter, hintOp);
  }

  // Drop all hints.
  for (auto hintOp : hintOps) {
    rewriter.replaceOp(hintOp, hintOp.getOperand());
  }
}

} // namespace mlir::iree_compiler
