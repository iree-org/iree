// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_RESOLVESWIZZLEHINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct ResolveSwizzleHintsPass final
    : impl::ResolveSwizzleHintsPassBase<ResolveSwizzleHintsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static Value createOrFoldNewStaticAdd(RewriterBase &rewriter, Value v,
                                      int64_t offset) {
  // Early exit for the common offset = 0 case.
  if (offset == 0) {
    return v;
  }

  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    llvm::APInt constant;
    if (matchPattern(add.getRhs(), m_ConstantInt(&constant))) {
      Value combined = arith::ConstantIndexOp::create(
          rewriter, add.getLoc(), offset + constant.getSExtValue());
      return arith::AddIOp::create(rewriter, add.getLoc(), add.getLhs(),
                                   combined, add.getOverflowFlags());
    }
  }
  Value offsetVal =
      arith::ConstantIndexOp::create(rewriter, v.getLoc(), offset);
  return arith::AddIOp::create(rewriter, v.getLoc(), v, offsetVal);
}

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

  // For loads smaller than accessWidth, swizzle the base offset and add
  // back the within-group offset. swizzleOffset drops |memrefOffset %
  // accessWidth| because it targets group-aligned accesses; for scalar
  // or sub-group loads we need to preserve the element's position within
  // the group so threads at different elements within the same group
  // read from distinct addresses.
  if (loadWidth < accessWidth) {
    Value swizzledBase = getValueOrCreateConstantIndexOp(
        rewriter, hintLoc,
        hintOp.getSwizzle().swizzleOffset(rewriter, hintOp.getLoc(),
                                          memrefOffset, hintOp.getOperand()));
    Value accessWidthVal =
        arith::ConstantIndexOp::create(rewriter, hintLoc, accessWidth);
    Value withinGroup =
        arith::RemUIOp::create(rewriter, hintLoc, memrefOffset, accessWidthVal);
    Value newOffset =
        arith::AddIOp::create(rewriter, hintLoc, swizzledBase, withinGroup);
    auto newLoad = vector::LoadOp::create(rewriter, load.getLoc(), type,
                                          load.getBase(), newOffset);
    rewriter.replaceOp(load, newLoad);
    return;
  }

  VectorType swizzledLoadType =
      VectorType::get({accessWidth}, type.getElementType());

  Value replacement = arith::ConstantOp::create(rewriter, hintLoc, type,
                                                rewriter.getZeroAttr(type));

  for (int64_t i = 0; i < loadWidth; i += accessWidth) {
    Value newBaseOffset = createOrFoldNewStaticAdd(rewriter, memrefOffset, i);
    Value newOffset = getValueOrCreateConstantIndexOp(
        rewriter, hintLoc,
        hintOp.getSwizzle().swizzleOffset(rewriter, hintOp.getLoc(),
                                          newBaseOffset, hintOp.getOperand()));
    auto subLoad = vector::LoadOp::create(
        rewriter, load.getLoc(), swizzledLoadType, load.getBase(), newOffset);

    replacement = vector::InsertStridedSliceOp::create(
        rewriter, load.getLoc(), subLoad, replacement, ArrayRef<int64_t>{i},
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

  // For stores smaller than accessWidth, swizzle the base offset and add
  // back the within-group offset (symmetric to the load path above).
  if (storeWidth < accessWidth) {
    Value swizzledBase = getValueOrCreateConstantIndexOp(
        rewriter, hintLoc,
        hintOp.getSwizzle().swizzleOffset(rewriter, hintOp.getLoc(),
                                          memrefOffset, hintOp.getOperand()));
    Value accessWidthVal =
        arith::ConstantIndexOp::create(rewriter, hintLoc, accessWidth);
    Value withinGroup =
        arith::RemUIOp::create(rewriter, hintLoc, memrefOffset, accessWidthVal);
    Value newOffset =
        arith::AddIOp::create(rewriter, hintLoc, swizzledBase, withinGroup);
    vector::StoreOp::create(rewriter, store.getLoc(), store.getValueToStore(),
                            store.getBase(), newOffset);
    rewriter.eraseOp(store);
    return;
  }

  for (int64_t i = 0; i < storeWidth; i += accessWidth) {
    Value subVec = vector::ExtractStridedSliceOp::create(
        rewriter, store.getLoc(), store.getValueToStore(), ArrayRef<int64_t>{i},
        ArrayRef<int64_t>{accessWidth}, ArrayRef<int64_t>{1});
    Value newBaseOffset = createOrFoldNewStaticAdd(rewriter, memrefOffset, i);

    Value newOffset = getValueOrCreateConstantIndexOp(
        rewriter, hintLoc,
        hintOp.getSwizzle().swizzleOffset(rewriter, hintOp.getLoc(),
                                          newBaseOffset, hintOp.getOperand()));
    vector::StoreOp::create(rewriter, store.getLoc(), subVec, store.getBase(),
                            newOffset);
  }
  rewriter.eraseOp(store);
}

static LogicalResult
verifyFlatContiguousSwizzleHintOp(IREE::Codegen::SwizzleHintOp hintOp) {
  auto memrefType = cast<MemRefType>(hintOp.getOperand().getType());
  // Swizzle hints require flat (rank 1) memrefs.
  // For rank 1, allow dynamic memrefs or static contiguous row-major memrefs.
  if (memrefType.getRank() != 1 ||
      (memrefType.hasStaticShape() &&
       !memref::isStaticShapeAndContiguousRowMajor(memrefType))) {
    hintOp.emitError()
        << "swizzle hint operand must be a contiguous flat memref, got "
        << hintOp.getOperand().getType();
    return failure();
  }
  return success();
}

/// Resolves all hints. Walks all direct users and splits them into loads and
/// stores. If any user is not a swizzle-able load or store, bail out and
/// silently drop the optimization hint.
static void resolveHintOp(RewriterBase &rewriter,
                          IREE::Codegen::SwizzleHintOp hintOp) {
  SmallVector<vector::LoadOp> loads;
  SmallVector<vector::StoreOp> stores;
  for (Operation *user : hintOp->getUsers()) {
    if (auto load = dyn_cast<vector::LoadOp>(user)) {
      VectorType loadType = load.getVectorType();
      // Guard on zero rank loads.
      if (loadType.getRank() != 1) {
        return;
      }
      loads.push_back(load);
      continue;
    }
    if (auto store = dyn_cast<vector::StoreOp>(user)) {
      VectorType storeType = store.getVectorType();
      // Guard on zero rank stores.
      if (storeType.getRank() != 1) {
        return;
      }
      stores.push_back(store);
      continue;
    }
    // Gather_to_lds destination-side swizzle is handled by
    // AMDGPULowerCoalescedDMAToGatherLDS, which applies the inverse swizzle
    // to source indices. Treat gather_to_lds and view-like ops as transparent
    // users that pass through the swizzled allocation.
    if (isa<amdgpu::GatherToLDSOp, ViewLikeOpInterface>(user)) {
      continue;
    }
    // Throw if we can't rewrite all users.
    hintOp.emitError() << "unsupported SwizzleHintOp user: " << user;
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
    if (failed(verifyFlatContiguousSwizzleHintOp(hintOp))) {
      return signalPassFailure();
    }
    resolveHintOp(rewriter, hintOp);
  }

  // Drop all hints.
  for (auto hintOp : hintOps) {
    rewriter.replaceOp(hintOp, hintOp.getOperand());
  }
}

} // namespace mlir::iree_compiler
