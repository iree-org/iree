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

/// Returns the swizzled memref offset for chunk `i` of an unrolled access.
static Value computeSwizzledChunkOffset(RewriterBase &rewriter,
                                        IREE::Codegen::SwizzleHintOp hintOp,
                                        Value memrefOffset, int64_t i) {
  Value newBaseOffset = createOrFoldNewStaticAdd(rewriter, memrefOffset, i);
  return getValueOrCreateConstantIndexOp(
      rewriter, hintOp.getLoc(),
      hintOp.getSwizzle().swizzleOffset(rewriter, hintOp.getLoc(),
                                        newBaseOffset, hintOp.getOperand()));
}

/// Extracts a contiguous chunk of `accessWidth` elements starting at index `i`
/// from a 1-d vector.
static Value extractChunk(RewriterBase &rewriter, Location loc, Value src,
                          int64_t i, int64_t accessWidth) {
  return vector::ExtractStridedSliceOp::create(
      rewriter, loc, src, ArrayRef<int64_t>{i}, ArrayRef<int64_t>{accessWidth},
      ArrayRef<int64_t>{1});
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
///
/// The masked variant (mask + passThru non-null) emits vector.maskedload ops
/// with the mask/passThru sliced along the access width.
static void swizzleLoad(RewriterBase &rewriter, Operation *loadOp,
                        IREE::Codegen::SwizzleHintOp hintOp, VectorType type,
                        Value base, Value memrefOffset, Value mask,
                        Value passThru) {
  Location loc = loadOp->getLoc();
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  int64_t loadWidth = type.getShape()[0];
  VectorType swizzledLoadType =
      VectorType::get({accessWidth}, type.getElementType());

  // ~ vector.undef, overwritten by unrolling.
  Value replacement = arith::ConstantOp::create(rewriter, hintOp.getLoc(), type,
                                                rewriter.getZeroAttr(type));

  // Load type = vector<C>, k = accessWidth
  // i = 0 -> C += k is the offset into the vector of a contiguous group of
  // swizzled elements.
  for (int64_t i = 0; i < loadWidth; i += accessWidth) {
    Value newOffset =
        computeSwizzledChunkOffset(rewriter, hintOp, memrefOffset, i);
    Value subLoad;
    if (mask) {
      subLoad =
          vector::MaskedLoadOp::create(
              rewriter, loc, swizzledLoadType, base, ValueRange{newOffset},
              extractChunk(rewriter, loc, mask, i, accessWidth),
              extractChunk(rewriter, loc, passThru, i, accessWidth))
              .getResult();
    } else {
      subLoad = vector::LoadOp::create(rewriter, loc, swizzledLoadType, base,
                                       newOffset);
    }
    replacement = vector::InsertStridedSliceOp::create(
        rewriter, loc, subLoad, replacement, ArrayRef<int64_t>{i},
        ArrayRef<int64_t>{1});
  }
  rewriter.replaceOp(loadOp, replacement);
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
///
/// The masked variant (mask non-null) emits vector.maskedstore ops with the
/// mask sliced along the access width.
static void swizzleStore(RewriterBase &rewriter, Operation *storeOp,
                         IREE::Codegen::SwizzleHintOp hintOp, VectorType type,
                         Value base, Value memrefOffset, Value valueToStore,
                         Value mask) {
  Location loc = storeOp->getLoc();
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  int64_t storeWidth = type.getShape()[0];

  // Store type = vector<C>, k = accessWidth
  // i = 0 -> C += k is the offset into the vector of a contiguous group of
  // swizzled elements.
  for (int64_t i = 0; i < storeWidth; i += accessWidth) {
    Value subVec = extractChunk(rewriter, loc, valueToStore, i, accessWidth);
    Value newOffset =
        computeSwizzledChunkOffset(rewriter, hintOp, memrefOffset, i);
    if (mask) {
      vector::MaskedStoreOp::create(
          rewriter, loc, base, ValueRange{newOffset},
          extractChunk(rewriter, loc, mask, i, accessWidth), subVec);
    } else {
      vector::StoreOp::create(rewriter, loc, subVec, base, newOffset);
    }
  }
  rewriter.eraseOp(storeOp);
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

/// Returns true if `vt` is rank 1 and its width divides evenly by
/// `accessWidth` — required for the unrolled per-chunk rewrite.
static bool isSwizzleableVectorType(VectorType vt, int64_t accessWidth) {
  return vt.getRank() == 1 && vt.getShape()[0] % accessWidth == 0;
}

/// Resolves all hints. Walks all direct users and splits them into loads and
/// stores. If any user is not a swizzle-able load or store, bail out and
/// silently drop the optimization hint.
static void resolveHintOp(RewriterBase &rewriter,
                          IREE::Codegen::SwizzleHintOp hintOp) {
  SmallVector<Operation *> loads;
  SmallVector<Operation *> stores;
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  for (Operation *user : hintOp->getUsers()) {
    if (auto load = dyn_cast<vector::LoadOp>(user)) {
      if (!isSwizzleableVectorType(load.getVectorType(), accessWidth)) {
        return;
      }
      loads.push_back(load);
      continue;
    }
    if (auto store = dyn_cast<vector::StoreOp>(user)) {
      if (!isSwizzleableVectorType(store.getVectorType(), accessWidth)) {
        return;
      }
      stores.push_back(store);
      continue;
    }
    if (auto mLoad = dyn_cast<vector::MaskedLoadOp>(user)) {
      if (!isSwizzleableVectorType(mLoad.getVectorType(), accessWidth)) {
        return;
      }
      loads.push_back(mLoad);
      continue;
    }
    if (auto mStore = dyn_cast<vector::MaskedStoreOp>(user)) {
      if (!isSwizzleableVectorType(mStore.getVectorType(), accessWidth)) {
        return;
      }
      stores.push_back(mStore);
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

  for (Operation *load : loads) {
    rewriter.setInsertionPoint(load);
    if (auto m = dyn_cast<vector::MaskedLoadOp>(load)) {
      swizzleLoad(rewriter, m, hintOp, m.getVectorType(), m.getBase(),
                  m.getIndices()[0], m.getMask(), m.getPassThru());
    } else {
      auto l = cast<vector::LoadOp>(load);
      swizzleLoad(rewriter, l, hintOp, l.getVectorType(), l.getBase(),
                  l.getIndices()[0], /*mask=*/Value{}, /*passThru=*/Value{});
    }
  }
  for (Operation *store : stores) {
    rewriter.setInsertionPoint(store);
    if (auto m = dyn_cast<vector::MaskedStoreOp>(store)) {
      swizzleStore(rewriter, m, hintOp, m.getVectorType(), m.getBase(),
                   m.getIndices()[0], m.getValueToStore(), m.getMask());
    } else {
      auto s = cast<vector::StoreOp>(store);
      swizzleStore(rewriter, s, hintOp, s.getVectorType(), s.getBase(),
                   s.getIndices()[0], s.getValueToStore(), /*mask=*/Value{});
    }
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
