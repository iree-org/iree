// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===-- GPUPushDownDMABoundsToConsumers.cpp -------------------------------===//
//
// For each iree_gpu.coalesced_gather_dma whose innermost in_bounds entry is
// false, inserts a tensor.extract_slice + tensor.pad chain between the DMA's
// outer scf.forall result and its consumer. Downstream vectorization
// (enable-vector-masking=true) then lowers the pad to a masked
// vector.transfer_read via vectorizeAsTensorPadOp, discarding straddle-
// corrupted columns and substituting zero.
//
// See docs/superpowers/specs/2026-04-29-dma-mask-pushdown-design.md.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPUSHDOWNDMABOUNDSTOCONSUMERSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

// Returns the static upper bound (in elements) for `src`'s dim `d`, if one
// can be derived from the ranked tensor type or the immediate defining
// `tensor.extract_slice`. For a dynamic dim whose size is a Value, looks
// through `arith.constant` and `affine.min` to extract a constant bound.
//
// Used by the pushdown skip predicates: when the inner extent's UB matches
// what the pass would otherwise pad against, both the validBytes wrap and
// the consumer-side pad become provably unnecessary (root alignment makes
// the wrap a no-op; AMDGPULowerCoalescedDMA's OOB clamping makes the pad a
// no-op). See padSourceBufferDescriptorToDWORD and rewriteOneDMA below.
static std::optional<int64_t> getInnermostStaticUpperBound(Value src,
                                                           unsigned dim) {
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  if (!srcTy) {
    return std::nullopt;
  }
  int64_t s = srcTy.getDimSize(dim);
  if (!ShapedType::isDynamic(s)) {
    return s;
  }

  auto sl = src.getDefiningOp<tensor::ExtractSliceOp>();
  if (!sl) {
    return std::nullopt;
  }
  OpFoldResult sz = sl.getMixedSizes()[dim];
  if (auto attr = dyn_cast<Attribute>(sz)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      return intAttr.getInt();
    }
    return std::nullopt;
  }
  Value szVal = cast<Value>(sz);
  if (auto cst = szVal.getDefiningOp<arith::ConstantIndexOp>()) {
    return cst.value();
  }
  if (auto am = szVal.getDefiningOp<affine::AffineMinOp>()) {
    std::optional<int64_t> ub;
    for (AffineExpr e : am.getMap().getResults()) {
      if (auto c = dyn_cast<AffineConstantExpr>(e)) {
        int64_t v = c.getValue();
        ub = ub ? std::min(*ub, v) : v;
      }
    }
    return ub;
  }
  return std::nullopt;
}

// Walk from the DMA's init operand (a forall sharedOut block argument) out
// through the chain of nested forall shared_outs and parallel_insert_slices
// until we reach an scf.forall whose result is read by a non-parallel_insert
// consumer. That result is the SSA value the matmul (or other consumer) sees.
//
// Two link types are followed:
//   1. BlockArgument (forall sharedOut) -> forall.getResult(idx)
//      (existing nested-forall handling)
//   2. forall result that is the source of a tensor.parallel_insert_slice
//      in a parent forall's terminator -> the parent forall's destination
//      sharedOut (a BlockArgument), which feeds back into case (1).
//
// The walk stops as soon as we hit a value whose only forward chain breaks
// (no enclosing forall, no propagating parallel_insert_slice, etc.).
static Value walkUpSharedOuts(Value v) {
  while (true) {
    if (auto bbarg = dyn_cast<BlockArgument>(v)) {
      auto forall = dyn_cast<scf::ForallOp>(bbarg.getOwner()->getParentOp());
      if (!forall) {
        return v;
      }
      unsigned argIdx = bbarg.getArgNumber();
      unsigned sharedOutsStart = forall.getRank();
      if (argIdx < sharedOutsStart) {
        return v;
      }
      v = forall.getResult(argIdx - sharedOutsStart);
      continue;
    }

    // v is an SSA value (typically an scf.forall result). If it's only
    // consumed by a parallel_insert_slice inside a parent forall, follow
    // that link to the parent forall's sharedOut BlockArg.
    auto definingForall = v.getDefiningOp<scf::ForallOp>();
    if (!definingForall) {
      return v;
    }
    auto parentForall = definingForall->getParentOfType<scf::ForallOp>();
    if (!parentForall) {
      return v;
    }

    Value nextSharedOut = nullptr;
    for (Operation &op : parentForall.getTerminator().getRegion().front()) {
      auto insert = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
      if (!insert) {
        continue;
      }
      if (insert.getSource() == v) {
        nextSharedOut = insert.getDest();
        break;
      }
    }
    if (!nextSharedOut) {
      return v;
    }
    v = nextSharedOut;
  }
}

// Wrap the DMA source's root tensor with a plain iree_gpu.buffer_resource_cast.
// Bufferization detects the misaligned innermost row from the bufferized
// memref's shape and emits the corresponding amdgpu.fat_raw_buffer_cast with
// a DWORD-rounded validBytes, keeping the trailing partial DWORD in-bounds.
// Garbage in the rounded tail lands in masked LDS columns and is discarded
// by the consumer-side tensor.pad inserted below.
//
// Returns failure when no rewrite is needed (already aligned, dynamic
// element bit width, etc.). Source is mutated in place.
static LogicalResult
padSourceBufferDescriptorToDWORD(IRRewriter &rewriter,
                                 IREE::GPU::CoalescedGatherDMAOp dma) {
  Value src = dma.getSource();
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  if (!srcTy) {
    return failure();
  }
  Type elemTy = srcTy.getElementType();
  if (!elemTy.isIntOrFloat()) {
    return failure();
  }
  unsigned elemBits = elemTy.getIntOrFloatBitWidth();
  if (elemBits == 0 || elemBits % 8 != 0) {
    return failure();
  }
  unsigned elemBytes = elemBits / 8;

  // If the innermost row is statically DWORD-aligned, the partial-DWORD
  // straddle issue cannot occur and we don't need to pad the descriptor.
  unsigned rank = srcTy.getRank();
  int64_t innermostStatic = srcTy.getDimSize(rank - 1);
  if (!ShapedType::isDynamic(innermostStatic) &&
      (innermostStatic * elemBytes) % 4 == 0) {
    return failure();
  }

  // The DMA source can be a per-K-block tensor.extract_slice of a larger
  // tensor (e.g. matmul tiled by a reduction dim). The buffer descriptor's
  // validBytes must reflect the *underlying* allocation, not the slice —
  // sizing it from the slice would clamp the descriptor below the size that
  // earlier accesses already need. Walk through extract_slices to the root.
  Value root = src;
  while (auto sl = root.getDefiningOp<tensor::ExtractSliceOp>()) {
    root = sl.getSource();
  }
  auto rootTy = dyn_cast<RankedTensorType>(root.getType());
  if (!rootTy) {
    return failure();
  }
  unsigned rootRank = rootTy.getRank();

  // Skip when the root's innermost row is statically DWORD-aligned — no
  // straddle is possible at the buffer end. Catches the "shape-dynamic but
  // root-aligned" case (e.g. K-block slice tensor<32x?xf16> of an even-N
  // root tensor<NxNxf16>).
  int64_t rootInnermost = rootTy.getDimSize(rootRank - 1);
  if (!ShapedType::isDynamic(rootInnermost) &&
      (rootInnermost * elemBytes) % 4 == 0) {
    return failure();
  }

  // Skip if the root is already wrapped — re-wrapping would be redundant.
  if (root.getDefiningOp<IREE::GPU::BufferResourceCastOp>()) {
    return failure();
  }

  // Place the cast in a scope that dominates the DMA but lives outside any
  // enclosing forall so the bufferized validBytes computation is hoisted out
  // of loops.
  if (auto *rootOp = root.getDefiningOp()) {
    rewriter.setInsertionPointAfter(rootOp);
  } else {
    // Block argument: insert at the start of its block (e.g. function entry).
    auto *block = cast<BlockArgument>(root).getOwner();
    rewriter.setInsertionPointToStart(block);
  }
  Location loc = dma.getLoc();

  auto castOp = IREE::GPU::BufferResourceCastOp::create(
      rewriter, loc, rootTy, root, /*cache_swizzle_stride=*/Value{});

  // Re-route uses of `root` to the cast.
  rewriter.replaceAllUsesExcept(root, castOp.getResult(), castOp);
  return success();
}

// Insert extract_slice + tensor.pad after the outermost forall for one DMA.
// Returns failure if the DMA doesn't match the expected pattern (no-op).
static LogicalResult rewriteOneDMA(IRRewriter &rewriter,
                                   IREE::GPU::CoalescedGatherDMAOp dma) {
  // Only act when innermost dimension is out-of-bounds.
  std::optional<ArrayAttr> inBoundsOpt = dma.getInBounds();
  if (!inBoundsOpt || inBoundsOpt->empty()) {
    return failure();
  }
  ArrayAttr inBounds = *inBoundsOpt;
  unsigned innermost = inBounds.size() - 1;
  if (cast<BoolAttr>(inBounds[innermost]).getValue()) {
    return failure(); // innermost is in-bounds; nothing to do
  }

  // Find the outermost forall result visible to consumers. The pad we insert
  // replaces this value, so the pad's static shape MUST match its type — not
  // the (possibly smaller) per-lane DMA init shape, which differs whenever
  // intermediate warp-level foralls split the workgroup tile.
  Value outerResult = walkUpSharedOuts(dma.getInit());
  auto outerForall =
      dyn_cast_or_null<scf::ForallOp>(outerResult.getDefiningOp());
  if (!outerForall) {
    return failure();
  }
  auto tileTy = dyn_cast<RankedTensorType>(outerResult.getType());
  if (!tileTy) {
    return failure();
  }
  unsigned rank = tileTy.getRank();
  int64_t innerTileSize = tileTy.getDimSize(innermost);
  if (ShapedType::isDynamic(innerTileSize)) {
    return failure(); // shouldn't occur; bail defensively
  }

  // Walk the DMA source up through extract_slices until we find one defined
  // outside outerForall. Two reasons:
  //   1. Dominance: dma.getSource() can live inside nested foralls; using its
  //      tensor.dim after outerForall would violate SSA dominance.
  //   2. Extent semantics: the inner-lane source carries a per-lane extent
  //      (e.g. min(K_remaining, 1)) — far smaller than the workgroup-K-block
  //      extent we need for the consumer pad.
  Value extentSrc = dma.getSource();
  while (auto sl = extentSrc.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (!outerForall->isAncestor(sl)) {
      break;
    }
    extentSrc = sl.getSource();
  }
  // Bail if the resulting value is still inside outerForall (shouldn't happen
  // for the supported pattern, but stay safe).
  if (auto *defOp = extentSrc.getDefiningOp()) {
    if (outerForall->isAncestor(defOp)) {
      return failure();
    }
  }

  // Skip when the source's innermost extent has a static upper bound equal
  // to the LDS inner tile size. Two cases:
  //   - Full-tile workgroups: the runtime extent equals innerTileSize, so
  //     the entire LDS tile is filled by valid data. The pad is a no-op.
  //   - Boundary workgroups (extent < UB): lanes assigned to OOB columns
  //     are clamped past the root buffer's end by
  //     AMDGPULowerCoalescedDMAToGatherLDS::applyOOBClamping; the HW
  //     fat_raw_buffer descriptor returns 0 for those reads, which the DMA
  //     writes to the LDS destination through the same swizzle the matmul
  //     reads through. The matmul then sees zeros in OOB columns —
  //     identical to what the consumer pad would have written. K-reduction
  //     contributions are zero (no pollution); N-OOB outputs are discarded
  //     by the downstream output extract_slice.
  // Removing the pad here avoids the masked-vector → tensor.empty alloca
  // round-trip emitted by vectorizeAsTensorPadOp.
  if (auto ub = getInnermostStaticUpperBound(extentSrc, innermost)) {
    if (*ub == innerTileSize) {
      return failure();
    }
  }

  // Insert after the outermost forall.
  rewriter.setInsertionPointAfter(outerForall);
  Location loc = dma.getLoc();

  // tensor.dim of the workgroup-tile source slice for the innermost dim.
  Value innerExtent =
      tensor::DimOp::create(rewriter, loc, extentSrc, innermost);
  Value innerTileV =
      arith::ConstantIndexOp::create(rewriter, loc, innerTileSize);
  Value padAmount =
      arith::SubIOp::create(rewriter, loc, innerTileV, innerExtent);

  // tensor.extract_slice: cut the valid columns out of the LDS tile.
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> validSizes;
  for (unsigned d = 0; d < rank; ++d) {
    if (d == innermost) {
      validSizes.push_back(OpFoldResult(innerExtent));
    } else {
      validSizes.push_back(rewriter.getIndexAttr(tileTy.getDimSize(d)));
    }
  }
  Value valid = tensor::ExtractSliceOp::create(rewriter, loc, outerResult,
                                               offsets, validSizes, strides);

  // tensor.pad: re-expand to the full tile shape with zero padding.
  SmallVector<OpFoldResult> lowPad(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> highPad(rank, rewriter.getIndexAttr(0));
  highPad[innermost] = OpFoldResult(padAmount);

  TypedAttr zeroAttr = rewriter.getZeroAttr(tileTy.getElementType());
  if (!zeroAttr) {
    return failure();
  }
  Value padCst = arith::ConstantOp::create(rewriter, loc, zeroAttr);

  auto padOp = tensor::PadOp::create(rewriter, loc, tileTy, valid, lowPad,
                                     highPad, padCst, /*nofold=*/false);

  // Replace all uses of the forall result with the re-padded tensor, except
  // for the extract_slice we just created which must read the original.
  outerResult.replaceAllUsesExcept(
      padOp.getResult(), valid.getDefiningOp<tensor::ExtractSliceOp>());
  return success();
}

struct GPUPushDownDMABoundsToConsumersPass final
    : impl::GPUPushDownDMABoundsToConsumersPassBase<
          GPUPushDownDMABoundsToConsumersPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    SmallVector<IREE::GPU::CoalescedGatherDMAOp> dmas;
    funcOp.walk(
        [&](IREE::GPU::CoalescedGatherDMAOp dma) { dmas.push_back(dma); });
    if (dmas.empty()) {
      return;
    }

    for (auto dma : dmas) {
      // Best-effort buffer-descriptor padding for non-DWORD-aligned sources.
      // Independent of the consumer-side pad rewrite below.
      (void)padSourceBufferDescriptorToDWORD(rewriter, dma);
      (void)rewriteOneDMA(rewriter, dma);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
