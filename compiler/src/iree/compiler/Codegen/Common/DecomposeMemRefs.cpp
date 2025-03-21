//===- DecomposeMemRefs.cpp - Decompose memrefs pass implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements decompose memrefs pass. It is an altered version of the
// upstream GPU/DecomposeMemRefs.cpp file. It adds a new option to not decompose
// it into 0-rank memrefs but instead single-ranked memrefs.
//
// Question to answer at this point:
// 1. should we disallow memrefs with non-identity layout? also cases where
// offset != 0 and stride != 1? if so we should update test cases.

// TODO:
// 1. update memref.subview.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Passes.h"

namespace mlir {
#define GEN_PASS_DEF_DECOMPOSEMEMREFSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static MemRefType inferCastResultType(Value source, OpFoldResult offset) {
  auto sourceType = cast<BaseMemRefType>(source.getType());
  SmallVector<int64_t> staticOffsets;
  SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offset, dynamicOffsets, staticOffsets);
  auto stridedLayout =
      StridedLayoutAttr::get(source.getContext(), staticOffsets.front(), {});
  return MemRefType::get({}, sourceType.getElementType(), stridedLayout,
                         sourceType.getMemorySpace());
}

// First version, it has to be contiguous.
static MemRefType inferCastResultType2(Value source, OpFoldResult offset) {
  auto sourceType = cast<BaseMemRefType>(source.getType());
  SmallVector<int64_t> staticOffsets;
  SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offset, dynamicOffsets, staticOffsets);
  int64_t collapsedShape = 1;
  for (auto dim : sourceType.getShape()) {
    collapsedShape *= dim;
  }
  return MemRefType::get({collapsedShape}, sourceType.getElementType(), nullptr,
                         sourceType.getMemorySpace());
}

static void setInsertionPointToStart(OpBuilder &builder, Value val) {
  if (auto *parentOp = val.getDefiningOp()) {
    builder.setInsertionPointAfter(parentOp);
  } else {
    builder.setInsertionPointToStart(val.getParentBlock());
  }
}

static std::tuple<Value, OpFoldResult, SmallVector<OpFoldResult>>
getFlatOffsetAndStrides(OpBuilder &rewriter, Location loc, Value source,
                        ArrayRef<OpFoldResult> subOffsets,
                        ArrayRef<OpFoldResult> subStrides = std::nullopt) {
  auto sourceType = cast<MemRefType>(source.getType());
  auto sourceRank = static_cast<unsigned>(sourceType.getRank());

  memref::ExtractStridedMetadataOp newExtractStridedMetadata;
  {
    OpBuilder::InsertionGuard g(rewriter);
    setInsertionPointToStart(rewriter, source);
    newExtractStridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, source);
  }

  auto &&[sourceStrides, sourceOffset] = sourceType.getStridesAndOffset();

  auto getDim = [&](int64_t dim, Value dimVal) -> OpFoldResult {
    return ShapedType::isDynamic(dim) ? getAsOpFoldResult(dimVal)
                                      : rewriter.getIndexAttr(dim);
  };

  OpFoldResult origOffset =
      getDim(sourceOffset, newExtractStridedMetadata.getOffset());
  ValueRange sourceStridesVals = newExtractStridedMetadata.getStrides();

  SmallVector<OpFoldResult> origStrides;
  origStrides.reserve(sourceRank);

  SmallVector<OpFoldResult> strides;
  strides.reserve(sourceRank);

  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
  for (auto i : llvm::seq(0u, sourceRank)) {
    OpFoldResult origStride = getDim(sourceStrides[i], sourceStridesVals[i]);

    if (!subStrides.empty()) {
      strides.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, loc, s0 * s1, {subStrides[i], origStride}));
    }

    origStrides.emplace_back(origStride);
  }

  auto &&[expr, values] =
      computeLinearIndex(origOffset, origStrides, subOffsets);
  OpFoldResult finalOffset =
      affine::makeComposedFoldedAffineApply(rewriter, loc, expr, values);
  return {newExtractStridedMetadata.getBaseBuffer(), finalOffset, strides};
}

static Value getValueFromOpFoldResult(OpBuilder &rewriter, Location loc,
                                      OpFoldResult in) {
  if (Attribute offsetAttr = in.dyn_cast<Attribute>()) {
    return rewriter.create<arith::ConstantIndexOp>(
        loc, cast<IntegerAttr>(offsetAttr).getInt());
  }
  return in.dyn_cast<Value>();
}

/// Returns a collapsed memref and the linearized index to access the element
/// at the specified indices.
static std::pair<Value, Value> getFlattenMemrefAndOffset(OpBuilder &rewriter,
                                                         Location loc,
                                                         Value source,
                                                         ValueRange indices) {
  MemRefType memrefType = cast<MemRefType>(source.getType());
  auto &&[base, index, _] = getFlatOffsetAndStrides(rewriter, loc, source,
                                                    getAsOpFoldResult(indices));
  // We do not support non-contiguous memrefs.
  int64_t collapsedShape = 1;
  for (auto dim : memrefType.getShape()) {
    collapsedShape *= dim;
  }
  MemRefType retType =
      MemRefType::get({collapsedShape}, memrefType.getElementType(), nullptr,
                      memrefType.getMemorySpace());

  Value indexValue = getValueFromOpFoldResult(rewriter, loc, index);

  // (lialan) TODO: should we keep `offset` in the result memref?
  return std::make_pair(rewriter.create<memref::ReinterpretCastOp>(
                            loc, retType, source, /* offset = */ 0,
                            /*shapes = */ ArrayRef<int64_t>{collapsedShape},
                            /* strides = */ ArrayRef<int64_t>{1}),
                        indexValue);
}

static bool needFlatten(Value val) {
  auto type = cast<MemRefType>(val.getType());
  return type.getRank() > 1;
}

static bool checkLayout(Value val) {
  auto type = cast<MemRefType>(val.getType());
  return type.getLayout().isIdentity() ||
         isa<StridedLayoutAttr>(type.getLayout());
}

namespace {
struct FlattenMemrefLoad : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getMemref();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    auto &&[flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op.getLoc(), memref, op.getIndices());

    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getType(), flatMemref,
                                                ValueRange{offset}/*,
                                                op.getNontemporal()*/);
    return success();
  }
};

struct FlattenVectorLoad : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getBase();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    auto &&[flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op.getLoc(), memref, op.getIndices());

    rewriter.replaceOpWithNewOp<vector::LoadOp>(op, op.getType(), flatMemref,
                                                ValueRange{offset}/*,
                                                op.getNontemporal()*/);
    return success();
  }
};

struct FlattenMemrefStore : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getMemref();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    auto &&[flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op.getLoc(), memref, op.getIndices());
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, op.getValue(), flatMemref,
                                                 ValueRange{offset});
    return success();
  }
};

struct FlattenVectorStore : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getBase();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    auto &&[flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op.getLoc(), memref, op.getIndices());
    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        op, op.getValueToStore(), flatMemref, ValueRange{offset});
    return success();
  }
};

struct FlattenVectorMaskedLoad : public OpRewritePattern<vector::MaskedLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedLoadOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getBase();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    auto &&[flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op.getLoc(), memref, op.getIndices());
    rewriter.replaceOpWithNewOp<vector::MaskedLoadOp>(
        op, op.getType(), flatMemref, ValueRange{offset}, op.getMask(),
        op.getPassThru());
    return success();
  }
};

struct FlattenVectorMaskedStore
    : public OpRewritePattern<vector::MaskedStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedStoreOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getBase();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    auto &&[flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op.getLoc(), memref, op.getIndices());
    rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
        op, flatMemref, ValueRange{offset}, op.getMask(), op.getValueToStore());
    return success();
  }
};
struct FlattenVectorTransferRead
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getSource();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    auto &&[flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op.getLoc(), memref, op.getIndices());

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, op.getType(), flatMemref, ValueRange{offset}, op.getPadding());
    return success();
  }
};

struct FlattenVectorTransferWrite
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getSource();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    auto &&[flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op.getLoc(), memref, op.getIndices());

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        op, op.getVector(), flatMemref, ValueRange{offset});
    return success();
  }
};

struct FlattenSubview : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getSource();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    Location loc = op.getLoc();
    SmallVector<OpFoldResult> subOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> subSizes = op.getMixedSizes();
    SmallVector<OpFoldResult> subStrides = op.getMixedStrides();
    auto &&[base, finalOffset, strides] =
        getFlatOffsetAndStrides(rewriter, loc, memref, subOffsets, subStrides);

    auto srcType = cast<MemRefType>(memref.getType());
    auto resultType = cast<MemRefType>(op.getType());
    unsigned subRank = static_cast<unsigned>(resultType.getRank());

    llvm::SmallBitVector droppedDims = op.getDroppedDims();

    SmallVector<OpFoldResult> finalSizes;
    finalSizes.reserve(subRank);

    SmallVector<OpFoldResult> finalStrides;
    finalStrides.reserve(subRank);

    for (auto i : llvm::seq(0u, static_cast<unsigned>(srcType.getRank()))) {
      if (droppedDims.test(i))
        continue;

      finalSizes.push_back(subSizes[i]);
      finalStrides.push_back(strides[i]);
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resultType, base, finalOffset, finalSizes, finalStrides);
    return success();
  }
};

struct DecomposeMemrefsPass
    : public impl::DecomposeMemrefsPassBase<DecomposeMemrefsPass> {
  using impl::DecomposeMemrefsPassBase<
      DecomposeMemrefsPass>::DecomposeMemrefsPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    mlir::iree_compiler::populateDecomposeMemrefsPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir::iree_compiler {
void populateDecomposeMemrefsPatterns(RewritePatternSet &patterns) {
  patterns.insert<FlattenMemrefLoad, FlattenMemrefStore, FlattenSubview,
                  FlattenVectorMaskedLoad, FlattenVectorMaskedStore,
                  FlattenVectorLoad, FlattenVectorStore,
                  FlattenVectorTransferRead, FlattenVectorTransferWrite>(
      patterns.getContext());
}

std::unique_ptr<Pass> createDecomposeMemrefsPass() {
  return std::make_unique<DecomposeMemrefsPass>();
}
} // namespace mlir::iree_compiler
