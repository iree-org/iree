// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/Transforms/MapPatterns.h"

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"
#include "iree/compiler/Codegen/Dialect/Map/Transforms/MapDistributionUtils.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/DistributionPatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

using IREE::Map::PackLayoutAttr;
using IREE::VectorExt::DistributionSignature;
using IREE::VectorExt::OpDistributionPattern;
using IREE::VectorExt::VectorLayoutInterface;
using VectorValue = TypedValue<VectorType>;

namespace {

//===----------------------------------------------------------------------===//
// ShapeCast
//===----------------------------------------------------------------------===//

struct MapDistributeShapeCast final
    : OpDistributionPattern<vector::ShapeCastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCast,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto src = cast<VectorValue>(shapeCast.getSource());
    auto dst = cast<VectorValue>(shapeCast.getResult());
    auto srcLayout = dyn_cast<PackLayoutAttr>(signature[src]);
    auto dstLayout = dyn_cast<PackLayoutAttr>(signature[dst]);
    if (!srcLayout || !dstLayout) {
      return rewriter.notifyMatchFailure(shapeCast, "not a PackLayout");
    }

    VectorValue disSrc = getDistributed(rewriter, src, signature[src]);
    SmallVector<int64_t> dstDisShape =
        cast<VectorLayoutInterface>(dstLayout).getDistributedShape();
    auto dstDisTy =
        VectorType::get(dstDisShape, src.getType().getElementType());
    Value result = vector::ShapeCastOp::create(rewriter, shapeCast.getLoc(),
                                               dstDisTy, disSrc);
    replaceOpWithDistributedValues(rewriter, shapeCast, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Broadcast
//===----------------------------------------------------------------------===//

struct MapDistributeBroadcast final
    : OpDistributionPattern<vector::BroadcastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto dst = cast<VectorValue>(broadcastOp.getVector());
    auto dstLayout = dyn_cast<PackLayoutAttr>(signature[dst]);
    if (!dstLayout) {
      return rewriter.notifyMatchFailure(broadcastOp, "not a PackLayout");
    }

    SmallVector<int64_t> dstDisShape =
        cast<VectorLayoutInterface>(dstLayout).getDistributedShape();
    auto dstDisTy =
        VectorType::get(dstDisShape, dst.getType().getElementType());

    // Distribute the source if it's a vector, otherwise use as-is (scalar).
    Value src = broadcastOp.getSource();
    if (auto srcVector = dyn_cast<VectorValue>(src)) {
      auto srcLayout = dyn_cast<PackLayoutAttr>(signature[srcVector]);
      if (!srcLayout) {
        return rewriter.notifyMatchFailure(broadcastOp,
                                           "source has no PackLayout");
      }
      src = getDistributed(rewriter, srcVector, signature[srcVector]);
    }

    Value result = vector::BroadcastOp::create(rewriter, broadcastOp.getLoc(),
                                               dstDisTy, src);
    replaceOpWithDistributedValues(rewriter, broadcastOp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Transpose
//===----------------------------------------------------------------------===//

struct MapDistributeTranspose final
    : OpDistributionPattern<vector::TransposeOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto src = cast<VectorValue>(transposeOp.getVector());
    auto srcLayout = dyn_cast<PackLayoutAttr>(signature[src]);
    if (!srcLayout) {
      return rewriter.notifyMatchFailure(transposeOp, "not a PackLayout");
    }

    VectorValue disSrc = getDistributed(rewriter, src, signature[src]);

    // Build expanded permutation over distributed dims.
    // Each original dim may expand to multiple distributed dims (value leaves).
    SmallVector<LeafDimInfo> leafMap = getLeafDimMap(srcLayout);
    int64_t rank = srcLayout.getRank();
    SmallVector<SmallVector<int64_t>> dimGroups(rank);
    for (auto [distIdx, info] : llvm::enumerate(leafMap)) {
      dimGroups[info.origDim].push_back(distIdx);
    }

    SmallVector<int64_t> expandedPerm;
    for (int64_t origDim : transposeOp.getPermutation()) {
      for (int64_t distIdx : dimGroups[origDim]) {
        expandedPerm.push_back(distIdx);
      }
    }

    Value result = vector::TransposeOp::create(rewriter, transposeOp.getLoc(),
                                               disSrc, expandedPerm);
    replaceOpWithDistributedValues(rewriter, transposeOp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Step
//===----------------------------------------------------------------------===//

struct MapDistributeStep final : OpDistributionPattern<vector::StepOp> {
  MapDistributeStep(MLIRContext *ctx, Value threadId)
      : OpDistributionPattern(ctx), threadId(threadId) {}

  LogicalResult matchAndRewrite(vector::StepOp stepOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue result = stepOp.getResult();
    auto layout = dyn_cast<PackLayoutAttr>(signature[result]);
    if (!layout) {
      return rewriter.notifyMatchFailure(stepOp, "not a PackLayout");
    }

    Location loc = stepOp.getLoc();
    IndexType indexType = rewriter.getIndexType();
    SmallVector<int64_t> distShape =
        cast<VectorLayoutInterface>(layout).getDistributedShape();
    auto distType = VectorType::get(distShape, indexType);
    SmallVector<LeafDimInfo> leafMap = getLeafDimMap(layout);

    // Value contribution: precompute all offsets as a constant.
    SmallVector<int64_t> ones(distShape.size(), 1);
    SmallVector<APInt> valOffsets;
    for (SmallVector<int64_t> idx : StaticTileOffsetRange(distShape, ones)) {
      int64_t offset = 0;
      for (auto [i, info] : llvm::enumerate(leafMap)) {
        offset += idx[i] * info.dataStride;
      }
      valOffsets.push_back(APInt(64, offset));
    }
    Value val = arith::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(distType, valOffsets));

    // Thread contribution: broadcast the per-dim thread offset and add.
    // If the offset is zero, canonicalization will fold the add away.
    SmallVector<Value> threadOffsets =
        buildThreadOffsets(rewriter, loc, layout, threadId);
    Value threadBcast =
        vector::BroadcastOp::create(rewriter, loc, distType, threadOffsets[0]);
    val = arith::AddIOp::create(rewriter, loc, val, threadBcast);

    replaceOpWithDistributedValues(rewriter, stepOp, val);
    return success();
  }

  Value threadId;
};

} // namespace

void populateMapDistributeGenericPatterns(RewritePatternSet &patterns,
                                          Value threadId) {
  patterns.add<MapDistributeShapeCast, MapDistributeBroadcast,
               MapDistributeTranspose>(patterns.getContext());
  patterns.add<MapDistributeStep>(patterns.getContext(), threadId);
}

} // namespace mlir::iree_compiler
