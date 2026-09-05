// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/Transforms/MapPatterns.h"

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IntTuple.h"
#include "iree/compiler/Codegen/Dialect/Map/Transforms/MapDistributionUtils.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/DistributionPatterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::iree_compiler {

using IREE::Map::getSize;
using IREE::Map::PackLayoutAttr;
using IREE::VectorExt::DistributionSignature;
using IREE::VectorExt::OpDistributionPattern;
using IREE::VectorExt::TransferGatherOp;
using IREE::VectorExt::VectorLayoutInterface;
using VectorValue = TypedValue<VectorType>;

namespace {

/// Expand an AffineMap's dim references from the original vector rank to the
/// distributed rank. A single original dim may become multiple distributed
/// dims when a layout mode has multiple value leaves.
///
/// Example: origToDistDims = [[0, 1], [2]] (orig dim 0 → dist dims 0,1)
///   (d0, d1) -> (d0, d1)  becomes  (d0, d1, d2) -> (d0, d1, d2)
///   (d0, d1) -> (d1)      becomes  (d0, d1, d2) -> (d2)
AffineMap expandDimsInMap(AffineMap map,
                          ArrayRef<SmallVector<int64_t>> origToDistDims,
                          int64_t distRank, int64_t numSymbols) {
  MLIRContext *ctx = map.getContext();
  SmallVector<AffineExpr> newResults;
  for (AffineExpr expr : map.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      for (int64_t distDim : origToDistDims[dimExpr.getPosition()]) {
        newResults.push_back(getAffineDimExpr(distDim, ctx));
      }
    } else {
      assert(isa<AffineConstantExpr>(expr) && "expected dim or constant expr");
      newResults.push_back(expr);
    }
  }
  return AffineMap::get(distRank, numSymbols, newResults, ctx);
}

//===----------------------------------------------------------------------===//
// TransferGather
//===----------------------------------------------------------------------===//

/// Distributes a transfer_gather under PackLayoutAttr by converting all
/// source dimensions into gathered (symbol) accesses and distributing the index
/// vectors:
///
///   - Contiguous dims (AffineDimExpr): replaced by a distributed vector.step
///     which is marked for redistribution.
///   - Gathered dims (AffineSymbolExpr): the existing index vec is distributed
///     via getDistributed.
///   - Broadcast dims (AffineConstantExpr): passed through unchanged.
///
/// The indexing maps are expanded from the original vector rank to the
/// distributed rank (one original dim may become multiple distributed dims
/// when a layout mode has multiple value leaves).
///
/// Example: layout ((4, 2), (4, 8)) : ((1, 0), (0, 4))
///   Distributed shape: [2, 4]
///
///   // Before:
///   transfer_gather %mem[%off0, %off1], %pad
///     {indexing_maps = [(d0, d1) -> (d0, d1)]}
///     : memref<8x32xf16>, vector<8x32xf16>
///
///   // After (pre-canonicalization):
///   %s0 = vector.step : vector<8xindex>   // redistributed → [tid_off + 0, 1]
///   %s1 = vector.step : vector<32xindex>  // redistributed → [tid_off + 0, 8,
///   16, 24] transfer_gather %mem[%off0, %off1] [%s0, %s1], %pad
///     {indexing_maps = [(d0, d1)[s0, s1] -> (s0, s1),
///                       (d0, d1)[s0, s1] -> (d0),
///                       (d0, d1)[s0, s1] -> (d1)]}
///     : memref<8x32xf16>, vector<2x4xf16>
struct MapDistributeTransferGather final
    : OpDistributionPattern<TransferGatherOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(TransferGatherOp gatherOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue result = gatherOp.getVector();
    auto layout = dyn_cast<PackLayoutAttr>(signature[result]);
    if (!layout) {
      return rewriter.notifyMatchFailure(gatherOp, "not a PackLayout");
    }
    if (!isa<MemRefType>(gatherOp.getBase().getType())) {
      return rewriter.notifyMatchFailure(gatherOp, "expects memrefs");
    }

    Location loc = gatherOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    int64_t origRank = layout.getRank();
    SmallVector<int64_t> distShape =
        cast<VectorLayoutInterface>(layout).getDistributedShape();
    int64_t distRank = distShape.size();

    // origToDistDims[i] = list of distributed dim indices for original dim i.
    SmallVector<LeafDimInfo> leafMap = getLeafDimMap(layout);
    SmallVector<SmallVector<int64_t>> origToDistDims(origRank);
    for (auto [distIdx, info] : llvm::enumerate(leafMap)) {
      origToDistDims[info.origDim].push_back(distIdx);
    }

    SmallVector<AffineMap> origMaps = gatherOp.getIndexingMapsArray();
    AffineMap origSourceMap = origMaps[0];
    OperandRange origIndexVecs = gatherOp.getIndexVecs();
    Value mask = gatherOp.getMask();

    // Walk source map results and convert each to a gathered symbol.
    SmallVector<AffineExpr> newSourceResults;
    SmallVector<Value> newIndexVecs;
    SmallVector<SmallVector<AffineExpr>> newIndexVecMapResults;
    int64_t nextSymbol = 0;
    SmallVector<int64_t> origDimToSymbol(origRank, -1);

    for (auto [srcDim, expr] : llvm::enumerate(origSourceMap.getResults())) {
      // Broadcast — pass through.
      if (!isa<AffineDimExpr, AffineSymbolExpr>(expr)) {
        newSourceResults.push_back(expr);
        continue;
      }

      // Already-gathered dim — distribute the existing index vec.
      if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
        int64_t origSym = symExpr.getPosition();
        int64_t sym = nextSymbol++;
        newSourceResults.push_back(getAffineSymbolExpr(sym, ctx));

        Value origIdxVec = origIndexVecs[origSym];
        if (auto vecVal = dyn_cast<VectorValue>(origIdxVec)) {
          if (auto idxLayout = signature[vecVal]) {
            newIndexVecs.push_back(getDistributed(rewriter, vecVal, idxLayout));
          } else {
            newIndexVecs.push_back(origIdxVec);
          }
        } else {
          newIndexVecs.push_back(origIdxVec);
        }

        AffineMap expanded =
            expandDimsInMap(origMaps[1 + origSym], origToDistDims, distRank,
                            /*numSymbols=*/0);
        newIndexVecMapResults.push_back(llvm::to_vector(expanded.getResults()));
        continue;
      }

      // Contiguous dim — convert to gathered via a distributed step.
      int64_t origDim = cast<AffineDimExpr>(expr).getPosition();

      if (origDimToSymbol[origDim] >= 0) {
        newSourceResults.push_back(
            getAffineSymbolExpr(origDimToSymbol[origDim], ctx));
        continue;
      }

      int64_t sym = nextSymbol++;
      origDimToSymbol[origDim] = sym;
      newSourceResults.push_back(getAffineSymbolExpr(sym, ctx));

      // Create a step [0..dimSize) and mark it for redistribution with the
      // projected layout for this dim. MapDistributeStep will resolve it
      // into thread_offset + value_offsets.
      int64_t dimSize = getSize(layout.getShapeMode(origDim));
      auto stepType = VectorType::get({dimSize}, rewriter.getIndexType());
      auto stepOp = vector::StepOp::create(rewriter, loc, stepType);

      SmallVector<bool> droppedDims(origRank, true);
      droppedDims[origDim] = false;
      PackLayoutAttr projLayout = layout.project(droppedDims);
      setSignatureForRedistribution(
          rewriter, stepOp, /*inputLayouts=*/{},
          /*outputLayouts=*/{cast<VectorLayoutInterface>(projLayout)});
      newIndexVecs.push_back(getDistributed(rewriter, stepOp.getResult(),
                                            VectorLayoutInterface(projLayout)));

      SmallVector<AffineExpr> mapResults;
      for (int64_t distDim : origToDistDims[origDim]) {
        mapResults.push_back(getAffineDimExpr(distDim, ctx));
      }
      newIndexVecMapResults.push_back(std::move(mapResults));
    }

    int64_t totalSymbols = nextSymbol;

    SmallVector<Attribute> allMapAttrs;
    allMapAttrs.push_back(AffineMapAttr::get(
        AffineMap::get(distRank, totalSymbols, newSourceResults, ctx)));
    for (auto &mapResults : newIndexVecMapResults) {
      allMapAttrs.push_back(AffineMapAttr::get(
          AffineMap::get(distRank, totalSymbols, mapResults, ctx)));
    }

    Value newMask;
    if (mask) {
      auto maskVec = cast<VectorValue>(mask);
      if (auto maskLayout = signature[maskVec]) {
        newMask = getDistributed(rewriter, maskVec, maskLayout);
      } else {
        newMask = mask;
      }
      allMapAttrs.push_back(AffineMapAttr::get(expandDimsInMap(
          origMaps.back(), origToDistDims, distRank, totalSymbols)));
    }

    Type elemType = result.getType().getElementType();
    auto distVecType = VectorType::get(distShape, elemType);
    auto newGather = TransferGatherOp::create(
        rewriter, loc, distVecType, gatherOp.getBase(), gatherOp.getOffsets(),
        newIndexVecs, rewriter.getArrayAttr(allMapAttrs), gatherOp.getPadding(),
        newMask);

    replaceOpWithDistributedValues(rewriter, gatherOp, newGather.getVector());
    return success();
  }
};

} // namespace

void populateMapDistributeMemoryPatterns(RewritePatternSet &patterns,
                                         Value threadId) {
  patterns.add<MapDistributeTransferGather>(patterns.getContext());
}

} // namespace mlir::iree_compiler
