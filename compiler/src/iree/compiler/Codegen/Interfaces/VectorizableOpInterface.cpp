// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/VectorizableOpInterface.h"

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

// clang-format off
#include "iree/compiler/Codegen/Interfaces/VectorizableOpInterface.cpp.inc"
// clang-format on

namespace mlir::iree_compiler {

namespace {

/// Extracts a boolean option from a DictionaryAttr.
static bool getBoolOption(DictionaryAttr options, StringRef name,
                          bool defaultValue = false) {
  if (!options) {
    return defaultValue;
  }
  if (auto attr = options.getAs<BoolAttr>(name)) {
    return attr.getValue();
  }
  return defaultValue;
}

struct GatherOpVectorizationModel
    : public VectorizableOpInterface::ExternalModel<GatherOpVectorizationModel,
                                                    IREE::LinalgExt::GatherOp> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    auto gatherOp = cast<IREE::LinalgExt::GatherOp>(op);
    // TODO: Support indexDepth > 1 by splitting the innermost dim of
    // `indices` into `indexDepth` vectors so that each independent index can
    // be passed to the transfer_gather op.
    if (gatherOp.getIndexDepth() != 1) {
      return false;
    }
    return true;
  }

  FailureOr<SmallVector<Value>> vectorize(Operation *op, RewriterBase &rewriter,
                                          ArrayRef<int64_t> vectorSizes,
                                          ArrayRef<bool> scalableDims,
                                          DictionaryAttr options) const {
    auto gatherOp = cast<IREE::LinalgExt::GatherOp>(op);
    int64_t batchRank = gatherOp.getBatchRank();
    Location loc = gatherOp.getLoc();
    RewriterBase::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(gatherOp);

    ShapedType indicesTy = gatherOp.getIndicesType();
    ShapedType gatherTy = gatherOp.getOutputType();
    ShapedType sourceTy = gatherOp.getSourceType();

    if (vectorSizes.empty()) {
      vectorSizes = gatherTy.getShape();
    }

    auto gatherVectorTy =
        VectorType::get(vectorSizes, gatherTy.getElementType());
    // Rank-reduced to remove the innermost unit dim.
    auto indicesVecTy =
        VectorType::get(vectorSizes.take_front(gatherOp.getBatchRank()),
                        rewriter.getIndexType());

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    VectorType indicesMaskType = indicesVecTy.clone(rewriter.getI1Type());
    SmallVector<OpFoldResult> gatherDims =
        tensor::getMixedSizes(rewriter, loc, gatherOp.getOutput());
    Value indicesMask = vector::CreateMaskOp::create(
        rewriter, loc, indicesMaskType,
        ArrayRef(gatherDims).take_front(gatherOp.getBatchRank()));
    auto indicesVecRead = vector::TransferReadOp::create(
        rewriter, loc, indicesVecTy.clone(indicesTy.getElementType()),
        gatherOp.getIndices(), SmallVector<Value>(indicesTy.getRank(), zero),
        std::nullopt);
    rewriter.modifyOpInPlace(indicesVecRead, [&] {
      indicesVecRead.getMaskMutable().assign(indicesMask);
    });
    Value indicesVec = indicesVecRead.getResult();
    indicesVec =
        arith::IndexCastOp::create(rewriter, loc, indicesVecTy, indicesVec);

    SmallVector<Value> baseOffsets(sourceTy.getRank(), zero);
    Value padding =
        ub::PoisonOp::create(rewriter, loc, gatherTy.getElementType());

    // Build indexing_maps for the transfer_gather.
    // Source map: (vector_dims)[s0] -> (s0, d_batch+1, ..., d_N)
    // First source dim is gathered (s0), rest are contiguous.
    MLIRContext *ctx = rewriter.getContext();
    int64_t vectorRank = vectorSizes.size();
    int64_t sourceRank = sourceTy.getRank();
    SmallVector<AffineExpr> sourceMapExprs;
    sourceMapExprs.push_back(getAffineSymbolExpr(0, ctx)); // gathered dim 0
    for (int64_t i = 1; i < sourceRank; ++i) {
      // Map remaining source dims to corresponding vector dims.
      // The batch dims come first, so source dim i maps to vector dim
      // (i - 1 + batchRank).
      sourceMapExprs.push_back(getAffineDimExpr(i - 1 + batchRank, ctx));
    }
    AffineMap sourceMap =
        AffineMap::get(vectorRank, /*symbolCount=*/1, sourceMapExprs, ctx);

    // Index vec map: (vector_dims)[s0] -> (d0, ..., d_{batchRank-1})
    SmallVector<AffineExpr> indexVecMapExprs;
    for (int64_t i = 0; i < batchRank; ++i) {
      indexVecMapExprs.push_back(getAffineDimExpr(i, ctx));
    }
    AffineMap indexVecMap =
        AffineMap::get(vectorRank, /*symbolCount=*/1, indexVecMapExprs, ctx);

    SmallVector<AffineMap> indexingMaps = {sourceMap, indexVecMap};

    VectorType gatherMaskType = gatherVectorTy.clone(rewriter.getI1Type());
    Value gatherMask =
        vector::CreateMaskOp::create(rewriter, loc, gatherMaskType, gatherDims);

    // Add a mask indexing map (identity) to the indexing_maps.
    // TODO: symbolCount is hardcoded to 1 because indexDepth != 1 bails out
    // above. All indexing maps must share the same symbol count (= number of
    // index vecs). Update this when indexDepth > 1 is supported.
    AffineMap maskMap =
        AffineMap::getMultiDimIdentityMap(vectorRank, rewriter.getContext());
    maskMap = AffineMap::get(vectorRank, /*symbolCount=*/1,
                             maskMap.getResults(), rewriter.getContext());
    indexingMaps.push_back(maskMap);

    auto transferGatherOp = IREE::VectorExt::TransferGatherOp::create(
        rewriter, loc, gatherVectorTy, gatherOp.getSource(), baseOffsets,
        ValueRange{indicesVec}, rewriter.getAffineMapArrayAttr(indexingMaps),
        padding, /*mask=*/gatherMask);
    SmallVector<Value> writeIndices(gatherTy.getRank(), zero);
    auto writeOp = vector::TransferWriteOp::create(
        rewriter, loc, transferGatherOp.getResult(), gatherOp.getOutput(),
        writeIndices);
    rewriter.modifyOpInPlace(
        writeOp, [&] { writeOp.getMaskMutable().assign(gatherMask); });

    return SmallVector<Value>{writeOp.getResult()};
  }
};

} // namespace

void registerVectorizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::LinalgExt::IREELinalgExtDialect *dialect) {
        IREE::LinalgExt::GatherOp::attachInterface<GatherOpVectorizationModel>(
            *ctx);
      });
}

} // namespace mlir::iree_compiler
