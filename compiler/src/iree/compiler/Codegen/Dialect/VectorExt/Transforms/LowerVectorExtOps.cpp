// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::VectorExt {

namespace {

/// Given a vector transfer op, calculate which dimension of the `source`
/// memref should be unpacked in the next application of TransferOpConversion.
/// A return value of std::nullopt indicates a broadcast.
static std::optional<int64_t> unpackedDim(VectorTransferOpInterface xferOp) {
  // TODO: support 0-d corner case.
  assert(xferOp.getTransferRank() > 0 && "unexpected 0-d transfer");
  auto map = xferOp.getPermutationMap();
  if (auto expr = dyn_cast<AffineDimExpr>(map.getResult(0))) {
    return expr.getPosition();
  }
  assert(xferOp.isBroadcastDim(0) &&
         "Expected AffineDimExpr or AffineConstantExpr");
  return std::nullopt;
}

/// Calculate the indices for the new vector transfer gather op.
static void getXferIndices(OpBuilder &b, TransferGatherOp xferOp, Value iv,
                           SmallVectorImpl<Value> &indices) {
  TransferGatherOp::Adaptor adaptor(xferOp);
  // Corresponding memref dim of the vector dim that is unpacked.
  auto dim = unpackedDim(xferOp);
  auto prevIndices = adaptor.getIndices();
  indices.append(prevIndices.begin(), prevIndices.end());

  auto indexed =
      llvm::to_vector(xferOp.getIndexed().getAsValueRange<BoolAttr>());

  // Iterate on the dimension if the value is not broadcasted or indexed.
  Location loc = xferOp.getLoc();
  bool isBroadcast = !dim.has_value();
  if (!isBroadcast && !indexed[dim.value()]) {
    AffineExpr d0, d1;
    bindDims(xferOp.getContext(), d0, d1);
    Value offset = adaptor.getIndices()[*dim];
    indices[*dim] =
        affine::makeComposedAffineApply(b, loc, d0 + d1, {offset, iv});
  }
}

static Value extractAtDim(OpBuilder &b, Value vec, Value iv, int64_t dim) {
  auto vecTy = cast<VectorType>(vec.getType());
  // Transpose to make this dimension the first dimension.
  auto perm = llvm::to_vector(llvm::seq<int64_t>(vecTy.getRank()));
  std::swap(perm[0], perm[dim]);
  Value transposed = b.create<vector::TransposeOp>(vec.getLoc(), vec, perm);
  // Extract at iv.
  Value extracted = b.create<vector::ExtractOp>(vec.getLoc(), transposed, iv);
  // Because we still haven't made up our mind on 0-D vectors.
  if (!isa<VectorType>(extracted.getType())) {
    extracted = b.create<vector::BroadcastOp>(
        vec.getLoc(), VectorType::get({}, extracted.getType()), extracted);
  }
  return extracted;
}

static void getXferIndexVecsAndMaps(OpBuilder &b, TransferGatherOp xferOp,
                                    Value iv, SmallVectorImpl<Value> &indexVecs,
                                    SmallVectorImpl<AffineMap> &indexedMaps) {
  // Corresponding memref dim of the vector dim that is unpacked.
  std::optional<int64_t> dim = unpackedDim(xferOp);
  SmallVector<Value> prevIndexVecs = xferOp.getIndexVecs();
  SmallVector<AffineMap> prevIndexedMaps = xferOp.getIndexedMapsArray();
  indexVecs.append(prevIndexVecs.begin(), prevIndexVecs.end());
  indexedMaps.append(prevIndexedMaps.begin(), prevIndexedMaps.end());

  // Nothing to do for broadcasted dimension unrolling.
  if (!dim.has_value()) {
    return;
  }

  for (auto [vec, map] : llvm::zip_equal(indexVecs, indexedMaps)) {
    if (!map.isFunctionOfDim(dim.value())) {
      continue;
    }

    int64_t resultDim =
        map.getResultPosition(b.getAffineDimExpr(dim.value())).value();
    vec = extractAtDim(b, vec, iv, resultDim);
    map = map.dropResult(resultDim);
  }
}

static void maybeYieldValue(OpBuilder &b, Location loc, bool hasRetVal,
                            Value value) {
  if (hasRetVal) {
    assert(value && "Expected non-empty value");
    b.create<scf::YieldOp>(loc, value);
  } else {
    b.create<scf::YieldOp>(loc);
  }
}

/// Generates a boolean Value that is true if the iv-th bit in xferOp's mask
/// is set to true. No such check is generated under following circumstances:
/// * xferOp does not have a mask.
/// * xferOp's mask is not 1D. (In case of (N>1)-D, a subvector of the mask is
///   computed and attached to the new transfer op in the pattern.)
/// * The to-be-unpacked dim of xferOp is a broadcast.
template <typename OpTy>
static Value generateMaskCheck(OpBuilder &b, OpTy xferOp, Value iv) {
  if (!xferOp.getMask())
    return Value();
  if (xferOp.getMaskType().getRank() != 1)
    return Value();
  if (xferOp.isBroadcastDim(0))
    return Value();

  Location loc = xferOp.getLoc();
  return b.create<vector::ExtractOp>(loc, xferOp.getMask(), iv);
}

/// Given an ArrayAttr, return a copy where the first element is dropped.
static ArrayAttr dropFirstElem(OpBuilder &b, ArrayAttr attr) {
  if (!attr)
    return attr;
  return ArrayAttr::get(b.getContext(), attr.getValue().drop_front());
}

/// If the original transfer op has a mask, compute the mask of the new transfer
/// op (for the current iteration `i`) and assign it.
template <typename OpTy>
static void maybeAssignMask(OpBuilder &b, OpTy xferOp, OpTy newXferOp,
                            int64_t i) {
  if (!xferOp.getMask())
    return;

  if (xferOp.isBroadcastDim(0)) {
    // To-be-unpacked dimension is a broadcast, which does not have a
    // corresponding mask dimension. Mask attribute remains unchanged.
    newXferOp.getMaskMutable().assign(xferOp.getMask());
    return;
  }

  if (xferOp.getMaskType().getRank() > 1) {
    // Unpack one dimension of the mask.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(newXferOp); // Insert load before newXfer.

    llvm::SmallVector<int64_t, 1> indices({i});
    Location loc = xferOp.getLoc();
    auto newMask = b.create<vector::ExtractOp>(loc, xferOp.getMask(), indices);
    newXferOp.getMaskMutable().assign(newMask);
  }

  // If we end up here: The mask of the old transfer op is 1D and the unpacked
  // dim is not a broadcast, so no mask is needed on the new transfer op.
  // `generateInBoundsCheck` will have evaluated the mask already.
}

/// Compute the permutation map for the new (N-1)-D vector transfer op. This
/// map is identical to the current permutation map, but the first result is
/// omitted.
template <typename OpTy>
static AffineMap unpackedPermutationMap(OpBuilder &b, OpTy xferOp) {
  // TODO: support 0-d corner case.
  assert(xferOp.getTransferRank() > 0 && "unexpected 0-d transfer");
  auto map = xferOp.getPermutationMap();
  return AffineMap::get(map.getNumDims(), 0, map.getResults().drop_front(),
                        b.getContext());
}

struct UnrollTransferGather : public OpRewritePattern<TransferGatherOp> {

  UnrollTransferGather(MLIRContext *context, int64_t maxUnrollRank,
                       PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), maxUnrollRank(maxUnrollRank) {}

  /// Get or build the vector into which the newly created TransferReadOp
  /// results are inserted.
  Value buildResultVector(PatternRewriter &rewriter,
                          TransferGatherOp xferOp) const {
    if (auto insertOp = getInsertOp(xferOp))
      return insertOp.getDest();
    Location loc = xferOp.getLoc();
    return rewriter.create<vector::SplatOp>(loc, xferOp.getVectorType(),
                                            xferOp.getPadding());
  }

  /// If the result of the TransferReadOp has exactly one user, which is a
  /// vector::InsertOp, return that operation.
  vector::InsertOp getInsertOp(TransferGatherOp xferOp) const {
    if (xferOp->hasOneUse()) {
      Operation *xferOpUser = *xferOp->getUsers().begin();
      if (auto insertOp = dyn_cast<vector::InsertOp>(xferOpUser))
        return insertOp;
    }

    return vector::InsertOp();
  }

  /// If the result of the TransferReadOp has exactly one user, which is a
  /// vector::InsertOp, return that operation's indices.
  void getInsertionIndices(TransferGatherOp xferOp,
                           SmallVectorImpl<OpFoldResult> &indices) const {
    if (auto insertOp = getInsertOp(xferOp)) {
      auto pos = insertOp.getMixedPosition();
      indices.append(pos.begin(), pos.end());
    }
  }

  /// Rewrite the op: Unpack one dimension. Can handle masks, out-of-bounds
  /// accesses, and broadcasts and transposes in permutation maps.
  LogicalResult matchAndRewrite(TransferGatherOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (xferOp.getTransferRank() <= maxUnrollRank) {
      return rewriter.notifyMatchFailure(xferOp, "no dim to unroll");
    }
    if (!llvm::all_of(xferOp.getInBoundsValues(), [](bool x) { return x; })) {
      return rewriter.notifyMatchFailure(
          xferOp, "in-bounds is not supported in transfer_gather unrolling.");
    }
    // Only support memref unrolling, unrolling on tensors can cause problems
    // with bufferization.
    if (isa<RankedTensorType>(xferOp.getSource().getType())) {
      return rewriter.notifyMatchFailure(
          xferOp, "unrolling for tensor types is disabled.");
    }
    auto xferVecType = xferOp.getVectorType();
    if (xferVecType.getScalableDims()[0]) {
      return rewriter.notifyMatchFailure(
          xferOp, "scalable dimensions cannot be unrolled at compile time");
    }

    auto insertOp = getInsertOp(xferOp);
    auto vec = buildResultVector(rewriter, xferOp);

    VectorType newXferVecType = VectorType::Builder(xferVecType).dropDim(0);

    int64_t dimSize = xferVecType.getShape()[0];

    // Generate fully unrolled loop of transfer ops.
    Location loc = xferOp.getLoc();
    for (int64_t i = 0; i < dimSize; ++i) {
      Value iv = rewriter.create<arith::ConstantIndexOp>(loc, i);

      auto inBoundsCase = [&](OpBuilder &b, Location loc) -> Value {
        // Indices for the new transfer op.
        SmallVector<Value, 8> xferIndices;
        getXferIndices(b, xferOp, iv, xferIndices);

        // Index vecs and indexed maps for the new transfer op.
        SmallVector<Value> indexVecs;
        SmallVector<AffineMap> indexedMaps;
        getXferIndexVecsAndMaps(b, xferOp, iv, indexVecs, indexedMaps);

        // Indices for the new vector.insert op.
        SmallVector<OpFoldResult, 8> insertionIndices;
        getInsertionIndices(xferOp, insertionIndices);
        insertionIndices.push_back(b.getIndexAttr(i));

        auto inBoundsAttr = dropFirstElem(b, xferOp.getInBoundsAttr());
        auto newXferOp = b.create<TransferGatherOp>(
            loc, newXferVecType, xferOp.getSource(), xferIndices, indexVecs,
            xferOp.getIndexed(), rewriter.getAffineMapArrayAttr(indexedMaps),
            AffineMapAttr::get(unpackedPermutationMap(b, xferOp)),
            xferOp.getPadding(), Value(), inBoundsAttr);
        maybeAssignMask(b, xferOp, newXferOp, i);
        return b.create<vector::InsertOp>(loc, newXferOp, vec,
                                          insertionIndices);
      };

      // For some reason, transfer_ops allow 0-D vectors, but not 0-D masks. Any
      // attempt to change this will just result in another discussion on 0-D
      // vectors. So, generate a mask condition check if the mask is 1-D.
      if (Value maskCond = generateMaskCheck(rewriter, xferOp, iv)) {
        auto check = rewriter.create<scf::IfOp>(
            loc, maskCond,
            /*thenBuilder=*/
            [&](OpBuilder &b, Location loc) {
              Value val = inBoundsCase(b, loc);
              b.create<scf::YieldOp>(loc, val);
            },
            /*elseBuilder=*/
            [&](OpBuilder &b, Location loc) { b.create<scf::YieldOp>(loc); });
        vec = check.getResult(0);
      } else {
        vec = inBoundsCase(rewriter, loc);
      }
    }

    if (insertOp) {
      // Rewrite single user of the old TransferReadOp, which was an InsertOp.
      rewriter.replaceOp(insertOp, vec);
      rewriter.eraseOp(xferOp);
    } else {
      rewriter.replaceOp(xferOp, vec);
    }

    return success();
  }

  int64_t maxUnrollRank;
};

} // namespace

void populateTransferGatherUnrollingPatterns(RewritePatternSet &patterns,
                                             int64_t maxUnrollRank) {
  TransferGatherOp::getCanonicalizationPatterns(patterns,
                                                patterns.getContext());
  patterns.add<UnrollTransferGather>(patterns.getContext(), maxUnrollRank);
}

}; // namespace mlir::iree_compiler::IREE::VectorExt
