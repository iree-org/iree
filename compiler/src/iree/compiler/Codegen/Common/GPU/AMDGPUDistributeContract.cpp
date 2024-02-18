// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-amdgpu-distribute-contract"

namespace mlir::iree_compiler {
namespace {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

/// A class for querying information about a contract op.
class ContractOpDetail {
public:
  enum class OpKind { MK_KN_MN, MK_NK_MN, UNKNOWN };

  explicit ContractOpDetail(vector::ContractionOp op) {
    opKind = inferOpKind(op.getContext(), op.getIndexingMapsArray());
  }

  OpKind getOpKind() const { return opKind; }

  // Returns the (LHS M, RHS N) dimension index pair.
  std::optional<std::pair<int, int>> getOperandMNIndex() const {
    switch (opKind) {
    case OpKind::MK_KN_MN:
      return std::make_pair(0, 1);
    case OpKind::MK_NK_MN:
      return std::make_pair(0, 0);
    case OpKind::UNKNOWN:
      break;
    }
    return std::nullopt;
  }

  // Returns the (LHS K, RHS K) dimension index pair.
  std::optional<std::pair<int, int>> getOperandKIndex() const {
    switch (opKind) {
    case OpKind::MK_KN_MN:
      return std::make_pair(1, 0);
    case OpKind::MK_NK_MN:
      return std::make_pair(1, 1);
    case OpKind::UNKNOWN:
      break;
    }
    return std::nullopt;
  }

  // Returns the result (M, N) dimension index pair.
  std::optional<std::pair<int, int>> getResultMNIndex() const {
    switch (opKind) {
    case OpKind::MK_KN_MN:
    case OpKind::MK_NK_MN:
      return std::make_pair(0, 1);
    default:
      break;
    }
    return std::nullopt;
  }

private:
  // Gets the kind of a contract op with the given indexing |maps|.
  OpKind inferOpKind(MLIRContext *ctx, SmallVector<AffineMap> maps) {
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [&](MapList m) {
      return AffineMap::inferFromExprList(m, ctx);
    };
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    if (maps == infer({{m, k}, {k, n}, {m, n}}))
      return OpKind::MK_KN_MN;
    if (maps == infer({{m, k}, {n, k}, {m, n}}))
      return OpKind::MK_NK_MN;
    return OpKind::UNKNOWN;
  }

private:
  OpKind opKind = OpKind::UNKNOWN;
};

/// Distributes `vector.contract` ops with nested layouts.
struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(contractOp.getResultType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          contractOp, "unhandled contraction to scalar value");
    }

    auto resultValue = cast<VectorValue>(contractOp.getResult());
    NestedLayoutAttr resultLayout =
        dyn_cast<NestedLayoutAttr>(signature[resultValue]);
    if (!resultLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction result");
    }
    int64_t rank = resultLayout.getBatchOrder().size();

    NestedLayoutAttr lhsLayout =
        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
    if (!lhsLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction lhs");
    }
    NestedLayoutAttr rhsLayout =
        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
    if (!rhsLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction rhs");
    }

    // We assume there is an decision made before regarding which mfma intrinsic
    // to use and it is attached as an attribute to this contract op.
    auto mfmaAttr =
        contractOp->getAttrOfType<IREE::GPU::MFMAAttr>("iree.amdgpu.mfma");
    if (!mfmaAttr) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing iree.amdgpu.mfma intrinsic attribute");
    }
    // Get the storage vector types that each thread is in charge of.
    auto [aVectorType, bVectorType, cVectorType] = mfmaAttr.getABCVectorTypes();
    // Get parameters for the amdgpu.mfma operation.
    MFMAParameters mfmaParams;
    std::tie(mfmaParams.m, mfmaParams.n, mfmaParams.k) = mfmaAttr.getMNKShape();
    mfmaParams.blocks = mfmaAttr.getBlockSize();

    // Infer the contract kind so that we know know to correlate M/N/K dims.
    ContractOpDetail opDetail(contractOp);
    if (opDetail.getOpKind() == ContractOpDetail::OpKind::UNKNOWN) {
      return rewriter.notifyMatchFailure(contractOp, "unknown contract kind");
    }

    SmallVector<int64_t> distShape = resultLayout.getDistributedShape();
    LLVM_DEBUG({
      llvm::dbgs() << "distributed shape: [";
      llvm::interleaveComma(distShape, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    // Create a zero vector with the full distributed vector shape for
    // accumulating unrolled contraction results.
    auto tileType = VectorType::get(distShape, resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(
        contractOp.getLoc(), tileType, rewriter.getZeroAttr(tileType));
    VectorValue finalTile = cast<VectorValue>(zero);
    LLVM_DEBUG(llvm::dbgs() << "init tile: " << finalTile << "\n");

    // Offsets into the LHS/RHS batches.
    SmallVector<int64_t, 2> lhsBatchOffsets(rank, 0);
    SmallVector<int64_t, 2> rhsBatchOffsets(rank, 0);

    // Offsets into the result batches.
    ArrayRef<int64_t> resultBatches = resultLayout.getBatchesPerSubgroup();
    SmallVector<int64_t, 2> resultBatchTileSizes(rank, 1);
    LLVM_DEBUG({
      llvm::dbgs() << "result batches: [";
      llvm::interleaveComma(resultBatches, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    Value acc = getDistributed(rewriter, cast<VectorValue>(contractOp.getAcc()),
                               resultLayout);
    Value lhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
    Value rhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);

    // Iterate over all result batches and unroll computation to direct MFMA
    // intrinsic ops.
    Location loc = contractOp.getLoc();
    auto resultTiles = StaticTileOffsetRange(
        resultBatches, resultBatchTileSizes, resultLayout.getBatchOrder());
    SmallVector<int64_t, 2> resultBatchOffsets;
    for (SmallVector<int64_t, 2> originalResultBatchOffsets : resultTiles) {
      // Permute the result batch offsets first to match the distributed shape
      // dim order for indexing.
      resultBatchOffsets = originalResultBatchOffsets;
      applyPermutationToVector(resultBatchOffsets,
                               resultLayout.getBatchOrder());
      LLVM_DEBUG({
        llvm::dbgs() << "current result batch offsets: [";
        llvm::interleaveComma(resultBatchOffsets, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      // Get the slice of the accumulator in this batch.
      Value accSlice =
          rewriter.create<vector::ExtractOp>(loc, acc, resultBatchOffsets);

      // Get the k batch size for LHS and RHS vector.
      std::optional<int64_t> kBatch =
          getKBatchSize(opDetail, lhsLayout, rhsLayout);
      LLVM_DEBUG(llvm::dbgs() << "k batch size = " << kBatch << "\n");
      if (!kBatch) {
        return rewriter.notifyMatchFailure(contractOp,
                                           "A/B vector k batch mismatch");
      }

      // Perform contraction by doing separate outer product with amdgpu.mfma
      // operation and accumulate to the same vector.
      for (int k = 0; k < kBatch; ++k) {
        // Fills the batch offsets for LHS and RHS. For the K dimension it's the
        // induction variable; for the M/N dimension we need to extract from the
        // result batch offsets.
        fillOperandBatchOffsets(opDetail, k, originalResultBatchOffsets,
                                resultLayout, lhsBatchOffsets, rhsBatchOffsets,
                                lhsLayout, rhsLayout);
        LLVM_DEBUG({
          llvm::dbgs() << "current lhs batch offsets: [";
          llvm::interleaveComma(lhsBatchOffsets, llvm::dbgs());
          llvm::dbgs() << "]\n";
          llvm::dbgs() << "current rhs batch offsets: [";
          llvm::interleaveComma(rhsBatchOffsets, llvm::dbgs());
          llvm::dbgs() << "]\n";
        });

        Value lhsSlice =
            rewriter.create<vector::ExtractOp>(loc, lhs, lhsBatchOffsets);
        Value rhsSlice =
            rewriter.create<vector::ExtractOp>(loc, rhs, rhsBatchOffsets);
        accSlice = computeMMA(rewriter, loc, mfmaParams, lhsSlice, rhsSlice,
                              accSlice, aVectorType, bVectorType, cVectorType);
      }
      finalTile = rewriter.create<vector::InsertOp>(loc, accSlice, finalTile,
                                                    resultBatchOffsets);
    }

    replaceOpWithDistributedValues(rewriter, contractOp, finalTile);
    return success();
  }

  // Gets the batch size for matmul K dimensions.
  std::optional<int64_t> getKBatchSize(const ContractOpDetail &opDetail,
                                       NestedLayoutAttr lhsLayout,
                                       NestedLayoutAttr rhsLayout) const {
    auto [lhsK, rhsK] = *opDetail.getOperandKIndex();
    int64_t lhsKBatch = lhsLayout.getBatchesPerSubgroup()[lhsK];
    int64_t rhsKBatch = rhsLayout.getBatchesPerSubgroup()[rhsK];

    if (lhsKBatch != rhsKBatch)
      return std::nullopt;
    return lhsKBatch;
  }

  // Given a contract op's batch |resultOffsets|, fills its batch offsets for
  // both LHS and RHS.
  void fillOperandBatchOffsets(const ContractOpDetail &opDetail,
                               int64_t kOffset, ArrayRef<int64_t> resultOffsets,
                               NestedLayoutAttr resultLayout,
                               SmallVector<int64_t, 2> &lhsOffsets,
                               SmallVector<int64_t, 2> &rhsOffsets,
                               NestedLayoutAttr lhsLayout,
                               NestedLayoutAttr rhsLayout) const {
    auto [lhsM, rhsN] = *opDetail.getOperandMNIndex();
    auto [lhsK, rhsK] = *opDetail.getOperandKIndex();
    auto [resultM, resultN] = *opDetail.getResultMNIndex();
    // resultOffsets contains batch indices into the C/D vector. It is a 2-D
    // index for both M and N. We need to split out for M and N, and add index
    // for K.
    lhsOffsets[lhsM] = resultOffsets[resultM];
    lhsOffsets[lhsK] = kOffset;
    rhsOffsets[rhsN] = resultOffsets[resultN];
    rhsOffsets[rhsK] = kOffset;

    // Now apply permutation on LHS/RHS according to their batch order.
    applyPermutationToVector(lhsOffsets, lhsLayout.getBatchOrder());
    applyPermutationToVector(rhsOffsets, rhsLayout.getBatchOrder());
  }

  struct MFMAParameters {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t blocks = 0;
  };

  // Generates amdgpu.mfma operation on the given inputs for the given MFMA
  // |intrinsic|.
  Value computeMMA(OpBuilder &builder, Location loc,
                   const MFMAParameters &mfmaParams, Value a, Value b, Value c,
                   VectorType aType, VectorType bType, VectorType cType) const {
    Value aCast = builder.create<vector::ShapeCastOp>(a.getLoc(), aType, a);
    Value bCast = builder.create<vector::ShapeCastOp>(b.getLoc(), bType, b);
    Value cCast = builder.create<vector::ShapeCastOp>(c.getLoc(), cType, c);

    Value mfmaOp = builder.create<amdgpu::MFMAOp>(
        loc, cType, mfmaParams.m, mfmaParams.n, mfmaParams.k, mfmaParams.blocks,
        aCast, bCast, cCast);
    return builder.create<vector::ShapeCastOp>(c.getLoc(), c.getType(), mfmaOp);
  }
};

} // namespace

void populateGPUDistributeNestedLayoutContractAMDGPUPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DistributeContract>(patterns.getContext());
}

} // namespace mlir::iree_compiler
