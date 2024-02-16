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

#define DEBUG_TYPE "iree-amdgpu-distribute-contract"

namespace mlir::iree_compiler {
namespace {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

enum class ContractKind { MK_KN_MN, UNKNOWN };

// Gets the kind of a contract op with the given indexing |maps|.
ContractKind inferContractKind(MLIRContext *ctx, SmallVector<AffineMap> maps) {
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [&](MapList m) { return AffineMap::inferFromExprList(m, ctx); };
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  if (maps == infer({{m, k}, {k, n}, {m, n}}))
    return ContractKind::MK_KN_MN;
  return ContractKind::UNKNOWN;
}

/// Distributes `vector.contract` ops with nested layouts.
struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeContract(MLIRContext *context, Value threadId)
      : OpDistributionPattern(context), threadId(threadId) {}

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

    auto mfmaAttr =
        contractOp->getAttrOfType<IREE::GPU::MFMAAttr>("iree.amdgpu.mfma");
    if (!mfmaAttr) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing iree.amdgpu.mfma intrinsic attribute");
    }

    ContractKind contractKind =
        inferContractKind(getContext(), contractOp.getIndexingMapsArray());
    if (contractKind == ContractKind::UNKNOWN) {
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

    SmallVector<int64_t, 2> lhsBatchOffsets(rank, 0);
    SmallVector<int64_t, 2> rhsBatchOffsets(rank, 0);

    ArrayRef<int64_t> resultBatches = resultLayout.getBatchesPerSubgroup();
    SmallVector<int64_t, 2> resultBatchTileSizes(rank, 1);
    LLVM_DEBUG({
      llvm::dbgs() << "result batches: [";
      llvm::interleaveComma(resultBatches, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    // Iterate over all result batches and unroll computation to direct MFMA
    // intrinsic ops.
    Location loc = contractOp.getLoc();
    auto resultTiles = StaticTileOffsetRange(
        resultBatches, resultBatchTileSizes, resultLayout.getBatchOrder());
    for (SmallVector<int64_t, 2> resultBatchOffsets : resultTiles) {
      LLVM_DEBUG({
        llvm::dbgs() << "current result batch offsets: [";
        llvm::interleaveComma(resultBatchOffsets, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      // Get the slice of the accumulator in this batch.
      Value acc = getDistributed(
          rewriter, cast<VectorValue>(contractOp.getAcc()), resultLayout);
      Value accSlice =
          rewriter.create<vector::ExtractOp>(loc, acc, resultBatchOffsets);

      // Get the k batch size for lhs and rhs vector.
      std::optional<int64_t> kBatch =
          getKBatchSize(contractKind, lhsLayout, rhsLayout);
      LLVM_DEBUG(llvm::dbgs() << "k batch size = " << kBatch << "\n");
      if (!kBatch) {
        return rewriter.notifyMatchFailure(contractOp,
                                           "A/B vector k batch mismatch");
      }

      // Perform contraction. Do separate outer product with mfma operation and
      // accumulate to the same vector.
      for (int k = 0; k < kBatch; ++k) {
        if (!getOperandBatchOffsets(contractKind, k, resultBatchOffsets,
                                    resultLayout, lhsBatchOffsets,
                                    rhsBatchOffsets, lhsLayout, rhsLayout)) {
          return rewriter.notifyMatchFailure(
              contractOp, "cannot deduce lhs/rhs batch offsets");
        }
        LLVM_DEBUG({
          llvm::dbgs() << "current lhs batch offsets: [";
          llvm::interleaveComma(lhsBatchOffsets, llvm::dbgs());
          llvm::dbgs() << "]\n";
          llvm::dbgs() << "current rhs batch offsets: [";
          llvm::interleaveComma(rhsBatchOffsets, llvm::dbgs());
          llvm::dbgs() << "]\n";
        });
        Value lhsSlice = rewriter.create<vector::ExtractOp>(
            loc, getDistributed(rewriter, contractOp.getLhs(), lhsLayout),
            lhsBatchOffsets);
        Value rhsSlice = rewriter.create<vector::ExtractOp>(
            loc, getDistributed(rewriter, contractOp.getRhs(), rhsLayout),
            rhsBatchOffsets);
        accSlice = computeMMA(rewriter, loc, lhsSlice, rhsSlice, accSlice,
                              mfmaAttr.getIntrinsic().getValue());
      }
      finalTile = rewriter.create<vector::InsertOp>(loc, accSlice, finalTile,
                                                    resultBatchOffsets);
    }

    replaceOpWithDistributedValues(rewriter, contractOp, finalTile);
    return success();
  }

  // Gets the batch size for matmul K dimensions.
  std::optional<int64_t> getKBatchSize(ContractKind kind,
                                       NestedLayoutAttr lhsLayout,
                                       NestedLayoutAttr rhsLayout) const {
    int64_t lhsKBatch = 0, rhsKBatch = 0;
    if (kind == ContractKind::MK_KN_MN) {
      lhsKBatch = lhsLayout.getBatchesPerSubgroup()[1];
      rhsKBatch = rhsLayout.getBatchesPerSubgroup()[0];
    } else {
      return std::nullopt;
    }

    if (lhsKBatch != rhsKBatch)
      return std::nullopt;
    return lhsKBatch;
  }

  // Given a contract op's batch |resultOffsets|, gets its batch offsets for
  // both LHS and RHS.
  bool getOperandBatchOffsets(
      ContractKind kind, int64_t kOffset,
      // Copy intentionally given we need to mutate in the function body
      SmallVector<int64_t, 2> resultOffsets, NestedLayoutAttr resultLayout,
      SmallVector<int64_t, 2> &lhsOffsets, SmallVector<int64_t, 2> &rhsOffsets,
      NestedLayoutAttr lhsLayout, NestedLayoutAttr rhsLayout) const {
    // The result offsets are permutated and we need to revert the permutation.
    // The following works based on the fact that for 2-D cases, a permuation
    // vector's reverse is just itself. So this only works for 2-D cases.
    applyPermutationToVector(resultOffsets, resultLayout.getBatchOrder());

    // resultOffsets contains batch indices into the C/D vector. It is a 2-D
    // index for both M and N. We need to split out for M and N, and add index
    // for K.
    if (kind == ContractKind::MK_KN_MN) {
      lhsOffsets[0] = resultOffsets[0];
      lhsOffsets[1] = kOffset;
      rhsOffsets[0] = kOffset;
      rhsOffsets[1] = resultOffsets[1];
    } else {
      return false;
    }

    // Now apply permutation on lhs/rhs according to their batch order.
    applyPermutationToVector(lhsOffsets, lhsLayout.getBatchOrder());
    applyPermutationToVector(rhsOffsets, rhsLayout.getBatchOrder());
    return true;
  }

  // Returns the vector type for the given |value| meeting mfma ops's
  // requirements.
  VectorType getMFMAVectorType(Value value) const {
    auto type = cast<VectorType>(value.getType());
    SmallVector<int64_t> shape;
    for (int64_t dim : type.getShape()) {
      if (dim != 1)
        shape.push_back(dim);
    }
    return VectorType::get(shape, type.getElementType());
  }

  Value computeMMA(OpBuilder &builder, Location loc, Value a, Value b, Value c,
                   IREE::GPU::MFMAIntrinsic intrinsic) const {
    // TODO: query these types from the intrinsic attribute outside of this
    // function.
    VectorType aType = getMFMAVectorType(a);
    VectorType bType = getMFMAVectorType(b);
    VectorType cType = getMFMAVectorType(c);
    Value aCast = builder.create<vector::ShapeCastOp>(a.getLoc(), aType, a);
    Value bCast = builder.create<vector::ShapeCastOp>(b.getLoc(), bType, b);
    Value cCast = builder.create<vector::ShapeCastOp>(c.getLoc(), cType, c);

    uint32_t m, n, k, blocks;
    switch (intrinsic) {
    case IREE::GPU::MFMAIntrinsic::F16_16x16x16_F32:
      m = n = k = 16;
      blocks = 1;
      break;
    case IREE::GPU::MFMAIntrinsic::F16_32x32x8_F32:
      m = n = 32;
      k = 8;
      blocks = 1;
      break;
    }
    Value mfmaOp = builder.create<amdgpu::MFMAOp>(loc, cType, m, n, k, blocks,
                                                  aCast, bCast, cCast);
    return builder.create<vector::ShapeCastOp>(c.getLoc(), c.getType(), mfmaOp);
  }

  Value threadId;
};

} // namespace

void populateGPUDistributeNestedLayoutContractAMDGPUPatterns(
    Value threadId, RewritePatternSet &patterns) {
  patterns.add<DistributeContract>(patterns.getContext(), threadId);
}

} // namespace mlir::iree_compiler
