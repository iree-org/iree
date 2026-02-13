// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-codegen-amdgpu-distribute-contract"

namespace mlir::iree_compiler {
namespace {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

static LogicalResult
isSubgroupLayoutCompatible(IREE::GPU::MMASingleSubgroupLayout subgroupLayout,
                           NestedLayoutAttr layout, int64_t dim1,
                           int64_t dim2) {
  SmallVector<int64_t> element = {layout.getElementTile()[dim1],
                                  layout.getElementTile()[dim2]};
  SmallVector<int64_t> thread = {layout.getThreadTile()[dim1],
                                 layout.getThreadTile()[dim2]};
  SmallVector<int64_t> tstrides = {layout.getThreadStrides()[dim1],
                                   layout.getThreadStrides()[dim2]};
  SmallVector<int64_t> outer = {layout.getOuterTile()[dim1],
                                layout.getOuterTile()[dim2]};

  if (subgroupLayout.element != element) {
    return failure();
  }
  if (subgroupLayout.thread != thread) {
    return failure();
  }
  if (subgroupLayout.tstrides != tstrides) {
    return failure();
  }
  if (subgroupLayout.outer != outer) {
    return failure();
  }

  return success();
}

static LogicalResult isIntrinsicLayoutCompatible(
    VectorContractOpInfo &opInfo, IREE::GPU::MmaInterfaceAttr intrinsic,
    NestedLayoutAttr lhsLayout, NestedLayoutAttr rhsLayout,
    NestedLayoutAttr accLayout) {
  auto [lhsM, rhsN] = opInfo.getOperandMNIndex();
  auto [lhsK, rhsK] = opInfo.getOperandKIndex();
  auto [accM, accN] = opInfo.getResultMNIndex();
  if (failed(isSubgroupLayoutCompatible(
          getSingleSubgroupLayout(intrinsic, IREE::GPU::kMMAOperandLhs),
          lhsLayout, lhsM, lhsK))) {
    return failure();
  }
  if (failed(isSubgroupLayoutCompatible(
          getSingleSubgroupLayout(intrinsic, IREE::GPU::kMMAOperandRhs),
          rhsLayout, rhsK, rhsN))) {
    return failure();
  }
  if (failed(isSubgroupLayoutCompatible(
          getSingleSubgroupLayout(intrinsic, IREE::GPU::kMMAOperandAcc),
          accLayout, accM, accN))) {
    return failure();
  }
  return success();
}

/// Distributes `vector.contract` ops with nested layouts.
struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // Infer the contract kind so that we know know to correlate M/N/K dims.
    auto maybeOpDetail = VectorContractOpInfo::inferFromIndexingMaps(
        contractOp.getIndexingMapsArray());
    if (failed(maybeOpDetail)) {
      return rewriter.notifyMatchFailure(contractOp, "invalid contraction");
    }
    VectorContractOpInfo opDetail = maybeOpDetail.value();

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
    int64_t rank = resultLayout.getRank();

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
    NestedLayoutAttr accLayout =
        dyn_cast<NestedLayoutAttr>(signature[resultValue]);
    if (!accLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing nested layout for contraction acc");
    }

    // We assume there is an decision made before regarding which mfma intrinsic
    // to use and it is attached as an attribute to this contract op.
    auto mmaKind =
        contractOp->getAttrOfType<IREE::GPU::MmaInterfaceAttr>("iree.gpu.mma");
    if (!mmaKind) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing iree.gpu.mma intrinsic attribute");
    }

    // Check if the given intrinsic can be distributed with the given
    // layouts.
    if (failed(isIntrinsicLayoutCompatible(opDetail, mmaKind, lhsLayout,
                                           rhsLayout, accLayout))) {
      return rewriter.notifyMatchFailure(
          contractOp, "the intrinsic does not match the expected layouts");
    }

    SmallVector<int64_t> distShape = resultLayout.getDistributedShape();
    LDBG() << "distributed shape: " << llvm::interleaved_array(distShape);

    // Create a zero vector with the full distributed vector shape for
    // accumulating unrolled contraction results.
    auto tileType = VectorType::get(distShape, resultType.getElementType());
    Value zero =
        arith::ConstantOp::create(rewriter, contractOp.getLoc(), tileType,
                                  rewriter.getZeroAttr(tileType));
    VectorValue finalTile = cast<VectorValue>(zero);
    LLVM_DEBUG(llvm::dbgs() << "init tile: " << finalTile << "\n");

    // Offsets into the LHS/RHS batches.
    SmallVector<int64_t> lhsBatchOffsets(lhsLayout.getRank(), 0);
    SmallVector<int64_t> rhsBatchOffsets(rhsLayout.getRank(), 0);

    // Offsets into the result batches.
    ArrayRef<int64_t> resultBatches = resultLayout.getBatchTile();
    SmallVector<int64_t> resultBatchTileSizes(rank, 1);
    LLVM_DEBUG({
      llvm::dbgs() << "result batches: [";
      llvm::interleaveComma(resultBatches, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    Value acc = getDistributed(rewriter, cast<VectorValue>(contractOp.getAcc()),
                               resultLayout);
    Value lhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
    Value rhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);

    SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
    AffineMap lhsMap = compressUnusedDims(indexingMaps[0]);
    AffineMap rhsMap = compressUnusedDims(indexingMaps[1]);
    AffineMap resMap = compressUnusedDims(indexingMaps[2]);

    SmallVector<int64_t> resBatchOrder(resMap.getNumResults());
    std::iota(resBatchOrder.begin(), resBatchOrder.end(), 0);
    resBatchOrder = applyPermutationMap(resMap, ArrayRef(resBatchOrder));

    // Iterate over all result batches and unroll computation to direct MFMA
    // intrinsic ops.
    Location loc = contractOp.getLoc();
    auto resultTiles = StaticTileOffsetRange(
        resultBatches, resultBatchTileSizes, resBatchOrder);
    SmallVector<int64_t, 2> resultBatchOffsets;
    for (SmallVector<int64_t, 2> resultBatchOffsets : resultTiles) {
      LLVM_DEBUG({
        llvm::dbgs() << "current result batch offsets: [";
        llvm::interleaveComma(resultBatchOffsets, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      // Get the slice of the accumulator in this batch.
      Value accSlice =
          vector::ExtractOp::create(rewriter, loc, acc, resultBatchOffsets);

      // Get the k batch size for LHS and RHS vector.
      std::optional<int64_t> kBatch =
          getKBatchSize(opDetail, lhsLayout, rhsLayout);
      LLVM_DEBUG(llvm::dbgs() << "k batch size = " << kBatch << "\n");
      if (!kBatch) {
        return rewriter.notifyMatchFailure(contractOp,
                                           "A/B vector k batch mismatch");
      }

      // Check if deferred accumulator collapse is applicable.
      // For VSMFMA intrinsics with kBatch > 1, we expand the ACC once before
      // the K-loop and collapse once after, avoiding redundant expand/collapse
      // on every iteration.
      auto virtualMma = dyn_cast<IREE::GPU::VirtualMMAAttr>(mmaKind);
      std::optional<VectorType> expandedAccType;
      if (virtualMma && *kBatch > 1) {
        expandedAccType = virtualMma.getExpandedAccType();
      }

      if (expandedAccType) {
        // === Deferred collapse path ===
        auto [aVectorType, bVectorType, cVectorType] =
            mmaKind.getABCVectorTypes();

        // Shape-cast to collapsed 1D, then expand once.
        Value cFlat =
            vector::ShapeCastOp::create(rewriter, loc, cVectorType, accSlice);
        Value workingAcc = virtualMma.expandAccumulator(rewriter, loc, cFlat);

        for (int k = 0; k < *kBatch; ++k) {
          fillOperandBatchOffsets(opDetail, k, resultBatchOffsets,
                                  lhsBatchOffsets, rhsBatchOffsets, lhsMap,
                                  rhsMap);
          LDBG() << "current lhs batch offsets: "
                 << llvm::interleaved_array(lhsBatchOffsets);
          LDBG() << "current rhs batch offsets: "
                 << llvm::interleaved_array(rhsBatchOffsets);

          Value lhsSlice =
              vector::ExtractOp::create(rewriter, loc, lhs, lhsBatchOffsets);
          Value rhsSlice =
              vector::ExtractOp::create(rewriter, loc, rhs, rhsBatchOffsets);
          Value aCast =
              vector::ShapeCastOp::create(rewriter, loc, aVectorType, lhsSlice);
          Value bCast =
              vector::ShapeCastOp::create(rewriter, loc, bVectorType, rhsSlice);

          SmallVector<Value> mmaResults;
          [[maybe_unused]] LogicalResult createdMmaOp =
              mmaKind.buildUnderlyingOperations(rewriter, loc, {aCast, bCast},
                                                {workingAcc}, mmaResults);
          assert(succeeded(createdMmaOp) &&
                 "Should never fail to construct mma op");
          workingAcc = mmaResults[0];
        }

        // Collapse once, then shape-cast back.
        Value collapsed =
            virtualMma.collapseAccumulator(rewriter, loc, workingAcc);
        accSlice = vector::ShapeCastOp::create(rewriter, loc,
                                               accSlice.getType(), collapsed);
      } else {
        // === Standard path ===
        for (int k = 0; k < *kBatch; ++k) {
          fillOperandBatchOffsets(opDetail, k, resultBatchOffsets,
                                  lhsBatchOffsets, rhsBatchOffsets, lhsMap,
                                  rhsMap);
          LDBG() << "current lhs batch offsets: "
                 << llvm::interleaved_array(lhsBatchOffsets);
          LDBG() << "current rhs batch offsets: "
                 << llvm::interleaved_array(rhsBatchOffsets);

          Value lhsSlice =
              vector::ExtractOp::create(rewriter, loc, lhs, lhsBatchOffsets);
          Value rhsSlice =
              vector::ExtractOp::create(rewriter, loc, rhs, rhsBatchOffsets);
          accSlice =
              computeMMA(rewriter, loc, mmaKind, lhsSlice, rhsSlice, accSlice);
        }
      }
      finalTile = vector::InsertOp::create(rewriter, loc, accSlice, finalTile,
                                           resultBatchOffsets);
    }

    replaceOpWithDistributedValues(rewriter, contractOp, finalTile);
    return success();
  }

  // Gets the batch size for matmul K dimensions.
  std::optional<int64_t> getKBatchSize(const VectorContractOpInfo &opDetail,
                                       NestedLayoutAttr lhsLayout,
                                       NestedLayoutAttr rhsLayout) const {
    auto [lhsK, rhsK] = opDetail.getOperandKIndex();
    int64_t lhsKBatch = lhsLayout.getBatchTile()[lhsK];
    int64_t rhsKBatch = rhsLayout.getBatchTile()[rhsK];

    if (lhsKBatch != rhsKBatch) {
      return std::nullopt;
    }
    return lhsKBatch;
  }

  // Given a contract op's batch |resultOffsets|, fills its batch offsets for
  // both LHS and RHS.
  void fillOperandBatchOffsets(const VectorContractOpInfo &opDetail,
                               int64_t kOffset, ArrayRef<int64_t> resultOffsets,
                               SmallVector<int64_t> &lhsOffsets,
                               SmallVector<int64_t> &rhsOffsets,
                               AffineMap lhsMap, AffineMap rhsMap) const {
    auto [lhsK, rhsK] = opDetail.getOperandKIndex();
    // resultOffsets contains batch indices into the C/D vector. It is a 2-D
    // index for both M and N. We need to split out for M and N, and add index
    // for K.
    for (auto [lhsM, resultM] :
         llvm::zip_equal(opDetail.lhsMDims, opDetail.outMDims)) {
      lhsOffsets[lhsM] = resultOffsets[resultM];
    }

    if (opDetail.getBatchCount() == 1) {
      rhsOffsets[0] = resultOffsets[0];
      lhsOffsets[0] = resultOffsets[0];
    }

    for (auto [rhsN, resultN] :
         llvm::zip_equal(opDetail.rhsNDims, opDetail.outNDims)) {
      rhsOffsets[rhsN] = resultOffsets[resultN];
    }

    lhsOffsets[lhsK] = kOffset;
    rhsOffsets[rhsK] = kOffset;
  }

  // Generates amdgpu.mfma operation on the given inputs for the given MFMA
  // |intrinsic|.
  Value computeMMA(OpBuilder &builder, Location loc,
                   IREE::GPU::MmaInterfaceAttr mmaKind, Value a, Value b,
                   Value c) const {
    // Get the storage vector types that each thread is in charge of.
    auto [aVectorType, bVectorType, cVectorType] = mmaKind.getABCVectorTypes();
    Value aCast =
        vector::ShapeCastOp::create(builder, a.getLoc(), aVectorType, a);
    Value bCast =
        vector::ShapeCastOp::create(builder, b.getLoc(), bVectorType, b);
    Value cCast =
        vector::ShapeCastOp::create(builder, c.getLoc(), cVectorType, c);
    SmallVector<Value> results;
    [[maybe_unused]] LogicalResult createdMmaOp =
        mmaKind.buildUnderlyingOperations(builder, loc, {aCast, bCast}, {cCast},
                                          results);
    assert(succeeded(createdMmaOp) && "Should never fail to construct mma op");
    return vector::ShapeCastOp::create(builder, c.getLoc(), c.getType(),
                                       results[0]);
  }
};

} // namespace

void populateGPUDistributeNestedLayoutContractAMDGPUPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DistributeContract>(patterns.getContext());
}

} // namespace mlir::iree_compiler
