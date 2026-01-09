// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dominance.h"

#define DEBUG_TYPE "iree-encoding-utils"

namespace mlir::iree_compiler::IREE::Encoding {

SerializableAttr getSerializableAttr(RankedTensorType type) {
  return dyn_cast_if_present<SerializableAttr>(type.getEncoding());
}

EncodingAttr getEncodingAttr(RankedTensorType type) {
  return dyn_cast_if_present<EncodingAttr>(type.getEncoding());
}

bool hasPackedStorageAttr(RankedTensorType type) {
  return dyn_cast_if_present<PackedStorageAttr>(type.getEncoding()) != nullptr;
}

FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(EncodingAttr encoding) {
  ArrayAttr indexingMapsAttr = encoding.getUserIndexingMaps();
  if (!indexingMapsAttr) {
    return failure();
  }
  // Derive the contraction dims from the first maps in every entry of the
  // `user_indexing_maps` as these contain the layout information about the
  // originally encoded operation.
  SmallVector<AffineMap> indexingMaps = encoding.getRootMaps();
  return linalg::inferContractionDims(indexingMaps);
}

FailureOr<IREE::LinalgExt::ScaledContractionDimensions>
getEncodingScaledContractionDims(EncodingAttr encoding) {
  ArrayAttr indexingMapsAttr = encoding.getUserIndexingMaps();
  if (!indexingMapsAttr) {
    return failure();
  }
  // Derive the contraction dims from the first maps in every entry of the
  // `user_indexing_maps` as these contain the layout information about the
  // originally encoded operation.
  SmallVector<AffineMap> indexingMaps = encoding.getRootMaps();
  return IREE::LinalgExt::inferScaledContractionDims(
      ArrayRef<AffineMap>(indexingMaps));
}

FailureOr<BxMxNxKxKb> getEncodingContractionLikeSizes(EncodingAttr encoding) {
  SmallVector<int64_t> iterationSizes = encoding.getIterationSizesArray();
  if (iterationSizes.empty()) {
    return failure();
  }
  // Try to get scaled contraction dimensions first (which includes kB),
  // otherwise fall back to regular contraction dimensions.
  FailureOr<IREE::LinalgExt::ScaledContractionDimensions> maybeScaledCDims =
      getEncodingScaledContractionDims(encoding);
  IREE::LinalgExt::ScaledContractionDimensions cDims;
  if (succeeded(maybeScaledCDims)) {
    cDims = *maybeScaledCDims;
  } else {
    // Fall back to regular contraction dimensions and convert to scaled format
    // with an empty kB dimension.
    FailureOr<linalg::ContractionDimensions> maybeCDims =
        getEncodingContractionDims(encoding);
    if (failed(maybeCDims)) {
      return failure();
    }
    // Convert ContractionDimensions to ScaledContractionDimensions.
    cDims.m = maybeCDims->m;
    cDims.n = maybeCDims->n;
    cDims.k = maybeCDims->k;
    cDims.kB = {}; // Empty for non-scaled matmuls
    cDims.batch = maybeCDims->batch;
  }
  // The following expects M, N, K, KB, and Batch sizes of at most 1 for now.
  // TODO: Extend this to multiple M/N/K/KB/Batch dims.
  assert(cDims.m.size() <= 1 && cDims.n.size() <= 1 && cDims.k.size() == 1 &&
         cDims.kB.size() <= 1 && cDims.batch.size() <= 1 &&
         "Expected at most one M, N, K, KB, and Batch dimension");
  const int64_t k = iterationSizes[cDims.k[0]];
  // M, N, or Batch can be empty instead of having an explicit dim size of 1
  // for matvec and vecmat, so set to 1 if empty.
  const int64_t m = cDims.m.empty() ? 1 : iterationSizes[cDims.m[0]];
  const int64_t n = cDims.n.empty() ? 1 : iterationSizes[cDims.n[0]];
  const int64_t batch =
      cDims.batch.empty() ? 1 : iterationSizes[cDims.batch[0]];
  // KB can be empty if the encoding is not a scaled matmul.
  const int64_t kb = cDims.kB.empty() ? 1 : iterationSizes[cDims.kB[0]];
  return BxMxNxKxKb{batch, m, n, k, kb};
}

MatmulNarrowDim getPo2MatmulNarrowDim(EncodingAttr encoding) {
  // Get the matmul sizes for the given encoding.
  FailureOr<BxMxNxKxKb> matmulSizes = getEncodingContractionLikeSizes(encoding);
  if (failed(matmulSizes)) {
    return {};
  }
  const int64_t m = matmulSizes->M;
  const int64_t n = matmulSizes->N;

  // If both dimensions are dynamic, return empty.
  if (ShapedType::isDynamic(m) && ShapedType::isDynamic(n)) {
    return {};
  }
  // If only one dimension is dynamic, pick the other as the narrow dimension.
  if (ShapedType::isDynamic(m)) {
    return {MatmulNarrowDim::Dim::N,
            static_cast<int64_t>(llvm::PowerOf2Ceil(n))};
  }
  if (ShapedType::isDynamic(n)) {
    return {MatmulNarrowDim::Dim::M,
            static_cast<int64_t>(llvm::PowerOf2Ceil(m))};
  }
  // If Both dimensions are static, pick the smaller one.
  if (n < m) {
    return {MatmulNarrowDim::Dim::N,
            static_cast<int64_t>(llvm::PowerOf2Ceil(n))};
  }
  if (m < n) {
    return {MatmulNarrowDim::Dim::M,
            static_cast<int64_t>(llvm::PowerOf2Ceil(m))};
  }
  // If dimensions are static and equal, return empty.
  return {};
}

bool isNarrowNResult(EncodingAttr encoding) {
  if (encoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RESULT) {
    return false;
  }

  return IREE::Encoding::getPo2MatmulNarrowDim(encoding).isN();
}

namespace {
/// Helper class for recursive rematerialization of encoding dims.
/// Tracks already-rematerialized values to avoid duplicate work.
class RematerializationHelper {
public:
  RematerializationHelper(RewriterBase &builder, Operation *insertionPoint,
                          Value propagationSource, Value newSource)
      : builder(builder), insertionPoint(insertionPoint),
        propagationSource(propagationSource), newSource(newSource),
        domInfo(insertionPoint->getParentOp()) {}

  /// Try to rematerialize a single value, recursively rematerializing
  /// operands if needed.
  FailureOr<Value> rematerialize(Value value) {
    // Check if already rematerialized.
    auto it = rematerialized.find(value);
    if (it != rematerialized.end()) {
      return it->second;
    }

    // If the value already dominates, use it directly.
    if (domInfo.properlyDominates(value, insertionPoint)) {
      rematerialized[value] = value;
      return value;
    }

    Operation *definingOp = value.getDefiningOp();
    if (!definingOp) {
      // Block argument that doesn't dominate - cannot rematerialize.
      LLVM_DEBUG(llvm::dbgs() << "Cannot rematerialize: block argument does "
                                 "not dominate\n");
      return failure();
    }

    // Special case: tensor.dim on the propagation source can be recreated
    // from the new source tensor.
    if (auto dimOp = dyn_cast<tensor::DimOp>(definingOp)) {
      if (dimOp.getSource() == propagationSource) {
        Value newDim = tensor::DimOp::create(builder, dimOp.getLoc(), newSource,
                                             dimOp.getIndex());
        rematerialized[value] = newDim;
        return newDim;
      }
    }

    // General case: any pure operation can be cloned if we can
    // rematerialize all its operands.
    if (!isPure(definingOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot rematerialize: operation is not pure\n");
      return failure();
    }

    // Recursively rematerialize operands.
    SmallVector<Value> newOperands;
    newOperands.reserve(definingOp->getNumOperands());
    for (Value operand : definingOp->getOperands()) {
      FailureOr<Value> newOperand = rematerialize(operand);
      if (failed(newOperand)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Cannot rematerialize operand: " << operand << "\n");
        return failure();
      }
      newOperands.push_back(*newOperand);
    }

    // Find which result index the value corresponds to.
    auto resultIt = llvm::find(definingOp->getResults(), value);
    if (resultIt == definingOp->getResults().end()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot rematerialize: value not found in op results\n");
      return failure();
    }
    unsigned resultIndex =
        std::distance(definingOp->getResults().begin(), resultIt);

    // Clone the operation with the rematerialized operands.
    Operation *cloned = builder.clone(*definingOp);
    for (auto [idx, newOperand] : llvm::enumerate(newOperands)) {
      cloned->setOperand(idx, newOperand);
    }
    Value result = cloned->getResult(resultIndex);
    rematerialized[value] = result;
    return result;
  }

private:
  RewriterBase &builder;
  Operation *insertionPoint;
  Value propagationSource;
  Value newSource;
  DominanceInfo domInfo;
  DenseMap<Value, Value> rematerialized;
};
} // namespace

FailureOr<SmallVector<Value>>
rematerializeEncodingDims(RewriterBase &builder, Operation *insertionPoint,
                          ValueRange encodingDims, Value propagationSource,
                          Value newSource) {
  if (encodingDims.empty()) {
    return SmallVector<Value>{};
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertionPoint);

  RematerializationHelper helper(builder, insertionPoint, propagationSource,
                                 newSource);
  SmallVector<Value> rematerializedDims;
  rematerializedDims.reserve(encodingDims.size());

  for (Value dim : encodingDims) {
    FailureOr<Value> result = helper.rematerialize(dim);
    if (failed(result)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot rematerialize encoding dim: " << dim << "\n");
      return failure();
    }
    rematerializedDims.push_back(*result);
  }

  return rematerializedDims;
}

} // namespace mlir::iree_compiler::IREE::Encoding
