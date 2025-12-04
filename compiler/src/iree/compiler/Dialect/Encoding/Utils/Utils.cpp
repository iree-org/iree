// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

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

} // namespace mlir::iree_compiler::IREE::Encoding
