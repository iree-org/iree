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
  return dyn_cast_or_null<SerializableAttr>(type.getEncoding());
}

EncodingAttr getEncodingAttr(RankedTensorType type) {
  return dyn_cast_or_null<EncodingAttr>(type.getEncoding());
}

bool hasPackedStorageAttr(RankedTensorType type) {
  return dyn_cast_or_null<PackedStorageAttr>(type.getEncoding()) != nullptr;
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

FailureOr<BxMxNxK> getMatmulSizes(EncodingAttr encoding) {
  if (encoding.getOpType().getValue() != EncodingOpType::matmul) {
    return failure();
  }
  SmallVector<int64_t> iterationSizes = encoding.getIterationSizesArray();
  if (iterationSizes.empty()) {
    return failure();
  }
  std::optional<linalg::ContractionDimensions> maybeCDims =
      getEncodingContractionDims(encoding);
  if (!maybeCDims) {
    return failure();
  }
  linalg::ContractionDimensions cDims = maybeCDims.value();
  // The following expects M, N, K, and Batch sizes of at most 1 for now.
  // TODO: Extend this to multiple M/N/K/Batch dims.
  assert(cDims.m.size() <= 1 && cDims.n.size() <= 1 && cDims.k.size() == 1 &&
         cDims.batch.size() <= 1 &&
         "Expected at most one M, N, K, and Batch dimension");
  const int64_t k = iterationSizes[cDims.k[0]];
  // M, N or Batch can be empty instead of having an explicit dim size of 1 for
  // matvec and vecmat, so set to 1 if empty.
  const int64_t m = cDims.m.empty() ? 1 : iterationSizes[cDims.m[0]];
  const int64_t n = cDims.n.empty() ? 1 : iterationSizes[cDims.n[0]];
  const int64_t batch =
      cDims.batch.empty() ? 1 : iterationSizes[cDims.batch[0]];
  return BxMxNxK{batch, m, n, k};
}

MatmulNarrowDim getPo2MatmulNarrowDim(EncodingAttr encoding) {
  // Get the matmul sizes for the given encoding.
  FailureOr<BxMxNxK> matmulSizes = getMatmulSizes(encoding);
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
