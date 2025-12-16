// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOpInterfaces.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtAttrInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOpInterfaces.cpp.inc" // IWYU pragma: keep
// clang-format on: must be included after all LLVM/MLIR headers

namespace mlir::iree_compiler::IREE::TensorExt {

LogicalResult verifySparseCastOpInterface(SparseCastOpInterface sparseOp) {
  // Check that the operation has only one result.
  if (sparseOp->getNumResults() != 1) {
    return sparseOp.emitOpError("sparse operations can only have one result");
  }

  // The sparse operation needs to return a shaped type that has an attribute
  // specifying the sparsity.
  Type resultType = sparseOp->getResult(0).getType();
  SparseShapeAttrInterface sparseAttr;
  if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
    sparseAttr =
        dyn_cast_if_present<SparseShapeAttrInterface>(tensorType.getEncoding());
    if (!sparseAttr) {
      return sparseOp.emitOpError(
          "expected result type to have an encoding attribute that implements "
          "the `SparseShapeAttrInterface`");
    }
  } else if (auto memrefType = dyn_cast<MemRefType>(resultType)) {
    sparseAttr =
        dyn_cast_if_present<SparseShapeAttrInterface>(memrefType.getLayout());
    if (!sparseAttr) {
      return sparseOp.emitOpError(
          "expected result type to have a layout attribute that implements "
          "the `SparseShapeAttrInterface`");
    }
  } else {
    return sparseOp->emitOpError("unhandled return type for sparse operation");
  }

  assert(sparseAttr);
  SmallVector<int64_t> sparseDimensions = sparseAttr.getSparseDimensions();
  if (sparseDimensions.size() < 2) {
    return sparseOp.emitOpError(
        "need at least two sparse dimensions for the result of a sparse op");
  }
  // Assert that the sparse dimensions are contiguous.
  for (int i = 1; i < sparseDimensions.size(); ++i) {
    if (sparseDimensions[i] != sparseDimensions[i - 1] + 1) {
      return sparseOp.emitOpError(
          "expected sparse dimensions to be contiguous");
    }
  }
  return success();
}

std::optional<SparseRangeResolver> getSparseRangeResolver(Range range) {
  Value upperBound = dyn_cast<Value>(range.size);
  if (!upperBound || isa<BlockArgument>(upperBound)) {
    return std::nullopt;
  }

  SparseCastOpInterface sparseOp;
  int64_t resultDim;

  // For now just check that the defining operation of the upper bound is a
  // `memref.dim` operation of a result of a sparse operation.
  Value ubSource;
  IntegerAttr dim;
  if (!matchPattern(upperBound, m_Op<memref::DimOp>(matchers::m_Any(&ubSource),
                                                    m_Constant(&dim)))) {
    return std::nullopt;
  }
  sparseOp = ubSource.getDefiningOp<SparseCastOpInterface>();
  if (!sparseOp) {
    return std::nullopt;
  }
  resultDim = dim.getInt();
  return SparseRangeResolver{sparseOp, resultDim};
}

SmallVector<SparseRangeResolver>
getSparseRangeResolvers(ArrayRef<Range> ranges) {
  SmallVector<SparseRangeResolver> resolvers;
  resolvers.reserve(ranges.size());
  for (Range range : ranges) {
    if (std::optional<SparseRangeResolver> resolver =
            getSparseRangeResolver(range)) {
      resolvers.push_back(*resolver);
    } else {
      resolvers.push_back(SparseRangeResolver{});
    }
  }
  return resolvers;
}

} // namespace mlir::iree_compiler::IREE::TensorExt
