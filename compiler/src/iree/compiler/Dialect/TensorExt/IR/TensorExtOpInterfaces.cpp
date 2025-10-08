// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOpInterfaces.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOpInterfaces.cpp.inc" // IWYU pragma: keep
// clang-format on: must be included after all LLVM/MLIR headers

namespace mlir::iree_compiler::IREE::TensorExt {

std::optional<SparseRangeResolver> getSparseRangeResolver(Range range) {
  Value upperBound = dyn_cast<Value>(range.size);
  if (!upperBound || isa<BlockArgument>(upperBound)) {
    return std::nullopt;
  }

  SparseOpInterface sparseOp;
  int64_t resultDim;

  // For now just check that the defining operation of the upper bound is a
  // `memref.dim` operation of a result of a sparse operation.
  Value ubSource;
  IntegerAttr dim;
  if (!matchPattern(upperBound, m_Op<memref::DimOp>(matchers::m_Any(&ubSource),
                                                    m_Constant(&dim)))) {
    return std::nullopt;
  }
  sparseOp = ubSource.getDefiningOp<SparseOpInterface>();
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
