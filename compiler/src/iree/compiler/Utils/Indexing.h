// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_INDEXING_H_
#define IREE_COMPILER_UTILS_INDEXING_H_

#include <algorithm>
#include <iterator>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler {

// Construct IR that extracts the linear index form a multi-index according to
// a shape.
inline OpFoldResult linearIndexFromShape(ArrayRef<OpFoldResult> multiIndex,
                                         ArrayRef<OpFoldResult> shape,
                                         ImplicitLocOpBuilder &builder) {
  assert(multiIndex.size() == shape.size());
  SmallVector<AffineExpr> shapeAffine;
  for (size_t i = 0; i < shape.size(); ++i) {
    shapeAffine.push_back(getAffineSymbolExpr(i, builder.getContext()));
  }

  SmallVector<AffineExpr> stridesAffine = computeStrides(shapeAffine);
  SmallVector<OpFoldResult> strides;
  strides.reserve(stridesAffine.size());
  llvm::transform(stridesAffine, std::back_inserter(strides),
                  [&builder, &shape](AffineExpr strideExpr) {
                    return affine::makeComposedFoldedAffineApply(
                        builder, builder.getLoc(), strideExpr, shape);
                  });

  auto &&[linearIndexExpr, multiIndexAndStrides] = computeLinearIndex(
      OpFoldResult(builder.getIndexAttr(0)), strides, multiIndex);
  return affine::makeComposedFoldedAffineApply(
      builder, builder.getLoc(), linearIndexExpr, multiIndexAndStrides);
}

/// Given a set of dimension `sizes` and `strides`, compute a `basis` - a list
/// of sizes suitable for passing to an `affine.delinearize_index` op without
/// outer bound that would produce the same effects as a `(x / strides[i]) %
/// sizes[i]` delinearization. The permutation mapping each dimension in `sizes`
/// to its corresponding delinearization result is in `dimToResult`.
///
/// That is, if there are `N` elements in the shape, after one builds
///
///     %r:(N+1) affine.delinearize_index %x by (basis) : index, index, ...
///
/// then, for all `i`
///
///    %r#(dimToResult[i]) == (%x floordiv strides[i]) mod sizes[i]
///
/// For example, sizes = {4, 16}, strides = {1, 4} will return basis = {4, 1}
/// and dimToResult = {2, 1}
///
/// This function does handle the case where the strides "skip over" elements.
/// For example, sizes = {16, 4} strides = {8, 1} will yield basis = {16, 2, 4}
/// and dimToResult = {1, 3}.
///
/// If a basis can't be found - for instance, if we have sizes = {4, 4}
/// strides = {3, 1}, returns failure().
///
/// As a special case, dimensions with stride 0 are treated as size-1
/// dimensions that are placed at the end of the delinearization, from where
/// they will canonicalize to 0.
LogicalResult basisFromSizesStrides(ArrayRef<int64_t> sizes,
                                    ArrayRef<int64_t> strides,
                                    SmallVectorImpl<int64_t> &basis,
                                    SmallVectorImpl<size_t> &dimToResult);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_INDEXING_H_
