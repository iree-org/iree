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

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_INDEXING_H_
