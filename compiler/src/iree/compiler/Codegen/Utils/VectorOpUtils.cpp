// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::iree_compiler {

std::pair<int, int> VectorContractOpInfo::getOperandMNIndex() const {
  return std::make_pair(lhsMDims.back(), rhsNDims.back());
}

// Returns the (LHS K, RHS K) dimension index pair.
std::pair<int, int> VectorContractOpInfo::getOperandKIndex() const {
  return std::make_pair(lhsKDim, rhsKDim);
}

// Returns the result (M, N) dimension index pair.
std::pair<int, int> VectorContractOpInfo::getResultMNIndex() const {
  return std::make_pair(outMDims.back(), outNDims.back());
}

VectorContractOpInfo::VectorContractOpInfo(vector::ContractionOp op) {
  contractionDims = *linalg::inferContractionDims(op.getIndexingMapsArray());

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  MLIRContext *ctx = op.getContext();

  for (auto m : contractionDims.m) {
    lhsMDims.push_back(*maps[0].getResultPosition(getAffineDimExpr(m, ctx)));
    outMDims.push_back(*maps[2].getResultPosition(getAffineDimExpr(m, ctx)));
  }
  for (auto n : contractionDims.n) {
    rhsNDims.push_back(*maps[1].getResultPosition(getAffineDimExpr(n, ctx)));
    outNDims.push_back(*maps[2].getResultPosition(getAffineDimExpr(n, ctx)));
  }

  int64_t k = contractionDims.k.back();
  lhsKDim = *maps[0].getResultPosition(getAffineDimExpr(k, ctx));
  rhsKDim = *maps[1].getResultPosition(getAffineDimExpr(k, ctx));
}

} // namespace mlir::iree_compiler
