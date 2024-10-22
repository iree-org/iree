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
  return std::make_pair(lhsKDim.back(), rhsKDim.back());
}

// Returns the result (M, N) dimension index pair.
std::pair<int, int> VectorContractOpInfo::getResultMNIndex() const {
  return std::make_pair(outMDims.back(), outNDims.back());
}

FailureOr<VectorContractOpInfo>
VectorContractOpInfo::inferFromIndexingMaps(ArrayRef<AffineMap> maps) {
  // Ensure all maps are projected permutations.
  if (!llvm::all_of(maps, [](AffineMap map) {
        return map.isProjectedPermutation(/*allowZeroInResults=*/true);
      })) {
    return failure();
  }

  auto maybeContractionDims = linalg::inferContractionDims(maps);
  if (failed(maybeContractionDims)) {
    return failure();
  }

  auto contractionDims = maybeContractionDims.value();
  MLIRContext *ctx = maps[0].getContext();
  VectorContractOpInfo opInfo;
  for (auto m : contractionDims.m) {
    opInfo.lhsMDims.push_back(
        *maps[0].getResultPosition(getAffineDimExpr(m, ctx)));
    opInfo.outMDims.push_back(
        *maps[2].getResultPosition(getAffineDimExpr(m, ctx)));
  }
  for (auto n : contractionDims.n) {
    opInfo.rhsNDims.push_back(
        *maps[1].getResultPosition(getAffineDimExpr(n, ctx)));
    opInfo.outNDims.push_back(
        *maps[2].getResultPosition(getAffineDimExpr(n, ctx)));
  }
  for (auto k : contractionDims.k) {
    opInfo.lhsKDim.push_back(
        *maps[0].getResultPosition(getAffineDimExpr(k, ctx)));
    opInfo.rhsKDim.push_back(
        *maps[1].getResultPosition(getAffineDimExpr(k, ctx)));
  }

  opInfo.lhsUnitDims = maps[0].getBroadcastDims();
  opInfo.rhsUnitDims = maps[1].getBroadcastDims();
  opInfo.accUnitDims = maps[2].getBroadcastDims();

  opInfo.contractionDims = contractionDims;

  return opInfo;
}

} // namespace mlir::iree_compiler
