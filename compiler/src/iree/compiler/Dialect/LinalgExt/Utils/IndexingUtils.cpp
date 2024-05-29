// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "llvm/ADT/SetOperations.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

namespace {

static llvm::SmallDenseSet<int64_t>
findPermutationsIndexingOperand(AffineMap indexingMap) {
  assert(indexingMap.isProjectedPermutation() &&
         "indexing maps for attention must be permutations.");
  llvm::SmallDenseSet<int64_t> res;
  for (AffineExpr e : indexingMap.getResults()) {
    auto d = cast<AffineDimExpr>(e);
    res.insert(d.getPosition());
  }
  return res;
}

}; // namespace

void AttentionOpDetail::inferFromIndexingMaps(
    ArrayRef<AffineMap> indexingMaps) {
  assert(indexingMaps.size() >= 4);
  AffineMap qMap = indexingMaps[0];
  AffineMap kMap = indexingMaps[1];
  AffineMap vMap = indexingMaps[2];
  AffineMap resMap = indexingMaps[3];

  // Q   = B x M x K1
  // K   = B x K2 x K1
  // V   = B x K2 x N
  // res = B x M x N
  llvm::SmallDenseSet<int64_t> qSet = findPermutationsIndexingOperand(qMap);
  llvm::SmallDenseSet<int64_t> vSet = findPermutationsIndexingOperand(vMap);
  llvm::SmallDenseSet<int64_t> kSet = findPermutationsIndexingOperand(kMap);
  llvm::SmallDenseSet<int64_t> resSet = findPermutationsIndexingOperand(resMap);

  // B = Q & K & V
  llvm::SmallDenseSet<int64_t> bSet = qSet;
  llvm::set_intersect(bSet, vSet);
  llvm::set_intersect(bSet, kSet);

  // K1 = Q & K - B
  llvm::SmallDenseSet<int64_t> k1Set = qSet;
  llvm::set_intersect(k1Set, kSet);
  llvm::set_subtract(k1Set, bSet);

  // K2 = K - B - K1
  llvm::SmallDenseSet<int64_t> k2Set = kSet;
  llvm::set_subtract(k2Set, bSet);
  llvm::set_subtract(k2Set, k1Set);

  // M = Q - B - K1
  llvm::SmallDenseSet<int64_t> mSet = qSet;
  llvm::set_subtract(mSet, bSet);
  llvm::set_subtract(mSet, k1Set);

  // N = V - B - K2
  llvm::SmallDenseSet<int64_t> nSet = vSet;
  llvm::set_subtract(nSet, bSet);
  llvm::set_subtract(nSet, k2Set);

  batch = SmallVector<int64_t>(bSet.begin(), bSet.end());
  m = SmallVector<int64_t>(mSet.begin(), mSet.end());
  k1 = SmallVector<int64_t>(k1Set.begin(), k1Set.end());
  k2 = SmallVector<int64_t>(k2Set.begin(), k2Set.end());
  n = SmallVector<int64_t>(nSet.begin(), nSet.end());
}

FailureOr<AttentionOpDetail>
AttentionOpDetail::get(ArrayRef<AffineMap> indexingMaps) {
  if (indexingMaps.size() != 4 && indexingMaps.size() != 6) {
    return failure();
  }

  AttentionOpDetail opInfo;
  opInfo.inferFromIndexingMaps(indexingMaps);
  return opInfo;
}

}; // namespace mlir::iree_compiler::IREE::LinalgExt
