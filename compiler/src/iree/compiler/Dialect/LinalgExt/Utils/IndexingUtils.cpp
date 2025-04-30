// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/raw_ostream.h"

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

void AttentionOpDetail::inferFromIndexingMaps(AffineMap qMap, AffineMap kMap,
                                              AffineMap vMap, AffineMap oMap) {
  // Q   = B x M x K1
  // K   = B x K2 x K1
  // V   = B x K2 x N
  // O   = B x M x N
  llvm::SmallDenseSet<int64_t> qSet = findPermutationsIndexingOperand(qMap);
  llvm::SmallDenseSet<int64_t> kSet = findPermutationsIndexingOperand(kMap);
  llvm::SmallDenseSet<int64_t> vSet = findPermutationsIndexingOperand(vMap);
  llvm::SmallDenseSet<int64_t> oSet = findPermutationsIndexingOperand(oMap);

  // B = (Q & V) U (K & O)
  llvm::SmallDenseSet<int64_t> b1Set = qSet;
  llvm::set_intersect(b1Set, vSet);
  llvm::SmallDenseSet<int64_t> b2Set = kSet;
  llvm::set_intersect(b2Set, oSet);
  llvm::SmallDenseSet<int64_t> bSet = b1Set;
  llvm::set_union(bSet, b2Set);

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

  // Sort to ensure that dims are in outermost to innermost order.
  llvm::sort(batch);
  llvm::sort(m);
  llvm::sort(k1);
  llvm::sort(k2);
  llvm::sort(n);
}

FailureOr<AttentionOpDetail> AttentionOpDetail::get(AffineMap qMap,
                                                    AffineMap kMap,
                                                    AffineMap vMap,
                                                    AffineMap oMap) {
  AttentionOpDetail opInfo;
  opInfo.inferFromIndexingMaps(qMap, kMap, vMap, oMap);
  opInfo.context = qMap.getContext();
  opInfo.domainRank = qMap.getNumDims();
  return opInfo;
}

AffineMap AttentionOpDetail::getSMap() const {
  // We need to create an indexing map for the intermediate result of first
  // matmul. There could be other options, but we choose to create a standard
  // indexing map:
  //   SMap = (batch, m, k1, k2, n) -> (batch, m, k2)
  AffineMap sMap = AffineMap::get(/*dimCount=*/getDomainRank(),
                                  /*symbolCount=*/0, getContext());
  for (auto dim :
       llvm::concat<const int64_t>(getBatchDims(), getMDims(), getK2Dims())) {
    AffineExpr dimExpr = getAffineDimExpr(dim, getContext());
    sMap = sMap.insertResult(dimExpr, sMap.getNumResults());
  }
  return sMap;
}

}; // namespace mlir::iree_compiler::IREE::LinalgExt
