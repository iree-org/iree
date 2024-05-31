// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_UTILS_INDEXINGUTILS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_UTILS_INDEXINGUTILS_H_

#include "mlir/IR/AffineMap.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Attention has the following shaped inputs:
///
/// Q  : B x M x K1
/// K  : B x K2 x K1
/// V  : B x K2 x N
/// res: B x M x N
///
/// For the purposes of Tiling, Attention can be thought of:
///
/// QKT = Q @ K.T
/// S = reduce QKT dim=2 keep_dims=True
/// att = S @ V
///
/// By this defination, K1 and K2 can be seen as reduction dimensions and
/// B, M, N can be seen as parallel dimensions.
///
/// Generally, K1 and N are really small (64/128), K2 and M are really large
/// (16k, 64k, 128k), and B is batch size and depends on input batch size.
///
/// Tiling on parallel dimensions is trivial.
///
/// Tiling on reduction dimensions on the other hand is much harder.
/// TileAndDecomposeAttention has an implementation to tile on K2 dimension
/// based on an online softmax technique proposed by Flash Attention V2
/// (https://arxiv.org/abs/2307.08691).
///
/// Tiling on K1 is generally not done because it's so small and is non-trivial.
class AttentionOpDetail {
public:
  static FailureOr<AttentionOpDetail> get(ArrayRef<AffineMap> indexingMaps);

  int64_t getDomainRank() const { return maps[0].getNumDims(); }
  ArrayRef<int64_t> getBatchDims() const { return batch; }
  ArrayRef<int64_t> getMDims() const { return m; }
  ArrayRef<int64_t> getK1Dims() const { return k1; }
  ArrayRef<int64_t> getK2Dims() const { return k2; }
  ArrayRef<int64_t> getNDims() const { return n; }

  ArrayRef<AffineMap> getIndexingMaps() const { return maps; }

  AffineMap getSMap() const;

  std::tuple<AffineMap, AffineMap, AffineMap> getQKMatmulCompressedMaps() const;

  std::tuple<AffineMap, AffineMap, AffineMap> getSVMatmulCompressedMaps() const;

private:
  void inferFromIndexingMaps(ArrayRef<AffineMap> indexingMaps);

  MLIRContext *getContext() const { return maps[0].getContext(); }

  SmallVector<int64_t> batch;
  SmallVector<int64_t> m;
  SmallVector<int64_t> k1;
  SmallVector<int64_t> k2;
  SmallVector<int64_t> n;

  SmallVector<AffineMap> maps;
};

}; // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_UTILS_INDEXINGUTILS_H_
