// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace llvm {
class raw_ostream;
}

namespace mlir {

class OpBuilder;

namespace iree_compiler {
namespace gpu {

struct GPUModel;

struct AbstractGemmLikeStrategy {
  AbstractGemmLikeStrategy() {}

  virtual ~AbstractGemmLikeStrategy();

  void initDefaultValues(bool optUseMmaSync = false);

  /// Tile sizes for the workgroup / determines grid size for all known
  /// reduction strategies. The initial values are set by initDefaultValues();
  SmallVector<int64_t> blockTileSizes;
  int64_t reductionTileSize;
  SmallVector<int64_t> numThreads;
  SmallVector<int64_t> numWarps;

  bool useAsyncCopies;
  bool useMmaSync;
  int64_t pipelineDepth;

  SmallVector<Type> paddingValueTypes;
  SmallVector<int64_t> paddingDimensions;
  SmallVector<int64_t> packingDimensions;

  ArrayAttr getZeroPadAttrFromElementalTypes(OpBuilder &b) const;

  int64_t lhsElementalBitWidth = 32;
  int64_t rhsElementalBitWidth = 32;
  int64_t resElementalBitWidth = 32;

  virtual int64_t m() const = 0;
  virtual int64_t n() const = 0;
  virtual int64_t k() const = 0;

  virtual int64_t blockTileM() const = 0;
  virtual int64_t blockTileN() const = 0;
  virtual int64_t blockTileK() const = 0;

  bool alignedLhs() const { return m() % 64 == 0 && k() % 16 == 0; }
  bool alignedRhs() const { return n() % 64 == 0 && k() % 16 == 0; }
  bool alignedRes() const { return m() % 64 == 0 && n() % 64 == 0; }

  /// Common values based on derived quantities.
  int64_t totalNumThreads() const {
    int64_t res = 1;
    for (auto v : numThreads) res *= v;
    return res;
  }
  int64_t totalNumWarps() const {
    int64_t res = 1;
    for (auto v : numWarps) res *= v;
    return res;
  }

  // Copy vector sizes based on inner most K/N dims.
  int64_t lhsCopyVectorSize() const;
  int64_t rhsCopyVectorSize() const;
  int64_t resCopyVectorSize() const;

  struct MappingInfo {
    SmallVector<int64_t> numThreads;
    // Explicitly computing the tileSizes is only needed until masked
    // vectorization properly computes the bounds automatically.
    SmallVector<int64_t> tileSizes;
    SmallVector<Attribute> threadMapping;
  };

  virtual MappingInfo getBlockMapping() const = 0;
  virtual MappingInfo lhsCopyMapping() const = 0;
  virtual MappingInfo rhsCopyMapping() const = 0;
  virtual MappingInfo resCopyMapping() const = 0;
  virtual MappingInfo computeMapping() const = 0;

  virtual void print(llvm::raw_ostream &os) const;
  virtual LLVM_DUMP_METHOD void dump() const;
};

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_
