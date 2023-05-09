// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/AbstractGpuStrategy.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

using iree_compiler::gpu::StrategyBase;

struct AbstractGemmLikeStrategy : StrategyBase {
  AbstractGemmLikeStrategy(MLIRContext *context) : StrategyBase(context) {}

  virtual ~AbstractGemmLikeStrategy();

  /// Tile sizes for the workgroup / determines grid size for all known
  /// reduction strategies. The initial values are set by initDefaultValues();
  SmallVector<int64_t> blockTileSizes;
  int64_t reductionTileSize;
  SmallVector<int64_t> numThreads;
  SmallVector<int64_t> numWarps;

  bool useAsyncCopies;
  bool useMmaSync;
  int64_t pipelineDepth;

  SmallVector<float> paddingValues;
  SmallVector<int64_t> paddingDimensions;
  SmallVector<int64_t> packingDimensions;

  virtual int64_t m() const = 0;
  virtual int64_t n() const = 0;
  virtual int64_t k() const = 0;

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
  int64_t lhsCopyVectorSize() const {
    if (k() % 4 == 0) return 4;
    if (k() % 2 == 0) return 2;
    return 1;
  }
  int64_t rhsCopyVectorSize() const {
    if (n() % 4 == 0) return 4;
    if (n() % 2 == 0) return 2;
    return 1;
  }
  int64_t resCopyVectorSize() const { return rhsCopyVectorSize(); }

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

  virtual void print(llvm::raw_ostream &os) const = 0;
  virtual LLVM_DUMP_METHOD void dump() const = 0;
};

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_
