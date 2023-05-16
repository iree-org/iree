// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/AbstractGemmLikeStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

class MatmulStrategy : public AbstractGemmLikeStrategy {
 public:
  MatmulStrategy(MLIRContext *context,
                 const transform_ext::MatchedMatmulCaptures &captures)
      : AbstractGemmLikeStrategy(), ctx(context), captures(captures) {
    initDefaultValues();
  }

  MatmulStrategy(const MatmulStrategy &) = default;
  MatmulStrategy &operator=(const MatmulStrategy &) = default;

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedMatmulCaptures captures;

  void initDefaultValues();

  LogicalResult verify() const;

  int64_t m() const override {
    assert(captures.matmulOpSizes.size() == 3 && "need 3 sizes");
    return captures.matmulOpSizes[0];
  }
  int64_t n() const override {
    assert(captures.matmulOpSizes.size() == 3 && "need 3 sizes");
    return captures.matmulOpSizes[1];
  }
  int64_t k() const override {
    assert(captures.matmulOpSizes.size() == 3 && "need 3 sizes");
    return captures.matmulOpSizes[2];
  }

  using AbstractGemmLikeStrategy::MappingInfo;

  MappingInfo getBlockMapping() const override {
    return MappingInfo{/*numThreads=*/{},
                       /*tileSizes=*/{blockTileSizes[0], blockTileSizes[1]},
                       /*threadMapping=*/{blockY(ctx), blockX(ctx)}};
  }

  // LHS copy is of size mxk.
  MappingInfo lhsCopyMapping() const override {
    assert(reductionTileSize % lhsCopyVectorSize() == 0 &&
           "vector size must divide reductionTileSize");
    int64_t numThreadsK = reductionTileSize / lhsCopyVectorSize();
    assert(totalNumThreads() % numThreadsK == 0 &&
           "num threads must be divisible by num threads along k");
    int64_t numThreadsM = totalNumThreads() / numThreadsK;
    assert(blockTileSizes[0] % numThreadsM == 0 &&
           "blockTileSizes[0] must be divisible by numThreadsM");
    assert(reductionTileSize % numThreadsK == 0 &&
           "reductionTileSize must be divisible by numThreadsK");
    return MappingInfo{
        /*numThreads=*/{numThreadsM, numThreadsK},
        /*tileSizes=*/
        {blockTileSizes[0] / numThreadsM, reductionTileSize / numThreadsK},
        /*threadMapping=*/{linearIdX(ctx), linearIdY(ctx)}};
  }
  // RHS copy is of size kxn.
  MappingInfo rhsCopyMapping() const override {
    assert(blockTileSizes[1] % rhsCopyVectorSize() == 0 &&
           "vector size must divide blockTileSizes[1]");
    int64_t numThreadsN = blockTileSizes[1] / rhsCopyVectorSize();
    assert(totalNumThreads() % numThreadsN == 0 &&
           "num threads must be divisible by num threads along n");
    int64_t numThreadsK = totalNumThreads() / numThreadsN;
    assert(reductionTileSize % numThreadsK == 0 &&
           "blockTileSizes[0] must be divisible by numThreadsK");
    assert(blockTileSizes[1] % numThreadsN == 0 &&
           "reductionTileSize must be divisible by numThreadsN");
    return MappingInfo{
        /*numThreads=*/{numThreadsK, numThreadsN},
        /*tileSizes=*/
        {reductionTileSize / numThreadsK, blockTileSizes[1] / numThreadsN},
        /*threadMapping=*/{linearIdY(ctx), linearIdX(ctx)}};
  }
  // RES copy is of size mxn.
  MappingInfo resCopyMapping() const override {
    assert(blockTileSizes[1] % resCopyVectorSize() == 0 &&
           "vector size must divide n");
    int64_t numThreadsN = blockTileSizes[1] / resCopyVectorSize();
    assert(totalNumThreads() % numThreadsN == 0 &&
           "num threads must be divisible by num threads along n");
    int64_t numThreadsM = totalNumThreads() / numThreadsN;
    assert(blockTileSizes[0] % numThreadsM == 0 &&
           "blockTileSizes[0] must be divisible by numThreadsM");
    assert(blockTileSizes[1] % numThreadsN == 0 &&
           "blockTileSizes[1] must be divisible by numThreadsN");
    return MappingInfo{
        /*numThreads=*/{numThreadsM, numThreadsN},
        /*tileSizes=*/
        {blockTileSizes[0] / numThreadsM, blockTileSizes[1] / numThreadsN},
        /*threadMapping=*/{linearIdY(ctx), linearIdX(ctx)}};
  }
  // COMPUTE is of size mxn.
  MappingInfo computeMapping() const override {
    return MappingInfo{/*numThreads=*/{numWarps[0], numWarps[1]},
                       /*tileSizes=*/{},
                       /*threadMapping=*/{warpY(ctx), warpX(ctx)}};
  }

  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

void buildMatmulTensorCoreStrategy(ImplicitLocOpBuilder &b, Value variantH,
                                   const MatmulStrategy &strategy);

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
