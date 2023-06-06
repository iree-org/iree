// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/AbstractGemmLikeStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Support/LogicalResult.h"

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

  int64_t blockTileM() const override {
    assert(blockTileSizes.size() >= 2 && "need at least 2 tile sizes");
    return blockTileSizes[0];
  }
  int64_t blockTileN() const override {
    assert(blockTileSizes.size() >= 2 && "need at least 2 tile sizes");
    return blockTileSizes[1];
  }

  int64_t numWarpsM() const override {
    assert(numWarps.size() >= 2 && "need at least 2 warp sizes");
    return numWarps[0];
  }
  int64_t numWarpsN() const override {
    assert(numWarps.size() >= 2 && "need at least 2 warp sizes");
    return numWarps[1];
  }

  using AbstractGemmLikeStrategy::MappingInfo;

  MappingInfo getBlockMapping() const override {
    return MappingInfo{/*numThreads=*/{},
                       /*tileSizes=*/{blockTileM(), blockTileN()},
                       /*threadMapping=*/{blockY(ctx), blockX(ctx)}};
  }

  // LHS copy is of size mxk.
  MappingInfo lhsCopyMapping() const override {
    int64_t numThreadsK = reductionTileSize / lhsCopyVectorSize();
    int64_t numThreadsM = totalNumThreads() / numThreadsK;
    return MappingInfo{/*numThreads=*/{numThreadsM, numThreadsK},
                       /*tileSizes=*/
                       {std::max(1l, blockTileM() / numThreadsM),
                        std::max(1l, reductionTileSize / numThreadsK)},
                       /*threadMapping=*/{linearIdX(ctx), linearIdY(ctx)}};
  }

  LogicalResult validateLhsCopyMapping() const override {
    MappingInfo mapping = lhsCopyMapping();
    bool cond1 = (reductionTileSize % lhsCopyVectorSize() == 0);
    bool cond2 =
        (totalNumThreads() == mapping.numThreads[0] * mapping.numThreads[1]);
    bool cond3 = (blockTileM() % mapping.numThreads[0] == 0);
    bool cond4 = (reductionTileSize % mapping.numThreads[1] == 0);
    return success(cond1 && cond2 && cond3 && cond4);
  }

  // RHS copy is of size kxn.
  MappingInfo rhsCopyMapping() const override {
    int64_t numThreadsN = blockTileN() / rhsCopyVectorSize();
    int64_t numThreadsK = totalNumThreads() / numThreadsN;
    return MappingInfo{/*numThreads=*/{numThreadsK, numThreadsN},
                       /*tileSizes=*/
                       {std::max(1l, reductionTileSize / numThreadsK),
                        std::max(1l, blockTileN() / numThreadsN)},
                       /*threadMapping=*/{linearIdY(ctx), linearIdX(ctx)}};
  }

  LogicalResult validateRhsCopyMapping() const override {
    MappingInfo mapping = rhsCopyMapping();
    bool cond1 = (blockTileN() % rhsCopyVectorSize() == 0);
    bool cond2 =
        (totalNumThreads() == mapping.numThreads[0] * mapping.numThreads[1]);
    bool cond3 = (reductionTileSize % mapping.numThreads[0] == 0);
    bool cond4 = (blockTileN() % mapping.numThreads[1] == 0);
    return success(cond1 && cond2 && cond3 && cond4);
  }

  // RES copy is of size mxn.
  MappingInfo resCopyMapping() const override {
    int64_t numThreadsN = blockTileN() / resCopyVectorSize();
    int64_t numThreadsM = totalNumThreads() / numThreadsN;
    return MappingInfo{/*numThreads=*/{numThreadsM, numThreadsN},
                       /*tileSizes=*/
                       {std::max(1l, blockTileM() / numThreadsM),
                        std::max(1l, blockTileN() / numThreadsN)},
                       /*threadMapping=*/{linearIdY(ctx), linearIdX(ctx)}};
  }

  LogicalResult validateResCopyMapping() const override {
    MappingInfo mapping = resCopyMapping();
    bool cond1 = (blockTileN() % resCopyVectorSize() == 0);
    bool cond2 =
        (totalNumThreads() == mapping.numThreads[0] * mapping.numThreads[1]);
    bool cond3 = (blockTileM() % mapping.numThreads[0] == 0);
    bool cond4 = (blockTileN() % mapping.numThreads[1] == 0);
    return success(cond1 && cond2 && cond3 && cond4);
  }

  // COMPUTE is of size mxn.
  MappingInfo computeMapping() const override {
    // Warps along M and N need to properly be ordered along the X and Y
    // dimensions respectively, otherwise we would improperly generate
    // predicated code.
    return MappingInfo{/*numThreads=*/{numWarpsM(), numWarpsN()},
                       /*tileSizes=*/{},
                       /*threadMapping=*/{warpX(ctx), warpY(ctx)}};
  }

  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
