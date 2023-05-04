// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

/// Base quantities generally useful for all GPU strategies.
// TODO: refactor into a common place.
struct StrategyBase {
  StrategyBase(MLIRContext *ctx) : ctx(ctx) {}

  /// Constructor quantities.
  MLIRContext *ctx;

  Attribute blockX() const {
    return mlir::gpu::GPUBlockMappingAttr::get(ctx, mlir::gpu::Blocks::DimX);
  }
  Attribute blockY() const {
    return mlir::gpu::GPUBlockMappingAttr::get(ctx, mlir::gpu::Blocks::DimY);
  }
  Attribute blockZ() const {
    return mlir::gpu::GPUBlockMappingAttr::get(ctx, mlir::gpu::Blocks::DimZ);
  }
  Attribute threadX() const {
    return mlir::gpu::GPUThreadMappingAttr::get(ctx, mlir::gpu::Threads::DimX);
  }
  Attribute threadY() const {
    return mlir::gpu::GPUThreadMappingAttr::get(ctx, mlir::gpu::Threads::DimY);
  }
  Attribute threadZ() const {
    return mlir::gpu::GPUThreadMappingAttr::get(ctx, mlir::gpu::Threads::DimZ);
  }
  Attribute warpX() const {
    return mlir::gpu::GPUWarpMappingAttr::get(ctx, mlir::gpu::Warps::DimX);
  }
  Attribute warpY() const {
    return mlir::gpu::GPUWarpMappingAttr::get(ctx, mlir::gpu::Warps::DimY);
  }
  Attribute warpZ() const {
    return mlir::gpu::GPUWarpMappingAttr::get(ctx, mlir::gpu::Warps::DimZ);
  }
  Attribute linearIdX() const {
    return mlir::gpu::GPULinearIdMappingAttr::get(ctx,
                                                  mlir::gpu::LinearId::DimX);
  }
  Attribute linearIdY() const {
    return mlir::gpu::GPULinearIdMappingAttr::get(ctx,
                                                  mlir::gpu::LinearId::DimY);
  }
  Attribute linearIdZ() const {
    return mlir::gpu::GPULinearIdMappingAttr::get(ctx,
                                                  mlir::gpu::LinearId::DimZ);
  }
};

struct MatmulStrategy : StrategyBase {
  MatmulStrategy(MLIRContext *context,
                 const transform_ext::MatchedMatmulCaptures &captures)
      : StrategyBase(context), captures(captures) {
    initDefaultValues();
  }

  /// Constructor quantities.
  transform_ext::MatchedMatmulCaptures captures;

  /// Tile sizes for the workgroup / determines grid size for all known
  /// reduction strategies. The initial values are set by initDefaultValues();
  SmallVector<int64_t> blockTileSizes;
  int64_t reductionTileSize;
  SmallVector<int64_t> numThreads;
  SmallVector<int64_t> numWarps;
  bool useAsyncCopies;
  bool useMmaSync;
  int64_t pipelineDepth;

  void initDefaultValues();

  int64_t m() const {
    assert(captures.matmulOpSizes.size() == 3 && "need 3 sizes");
    return captures.matmulOpSizes[0];
  }
  int64_t n() const {
    assert(captures.matmulOpSizes.size() == 3 && "need 3 sizes");
    return captures.matmulOpSizes[1];
  }
  int64_t k() const {
    assert(captures.matmulOpSizes.size() == 3 && "need 3 sizes");
    return captures.matmulOpSizes[2];
  }
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
  MappingInfo getBlockMapping() const {
    return MappingInfo{/*numThreads=*/{},
                       /*tileSizes=*/{blockTileSizes[0], blockTileSizes[1]},
                       /*threadMapping=*/{blockY(), blockX()}};
  }
  // LHS copy is of size mxk.
  MappingInfo lhsCopyMapping() const {
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
        /*threadMapping=*/{linearIdX(), linearIdY()}};
  }
  // RHS copy is of size kxn.
  MappingInfo rhsCopyMapping() const {
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
        /*threadMapping=*/{linearIdY(), linearIdX()}};
  }
  // RES copy is of size mxn.
  MappingInfo resCopyMapping() const {
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
        /*threadMapping=*/{linearIdY(), linearIdX()}};
  }
  // COMPUTE is of size mxn.
  MappingInfo computeMapping() const {
    return MappingInfo{/*numThreads=*/{numWarps[0], numWarps[1]},
                       /*tileSizes=*/{},
                       /*threadMapping=*/{warpY(), warpX()}};
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
