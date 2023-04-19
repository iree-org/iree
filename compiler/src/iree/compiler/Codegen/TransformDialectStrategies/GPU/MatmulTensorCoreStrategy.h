// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

struct MatmulStrategy {
  MatmulStrategy(MLIRContext *context,
                 const transform_ext::MatchedMatmulCaptures &captures)
      : context(context), captures(captures) {}

  /// Constructor quantities.
  MLIRContext *context;
  transform_ext::MatchedMatmulCaptures captures;

  /// Tile sizes for the workgroup / determines grid size for all known
  /// reduction strategies.
  SmallVector<int64_t> blockTileSizes = {128, 128, 1};
  int64_t reductionTileSize = 16;
  SmallVector<int64_t> numThreads = {32, 4, 1};
  SmallVector<int64_t> numWarps = {2, 2, 1};
  bool useMmaSync = false;
  int64_t pipelineDepth = 3;

  Attribute blockX(MLIRContext *ctx) const {
    return mlir::gpu::GPUBlockMappingAttr::get(context,
                                               mlir::gpu::Blocks::DimX);
  }
  Attribute blockY(MLIRContext *ctx) const {
    return mlir::gpu::GPUBlockMappingAttr::get(context,
                                               mlir::gpu::Blocks::DimY);
  }
  Attribute blockZ(MLIRContext *ctx) const {
    return mlir::gpu::GPUBlockMappingAttr::get(context,
                                               mlir::gpu::Blocks::DimZ);
  }
  Attribute threadX(MLIRContext *ctx) const {
    return mlir::gpu::GPUThreadMappingAttr::get(context,
                                                mlir::gpu::Threads::DimX);
  }
  Attribute threadY(MLIRContext *ctx) const {
    return mlir::gpu::GPUThreadMappingAttr::get(context,
                                                mlir::gpu::Threads::DimY);
  }
  Attribute threadZ(MLIRContext *ctx) const {
    return mlir::gpu::GPUThreadMappingAttr::get(context,
                                                mlir::gpu::Threads::DimZ);
  }
  Attribute warpX(MLIRContext *ctx) const {
    return mlir::gpu::GPUWarpMappingAttr::get(context, mlir::gpu::Warps::DimX);
  }
  Attribute warpY(MLIRContext *ctx) const {
    return mlir::gpu::GPUWarpMappingAttr::get(context, mlir::gpu::Warps::DimY);
  }
  Attribute warpZ(MLIRContext *ctx) const {
    return mlir::gpu::GPUWarpMappingAttr::get(context, mlir::gpu::Warps::DimZ);
  }
  Attribute linearIdX(MLIRContext *ctx) const {
    return mlir::gpu::GPULinearIdMappingAttr::get(context,
                                                  mlir::gpu::LinearId::DimX);
  }
  Attribute linearIdY(MLIRContext *ctx) const {
    return mlir::gpu::GPULinearIdMappingAttr::get(context,
                                                  mlir::gpu::LinearId::DimY);
  }
  Attribute linearIdZ(MLIRContext *ctx) const {
    return mlir::gpu::GPULinearIdMappingAttr::get(context,
                                                  mlir::gpu::LinearId::DimZ);
  }

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
    SmallVector<Attribute> threadMapping;
  };
  MappingInfo getBlockMapping(MLIRContext *ctx) const {
    ArrayRef<int64_t> blocks{blockTileSizes};
    return MappingInfo{SmallVector<int64_t>{blocks.take_front(2)},
                       {blockY(ctx), blockX(ctx)}};
  }
  // LHS copy is of size mxk.
  MappingInfo lhsCopyMapping(MLIRContext *ctx) const {
    assert(reductionTileSize % lhsCopyVectorSize() == 0 &&
           "vector size must divide reductionTileSize");
    int64_t numThreadsK = reductionTileSize / lhsCopyVectorSize();
    assert(totalNumThreads() % numThreadsK == 0 &&
           "num threads must be divisible by num threads along k");
    int64_t numThreadsM = totalNumThreads() / numThreadsK;
    return MappingInfo{{numThreadsM, numThreadsK},
                       {threadX(ctx), threadY(ctx)}};
  }
  // RHS copy is of size kxn.
  MappingInfo rhsCopyMapping(MLIRContext *ctx) const {
    assert(blockTileSizes[1] % rhsCopyVectorSize() == 0 &&
           "vector size must divide blockTileSizes[1]");
    int64_t numThreadsN = blockTileSizes[1] / rhsCopyVectorSize();
    assert(totalNumThreads() % numThreadsN == 0 &&
           "num threads must be divisible by num threads along n");
    int64_t numThreadsK = totalNumThreads() / numThreadsN;
    return MappingInfo{{numThreadsK, numThreadsN},
                       {threadY(ctx), threadX(ctx)}};
  }
  // RES copy is of size mxn.
  MappingInfo resCopyMapping(MLIRContext *ctx) const {
    assert(blockTileSizes[1] % resCopyVectorSize() == 0 &&
           "vector size must divide n");
    int64_t numThreadsN = blockTileSizes[1] / resCopyVectorSize();
    assert(totalNumThreads() % numThreadsN == 0 &&
           "num threads must be divisible by num threads along n");
    int64_t numThreadsM = totalNumThreads() / numThreadsN;
    return MappingInfo{{numThreadsM, numThreadsN},
                       {threadY(ctx), threadX(ctx)}};
  }
  // COMPUTE is of size mxn.
  MappingInfo computeMapping(MLIRContext *ctx) const {
    ArrayRef<int64_t> warps{numWarps};
    return MappingInfo{SmallVector<int64_t>{warps.take_front(2)},
                       {warpY(ctx), warpX(ctx)}};
  }
};

void buildMatmulTensorCoreStrategy(ImplicitLocOpBuilder &b, Value variantH,
                                   const MatmulStrategy &strategy);

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
