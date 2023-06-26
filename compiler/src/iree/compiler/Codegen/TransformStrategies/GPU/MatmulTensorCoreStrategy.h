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
#include "iree/compiler/Codegen/TransformStrategies/GPU/CopyMapping.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"

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

  /// Initialize values from the CLI. Set cliOptionsSpecified to true if the
  /// default CLI values have been overriden.
  void initDefaultValues() override;

  LogicalResult validate(const GPUModel &gpuModel) const override;

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

  int64_t numWarpsX() const override {
    assert(numWarps.size() >= 2 && "need at least 2 warp sizes");
    return numWarps[0];
  }
  int64_t numWarpsY() const override {
    assert(numWarps.size() >= 2 && "need at least 2 warp sizes");
    return numWarps[1];
  }

  MappingInfo getBlockMapping() const override {
    return MappingInfo{/*numThreads=*/{},
                       /*tileSizes=*/{blockTileM(), blockTileN()},
                       /*threadMapping=*/{blockY(ctx), blockX(ctx)},
                       /*vectorSize=*/std::nullopt};
  }

  // LHS copy is of size mxk.
  MappingInfo lhsCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/k(),
        /*copySizes=*/ArrayRef<int64_t>{blockTileM(), reductionTileSize},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/lhsElementalBitWidth);
  }
  LogicalResult validateLhsCopyMapping() const override {
    MappingInfo mapping = lhsCopyMapping();
    // It is fine to use fewer threads to copy the LHS.
    if (totalNumThreads() < mapping.numThreads[0] * mapping.numThreads[1]) {
      llvm::errs() << "too many threads used for transferring lhs: "
                   << mapping.numThreads[0] << " * " << mapping.numThreads[1]
                   << " > " << totalNumThreads() << "\n";
      return failure();
    }
    return success();
  }

  // RHS copy is of size kxn.
  MappingInfo rhsCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/n(),
        /*copySizes=*/ArrayRef<int64_t>{reductionTileSize, blockTileN()},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/rhsElementalBitWidth);
  }
  LogicalResult validateRhsCopyMapping() const override {
    MappingInfo mapping = rhsCopyMapping();
    // It is fine to use fewer threads to copy the RHS.
    if (totalNumThreads() < mapping.numThreads[0] * mapping.numThreads[1]) {
      llvm::errs() << "too many threads used for transferring rhs: "
                   << mapping.numThreads[0] << " * " << mapping.numThreads[1]
                   << " > " << totalNumThreads() << "\n";
      return failure();
    }
    return success();
  }

  // RES copy is of size mxn.
  MappingInfo resCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/n(),
        /*copySizes=*/ArrayRef<int64_t>{blockTileM(), blockTileN()},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/resElementalBitWidth);
  }

  LogicalResult validateResCopyMapping() const override {
    MappingInfo mapping = resCopyMapping();
    // It is fine to use fewer threads to copy the RES.
    if (totalNumThreads() < mapping.numThreads[0] * mapping.numThreads[1]) {
      llvm::errs() << "too many threads used for transferring res: "
                   << mapping.numThreads[0] << " * " << mapping.numThreads[1]
                   << " > " << totalNumThreads() << "\n";
      return failure();
    }
    return success();
  }

  // COMPUTE is of size mxn.
  MappingInfo computeMapping() const override {
    if (useFma) {
      // When using FMA we don't need to map to warps, instead just match what
      // the copy does.
      return CopyMapping::getMappingInfo(ctx, totalNumThreads(),
                                         /*alignment=*/n(),
                                         {blockTileM(), blockTileN()});
    }
    return MappingInfo{/*numThreads=*/{numWarpsY(), numWarpsX()},
                       /*tileSizes=*/{},
                       /*threadMapping=*/{warpY(ctx), warpX(ctx)},
                       /*vectorSize=*/std::nullopt};
  }

  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

} // namespace gpu
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
