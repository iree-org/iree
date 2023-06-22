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
      : AbstractGemmLikeStrategy(),
        ctx(context),
        captures(captures),
        cliOptionsSpecified(false) {
    initDefaultValues();
  }

  MatmulStrategy(const MatmulStrategy &) = default;
  MatmulStrategy &operator=(const MatmulStrategy &) = default;

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedMatmulCaptures captures;

  /// Encodes whether the user has specified any CLI options. When true, the
  /// strategy should just run what was specified and is not allowed to
  /// override the user's choices.
  bool cliOptionsSpecified;

  /// Initialize values from the CLI. Set cliOptionsSpecified to true if the
  /// default CLI values have been overriden.
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
        /*copySizes=*/ArrayRef<int64_t>{blockTileM(), reductionTileSize});
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
        /*copySizes=*/ArrayRef<int64_t>{reductionTileSize, blockTileN()});
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
    return CopyMapping::getMappingInfo(ctx, totalNumThreads(),
                                       /*alignment=*/n(),
                                       {blockTileM(), blockTileN()});
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

  LogicalResult validate() const override {
    if (totalNumThreads() != totalNumWarps() * kCudaWarpSize) {
      llvm::errs() << "Number of threads specified by warps must match total "
                      "number of threads\n";
      return failure();
    }
    if (m() < blockTileM()) {
      llvm::errs() << "m(" << m() << ") < blockTileM(" << blockTileM() << ") ";
      llvm::errs() << "this is at risk of not vectorizing and is NYI";
      return failure();
    }
    if (n() < blockTileN()) {
      llvm::errs() << "n(" << n() << ") < blockTileN(" << blockTileN() << ") ";
      llvm::errs() << "this is at risk of not vectorizing and is NYI";
      return failure();
    }
    if (k() < reductionTileSize) {
      llvm::errs() << "k(" << k() << ") < reductionTileSize("
                   << reductionTileSize << ") ";
      llvm::errs() << "this is at risk of not vectorizing and is NYI";
      return failure();
    }

    if (failed(validateLhsCopyMapping())) {
      llvm::errs() << "invalid lhs copy mapping";
      return failure();
    }
    if (failed(validateRhsCopyMapping())) {
      llvm::errs() << "invalid rhs copy mapping";
      return failure();
    }
    if (failed(validateResCopyMapping())) {
      llvm::errs() << "invalid res copy mapping";
      return failure();
    }

    if (pipelineDepth > 1 && reductionTileSize * pipelineDepth > k()) {
      llvm::errs() << "pipeline depth too large for reduction tile size";
      return failure();
    }

    bool oneOption =
        (useMmaSync ^ useWmma ^ useFma) && !(useMmaSync && useWmma && useFma);
    if (!oneOption) {
      llvm::errs() << "at most one of useMmaSync, useWmma, useFma can be true";
      return failure();
    }

    if (useMmaSync) {
      if (blockTileM() < kMinMmaSyncMinM) {
        llvm::errs() << "mma.sync requires at least " << kMinMmaSyncMinM
                     << " block tile size in M";
        return failure();
      }
      if (blockTileN() < kMinMmaSyncMinN) {
        llvm::errs() << "mma.sync requires at least " << kMinMmaSyncMinN
                     << " block tile size in N";
        return failure();
      }
      if (reductionTileSize < kMinMmaSyncMinK) {
        llvm::errs() << "mma.sync requires at least " << kMinMmaSyncMinK
                     << " block tile size in K";
        return failure();
      }
      if (pipelineDepth > 1 && pipelineDepth < kMinMmaSyncPipelineDepth) {
        llvm::errs() << "mma.sync pipelining requires at least "
                     << kMinMmaSyncPipelineDepth << " stages";
        return failure();
      }
      if (pipelineDepth > 1 && reductionTileSize * kMinMmaSyncGroups > k()) {
        llvm::errs() << "mma.sync pipelining requires at least "
                     << kMinMmaSyncGroups << " k groups";
        return failure();
      }
    } else if (useWmma) {
      if (blockTileM() < kMinWmmaMinM) {
        llvm::errs() << "wmma requires at least " << kMinWmmaMinM
                     << " block tile size in M";
        return failure();
      }
      if (blockTileN() < kMinWmmaMinN) {
        llvm::errs() << "wmma requires at least " << kMinWmmaMinN
                     << " block tile size in N";
        return failure();
      }
      if (reductionTileSize < kMinWmmaMinK) {
        llvm::errs() << "wmma requires at least " << kMinWmmaMinK
                     << " block tile size in K";
        return failure();
      }
    }
    return success();
  }

  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
