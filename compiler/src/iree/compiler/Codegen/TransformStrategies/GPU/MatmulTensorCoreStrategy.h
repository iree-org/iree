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

namespace mlir::iree_compiler::gpu {

struct GPUModel;

class MatmulStrategy : public AbstractGemmLikeStrategy {
public:
  MatmulStrategy(MLIRContext *context,
                 const transform_ext::MatchedMatmulCaptures &captures,
                 const GPUModel &gpuModel)
      : AbstractGemmLikeStrategy(gpuModel), ctx(context), captures(captures) {
    initDefaultValues(gpuModel);
  }

  MatmulStrategy(const MatmulStrategy &) = default;
  MatmulStrategy &operator=(const MatmulStrategy &) = default;

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedMatmulCaptures captures;

  /// Initialize values from the CLI. Set cliOptionsSpecified to true if the
  /// default CLI values have been overriden.
  void initDefaultValues(const GPUModel &gpuModel) override;

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

  Type getLhsElementalType() const override { return captures.lhsElementType; }
  Type getRhsElementalType() const override { return captures.rhsElementType; }
  Type getResElementalType() const override {
    return captures.outputElementType;
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
        /*elementalBitWidth=*/lhsElementalBitWidth());
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
        /*elementalBitWidth=*/rhsElementalBitWidth());
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
        /*elementalBitWidth=*/resElementalBitWidth());
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

  void print(llvm::raw_ostream &os) const override;
  LLVM_DUMP_METHOD void dump() const override;
};

/// An extension of the matmul strategy to batched matrix multiplications.
class BatchMatmulStrategy : public MatmulStrategy {
public:
  /// Construct the default strategy, pulling options from the command-line
  /// arguments if provided and using the defaults otherwise.
  BatchMatmulStrategy(MLIRContext *context, const GPUModel &gpuModel,
                      const transform_ext::MatchedMatmulCaptures &captures)
      : MatmulStrategy(context, captures, gpuModel) {
    initDefaultValues(gpuModel);
  }

  /// Initialize the default values of the strategy.
  void initDefaultValues(const GPUModel &gpuModel) override {
    // First, initialize as if this was a simple matmul.
    MatmulStrategy::initDefaultValues(gpuModel);

    // Make sure we pad along all dimensions.
    paddingDimensions = {0, 1, 2, 3};
    packingDimensions = {1, 1, 1, 1};
  }

  /// Check that the strategy is valid for the captures and the model.
  LogicalResult validate(const GPUModel &gpuModel) const override;

  /// Named accessors to shapes.
  int64_t batch() const { return captures.matmulOpSizes[0]; }
  int64_t m() const override { return captures.matmulOpSizes[1]; }
  int64_t n() const override { return captures.matmulOpSizes[2]; }
  int64_t k() const override { return captures.matmulOpSizes[3]; }

  /// Named accessors to block tile sizes associated with shapes.
  int64_t blockTileBatch() const { return blockTileSizes[0]; }
  int64_t blockTileM() const override { return blockTileSizes[1]; }
  int64_t blockTileN() const override { return blockTileSizes[2]; }

  /// Number of threads to use.
  int64_t numThreadsX() const { return numThreads[0]; }
  int64_t numThreadsY() const { return numThreads[1]; }
  int64_t numThreadsZ() const { return numThreads[2]; }

  /// Number of warps to use.
  int64_t numWarpsX() const override { return numWarps[0]; }
  int64_t numWarpsY() const override { return numWarps[1]; }
  int64_t numWarpsZ() const { return numWarps[2]; }

  MappingInfo getBlockMapping() const override {
    return MappingInfo{
        /*numThreads=*/
        {},
        /*tileSizes=*/{blockTileBatch(), blockTileM(), blockTileN()},
        /*threadMapping=*/{blockZ(ctx), blockY(ctx), blockX(ctx)},
        /*vectorSize=*/std::nullopt};
  }

  // LHS copy is batch x M x K.
  MappingInfo lhsCopyMapping() const override {
    // TODO: generalize to transpositions, here and below.
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(), k(),
        {blockTileBatch(), blockTileM(), reductionTileSize},
        /*favorPredication=*/false,
        captures.lhsElementType.getIntOrFloatBitWidth());
  }

  // RHS copy is batch x K x N.
  MappingInfo rhsCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(), n(),
        {blockTileBatch(), reductionTileSize, blockTileN()},
        /*favorPredication=*/false,
        captures.rhsElementType.getIntOrFloatBitWidth());
  }

  // RES copy is batch x M x N.
  MappingInfo resCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(), n(),
        {blockTileBatch(), blockTileM(), blockTileN()},
        /*favorPredication=*/false,
        captures.outputElementType.getIntOrFloatBitWidth());
  }

  /// Check that the mapping computed for a copy is valid.
  LogicalResult validateLhsCopyMapping() const override {
    return validateCopyMapping(ctx, lhsCopyMapping(), "lhs");
  }
  LogicalResult validateRhsCopyMapping() const override {
    return validateCopyMapping(ctx, rhsCopyMapping(), "rhs");
  }
  LogicalResult validateResCopyMapping() const override {
    return validateCopyMapping(ctx, resCopyMapping(), "result");
  }

  // Compute is of the size batch x M x N.
  MappingInfo computeMapping() const override {
    assert(useFma && "only fma is currently supported");
    return MappingInfo{{numThreadsZ(), numThreadsY(), numThreadsX()},
                       {},
                       {threadZ(ctx), threadY(ctx), threadX(ctx)},
                       std::nullopt};
  }
};

} // namespace mlir::iree_compiler::gpu

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
