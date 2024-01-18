// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/AbstractGemmLikeStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/CopyMapping.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::gpu {

struct GPUModel;

class ImplicitGemmStrategy : public AbstractGemmLikeStrategy {
public:
  ImplicitGemmStrategy(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures,
      const GPUModel &gpuModel)
      : AbstractGemmLikeStrategy(gpuModel), ctx(context), captures(captures) {
    initDefaultValues(gpuModel);
  }

  ImplicitGemmStrategy(const ImplicitGemmStrategy &) = default;
  ImplicitGemmStrategy &operator=(const ImplicitGemmStrategy &) = default;

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedConvolutionCaptures captures;

  /// Initialize values from the CLI. Set cliOptionsSpecified to true if the
  /// default CLI values have been overriden.
  void initDefaultValues(const GPUModel &gpuModel) override;

  LogicalResult validate(const GPUModel &gpuModel) const override;

  int64_t batch() const { return captures.convolutionOpSizes[0]; }
  int64_t m() const override { return derivedM; }
  int64_t n() const override { return derivedN; }
  int64_t k() const override { return derivedK; }

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

  Type getLhsElementalType() const override {
    return filterLHS ? captures.filterElementType : captures.inputElementType;
  }
  Type getRhsElementalType() const override {
    return filterLHS ? captures.inputElementType : captures.filterElementType;
  }
  Type getResElementalType() const override {
    return captures.outputElementType;
  }

  MappingInfo getBlockMapping() const override {
    // 2D named convolutions are always batched.
    return MappingInfo{
        /*numThreads=*/{},
        /*tileSizes=*/{blockTileBatch(), blockTileM(), blockTileN()},
        /*threadMapping=*/{blockZ(ctx), blockY(ctx), blockX(ctx)}};
  }

  // LHS copy is of size (batch) x M x K.
  MappingInfo lhsCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/k(),
        /*copySizes=*/
        filterLHS ? ArrayRef<int64_t>{blockTileM(), reductionTileSize}
                  : ArrayRef<int64_t>{blockTileBatch(), blockTileM(),
                                      reductionTileSize},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/lhsElementalBitWidth());
  }

  // RHS copy is of size (batch) x K x N.
  MappingInfo rhsCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/n(),
        /*copySizes=*/
        filterLHS ? ArrayRef<int64_t>{blockTileBatch(), reductionTileSize,
                                      blockTileN()}
                  : ArrayRef<int64_t>{reductionTileSize, blockTileN()},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/rhsElementalBitWidth());
  }

  // RES copy is of size batch x M x N.
  MappingInfo resCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/n(),
        /*copySizes=*/{blockTileBatch(), blockTileM(), blockTileN()},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/resElementalBitWidth());
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

  // COMPUTE is of size batch x M x N.
  MappingInfo computeMapping() const override {
    if (useFma) {
      return MappingInfo{
          /*numThreads=*/{numThreadsZ(), numThreadsY(), numThreadsX()},
          /*tileSizes=*/{},
          /*threadMapping=*/{threadZ(ctx), threadY(ctx), threadX(ctx)},
          /*vectorSize=*/std::nullopt};
    }
    return MappingInfo{/*numThreads=*/{numWarpsZ(), numWarpsY(), numWarpsX()},
                       /*tileSizes=*/{},
                       /*threadMapping=*/{warpZ(ctx), warpY(ctx), warpX(ctx)},
                       /*vectorSize=*/std::nullopt};
  }

  void print(llvm::raw_ostream &os) const override;
  LLVM_DUMP_METHOD void dump() const override;

private:
  // For NCHW convolutions, the filter will be the LHS of the GEMM.
  bool filterLHS = false;

  int64_t derivedM = 0;
  int64_t derivedN = 0;
  int64_t derivedK = 0;
};

void buildConvolutionImplicitGemmStrategy(ImplicitLocOpBuilder &b,
                                          Value variantH,
                                          const ImplicitGemmStrategy &strategy);

} // namespace mlir::iree_compiler::gpu

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_
