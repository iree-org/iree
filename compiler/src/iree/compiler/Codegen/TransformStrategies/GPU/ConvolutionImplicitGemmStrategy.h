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

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace iree_compiler {
namespace gpu {

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

  int64_t m() const override { return derivedM; }
  int64_t n() const override { return derivedN; }
  int64_t k() const override { return derivedK; }

  int64_t blockTileM() const override {
    assert(blockTileSizes.size() >= 3 && "need at least 3 tile sizes");
    return blockTileSizes[0];
  }
  int64_t blockTileN() const override {
    assert(blockTileSizes.size() >= 3 && "need at least 3 tile sizes");
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
        /*tileSizes=*/{blockTileSizes[2], blockTileM(), blockTileN()},
        /*threadMapping=*/{blockZ(ctx), blockY(ctx), blockX(ctx)}};
  }

  // LHS copy is of size mxk.
  MappingInfo lhsCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/k(),
        /*copySizes=*/
        filterLHS ? ArrayRef<int64_t>{blockTileM(), reductionTileSize}
                  : ArrayRef<int64_t>{1, blockTileM(), reductionTileSize},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/lhsElementalBitWidth());
  }
  LogicalResult validateLhsCopyMapping() const override {
    MappingInfo mapping = lhsCopyMapping();
    // It is fine to use fewer threads to copy the LHS.
    int64_t mappingThreadCount = 1;
    for (auto numThread : mapping.numThreads)
      mappingThreadCount *= numThread;
    if (totalNumThreads() < mappingThreadCount) {
      llvm::errs() << "too many threads used for transferring lhs: "
                   << mappingThreadCount << " > " << totalNumThreads() << "\n";
      return failure();
    }
    return success();
  }

  // RHS copy is of size kxn.
  MappingInfo rhsCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/n(),
        /*copySizes=*/
        filterLHS ? ArrayRef<int64_t>{1, reductionTileSize, blockTileN()}
                  : ArrayRef<int64_t>{reductionTileSize, blockTileN()},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/rhsElementalBitWidth());
  }
  LogicalResult validateRhsCopyMapping() const override {
    MappingInfo mapping = rhsCopyMapping();
    // It is fine to use fewer threads to copy the RHS.
    int64_t mappingThreadCount = 1;
    for (auto numThread : mapping.numThreads)
      mappingThreadCount *= numThread;
    if (totalNumThreads() < mappingThreadCount) {
      llvm::errs() << "too many threads used for transferring rhs: "
                   << mappingThreadCount << " > " << totalNumThreads() << "\n";
      return failure();
    }
    return success();
  }

  // RES copy is of size mxn.
  MappingInfo resCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/n(),
        /*copySizes=*/{1, blockTileM(), blockTileN()},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/resElementalBitWidth());
  }
  LogicalResult validateResCopyMapping() const override {
    MappingInfo mapping = resCopyMapping();
    // It is fine to use fewer threads to copy the RES.
    int64_t mappingThreadCount = 1;
    for (auto numThread : mapping.numThreads)
      mappingThreadCount *= numThread;
    if (totalNumThreads() < mappingThreadCount) {
      llvm::errs() << "too many threads used for transferring res: "
                   << mappingThreadCount << " > " << totalNumThreads() << "\n";
      return failure();
    }
    return success();
  }

  // COMPUTE is of size mxn.
  MappingInfo computeMapping() const override {
    return MappingInfo{/*numThreads=*/{0, numWarpsY(), numWarpsX()},
                       /*tileSizes=*/{},
                       /*threadMapping=*/{warpY(ctx), warpX(ctx)},
                       /*vectorSize=*/std::nullopt};
  }

  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;

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

} // namespace gpu
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_
