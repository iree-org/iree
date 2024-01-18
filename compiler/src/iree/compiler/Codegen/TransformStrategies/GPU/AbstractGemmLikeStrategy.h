// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/MappingInfo.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir::iree_compiler::gpu {

struct AbstractGemmLikeStrategy : GPUStrategy {
  AbstractGemmLikeStrategy(const GPUModel &gpuModel) : GPUStrategy(gpuModel) {}

  virtual ~AbstractGemmLikeStrategy();

  //===--------------------------------------------------------------------===//
  // Helpers and parameters for configuring the strategy.
  //===--------------------------------------------------------------------===//

  /// Initialize values from the CLI. Set cliOptionsSpecified to true if the
  /// default CLI values have been overriden.
  virtual void initDefaultValues(const GPUModel &gpuModel);

  /// Encodes whether the user has specified any CLI options. When true, the
  /// strategy should just run what was specified and is not allowed to
  /// override the user's choices.
  bool cliOptionsSpecified = false;

  /// Non-default subgroup size to use configured based on hardware supported
  /// values.
  std::optional<int64_t> targetSubgroupSize = std::nullopt;

  int64_t getSubgroupSize() const {
    return targetSubgroupSize ? *targetSubgroupSize : subgroupSize;
  }

  //===--------------------------------------------------------------------===//
  // Parameters that control the tiling and mapping.
  //===--------------------------------------------------------------------===//

  /// Tile sizes for the workgroup / determines grid size for all known
  /// reduction strategies. The initial values are set by initDefaultValues();
  SmallVector<int64_t> blockTileSizes;
  int64_t reductionTileSize;
  SmallVector<int64_t> numThreads;
  SmallVector<int64_t> numWarps;
  virtual int64_t blockTileM() const = 0;
  virtual int64_t blockTileN() const = 0;

  virtual int64_t numWarpsX() const = 0;
  virtual int64_t numWarpsY() const = 0;

  virtual MappingInfo getBlockMapping() const = 0;

  /// Common values based on derived quantities.
  int64_t totalNumThreads() const {
    int64_t res = 1;
    for (auto v : numThreads)
      res *= v;
    return res;
  }

  int64_t totalNumWarps() const {
    int64_t res = 1;
    for (auto v : numWarps)
      res *= v;
    return res;
  }

  //===--------------------------------------------------------------------===//
  // Parameters that control copy/padding transfers from global to shared.
  //===--------------------------------------------------------------------===//
  SmallVector<Type> paddingValueTypes;
  SmallVector<int64_t> paddingDimensions;
  SmallVector<int64_t> packingDimensions;

  ArrayAttr getZeroPadAttrFromElementalTypes(OpBuilder &b) const;

  virtual Type getLhsElementalType() const = 0;
  virtual Type getRhsElementalType() const = 0;
  virtual Type getResElementalType() const = 0;

  int64_t lhsElementalBitWidth() const {
    return getLhsElementalType().getIntOrFloatBitWidth();
  }
  int64_t rhsElementalBitWidth() const {
    return getRhsElementalType().getIntOrFloatBitWidth();
  }
  int64_t resElementalBitWidth() const {
    return getResElementalType().getIntOrFloatBitWidth();
  }

  bool alignedLhs() const {
    return m() % blockTileM() == 0 && k() % reductionTileSize == 0;
  }
  bool alignedRhs() const {
    return n() % blockTileN() == 0 && k() % reductionTileSize == 0;
  }
  bool alignedRes() const {
    return m() % blockTileM() == 0 && n() % blockTileN() == 0;
  }

  virtual MappingInfo lhsCopyMapping() const = 0;
  virtual LogicalResult validateLhsCopyMapping() const = 0;
  virtual MappingInfo rhsCopyMapping() const = 0;
  virtual LogicalResult validateRhsCopyMapping() const = 0;
  virtual MappingInfo resCopyMapping() const = 0;
  virtual LogicalResult validateResCopyMapping() const = 0;

  /// Validates the mapping and emits a diagnostic on failure.
  LogicalResult validateCopyMapping(MLIRContext *ctx,
                                    const MappingInfo &mapping,
                                    StringRef name) const;

  //===--------------------------------------------------------------------===//
  // Parameters that control compute mapping decisions.
  //===--------------------------------------------------------------------===//
  bool useAsyncCopies;
  bool useMmaSync;
  bool useWmma;
  bool useFma;
  int64_t pipelineDepth;
  bool peelPipelineEpilogue;
  virtual MappingInfo computeMapping() const = 0;

  virtual LogicalResult validate(const GPUModel &gpuModel) const;

  //===--------------------------------------------------------------------===//
  // Problem-related quantities.
  //===--------------------------------------------------------------------===//
  virtual int64_t m() const = 0;
  virtual int64_t n() const = 0;
  virtual int64_t k() const = 0;

  virtual void print(llvm::raw_ostream &os) const = 0;
  virtual LLVM_DUMP_METHOD void dump() const = 0;

  //===--------------------------------------------------------------------===//
  // Preconditions of internal transforms lifted to the top-level for more
  // actionnable error messages. In the fullness of time, transforms should
  // expose preconditions and we should aggregate them automatically.
  //===--------------------------------------------------------------------===//

  // TODO: To handle different element types efficiently, it would be much
  // better to expose the unrolling to native size explicitly to the transforms
  // rather than hide it behind an opaque transform.

  // wmma preconditions that we want to lift out in an actionnable top-level
  // error message instead of failing late in the transformation schedule.
  // TODO: These are now hardcoded for f32 but are element-type dependent.
  // Precondition: the pipeline transformation for wmma requires at least 2
  // k-groups.
  constexpr static int64_t kMinWmmaMinM = 16;
  constexpr static int64_t kMinWmmaMinN = 16;
  constexpr static int64_t kMinWmmaMinK = 8;

  // mma.sync preconditions that we want to lift out in an actionnable top-level
  // error message instead of failing late in the transformation schedule.
  // TODO: These are now hardcoded for f32 but are element-type dependent.
  // Precondition: the pipeline transformation for mma.sync requires at least 2
  // k-groups.
  constexpr static int64_t kMinMmaSyncGroups = 2;
  // Precondition: the pipeline transformation for mma.sync requires at least a
  // pipeline depth of 3.
  constexpr static int64_t kMinMmaSyncPipelineDepth = 3;
  // Precondition: if mma.sync is used, the tile sizes must be at least 8x8x4.
  constexpr static int64_t kMinMmaSyncMinM = 8;
  constexpr static int64_t kMinMmaSyncMinN = 8;
  constexpr static int64_t kMinMmaSyncMinK = 4;
};

} // namespace mlir::iree_compiler::gpu

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_
