// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"

#include <tuple>

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/ConvolutionImplicitGemmStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/MatmulTensorCoreStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/PadStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/SmallReductionStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/StagedReductionStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/MathExtras.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << '[' << DEBUG_TYPE << "] " << X)

llvm::cl::opt<bool> clGPUEnableTransformDialectMatmulTensorCoreStrategy(
    "iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy",
    llvm::cl::desc("activate the matmul tensorcore strategy"),
    llvm::cl::init(true));
llvm::cl::opt<bool> clGPUEnableTransformDialectImplicitGemmStrategy(
    "iree-codegen-llvmgpu-enable-transform-dialect-implicit-gemm-strategy",
    llvm::cl::desc("activate the convolution implicit gemm strategy"),
    llvm::cl::init(false));
llvm::cl::opt<bool> clGPUEnableTransformDialectAlignedMatmul(
    "iree-codegen-llvmgpu-enable-transform-dialect-aligned-matmul",
    llvm::cl::desc(
        "activate the matmul tensorcore strategy for tile aligned shapes"),
    llvm::cl::init(false));
llvm::cl::opt<bool> clGPUEnableTransformDialectSmallMatmul(
    "iree-codegen-llvmgpu-enable-transform-dialect-small-matmul",
    llvm::cl::desc("activate the matmul tensorcore strategy for small shapes "
                   "(< 16) in at least a dimension"),
    llvm::cl::init(false));
llvm::cl::opt<bool> clGPUEnableTransformDialectPadStrategy(
    "iree-codegen-llvmgpu-enable-transform-dialect-pad-strategy",
    llvm::cl::desc("activate the pad strategy"), llvm::cl::init(false));
llvm::cl::opt<bool> clGPUEnableTransformDialectBatchMatmulStrategy(
    "iree-codegen-llvmgpu-enable-transform-dialect-batch-matmul-strategy",
    llvm::cl::desc("activate the batch matmul strategy, additional "
                   "configuration flags are shared with matmul"),
    llvm::cl::init(false));

// TODO: significantly better namespacing.
using iree_compiler::gpu::AbstractGemmLikeStrategy;
using iree_compiler::gpu::BatchMatmulStrategy;
using iree_compiler::gpu::GPUModel;
using iree_compiler::gpu::ImplicitGemmStrategy;
using iree_compiler::gpu::kCudaMaxVectorLoadBitWidth;
using iree_compiler::gpu::MatmulStrategy;
using iree_compiler::gpu::PadConfig;
using iree_compiler::gpu::PadStrategy;
using iree_compiler::gpu::ReductionConfig;
using iree_compiler::gpu::ReductionStrategy;
using iree_compiler::gpu::scaleUpByBitWidth;
using iree_compiler::gpu::SmallReductionStrategy;
using iree_compiler::gpu::StagedReductionStrategy;
using transform_ext::CapturingOpMatcher;
using transform_ext::MatchCallbackOp;
using transform_ext::MatchedMatmulCaptures;
using transform_ext::MatchedPadCaptures;
using transform_ext::MatchedReductionCaptures;
using transform_ext::MatcherContext;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::StructuredOpMatcher;

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
// Reduction strategies.
//===--------------------------------------------------------------------===//
/// Placeholder to encode fixed reductions that should take finer-grained
/// precedence over other heuristics. In the future, this could be lifted to
/// e.g. `gpuModel` or higher up in some transform dialect database summary of
/// "known good things".
static FailureOr<ReductionConfig> applyKnownGoodReductionConfigurations(
    const transform_ext::MatchedReductionCaptures &captures,
    const GPUModel &gpuModel) {
  auto staged = ReductionStrategy::Staged;
  int64_t reductionSize = captures.reductionOpSizes.back();
  if (gpuModel.model == GPUModel::kDefaultGPU) {
    if (captures.reductionOutputElementalTypeBitWidth == 32) {
      if (reductionSize == 64)
        return ReductionConfig{/*maxNumThreads=*/64, /*vectorSize=*/1, staged};
      if (reductionSize == 128)
        return ReductionConfig{/*maxNumThreads=*/32, /*vectorSize=*/4, staged};
      if (reductionSize == 512)
        return ReductionConfig{/*maxNumThreads=*/256, /*vectorSize=*/2, staged};
    }
  }
  return failure();
}

/// The configurations below have been determined empirically by performing a
/// manual tradeoff between problem size, amount of parallelism and vector
/// size on a particular NVIDIA RTX2080Ti 12GB card. This is a coarse tradeoff
/// that should generally give reasonably good results but that begs to be
/// complemented by hardcoded known good configurations and ultimately a
/// database and/or a random forest compression of configurations with
/// guaranteed performance.
// TODO: Lift some of the strategy sizing logic as hints and/or heuristics to
// also work properly in the dynamic case.
// TODO: Support more HW configs and make it more pluggable.
static ReductionConfig
getReductionConfig(const transform_ext::MatchedReductionCaptures &captures,
                   const GPUModel &gpuModel) {
  auto maybeHardcodedConfiguration =
      applyKnownGoodReductionConfigurations(captures, gpuModel);
  if (succeeded(maybeHardcodedConfiguration))
    return *maybeHardcodedConfiguration;

  //===--------------------------------------------------------------------===//
  // Small reduction strategy.
  //===--------------------------------------------------------------------===//
  // Dynamic reductions are never supported by default because we can
  // never know offhand whether we are in a small-reduction regime mode.
  // Since this mode does not coalesce reads, perf will suffer
  // catastrophically on larger runtime reduction.
  // TODO: explicit hint from above that we really want to do that.
  int64_t redSize = captures.reductionOpSizes.back();
  bool isDynamicReduction = ShapedType::isDynamic(redSize);
  // Otherwise, still only support the small cases for now and fall back to
  // other strategies otherwise.
  bool isSmallReduction = (redSize < 2 * gpuModel.subgroupSize);
  if (!isDynamicReduction && isSmallReduction) {
    int64_t maxNumThreads = 4 * gpuModel.subgroupSize;
    return ReductionConfig{maxNumThreads, 0, ReductionStrategy::Small};
  }

  //===--------------------------------------------------------------------===//
  // Staged reduction strategy.
  //===--------------------------------------------------------------------===//
  int64_t bitWidth = captures.reductionOutputElementalTypeBitWidth;
  int64_t vectorSize = scaleUpByBitWidth(4, bitWidth);
  int64_t maxNumThreads = 8 * gpuModel.subgroupSize;
  // No adjustments in the dynamic case, we need extra information to make a
  // good decision.
  if (ShapedType::isDynamic(redSize))
    return ReductionConfig{maxNumThreads, vectorSize,
                           ReductionStrategy::Staged};
  // Scale down to smaller sizes (4, 8, 16)-warps.
  if (scaleUpByBitWidth(redSize, bitWidth) <= 4 * gpuModel.subgroupSize) {
    vectorSize = scaleUpByBitWidth(1, bitWidth);
    maxNumThreads = 4 * gpuModel.subgroupSize;
  } else if (scaleUpByBitWidth(redSize, bitWidth) <=
             8 * gpuModel.subgroupSize) {
    vectorSize = scaleUpByBitWidth(2, bitWidth);
    maxNumThreads = 4 * gpuModel.subgroupSize;
  } else if (scaleUpByBitWidth(redSize, bitWidth) <=
             8 * 2 * gpuModel.subgroupSize) {
    vectorSize = scaleUpByBitWidth(4, bitWidth);
    maxNumThreads = 4 * gpuModel.subgroupSize;
  }
  // Scale up to larger sizes (32, 64, 128+)-warps, using vector-4.
  if (!captures.trailingOpSizes.empty()) {
    if (scaleUpByBitWidth(redSize, bitWidth) >=
        128 * 4 * gpuModel.subgroupSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 32 * gpuModel.subgroupSize;
    } else if (scaleUpByBitWidth(redSize, bitWidth) >=
               64 * 4 * gpuModel.subgroupSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 16 * gpuModel.subgroupSize;
    } else if (scaleUpByBitWidth(redSize, bitWidth) >=
               32 * 4 * gpuModel.subgroupSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 8 * gpuModel.subgroupSize;
    } else if (scaleUpByBitWidth(redSize, bitWidth) >=
               16 * 4 * gpuModel.subgroupSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 4 * gpuModel.subgroupSize;
    }
  }
  return ReductionConfig{maxNumThreads, vectorSize, ReductionStrategy::Staged};
}

/// Map an N-D parallel, 1-D reduction operation with optional leading and
/// optional trailing elementwise operations.
/// The 1-D reduction dimension must be in the most minor dimension.
/// The innermost dimensions of the leading and trailing operations must be
/// most minor along all accesses. Return failure if matching fails. On a
/// successful match, configure a reduction strategy based on a proxy model of
/// the hardware and construct transform dialect IR that implements the
/// reduction strategy. The transform dialect IR is added in a top-level
/// ModuleOp after the `entryPoint` mlir::FunctionOpInterface.
static LogicalResult
matchAndSetReductionStrategy(mlir::FunctionOpInterface entryPoint,
                             linalg::LinalgOp op, const GPUModel &gpuModel) {
  if (!gpuModel.hasWarpShuffle) {
    LDBG("--Reduction strategy no warp shuffle\n");
    return failure();
  }

  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher *reduction;
  transform_ext::MatchedReductionCaptures captures;
  transform_ext::MatcherContext matcherContext;
  makeReductionMatcher(matcherContext, reduction, captures,
                       /*mustMatchEntireFunc=*/true);
  if (!matchPattern(op, *reduction)) {
    LDBG("--Reduction strategy failed to match\n");
    return failure();
  }

  // 2. Construct the configuration and the strategy builder.
  // TODO: Generalize along the HW axis.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    ReductionConfig reductionConfig = getReductionConfig(captures, gpuModel);
    if (reductionConfig.strategy == ReductionStrategy::Small) {
      SmallReductionStrategy strategy(captures, reductionConfig, gpuModel);
      return buildSmallReductionStrategy(b, variant, strategy);
    } else if (reductionConfig.strategy == ReductionStrategy::Staged) {
      // Otherwise, always fallback to the staged strategy.
      StagedReductionStrategy strategy(captures, reductionConfig, gpuModel);
      return buildStagedReductionStrategy(b, variant, strategy);
    } else {
      return llvm_unreachable("Unknown strategy");
    }
  };

  // 3. Build strategy embedded into the IR.
  mlir::iree_compiler::createTransformRegion(entryPoint, strategyBuilder);

  return success();
}

//===--------------------------------------------------------------------===//
// Matmul strategies.
//===--------------------------------------------------------------------===//
/// Placeholder to encode fixed matmuls that should take finer-grained
/// precedence over other heuristics. In the future, this could be lifted to
/// e.g. `gpuModel` or higher up in some transform dialect database summary of
/// "known good things".
static FailureOr<MatmulStrategy> applyKnownGoodMatmulConfigurations(
    const transform_ext::MatchedMatmulCaptures &captures,
    const GPUModel &gpuModel) {
  return failure();
}

static int64_t
selectLargestFailsafeValueIfNeeded(int64_t value, int64_t limit,
                                   ArrayRef<int64_t> thresholds,
                                   ArrayRef<int64_t> failSafeValues) {
  for (auto [threshold, failSafeValue] :
       llvm::zip(thresholds, failSafeValues)) {
    if (limit < threshold && value > failSafeValue)
      return failSafeValue;
  }
  return value;
}

static void failSafeOverrides(MatmulStrategy &strategy,
                              const GPUModel &gpuModel) {
  // Failsafe for blockTileM to avoid tiling by > size (i.e. no tiling).
  int64_t blockTileM = selectLargestFailsafeValueIfNeeded(
      /*value=*/strategy.blockTileM(),
      /*limit=*/strategy.m(),
      /*thresholds=*/{2, 4, 8, 16, 32, 64, 128},
      /*failSafeValues=*/{1, 2, 4, 8, 16, 32, 64});
  // Failsafe for blockTileN to avoid tiling by > size (i.e. no tiling).
  int64_t blockTileN = selectLargestFailsafeValueIfNeeded(
      /*value=*/strategy.blockTileN(),
      /*limit=*/strategy.n(),
      /*thresholds=*/{2, 4, 8, 16, 32, 64, 128},
      /*failSafeValues=*/{1, 2, 4, 8, 16, 32, 64});
  // Failsafe for reductionSize to avoid tiling by > size (i.e. no tiling).
  int64_t reductionTileSize = selectLargestFailsafeValueIfNeeded(
      /*value=*/strategy.reductionTileSize,
      /*limit=*/strategy.k(),
      /*thresholds=*/{2, 4, 8, 16, 24, 32, 40, 48, 56, 64},
      /*failSafeValues=*/{1, 2, 4, 8, 16, 24, 32, 40, 48, 56});

  // If some dimension is small, use fmas.
  // TODO: more parallelism by locally splitting the K-loop and reducing in the
  // fma case.
  if (blockTileM < 16 || blockTileN < 16 || reductionTileSize < 16) {
    strategy.useMmaSync = false;
    strategy.useWmma = false;
    strategy.useFma = true;
  }

  strategy.blockTileSizes = {blockTileM, blockTileN};
  strategy.reductionTileSize = reductionTileSize;

  // Avoid too deep pipelines. This should also look at shared memory usage in
  // the future.
  if (strategy.pipelineDepth * strategy.reductionTileSize > strategy.k()) {
    strategy.pipelineDepth =
        mlir::floorDiv(strategy.k(), strategy.reductionTileSize);
  }
}

/// The configurations below have been determined empirically.
// TODO: Significantly improve these heuristics.
static MatmulStrategy
getMatmulConfig(MLIRContext *context,
                const transform_ext::MatchedMatmulCaptures &captures,
                const GPUModel &gpuModel) {
  MatmulStrategy strategy(context, captures, gpuModel);
  if (strategy.cliOptionsSpecified)
    return strategy;

  auto maybeHardcodedConfiguration =
      applyKnownGoodMatmulConfigurations(captures, gpuModel);
  if (succeeded(maybeHardcodedConfiguration))
    return *maybeHardcodedConfiguration;

  // TODO: encode a decision tree of reasonnable heuristics here.

  // Apply failsafe overrides to avoid identified bad corner cases.
  failSafeOverrides(strategy, gpuModel);

  return strategy;
}

/// Update the strategy to make sure it can be consumed by the codegen. In
/// particular, make sure that tile sizes are smaller than the problem sizes to
/// actually trigger tiling and mapping to blocks and threads.
static void failSafeOverrides(BatchMatmulStrategy &strategy,
                              const GPUModel &gpuModel) {
  // Configure the strategy as if for a matmul.
  failSafeOverrides(static_cast<MatmulStrategy &>(strategy), gpuModel);

  // Failsafe for blockTileBatch to avoid tiling by > size (i.e. no tiling).
  int64_t blockTileBatch = selectLargestFailsafeValueIfNeeded(
      /*value=*/strategy.blockTileBatch(),
      /*limit=*/strategy.batch(),
      /*thresholds=*/{2, 4, 8, 16, 32, 64, 128},
      /*failSafeValues=*/{1, 2, 4, 8, 16, 32, 64});

  // Override the matmul configuration to be suitable for batch matmul.
  // Specifically, prepend the tile size for the batch dimension and force FMA.
  strategy.blockTileSizes.insert(strategy.blockTileSizes.begin(),
                                 blockTileBatch);

  strategy.useMmaSync = false;
  strategy.useWmma = false;
  strategy.useFma = true;
}

/// Produce a strategy for the batch matmul characterized by the given capture
/// list (shapes and types).
static BatchMatmulStrategy getBatchMatmulConfig(MLIRContext *context,
                                                MatchedMatmulCaptures &captures,
                                                const GPUModel &gpuModel) {
  // Command-line arguments trump everything.
  BatchMatmulStrategy strategy(context, gpuModel, captures);
  if (strategy.cliOptionsSpecified)
    return strategy;

  // TODO: fixed strategies and decision tree/heuristic.

  failSafeOverrides(strategy, gpuModel);
  return strategy;
}

/// Match the supported batch matmuls and set the transform dialect strategy for
/// them.
static LogicalResult
matchAndSetBatchMatmulStrategy(mlir::FunctionOpInterface entryPoint,
                               linalg::LinalgOp op, const GPUModel &gpuModel) {
  if (!clGPUEnableTransformDialectBatchMatmulStrategy) {
    LDBG("--Batch matmul strategy flag turned off\n");
    return failure();
  }

  StructuredOpMatcher *fill;
  StructuredOpMatcher *bmm;
  transform_ext::MatchedMatmulCaptures captures;
  transform_ext::MatcherContext matcherContext;
  transform_ext::makeBatchMatmulMatcher(matcherContext, bmm, fill, captures,
                                        /*mustMatchEntireFunc=*/true);
  if (!matchPattern(op, *bmm)) {
    LDBG("--Batch matmul strategy failed to match\n");
    return failure();
  }

  if (captures.contractionDims.batch.size() != 1 ||
      captures.contractionDims.m.size() != 1 ||
      captures.contractionDims.n.size() != 1 ||
      captures.contractionDims.k.size() != 1 || captures.batches()[0] != 0 ||
      captures.m() != 1 || captures.n() != 2 || captures.k() != 3) {
    LDBG("--Only support batch matmul with b, m, n, k iterator order atm\n");
    return failure();
  }

  BatchMatmulStrategy strategy =
      getBatchMatmulConfig(entryPoint->getContext(), captures, gpuModel);
  if (failed(strategy.validate(gpuModel))) {
    LDBG("--Batch matmul strategy failed to validate\n");
    return failure();
  }

  iree_compiler::createTransformRegion(entryPoint, [&](ImplicitLocOpBuilder &b,
                                                       Value variantH) {
    return iree_compiler::gpu::buildBatchMatmulStrategy(b, variantH, strategy);
  });
  return success();
}

static LogicalResult
matchAndSetMatmulStrategy(mlir::FunctionOpInterface entryPoint,
                          linalg::LinalgOp op, const GPUModel &gpuModel) {
  if (!clGPUEnableTransformDialectMatmulTensorCoreStrategy) {
    LDBG("--Matmul strategy flag turned off\n");
    return failure();
  }

  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher *fill;
  StructuredOpMatcher *matmul;
  StructuredOpMatcher *trailing;
  transform_ext::MatchedMatmulCaptures captures;
  transform_ext::MatcherContext matcherContext;
  makeMatmulMatcher(matcherContext, matmul, fill, trailing, captures,
                    /*mustMatchEntireFunc=*/true);
  if (!matchPattern(op, *matmul)) {
    LDBG("--Matmul strategy fail to match\n");
    return failure();
  }

  // We are very peculiar about the dispatches we want to match for now:
  //   - f32 only atm.
  //   - Mandatory fill op.
  //   - No trailing op.
  //   - If the matmul is "too aligned", then guard on the alignment flag.
  //   - If the matmul is "too small", then use the default IREE strategy.
  //   - Otherwise, we take it.
  if (!fill->getCaptured() || trailing->getCaptured()) {
    LDBG("--Matmul strategy fill / trailing preconditions failed\n");
    return failure();
  }

  // TODO: Generalize to a good mix of sizes, alignments and element types.
  const auto &matmulSize = captures.matmulOpSizes;
  if (matmulSize.size() != 3) {
    LDBG("--Matmul strategy size capture failed\n");
    return failure();
  }

  // Currently the unaligned transform strategy does not properly handle
  // degenerate dimensions that should have been rank-reduced (e.g. `1`).
  // Also, it is unprofitable to force small matmuls through a high latency
  // tensorcore path, we are better off with a simple simt strategy.
  // TODO: profitability details can be ironed out in the future when we have a
  // heuristic to better select strategy parameters.
  bool smallCases = (matmulSize[0] > 0 && matmulSize[0] < 16) ||
                    (matmulSize[1] > 0 && matmulSize[1] < 16) ||
                    (matmulSize[2] > 0 && matmulSize[2] < 16);
  if (smallCases && !clGPUEnableTransformDialectSmallMatmul) {
    LDBG("--Matmul strategy small size check failed\n");
    return failure();
  }

  // Currently the fully aligned case still lags behind the current default
  // pipeline and thus is guarded by a flag. This is the case when at least one
  // of the following holds
  //   - m is tile aligned (conservatively, take 64)
  //   - n is tile aligned (conservatively, take 64)
  //   - k is tile aligned (conservatively, take 16)
  bool guardedAlignedCases = matmulSize[0] % 64 == 0 ||
                             matmulSize[1] % 64 == 0 || matmulSize[2] % 16 == 0;

  if (!smallCases && guardedAlignedCases &&
      !clGPUEnableTransformDialectAlignedMatmul) {
    LDBG("--Matmul strategy alignment check failed\n");
    return failure();
  }

  iree_compiler::gpu::MatmulStrategy strategy =
      getMatmulConfig(op->getContext(), captures, gpuModel);
  LLVM_DEBUG(strategy.dump());

  // Validate the strategy configuration against the compilation target.
  if (failed(strategy.validate(gpuModel))) {
    LDBG("--Matmul strategy failed to validate\n");
    return failure();
  }

  // Limit the types that we choose to support without user intervention for
  // tensor core.
  if (!strategy.useFma && !strategy.cliOptionsSpecified &&
      (!captures.lhsElementType.isF32() || !captures.rhsElementType.isF32() ||
       !captures.outputElementType.isF32())) {
    LDBG("--Matmul strategy elemental type check failed\n");
    return failure();
  }

  // 2. Construct the configuration and the strategy builder.
  // TODO: Generalize along the HW axis.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return buildMatmulTensorCoreStrategy(b, variant, strategy);
  };

  // 3. Build strategy embedded into the IR.
  mlir::iree_compiler::createTransformRegion(entryPoint, strategyBuilder);

  return success();
}

//===--------------------------------------------------------------------===//
// Convolution strategies.
//===--------------------------------------------------------------------===//
/// Placeholder to encode fixed convolutions that should take finer-grained
/// precedence over other heuristics. In the future, this could be lifted to
/// e.g. `gpuModel` or higher up in some transform dialect database summary of
/// "known good things".
static FailureOr<ImplicitGemmStrategy> applyKnownGoodConvolutionConfigurations(
    const transform_ext::MatchedConvolutionCaptures &captures,
    const GPUModel &gpuModel) {
  return failure();
}

static void failSafeOverrides(ImplicitGemmStrategy &strategy,
                              const GPUModel &gpuModel) {
  // Prefer a default block tile of 1 for the batch.
  strategy.blockTileSizes = SmallVector<int64_t>{1, 128, 128};
  // Failsafe for blockTileM to avoid tiling by > size (i.e. no tiling).
  int64_t blockTileM = selectLargestFailsafeValueIfNeeded(
      strategy.blockTileM(), strategy.m(), {16, 32, 64, 128}, {1, 16, 32, 64});
  // Failsafe for blockTileN to avoid tiling by > size (i.e. no tiling).
  int64_t blockTileN = selectLargestFailsafeValueIfNeeded(
      strategy.blockTileN(), strategy.n(), {16, 32, 64, 128}, {1, 16, 32, 64});
  // Failsafe for reductionSize to avoid tiling by > size (i.e. no tiling).
  int64_t reductionTileSize = selectLargestFailsafeValueIfNeeded(
      strategy.reductionTileSize, strategy.k(), {8, 16, 24, 32, 40, 48, 56, 64},
      {1, 8, 16, 24, 32, 40, 48, 56});
  // Failsafe for blockTileBatch to avoid tiling by > size (i.e. no tiling).
  int64_t blockTileBatch = selectLargestFailsafeValueIfNeeded(
      /*value=*/strategy.blockTileBatch(),
      /*limit=*/strategy.batch(),
      /*thresholds=*/{2, 4, 8, 16, 32, 64, 128},
      /*failSafeValues=*/{1, 2, 4, 8, 16, 32, 64});
  strategy.blockTileSizes = {blockTileBatch, blockTileM, blockTileN};
  strategy.reductionTileSize = reductionTileSize;
  // Avoid too deep pipelines. This should also look at shared memory usage in
  // the future.
  if (strategy.pipelineDepth * strategy.reductionTileSize > strategy.k()) {
    strategy.pipelineDepth =
        mlir::floorDiv(strategy.k(), strategy.reductionTileSize);
  }
}

/// The configurations below have been determined empirically.
// TODO: Significantly improve these heuristics.
static ImplicitGemmStrategy
getConvolutionConfig(MLIRContext *context,
                     const transform_ext::MatchedConvolutionCaptures &captures,
                     const GPUModel &gpuModel) {
  ImplicitGemmStrategy strategy(context, captures, gpuModel);
  if (strategy.cliOptionsSpecified)
    return strategy;

  auto maybeHardcodedConfiguration =
      applyKnownGoodConvolutionConfigurations(captures, gpuModel);
  if (succeeded(maybeHardcodedConfiguration))
    return *maybeHardcodedConfiguration;

  // TODO: encode a decision tree of reasonnable heuristics here.

  // Apply failsafe overrides to avoid identified bad corner cases.
  failSafeOverrides(strategy, gpuModel);

  return strategy;
}

static LogicalResult
matchAndSetConvolutionStrategy(mlir::FunctionOpInterface entryPoint,
                               linalg::LinalgOp op, const GPUModel &gpuModel) {
  if (!clGPUEnableTransformDialectImplicitGemmStrategy) {
    LDBG("--Implicit gemm strategy flag turned off\n");
    return failure();
  }

  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher *fill;
  StructuredOpMatcher *convolution;
  StructuredOpMatcher *trailing;
  transform_ext::MatchedConvolutionCaptures captures;
  transform_ext::MatcherContext matcherContext;
  makeConvolutionMatcher(matcherContext, convolution, fill, trailing, captures,
                         /*mustMatchEntireFunc=*/true);
  if (!matchPattern(op, *convolution)) {
    LDBG("--Implicit gemm strategy fail to match\n");
    return failure();
  }

  // We are very peculiar about the dispatches we want to match for now:
  //   - f32 or f16 only atm.
  //   - Mandatory fill op.
  //   - Require minimum tile alignment due to img2col.
  //   - Otherwise, we take it.
  if (!fill->getCaptured() || trailing->getCaptured()) {
    LDBG("--Implicit gemm strategy fill / trailing preconditions failed\n");
    return failure();
  }

  // Currently requires a typical 2d named convolution (conv_2d_nchw/nhwc).
  if (captures.convolutionDims.outputChannel.size() != 1) {
    return failure();
  }
  if (captures.convolutionDims.inputChannel.size() != 1) {
    return failure();
  }
  if (captures.convolutionDims.outputImage.size() != 2) {
    return failure();
  }
  if (captures.convolutionDims.filterLoop.size() != 2) {
    return failure();
  }
  if (captures.convolutionDims.batch.size() != 1) {
    return failure();
  }

  int64_t channelSize = 1;
  for (auto dim : captures.convolutionDims.outputChannel)
    channelSize *= captures.convolutionOpSizes[dim];
  int64_t imageSize = 1;
  for (auto dim : captures.convolutionDims.outputImage)
    imageSize *= captures.convolutionOpSizes[dim];

  int64_t derivedK = 1;
  for (auto dim : captures.convolutionDims.filterLoop)
    derivedK *= captures.convolutionOpSizes[dim];
  for (auto dim : captures.convolutionDims.inputChannel)
    derivedK *= captures.convolutionOpSizes[dim];

  // Require tile-aligned due to the img2col op.
  if (channelSize % 64 || imageSize % 64 || derivedK % 16) {
    LDBG("--Implicit gemm strategy alignment check failed\n");
    return failure();
  }

  iree_compiler::gpu::ImplicitGemmStrategy strategy =
      getConvolutionConfig(op->getContext(), captures, gpuModel);

  // Validate the strategy configuration against the compilation target.
  if (failed(strategy.validate(gpuModel))) {
    LDBG("--Implicit gemm strategy failed to validate\n");
    return failure();
  }

  // 2. Construct the configuration and the strategy builder.
  // TODO: Generalize along the HW axis.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return buildConvolutionImplicitGemmStrategy(b, variant, strategy);
  };

  // 3. Build strategy embedded into the IR.
  mlir::iree_compiler::createTransformRegion(entryPoint, strategyBuilder);

  return success();
}

//===--------------------------------------------------------------------===//
// Pad strategies.
//===--------------------------------------------------------------------===//

/// Placeholder to encode fixed pads that should take finer-grained precedence
/// over other heuristics. In the future, this could be lifted to
/// e.g. `gpuModel` or higher up in some transform dialect database summary of
/// "known good things".
static FailureOr<PadConfig> applyKnownGoodPadConfigurations(
    const transform_ext::MatchedPadCaptures &captures,
    const GPUModel &gpuModel) {
  if (ArrayRef<int64_t>{captures.dims} == ArrayRef<int64_t>{1024, 1024}) {
    return PadConfig{};
  }
  return failure();
}

/// Placeholder to encode simple heuristics.
static PadConfig getPadConfig(const transform_ext::MatchedPadCaptures &captures,
                              const GPUModel &gpuModel) {
  auto maybeHardcodedConfiguration =
      applyKnownGoodPadConfigurations(captures, gpuModel);
  if (succeeded(maybeHardcodedConfiguration))
    return *maybeHardcodedConfiguration;
  return PadConfig{};
}

static LogicalResult
matchAndSetPadStrategy(mlir::FunctionOpInterface entryPoint, tensor::PadOp op,
                       const GPUModel &gpuModel) {
  if (!clGPUEnableTransformDialectPadStrategy) {
    LDBG("--Pad strategy flag turned off\n");
    return failure();
  }

  // 1. Match a padOp.
  CapturingOpMatcher *pad;
  MatchedPadCaptures captures;
  MatcherContext matcherContext;
  makePadMatcher(matcherContext, pad, captures, /*mustMatchEntireFunc=*/true);

  if (!matchPattern(op.getOperation(), *pad)) {
    LDBG("--Pad strategy failed to match\n");
    return failure();
  }
  if (captures.rank != 2) {
    LDBG("--Pad strategy supported ranks check failed\n");
    return failure();
  }
  if (!captures.elementType.isF32()) {
    LDBG("--Pad strategy elemental type check failed\n");
    return failure();
  }

  // 2. Construct the strategy builder.
  PadConfig padConfig = getPadConfig(captures, gpuModel);
  iree_compiler::gpu::PadStrategy strategy(op->getContext(), captures,
                                           padConfig, gpuModel);
  if (strategy.useAsyncCopies) {
    LDBG("--Async copies not supported yet\n");
    return failure();
  }
  if (strategy.numThreads.size() > 3) {
    LDBG("--Can only assign 3 num threads\n");
    return failure();
  }
  // Make sure all thread numbers are set.
  if (strategy.numThreads.size() != 3) {
    strategy.numThreads.resize(3, 1);
  }

  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return buildPadStrategy(b, variant, strategy);
  };

  // 3. Build strategy embedded into the IR.
  mlir::iree_compiler::createTransformRegion(entryPoint, strategyBuilder);

  return success();
}

//===--------------------------------------------------------------------===//
// Switch between strategies depending on matched IR.
//===--------------------------------------------------------------------===//
LogicalResult mlir::iree_compiler::gpu::matchAndSetTransformStrategy(
    mlir::FunctionOpInterface entryPoint, Operation *op,
    const GPUModel &gpuModel) {
  LDBG("Look up a TD strategy for entryPoint:\n" << entryPoint << "\n");
  auto padOp = dyn_cast<tensor::PadOp>(op);
  if (padOp) {
    if (succeeded(matchAndSetPadStrategy(entryPoint, padOp, gpuModel))) {
      LDBG("Activate pad strategy\n");
      return success();
    }
    LDBG("Unmatched pad strategy\n");
    return failure();
  }
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    LDBG("Not a Linalg op: " << *op << " -> Fail\n");
    return failure();
  }
  if (succeeded(matchAndSetReductionStrategy(entryPoint, linalgOp, gpuModel))) {
    LDBG("Activate reduction strategy\n");
    return success();
  }
  if (succeeded(matchAndSetMatmulStrategy(entryPoint, linalgOp, gpuModel))) {
    LDBG("Activate matmul\n");
    return success();
  }
  if (succeeded(
          matchAndSetBatchMatmulStrategy(entryPoint, linalgOp, gpuModel))) {
    LDBG("Activate batch matmul\n");
    return success();
  }
  if (succeeded(
          matchAndSetConvolutionStrategy(entryPoint, linalgOp, gpuModel))) {
    LDBG("Activate convolution\n");
    return success();
  }
  // TODO: Add more transform dialect strategy for other kind of dispatch
  // regions.
  LDBG("No suitable strategy found\n");
  return failure();
}
